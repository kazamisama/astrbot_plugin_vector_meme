import numpy as np
import pytest

from core.database import MemeDatabase
from core.embedder import DummyEmbedder
from core.retriever import MemeRetriever
from core.retriever_dual import DualRetriever


@pytest.fixture()
def library(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=7)
    vecs = np.stack(
        [
            emb.embed_text("happy cat"),
            emb.embed_text("angry dog"),
            emb.embed_text("sleepy bird"),
        ]
    )
    vids = db.add_vectors(vecs)
    m1 = db.upsert_meme(str(tmp_path / "happy.png"), "h1", "happy", int(vids[0]))
    db.upsert_meme(str(tmp_path / "angry.png"), "h2", "angry", int(vids[1]))
    db.upsert_meme(str(tmp_path / "sleepy.png"), "h3", "sleepy", int(vids[2]))
    return db, emb, m1


def test_retrieve_hit_and_tag_filter(library):
    db, emb, m1 = library
    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)
    res = retriever.retrieve("happy cat", tag="happy", topk=2)
    assert res.hits
    assert res.hits[0].meme_id == m1


def test_retrieve_fallback_when_tag_missing(library):
    db, emb, _ = library
    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)
    res = retriever.retrieve("happy cat", tag="not_exist", topk=1)
    assert res.used_fallback
    assert len(res.hits) == 1


def test_dual_retriever_caption_only_applies_rerank(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=11)
    cap_ids = db.add_vectors(np.stack([emb.embed_text("smiling cat")]))
    mid = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", -1)
    db.set_meme_caption(mid, "smiling cat", int(cap_ids[0]))
    db.mark_used(mid)
    retriever = DualRetriever(db, emb, caption_weight=1.0, candidate_pool_size=2, random_jitter=0.0)
    res = retriever.retrieve("smiling cat", tag="happy", topk=1)
    assert res.hits
    assert res.hits[0].meme_id == mid
    assert res.hits[0].repeat_penalty > 0


def test_dual_retriever_uses_caption_path(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=9)
    img_ids = db.add_vectors(np.stack([emb.embed_text("cat"), emb.embed_text("dog")]))
    m1 = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", int(img_ids[0]))
    db.upsert_meme(str(tmp_path / "b.png"), "b", "happy", int(img_ids[1]))
    cap_ids = db.add_vectors(
        np.stack([emb.embed_text("smiling cat"), emb.embed_text("sleepy dog")])
    )
    db.set_meme_caption(m1, "smiling cat", int(cap_ids[0]))
    db.set_meme_caption(
        db.get_meme_by_path(str(tmp_path / "b.png"))["id"],
        "sleepy dog",
        int(cap_ids[1]),
    )
    retriever = DualRetriever(
        db,
        emb,
        caption_weight=0.8,
        candidate_pool_size=2,
        random_jitter=0.0,
    )
    res = retriever.retrieve("smiling cat", tag="happy", topk=2)
    assert res.hits
    assert res.hits[0].meme_id == m1


def test_retrieve_rerank_false_is_deterministic(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=3)
    base = emb.embed_text("happy")
    arr = np.repeat(base[None, :], 6, axis=0)
    vids = db.add_vectors(arr)
    for i, vid in enumerate(vids):
        db.upsert_meme(str(tmp_path / f"{i}.png"), str(i), "happy", int(vid))

    retriever = MemeRetriever(
        db, emb, anti_repeat_window=0, candidate_pool_size=6, random_jitter=0.1
    )
    choices = []
    for _ in range(10):
        result = retriever.retrieve(
            "happy", tag="happy", topk=6, anti_repeat=False, rerank=False
        )
        hit = max(result.hits, key=lambda h: h.raw_similarity)
        choices.append(hit.meme_id)
    assert len(set(choices)) == 1

def test_retrieve_small_pool_expansion(tmp_path):
    """小 tag 候选不足时扩展到相关 tag；semantic 更接近的跨 tag 图片可胜出。"""
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=21)
    v_tiny = emb.embed_text("alpha cat")
    v_beta = emb.embed_text("beta dog")
    v_gamma = emb.embed_text("gamma bird")
    vids = db.add_vectors(np.stack([v_tiny, v_beta, v_gamma]))
    tiny = db.upsert_meme(str(tmp_path / "tiny.png"), "ht", "tiny", int(vids[0]))
    big1 = db.upsert_meme(str(tmp_path / "big1.png"), "hb1", "big", int(vids[1]))
    db.upsert_meme(str(tmp_path / "big2.png"), "hb2", "big", int(vids[2]))

    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)
    # 扩展模式下：查询精确命中 big 的向量，即使 tiny 是目标 tag
    res = retriever.retrieve("beta dog", tag="tiny", topk=3, anti_repeat=False)
    assert res.expanded_tags and "big" in res.expanded_tags
    assert res.hits[0].meme_id == big1

    # 关闭扩展：tiny 只有一个候选，且不会引入 big
    res2 = retriever.retrieve("beta dog", tag="tiny", topk=3, anti_repeat=False, expand_pool=False)
    assert res2.expanded_tags is None
    assert [h.meme_id for h in res2.hits] == [tiny]


def test_diversify_orders_near_duplicates_apart(tmp_path):
    """MMR：与已选候选近重复的图被压后，让真正的不同图进入前列。"""
    from core.retriever import MemeHit

    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=31)
    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)
    rng = np.random.RandomState(0)
    v_a = rng.randn(16); v_a /= np.linalg.norm(v_a)
    v_b = v_a.copy()  # 完全相同
    v_c = rng.randn(16); v_c /= np.linalg.norm(v_c)
    hits = [
        MemeHit(1, "a.png", "x", similarity=0.90),
        MemeHit(2, "b.png", "x", similarity=0.80),
        MemeHit(3, "c.png", "x", similarity=0.70),
    ]
    vecs = {1: v_a.astype("float32"), 2: v_b.astype("float32"), 3: v_c.astype("float32")}
    out = retriever._diversify(hits, vecs)
    assert out[0].meme_id == 1
    assert out[1].meme_id == 3  # 与 1 近重复的 2 被抑制，不同的 3 上位


def test_pick_hard_exclude_recent(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=41)
    vq = emb.embed_text("query text")
    v_oth = emb.embed_text("other text")
    vids = db.add_vectors(np.stack([vq, v_oth]))
    m1 = db.upsert_meme(str(tmp_path / "a.png"), "ha", "happy", int(vids[0]))
    m2 = db.upsert_meme(str(tmp_path / "b.png"), "hb", "happy", int(vids[1]))
    db.mark_used(m1)

    retriever = MemeRetriever(
        db, emb, anti_repeat_window=5, candidate_pool_size=2,
        random_jitter=0.0, hard_exclude_recent=True,
    )
    hit = retriever.pick("query text", tag="happy", stochastic=False)
    assert hit is not None and hit.meme_id == m2

    # 关闭硬排除 → 回到最高分（m1）
    retriever2 = MemeRetriever(
        db, emb, anti_repeat_window=5, candidate_pool_size=2,
        random_jitter=0.0, hard_exclude_recent=False,
    )
    hit2 = retriever2.pick("query text", tag="happy", stochastic=False)
    assert hit2.meme_id == m1


def test_pick_accepts_query_vector(tmp_path):
    """外部传入 query_vector 时跳过 embed_text，语义由调用方控制。"""
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=51)
    v1 = emb.embed_text("cat")
    v2 = emb.embed_text("dog")
    vids = db.add_vectors(np.stack([v1, v2]))
    m1 = db.upsert_meme(str(tmp_path / "a.png"), "ha", "happy", int(vids[0]))
    m2 = db.upsert_meme(str(tmp_path / "b.png"), "hb", "happy", int(vids[1]))

    retriever = MemeRetriever(db, emb, candidate_pool_size=2, random_jitter=0.0)
    hit = retriever.pick(
        "完全不相关的一句话",
        tag="happy",
        stochastic=False,
        query_vector=v2,
    )
    assert hit.meme_id == m2
    _ = m1

