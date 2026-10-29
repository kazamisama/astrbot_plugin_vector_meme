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


def test_dual_retriever_skips_caption_when_disabled_or_empty(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=71)
    vids = db.add_vectors(np.stack([emb.embed_text("happy cat")]))
    mid = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", int(vids[0]))

    disabled = DualRetriever(
        db, emb, caption_weight=0.0, candidate_pool_size=2, random_jitter=0.0
    )
    base = disabled.retrieve("happy cat", tag="happy", topk=1)
    assert base.hits and base.hits[0].meme_id == mid

    empty_captions = DualRetriever(
        db, emb, caption_weight=0.6, candidate_pool_size=2, random_jitter=0.0
    )
    base2 = empty_captions.retrieve("happy cat", tag="happy", topk=1)
    assert base2.hits and base2.hits[0].meme_id == mid


def test_dual_retriever_caption_only_without_rerank(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=73)
    cap_ids = db.add_vectors(np.stack([emb.embed_text("smiling cat")]))
    mid = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", -1)
    db.set_meme_caption(mid, "smiling cat", int(cap_ids[0]))
    retriever = DualRetriever(
        db, emb, caption_weight=1.0, candidate_pool_size=2, random_jitter=0.0
    )

    result = retriever.retrieve("smiling cat", tag="happy", topk=1, rerank=False)

    assert result.hits
    assert result.hits[0].meme_id == mid


def test_dual_minmax_edges():
    assert DualRetriever._minmax({}) == {}
    assert DualRetriever._minmax({1: 0.5, 2: 0.5}) == {1: 1.0, 2: 1.0}


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


def test_weighted_pick_edges_and_pick_multiple(library):
    from core.retriever import MemeHit

    db, emb, _ = library
    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)

    assert retriever._weighted_pick([]) is None
    single = MemeHit(1, "a.png", "x", 0.5)
    assert retriever._weighted_pick([single]) is single

    retriever.weighted_sampling_power = 1.0
    picked_one = retriever._weighted_pick(
        [MemeHit(1, "a.png", "x", 0.2), MemeHit(2, "b.png", "x", 0.9)]
    )
    assert picked_one is not None

    picked = retriever.pick_multiple("happy cat", tag="happy", n=2, stochastic=False)
    assert len(picked) == 2
    assert len({hit.meme_id for hit in picked}) == 2
    assert db.get_meme(picked[0].meme_id)["usage_count"] >= 1

    exhausted = retriever.pick_multiple("happy cat", n=10, stochastic=False)
    assert len(exhausted) == 3


def test_classify_vector_and_classify_meme_branches(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=61)
    retriever = MemeRetriever(db, emb, candidate_pool_size=3, random_jitter=0.0)

    assert retriever.classify_vector(emb.embed_text("anything")) == []

    tags = ("alpha", "beta", "gamma")
    vids = db.add_vectors(np.stack([emb.embed_text(tag) for tag in tags]))
    ids = {}
    for i, (tag, vid) in enumerate(zip(tags, vids)):
        ids[tag] = db.upsert_meme(
            str(tmp_path / f"{i}.png"), str(i), tag, int(vid)
        )

    query = emb.embed_text("alpha")
    no_proto = retriever.classify_vector(
        query, topk=5, knn_k=1, prototype_min_samples=2
    )
    assert len(no_proto) == 3
    assert no_proto[0]["tag"] == "alpha"
    assert no_proto[0]["prototype_similarity"] is None
    assert no_proto[0]["knn_score"] is not None

    mixed = retriever.classify_vector(
        query, topk=2, knn_k=1, prototype_min_samples=1
    )
    assert len(mixed) == 2
    assert any(p["knn_score"] is None for p in mixed)

    predicted = retriever.classify_meme(ids["beta"], topk=3)
    assert predicted
    assert retriever.classify_meme(9999) == []

    no_vector = db.upsert_meme("nv.png", "hn", "misc", -1)
    assert retriever.classify_meme(no_vector) == []


def test_retriever_helper_guards_and_power_branch(tmp_path):
    from core.retriever import MemeHit

    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=5)
    retriever = MemeRetriever(db, emb, candidate_pool_size=2, random_jitter=0.0)

    assert retriever._search_meme_scores(np.ones(16), [], 3) == []
    assert retriever._load_meme_rows([]) == {}
    assert retriever._build_hits([(1, 0.5)], {}) == []
    retriever._log_search("q", None, [])
    assert retriever._vectors_by_meme_id([]) == {}
    assert retriever._diversify([], {}) == []
    assert retriever._rerank([]) == []

    vids = db.add_vectors(np.stack([emb.embed_text("a")]))
    db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", int(vids[0]))
    assert retriever._related_tags(np.ones(16), None, 0) == []
    assert retriever._related_tags(np.zeros(16), None, 2) == []

    db.active_vector_rows = lambda: (
        [{"tag": "zero"}],
        np.zeros((1, 16), dtype="float32"),
    )
    assert retriever._related_tags(np.ones(16), None, 2) == []

    h1 = MemeHit(1, "a.png", "x", 0.9)
    h2 = MemeHit(2, "b.png", "x", 0.8)
    diversified = retriever._diversify([h1, h2], {1: np.ones(16, dtype="float32")})
    assert [h.meme_id for h in diversified] == [1, 2]

    retriever.expand_small_pool = True
    retriever._related_tags = lambda *args, **kwargs: [("other", 0.5)]
    expanded, added = retriever._expand_candidates(
        np.ones(16),
        "tiny",
        [(1, 10)],
        lambda tag: [(1, 10), (2, 11)] if tag == "other" else [],
    )
    assert expanded == [(1, 10), (2, 11)]
    assert added == ["other"]

    retriever.weighted_sampling_power = 2.0
    assert retriever._weighted_pick(
        [MemeHit(1, "a.png", "x", 0.2), MemeHit(2, "b.png", "x", 0.9)]
    ) is not None
    assert retriever.pick_multiple("a", tag="happy", n=1, stochastic=True)


def test_dual_caption_candidate_and_empty_score_edges(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=77)
    cap_ids = db.add_vectors(np.stack([emb.embed_text("smiling")]))
    mid = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", -1)
    db.set_meme_caption(mid, "smiling", int(cap_ids[0]))
    retriever = DualRetriever(
        db, emb, caption_weight=1.0, candidate_pool_size=2, random_jitter=0.0
    )
    assert retriever._caption_candidates(None) == [(int(cap_ids[0]), mid)]
    retriever._search_caption_path = lambda *args, **kwargs: {}
    result = retriever.retrieve("smiling", tag="happy", topk=1)
    assert result.hits == []


def test_dual_fusion_without_rerank(tmp_path):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=79)
    img_ids = db.add_vectors(
        np.stack([emb.embed_text("cat"), emb.embed_text("dog")])
    )
    m1 = db.upsert_meme(str(tmp_path / "a.png"), "a", "happy", int(img_ids[0]))
    m2 = db.upsert_meme(str(tmp_path / "b.png"), "b", "happy", int(img_ids[1]))
    cap_ids = db.add_vectors(
        np.stack([emb.embed_text("smiling cat"), emb.embed_text("sleepy dog")])
    )
    db.set_meme_caption(m1, "smiling cat", int(cap_ids[0]))
    db.set_meme_caption(m2, "sleepy dog", int(cap_ids[1]))
    retriever = DualRetriever(
        db, emb, caption_weight=0.7, candidate_pool_size=2, random_jitter=0.0
    )
    result = retriever.retrieve("smiling cat", tag="happy", topk=2, rerank=False)
    assert {hit.meme_id for hit in result.hits} == {m1, m2}
    assert result.hits[0].meme_id == m1


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

