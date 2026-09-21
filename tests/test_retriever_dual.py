"""DualRetriever 回归测试。

背景：v0.8.0 给 `MemeRetriever.pick()/pick_multiple()` 加了 `query_vector` 直传，
但 `DualRetriever.retrieve()` 漏了同名参数。api 后端恒走 DualRetriever，于是每次
自动表情都抛 `TypeError: DualRetriever.retrieve() got an unexpected keyword
argument 'query_vector'`，被 main.py 的兜底 except 吞掉——内容检索 / 反重复 /
加权采样整条路径静默失效。这里把调用形状钉死。
"""
from __future__ import annotations

from core.database import MemeDatabase
from core.embedder import DummyEmbedder
from core.retriever_dual import DualRetriever

CAPTIONS = (
    ("happy", "开心大笑"),
    ("happy", "平静微笑"),
    ("sad", "委屈掉泪"),
)


def _make_caption_db(tmp_path, dim: int = 16):
    """造一个 api 后端形状的库：vector_id 全 -1，只有 caption_vector_id。"""
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=dim)
    emb = DummyEmbedder(dim=dim, seed=11)
    ids: dict[str, int] = {}
    for i, (tag, caption) in enumerate(CAPTIONS):
        path = tmp_path / f"{i}.png"
        mid = db.upsert_meme(
            file_path=str(path),
            file_hash=f"h{i}",
            tag=tag,
            vector_id=-1,
            file_name=path.name,
        )
        cvid = int(db.add_vectors(emb.embed_text(caption))[0])
        db.set_meme_caption(mid, caption, cvid)
        ids[caption] = mid
    db.save_index()
    return db, emb, ids


def test_dual_retrieve_accepts_query_vector(tmp_path):
    db, emb, ids = _make_caption_db(tmp_path)
    retriever = DualRetriever(db, emb, caption_weight=1.0, candidate_pool_size=8)

    result = retriever.retrieve(
        text="这句话和 caption 毫无关系",
        tag="happy",
        topk=4,
        query_vector=emb.embed_text("开心大笑"),
    )
    assert result.hits
    assert result.hits[0].meme_id == ids["开心大笑"]


def test_dual_pick_accepts_query_vector(tmp_path):
    """main.py 的调用形状：pick(..., selection_pool_size=..., query_vector=qvec)。"""
    db, emb, ids = _make_caption_db(tmp_path)
    retriever = DualRetriever(
        db, emb, caption_weight=1.0, candidate_pool_size=4, random_jitter=0.0
    )

    hit = retriever.pick(
        text="某句话",
        tag="happy",
        selection_pool_size=4,
        stochastic=False,
        query_vector=emb.embed_text("开心大笑"),
    )
    assert hit is not None
    assert hit.meme_id == ids["开心大笑"]
    assert db.get_meme(hit.meme_id)["usage_count"] == 1


def test_dual_pick_multiple_accepts_query_vector(tmp_path):
    db, emb, ids = _make_caption_db(tmp_path)
    retriever = DualRetriever(
        db, emb, caption_weight=1.0, candidate_pool_size=4, random_jitter=0.0
    )

    hits = retriever.pick_multiple(
        text="某句话",
        tag="happy",
        n=2,
        stochastic=False,
        query_vector=emb.embed_text("开心大笑"),
    )
    assert len(hits) == 2
    assert {h.meme_id for h in hits} == {ids["开心大笑"], ids["平静微笑"]}


def test_dual_retrieve_without_query_vector_still_embeds_text(tmp_path):
    """不传 query_vector 时保持原行为（内部自己编码 query 文本）。"""
    db, emb, _ = _make_caption_db(tmp_path)
    retriever = DualRetriever(db, emb, caption_weight=1.0, candidate_pool_size=8)

    result = retriever.retrieve(text="开心大笑", tag="happy", topk=4)
    assert result.hits
    assert result.query_text == "开心大笑"
