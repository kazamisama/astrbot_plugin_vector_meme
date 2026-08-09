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
