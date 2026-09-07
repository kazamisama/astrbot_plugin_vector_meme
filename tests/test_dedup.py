import numpy as np
import pytest

from core.database import MemeDatabase
from core.dedup import DEDUP_TAG_PREFIX, apply_dedup, cluster_near_duplicates, undo_dedup
from core.embedder import DummyEmbedder


def _build(tmp_path, n_dup=2, n_other=1):
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    emb = DummyEmbedder(16, seed=61)
    vec_a = emb.embed_text("same meme")
    vec_b = emb.embed_text("other meme")
    vecs = np.stack([vec_a] * n_dup + [vec_b] * n_other)
    vids = db.add_vectors(vecs)
    ids = []
    for i, vid in enumerate(vids):
        tag = "dup" if i < n_dup else "other"
        ids.append(db.upsert_meme(str(tmp_path / f"{i}.png"), f"h{i}", tag, int(vid)))
    return db, ids


def test_cluster_finds_near_duplicate_group(tmp_path):
    db, ids = _build(tmp_path)
    groups = cluster_near_duplicates(db, threshold=0.95, min_group=2)
    assert len(groups) >= 1
    big = max(groups, key=lambda g: len(g.member_ids))
    assert sorted(big.member_ids) == [ids[0], ids[1]]


def test_apply_dedup_disables_dups_and_undo_restores(tmp_path):
    db, ids = _build(tmp_path)
    stats = apply_dedup(db, threshold=0.95, min_group=2, dry_run=True)
    assert stats["dry_run"]
    assert len(stats["disabled"]) == 1
    # dry-run 不修改
    active = [r["id"] for r in db.list_memes(limit=100)]
    assert len(active) == 3

    stats2 = apply_dedup(db, threshold=0.95, min_group=2, dry_run=False)
    assert len(stats2["disabled"]) == 1
    dup_id = stats2["disabled"][0]
    row = db.get_meme(dup_id)
    assert row["disabled"] == 1
    assert DEDUP_TAG_PREFIX in (row["sub_tags"] or "")

    # 去重后被禁用的图不再出现在候选
    assert int(row["vector_id"]) not in [v for v, _ in db.list_candidate_vector_ids()]

    res = undo_dedup(db)
    assert res["count"] == 1
    assert db.get_meme(dup_id)["disabled"] == 0
    assert DEDUP_TAG_PREFIX not in (db.get_meme(dup_id)["sub_tags"] or "")
