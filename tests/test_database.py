import json

import numpy as np
import pytest

from core.database import MemeDatabase


@pytest.fixture()
def db(tmp_path):
    return MemeDatabase(tmp_path / "memes.db", tmp_path / "memes.faiss", dim=8)


def test_upsert_and_hash_update(db):
    v = np.ones((1, 8), dtype="float32")
    vids = db.add_vectors(v)
    mid1 = db.upsert_meme("a.png", "h1", "happy", int(vids[0]), file_name="a.png")
    assert db.get_meme(mid1)["tag"] == "happy"

    vids2 = db.add_vectors(v * 2)
    mid2 = db.upsert_meme("a.png", "h2", "sad", int(vids2[0]), file_name="a.png")
    assert mid2 != mid1
    assert db.get_meme(mid1) is None
    assert db.get_meme(mid2)["file_hash"] == "h2"


def test_health_and_compact_index_removes_orphans(db):
    img_ids = db.add_vectors(np.ones((3, 8), dtype="float32"))
    cap_ids = db.add_vectors(np.full((2, 8), 2.0, dtype="float32"))
    m1 = db.upsert_meme("m1.png", "h1", "happy", int(img_ids[0]))
    m2 = db.upsert_meme("m2.png", "h2", "sad", int(img_ids[1]))
    m3 = db.upsert_meme("m3.png", "h3", "happy", int(img_ids[2]))
    db.set_meme_caption(m1, "c1", int(cap_ids[0]))
    db.set_meme_caption(m2, "c2", int(cap_ids[1]))
    db.set_disabled(m2, True)
    db.remove_meme(m3)

    assert db.index_size == 5
    assert db.live_vector_count() == 4
    assert db.health_check()["orphan_vector_count"] == 1

    result = db.compact_index()
    assert result["image_vectors"] == 2
    assert result["caption_vectors"] == 2
    assert db.index_size == 4
    assert db.health_check()["orphan_vector_count"] == 0
    for mid in (m1, m2):
        row = db.get_meme(mid)
        assert 0 <= row["vector_id"] < db.index_size
        assert 0 <= row["caption_vector_id"] < db.index_size


def test_health_detects_bad_caption_vector_id(db):
    vids = db.add_vectors(np.ones((1, 8), dtype="float32"))
    mid = db.upsert_meme("x.png", "h", "happy", int(vids[0]))
    db.set_meme_caption(mid, "caption", 999)
    h = db.health_check()
    assert h["bad_caption_vector_ids_count"] == 1
    assert not h["ok"]


def test_health_does_not_flag_missing_caption_as_bad(db):
    vids = db.add_vectors(np.ones((1, 8), dtype="float32"))
    db.upsert_meme("x.png", "h", "happy", int(vids[0]))
    h = db.health_check()
    assert h["bad_caption_vector_ids_count"] == 0


def test_corrupted_index_is_flagged(tmp_path):
    index_path = tmp_path / "bad.faiss"
    index_path.write_text("not a faiss index", encoding="utf-8")
    db = MemeDatabase(tmp_path / "m.db", index_path, dim=8)
    assert db.index_corrupted
    assert db.health_check()["index_corrupted"]


def test_relabel_keeps_old_tag_as_subtag(db):
    vids = db.add_vectors(np.ones((1, 8), dtype="float32"))
    mid = db.upsert_meme("r.png", "h", "happy", int(vids[0]))
    ok, old, new = db.relabel_meme(mid, "laugh")
    assert ok and old == "happy" and new == "laugh"
    sub_tags = json.loads(db.get_meme(mid)["sub_tags"] or "[]")
    assert "happy" in sub_tags
