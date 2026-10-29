from pathlib import Path

import numpy as np
from PIL import Image

from helpers import write_png

from core.database import MemeDatabase
from core.embedder import DummyEmbedder
from core.indexer import (
    IndexProgress,
    MemeIndexer,
    _read_image_meta,
    iter_image_files,
    tag_meta,
)


def test_indexer_stores_absolute_paths(tmp_path):
    meme_root = tmp_path / "memes"
    (meme_root / "happy").mkdir(parents=True)
    write_png(meme_root / "happy" / "a.png")
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    indexer = MemeIndexer(db, DummyEmbedder(8), use_subdir_as_tag=True, default_tag="misc")

    progress = indexer.index_directory(meme_root)
    assert progress.added == 1
    rows = db.list_memes()
    assert len(rows) == 1
    assert rows[0]["tag"] == "happy"
    stored = Path(rows[0]["file_path"])
    assert stored.is_absolute()
    assert stored.exists()


def test_remove_missing_cleans_deleted_files(tmp_path):
    meme_root = tmp_path / "memes"
    (meme_root / "happy").mkdir(parents=True)
    file_path = meme_root / "happy" / "a.png"
    write_png(file_path)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    indexer = MemeIndexer(db, DummyEmbedder(8), use_subdir_as_tag=True, default_tag="misc")
    indexer.index_directory(meme_root)
    assert db.total_count() == 1

    file_path.unlink()
    removed = indexer.remove_missing(meme_root)
    assert removed == 1
    assert db.total_count() == 0


class _FailingEmbedder(DummyEmbedder):
    def embed_image(self, image):
        raise RuntimeError("boom")


class _ApiLikeEmbedder(DummyEmbedder):
    def embed_image(self, image):
        raise NotImplementedError


def test_iter_image_files_and_tag_resolution_edges(tmp_path):
    root = tmp_path / "memes"
    root.mkdir()
    write_png(root / "top.png")
    write_png(root / "happy" / "a.png")
    (root / "note.txt").write_text("x", encoding="utf-8")

    assert list(iter_image_files(tmp_path / "missing")) == []
    assert [p.name for p in iter_image_files(root, recursive=False)] == ["top.png"]
    assert {p.name for p in iter_image_files(root)} == {"top.png", "a.png"}

    bad = root / "bad.png"
    bad.write_bytes(b"not-an-image")
    assert _read_image_meta(bad) is None
    assert _read_image_meta(root / "top.png")[:2] == (16, 16)

    assert tag_meta("x", {"x": "not-a-dict"}) == {
        "description": "",
        "category": "",
        "color": "",
    }

    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    indexer = MemeIndexer(db, DummyEmbedder(8), use_subdir_as_tag=True, default_tag="misc")
    assert indexer._resolve_tag(root / "top.png", root) == "misc"
    assert indexer._resolve_tag(tmp_path / "elsewhere.png", root) == "misc"
    flat = MemeIndexer(db, DummyEmbedder(8), use_subdir_as_tag=False, default_tag="misc")
    assert flat._resolve_tag(root / "happy" / "a.png", root) == "misc"


def test_indexer_records_unreadable_files_and_embed_failures(tmp_path):
    root = tmp_path / "memes"
    write_png(root / "happy" / "ok.png")
    (root / "happy" / "bad.png").write_bytes(b"not-an-image")
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)

    progress = MemeIndexer(db, _FailingEmbedder(8)).index_directory(root)

    assert progress.total == 2
    assert progress.failed == 2
    assert db.total_count() == 0


def test_indexer_api_backend_upserts_without_vectors(tmp_path):
    root = tmp_path / "memes"
    path = write_png(root / "happy" / "a.png")
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    indexer = MemeIndexer(db, _ApiLikeEmbedder(8))

    first = indexer.index_directory(root)
    assert first.added == 1
    row = db.get_meme_by_path(str(path.resolve()))
    assert row["vector_id"] == -1
    assert db.index_size == 0

    write_png(path, color=(0, 0, 255))
    second = indexer.index_directory(root)
    assert second.updated == 1


def test_indexer_batches_updates_and_reports_progress(tmp_path):
    root = tmp_path / "memes"
    for i in range(17):
        write_png(root / "happy" / f"{i}.png", color=(i, 0, 0))
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    seen = []
    progress = IndexProgress(lambda done, total, file: seen.append(done))

    first = MemeIndexer(db, DummyEmbedder(8)).index_directory(
        root, batch_size=5, progress=progress
    )
    assert first.added == 17
    assert seen == list(range(1, 18))

    write_png(root / "happy" / "0.png", color=(255, 255, 0))
    second = MemeIndexer(db, DummyEmbedder(8)).index_directory(root, batch_size=5)
    assert second.updated == 1
    assert second.skipped == 16


def test_remove_missing_skips_paths_outside_root(tmp_path):
    root = tmp_path / "memes"
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    vids = db.add_vectors(np.ones((2, 8), dtype="float32"))
    inside = root / "happy" / "missing.png"
    outside = tmp_path / "elsewhere" / "missing.png"
    db.upsert_meme(str(inside), "h1", "happy", int(vids[0]))
    db.upsert_meme(str(outside), "h2", "happy", int(vids[1]))

    removed = MemeIndexer(db, DummyEmbedder(8)).remove_missing(root)

    assert removed == 1
    assert db.get_meme_by_path(str(outside)) is not None


def test_indexer_handles_animated_gif(tmp_path):
    root = tmp_path / "memes"
    path = root / "happy" / "a.gif"
    path.parent.mkdir(parents=True)
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)

    progress = MemeIndexer(db, DummyEmbedder(8)).index_directory(root)

    assert progress.added == 1
    assert progress.failed == 0
