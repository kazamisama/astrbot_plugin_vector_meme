from pathlib import Path

from helpers import write_png

from core.database import MemeDatabase
from core.embedder import DummyEmbedder
from core.indexer import MemeIndexer


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
