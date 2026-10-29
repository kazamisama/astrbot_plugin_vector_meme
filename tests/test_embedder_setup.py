"""embedder 解析、库准备与 caption 读写分支测试。"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pytest

from helpers import make_plugin

from core.database import MemeDatabase
from core.embedder import DummyEmbedder


def test_embedder_kwargs_backend_variants(tmp_path, main_module):
    api = make_plugin(
        tmp_path / "api",
        main_module,
        backend="api",
        config={"embedding_provider_id": "prov-1"},
    )
    kwargs = api._embedder_kwargs()
    assert kwargs["context"] is api.context
    assert kwargs["provider_id"] == "prov-1"

    dummy = make_plugin(tmp_path / "dummy", main_module, backend="dummy")
    assert dummy._embedder_kwargs() == {}

    cache = tmp_path / "cache"
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"x")
    open_clip = make_plugin(
        tmp_path / "openclip",
        main_module,
        backend="open_clip",
        config={
            "open_clip_cache_dir": str(cache),
            "open_clip_local_weights": f'"{weights}"',
            "hf_endpoint": "https://hf.example",
            "device": "cuda",
            "open_clip_model": "ViT-L-14",
        },
    )
    kwargs = open_clip._embedder_kwargs()
    assert kwargs["pretrained"] == str(weights)
    assert kwargs["cache_dir"] == str(cache)
    assert kwargs["hf_endpoint"] == "https://hf.example"
    assert kwargs["device"] == "cuda"
    assert kwargs["model_name"] == "ViT-L-14"
    assert cache.is_dir()

    default = make_plugin(
        tmp_path / "default",
        main_module,
        backend="open_clip",
        config={
            "open_clip_cache_dir": str(tmp_path / "cache2"),
            "open_clip_pretrained": "laion",
        },
    )
    assert default._embedder_kwargs()["pretrained"] == "laion"

    missing = make_plugin(
        tmp_path / "missing",
        main_module,
        backend="open_clip",
        config={
            "open_clip_cache_dir": str(tmp_path / "cache3"),
            "open_clip_local_weights": str(tmp_path / "nope.pt"),
        },
    )
    with pytest.raises(FileNotFoundError, match="open_clip_local_weights"):
        missing._embedder_kwargs()


def test_prepare_library_sets_fingerprint_and_components(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin._embedder = DummyEmbedder(16)

    plugin._prepare_library_for_embedder()

    assert plugin._db is not None
    assert plugin._db.dim == 16
    assert plugin._db.get_meta("embedder_fingerprint")
    assert plugin._indexer is not None
    assert plugin._retriever is not None


def test_prepare_library_rejects_fingerprint_change(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin._embedder = DummyEmbedder(16)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    vids = db.add_vectors(np.ones((1, 16), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.set_meta("embedder_fingerprint", "old:16")
    plugin._db = db

    with pytest.raises(RuntimeError, match="指纹已变化"):
        plugin._prepare_library_for_embedder()
    assert plugin._embedder is None


def test_prepare_library_rejects_dimension_change_with_rows(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin._embedder = DummyEmbedder(8)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    vids = db.add_vectors(np.ones((1, 16), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.set_meta("embedder_fingerprint", plugin._embedder_fingerprint())
    plugin._db = db

    with pytest.raises(RuntimeError, match="维度"):
        plugin._prepare_library_for_embedder()
    assert plugin._embedder is None


def test_prepare_library_warns_on_missing_fingerprint(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin._embedder = DummyEmbedder(8)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    vids = db.add_vectors(np.ones((1, 8), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    plugin._db = db

    plugin._prepare_library_for_embedder()

    assert db.get_meta("embedder_fingerprint")
    assert plugin._retriever is not None


def test_embedder_fingerprint_fallback_and_env_scope(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert plugin._embedder_fingerprint() == ""

    plugin._embedder = DummyEmbedder(8)
    assert plugin._embedder_fingerprint() == "DummyEmbedder:8"

    class WeirdEmbedder:
        dim = 8

        @property
        def fingerprint(self):
            raise RuntimeError("boom")

    plugin._embedder = WeirdEmbedder()
    assert plugin._embedder_fingerprint() == "dummy:8"

    os.environ.pop("VM_TEST_ENV", None)
    try:
        with plugin._env_scope({"VM_TEST_ENV": "1", "VM_MISSING": None}):
            assert os.environ["VM_TEST_ENV"] == "1"
            os.environ["VM_TEST_ENV"] = "2"
        assert "VM_TEST_ENV" not in os.environ

        os.environ["VM_TEST_ENV"] = "before"
        with plugin._env_scope({"VM_TEST_ENV": None}):
            assert "VM_TEST_ENV" not in os.environ
        assert os.environ["VM_TEST_ENV"] == "before"
    finally:
        os.environ.pop("VM_TEST_ENV", None)


def test_ensure_embedder_prepare_library_false_then_reuse(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    embedder = asyncio.run(plugin._ensure_embedder(prepare_library=False))
    assert embedder is not None
    assert plugin._indexer is None and plugin._retriever is None

    reused = asyncio.run(plugin._ensure_embedder())
    assert reused is embedder
    assert plugin._indexer is not None and plugin._retriever is not None


def test_make_retriever_uses_dual_when_caption_enabled(tmp_path, main_module):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    retriever = plugin._make_retriever(plugin._db)
    assert isinstance(retriever, main_module.DualRetriever)


def test_caption_io_edges(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    monkeypatch.setattr(main_module, "__file__", str(tmp_path / "main.py"))
    captions = tmp_path / "memes" / "captions.json"
    captions.parent.mkdir(parents=True, exist_ok=True)

    assert plugin._load_external_captions() == {}
    captions.write_text("{bad json", encoding="utf-8")
    assert plugin._load_external_captions() == {}
    captions.write_text("[1, 2]", encoding="utf-8")
    assert plugin._load_external_captions() == {}
    captions.write_text(
        json.dumps({"happy/a.png": "  ", "happy/b.png": "caption"}),
        encoding="utf-8",
    )
    loaded = plugin._load_external_captions()
    assert list(loaded.values()) == ["caption"]

    plugin._db = None
    assert plugin._export_captions() == 0

    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    inside = db.upsert_meme(str(plugin.meme_dir / "happy" / "b.png"), "h1", "happy", -1)
    outside = db.upsert_meme(str(tmp_path / "outside.png"), "h2", "happy", -1)
    db.set_meme_caption(inside, "cap", None)
    db.set_meme_caption(outside, "outside", None)
    plugin._db = db

    assert plugin._export_captions() == 1
    assert json.loads(captions.read_text(encoding="utf-8")) == {"happy/b.png": "cap"}


def test_collect_db_captions_and_restore_backup_errors(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=8)
    with_caption = db.upsert_meme("a.png", "h1", "happy", -1)
    db.upsert_meme("b.png", "h2", "sad", -1)
    db.set_meme_caption(with_caption, "cap", None)

    collected = plugin._collect_db_captions(db)
    assert list(collected.values()) == ["cap"]

    class BrokenDB:
        def list_memes(self, **kwargs):
            raise RuntimeError("boom")

    assert plugin._collect_db_captions(BrokenDB()) == {}

    original = tmp_path / "orig.png"
    original.write_bytes(b"new")
    backup = tmp_path / "orig.png.bak"
    backup.write_bytes(b"old")
    gone = tmp_path / "gone.png"
    errors = plugin._restore_backup_files(
        [(original, backup, True), (gone, tmp_path / "missing.bak", True)]
    )
    assert original.read_bytes() == b"old"
    assert errors and "gone.png" in errors[0]