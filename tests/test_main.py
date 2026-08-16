import asyncio
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def test_version_consistency():
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    meta = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert '"0.7.4"' in main_src
    assert re.search(r"^version:\s*0\.7\.4\s*$", meta, re.MULTILINE)
    assert changelog.startswith("# 更新日志")
    assert "## [0.7.4]" in changelog


def test_migrate_script_defines_config_file():
    path = ROOT / "scripts" / "migrate_config_v041.py"
    spec = importlib.util.spec_from_file_location("migrate_config_v041", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert hasattr(mod, "CONFIG_FILE")
    assert isinstance(mod.CONFIG_FILE, Path)


def test_strip_tags_keeps_sticker_block(main_module):
    text = "hi %%happy%% <sticker>%%happy%%</sticker> tail"
    out = main_module.strip_tags_outside_stickers(text)
    assert out.startswith("hi ")
    assert "<sticker>%%happy%%</sticker>" in out
    assert out.count("%%happy%%") == 1


def test_persona_marker_roundtrip(main_module):
    start = main_module.PERSONA_MARKER_START
    end = main_module.PERSONA_MARKER_END
    prompt = "base prompt"
    injected = prompt + f"\n{start}\nbody\n{end}"
    assert start in injected
    assert main_module.PERSONA_INJECT_RE.sub("", injected) == "base prompt"


def _make_plugin(tmp_path, personas, main_module):
    class FakeProviderManager:
        def __init__(self):
            self.personas = personas

    class FakeContext:
        def __init__(self):
            self.provider_manager = FakeProviderManager()

    meme_dir = tmp_path / "memes"
    meme_dir.mkdir(exist_ok=True)
    return main_module.VectorMemePlugin(
        FakeContext(),
        {
            "data_dir": str(tmp_path / "data"),
            "meme_dir": str(meme_dir),
            "embedder_backend": "dummy",
            "enable_prompt_injection": True,
        },
    )


def test_persona_marker_injection_idempotent(tmp_path, main_module):
    personas = [{"prompt": "base prompt"}]
    plugin = _make_plugin(tmp_path, personas, main_module)
    plugin._db.upsert_tag("happy", "开心")
    plugin._inject_into_personas()
    assert personas[0]["prompt"].count(main_module.PERSONA_MARKER_START) == 1
    plugin._inject_into_personas()
    assert personas[0]["prompt"].count(main_module.PERSONA_MARKER_START) == 1
    plugin._restore_personas()
    assert main_module.PERSONA_MARKER_START not in personas[0]["prompt"]
    assert personas[0]["prompt"].startswith("base prompt")


def test_rebuild_preserves_db_when_embedder_fails(tmp_path, monkeypatch, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    db = plugin._db
    vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.save_index()
    db_path = plugin.data_dir / "memes.db"
    bak_path = db_path.with_name("memes.db.bak.v071")
    assert db_path.exists()

    async def _fail(*_args, **_kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(plugin, "_ensure_embedder", _fail)

    class FakeEvent:
        def __init__(self):
            self.sent = []

        def plain_result(self, text):
            return ("plain", str(text))

        async def send(self, chain):
            self.sent.append(chain)

    event = FakeEvent()
    results = []

    async def _collect():
        async for item in plugin._cmd_rebuild(event, []):
            results.append(item)

    asyncio.run(_collect())
    assert any("已保留原库" in str(r) for r in results)
    assert db_path.exists()
    assert bak_path.exists()
    with db._conn() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM memes").fetchone()["n"]
    assert n == 1


def _make_fake_event(admin=True):
    class FakeEvent:
        def __init__(self):
            self.sent = []
            self.stopped = False
            self.extras = {}

        def is_admin(self):
            return admin

        def plain_result(self, text):
            return ("plain", str(text))

        def stop_event(self):
            self.stopped = True

        def set_extra(self, key, value):
            self.extras[key] = value

        def get_extra(self, key):
            return self.extras.get(key)

        async def send(self, chain):
            self.sent.append(chain)

    return FakeEvent()


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_dispatch_blocks_admin_commands_for_members(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    event = _make_fake_event(admin=False)
    results = asyncio.run(_collect(plugin._dispatch_command(event, ["索引"])))
    assert event.stopped
    assert any("权限不足" in str(r) for r in results)


def test_index_rejects_path_outside_meme_dir(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    outside = tmp_path / "outside"
    outside.mkdir()
    event = _make_fake_event(admin=True)
    results = asyncio.run(_collect(plugin._cmd_index(event, [str(outside)])))
    assert any("无权索引" in str(r) for r in results)


def test_embedder_fingerprint_change_blocks_existing_library(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    db = plugin._db
    db.set_meta("embedder_fingerprint", "old_backend:512")
    vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.save_index()

    with pytest.raises(RuntimeError, match="指纹已变化"):
        asyncio.run(plugin._ensure_ready())


def test_unknown_llm_tag_is_filtered_and_not_queued(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)

    async def _setup_and_run():
        await plugin._ensure_embedder()
        db = plugin._db
        vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
        db.upsert_meme("happy.png", "h", "happy", int(vids[0]))
        db.save_index()

        class FakeResponse:
            def __init__(self, text):
                self.completion_text = text

        event = _make_fake_event(admin=True)
        response = FakeResponse("hello %%unknown_tag%% %%happy%%")
        await plugin.on_llm_response(event, response)
        return event, response

    event, response = asyncio.run(_setup_and_run())
    assert response.completion_text == "hello"
    assert event.extras.get("vector_meme_pending_tags") == ["happy"]


def test_rebuild_indexing_failure_keeps_original_and_backup(tmp_path, monkeypatch, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    meme_dir = tmp_path / "memes"
    (meme_dir / "happy").mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(meme_dir / "happy" / "a.png")

    db = plugin._db
    vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
    db.upsert_meme(str(meme_dir / "old.png"), "h", "happy", int(vids[0]))
    db.save_index()

    def _fail(self, *_args, **_kwargs):
        raise RuntimeError("indexing boom")

    monkeypatch.setattr(main_module.MemeIndexer, "index_directory", _fail)

    event = _make_fake_event(admin=True)
    results = asyncio.run(_collect(plugin._cmd_rebuild(event, [])))
    assert any("重建失败" in str(r) for r in results)
    assert plugin._db.db_path.exists()
    assert plugin._db.index_path.exists()
    assert plugin._db.db_path.with_name("memes.db.bak.v071").exists()
    assert plugin._db.index_path.with_name("memes.faiss.bak.v071").exists()
    assert plugin._db.total_count() == 1
    assert plugin._db.index_size == 1


def test_rebuild_empty_library_succeeds_and_persists_empty_index(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    db = plugin._db
    vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
    db.upsert_meme(str(tmp_path / "old.png"), "h", "happy", int(vids[0]))
    db.save_index()

    event = _make_fake_event(admin=True)
    results = asyncio.run(_collect(plugin._cmd_rebuild(event, [])))
    assert any("重建完成" in str(r) for r in results)
    assert plugin._db.total_count() == 0
    assert plugin._db.index_size == 0
    assert plugin._db.index_path.exists()
    assert not plugin._db.index_path.with_name("memes.faiss.bak.v071").exists()


def test_rebuild_preserves_runtime_state_and_logs(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    meme_dir = tmp_path / "memes"
    (meme_dir / "happy").mkdir(parents=True)
    image_path = meme_dir / "happy" / "a.png"
    Image.new("RGB", (4, 4), "red").save(image_path)

    db = plugin._db
    vids = db.add_vectors(np.ones((1, 512), dtype="float32"))
    mid = db.upsert_meme(str(image_path), "h", "happy", int(vids[0]))
    db.mark_used(mid)
    db.set_disabled(mid, True)
    db.log_search("hello", "happy", [mid], mid, 0.9)
    db.log_index_action("add", str(image_path), "success", "test")

    event = _make_fake_event(admin=True)
    results = asyncio.run(_collect(plugin._cmd_rebuild(event, [])))
    assert any("重建完成" in str(r) for r in results)

    row = plugin._db.get_meme_by_path(str(image_path))
    assert row is not None
    assert row["usage_count"] == 1
    assert row["last_used_at"] is not None
    assert int(row["disabled"]) == 1
    assert len(plugin._db.list_search_logs(limit=10)) == 1
    with plugin._db._conn() as c:
        old_log = c.execute(
            "SELECT COUNT(*) AS n FROM index_log WHERE message = ?", ("test",)
        ).fetchone()["n"]
    assert old_log == 1


def test_member_search_does_not_cold_load_embedder(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, [], main_module)
    event = _make_fake_event(admin=False)
    results = asyncio.run(_collect(plugin._dispatch_command(event, ["搜索", "开心"])))
    assert event.stopped
    assert plugin._embedder is None
    assert any("尚未预热" in str(r) for r in results)
