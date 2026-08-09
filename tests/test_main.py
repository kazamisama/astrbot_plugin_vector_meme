import asyncio
import importlib.util
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def test_version_consistency():
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    meta = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert '"0.7.1"' in main_src
    assert re.search(r"^version:\s*0\.7\.1\s*$", meta, re.MULTILINE)
    assert changelog.startswith("# 更新日志")
    assert "## [0.7.1]" in changelog


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
            "_temp_dim": 8,
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
    vids = db.add_vectors(np.ones((1, 8), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.save_index()
    db_path = plugin.data_dir / "memes.db"
    bak_path = db_path.with_name("memes.db.bak.v071")
    assert db_path.exists()

    async def _fail(*_args, **_kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(plugin, "_ensure_ready", _fail)

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
