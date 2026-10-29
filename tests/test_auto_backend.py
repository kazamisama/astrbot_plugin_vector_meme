"""embedder_backend=auto 的解析规则。"""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class FakeEvent:
    def __init__(self):
        self.sent = []

    @staticmethod
    def plain_result(text: str):
        return ("plain", text)

    async def send(self, result) -> None:
        self.sent.append(result)


def _run(gen):
    async def collect():
        return [item async for item in gen]

    return asyncio.run(collect())


def _texts(out) -> list[str]:
    return [item[1] if isinstance(item, tuple) else str(item) for item in out]


def _make_plugin(tmp_path, main_module, *, backend="auto", providers=None, getter=None):
    class FakeContext:
        def __init__(self):
            self.provider_manager = types.SimpleNamespace(personas=[])
            if getter is not None:
                self.get_all_embedding_providers = getter
            elif providers is not None:
                self.get_all_embedding_providers = lambda: providers

    meme_dir = tmp_path / "memes"
    meme_dir.mkdir(exist_ok=True)
    return main_module.VectorMemePlugin(
        FakeContext(),
        {
            "data_dir": str(tmp_path / "data"),
            "meme_dir": str(meme_dir),
            "embedder_backend": backend,
            "enable_prompt_injection": False,
            "enable_vision_caption": False,
            "enable_stochastic_selection": False,
        },
    )


def _patch_modules(main_module, monkeypatch, modules):
    monkeypatch.setattr(main_module, "_module_available", lambda name: name in set(modules))


def test_module_available_probes_real_modules():
    from vector_meme.main import _module_available

    assert _module_available("numpy") is True
    assert _module_available("definitely_missing_module_xyz") is False


def test_explicit_backend_is_kept(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, {"open_clip", "torch"})
    plugin = _make_plugin(tmp_path, main_module, backend="dummy", providers=[object()])
    assert plugin._requested_backend == "dummy"
    assert plugin._backend_name == "dummy"
    assert plugin.caption_enabled is False

    plugin = _make_plugin(tmp_path, main_module, backend="api", providers=[object()])
    assert plugin._backend_name == "api"
    assert plugin.caption_enabled is True


def test_auto_prefers_open_clip_when_installed(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, {"open_clip", "torch"})
    plugin = _make_plugin(tmp_path, main_module, providers=[object()])
    assert plugin._backend_name == "open_clip"
    assert plugin.caption_enabled is False
    assert plugin._backend_label() == "auto→open_clip"


def test_auto_requires_both_open_clip_and_torch(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, {"open_clip"})
    plugin = _make_plugin(tmp_path, main_module, providers=[object()])
    assert plugin._backend_name == "api"


def test_auto_falls_back_to_api_with_provider(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, set())
    plugin = _make_plugin(tmp_path, main_module, providers=[object()])
    assert plugin._backend_name == "api"
    assert plugin.caption_enabled is True
    assert plugin._backend_label() == "auto→api"


def test_auto_falls_back_to_open_clip_without_provider(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, set())
    plugin = _make_plugin(tmp_path, main_module, providers=[])
    assert plugin._backend_name == "open_clip"
    assert plugin.caption_enabled is False


def test_auto_without_provider_getter(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, set())
    plugin = _make_plugin(tmp_path, main_module)
    assert plugin._backend_name == "open_clip"


def test_auto_tolerates_provider_getter_exception(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, set())

    def broken():
        raise RuntimeError("provider manager not ready")

    plugin = _make_plugin(tmp_path, main_module, getter=broken)
    assert plugin._backend_name == "open_clip"


def test_status_reports_auto_resolution(tmp_path, main_module, monkeypatch):
    _patch_modules(main_module, monkeypatch, set())
    plugin = _make_plugin(tmp_path, main_module, providers=[object()])
    texts = _texts(_run(plugin._cmd_status(FakeEvent(), [])))
    assert "- 后端: auto→api" in texts[-1]


def test_config_schema_offers_auto_as_default():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    backend = schema["model"]["items"]["embedder_backend"]
    assert backend["default"] == "auto"
    assert backend["options"][0] == "auto"
    assert set(backend["options"]) == {"auto", "open_clip", "api", "dummy"}
    assert "auto" in backend["hint"]
