"""命令分发路由的回归测试。

分发表、管理员门控、冷启动门控、未知子命令回退都由 core/command_spec.py 驱动，
这里从 `_dispatch_command` 这一层验证，避免只测渲染文本、不测真实路由。
"""
from __future__ import annotations

import asyncio

from core.command_spec import render_vm_help


def _make_plugin(tmp_path, main_module):
    class FakeProviderManager:
        def __init__(self):
            self.personas = []

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
            "enable_prompt_injection": False,
        },
    )


class FakeEvent:
    def __init__(self, admin: bool = False):
        self._admin = admin
        self.stopped = False

    def is_admin(self) -> bool:
        return self._admin

    @staticmethod
    def plain_result(text: str):
        return ("plain", text)

    def stop_event(self) -> None:
        self.stopped = True


def _dispatch(plugin, event, args):
    async def run():
        out = []
        async for result in plugin._dispatch_command(event, args):
            out.append(result)
        return out

    return asyncio.run(run())


def test_unknown_subcommand_falls_back_to_help(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    event = FakeEvent()
    out = _dispatch(plugin, event, ["不存在的子命令"])
    assert out == [("plain", render_vm_help("vector_meme"))]
    assert event.stopped


def test_help_subcommand_returns_generated_help(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    out = _dispatch(plugin, FakeEvent(), ["帮助"])
    assert out[0][1] == render_vm_help("vector_meme")


def test_non_admin_is_rejected_for_admin_command(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    out = _dispatch(plugin, FakeEvent(admin=False), ["预热"])
    assert len(out) == 1
    assert "权限不足" in out[0][1]


def test_alias_keeps_admin_gating(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    # 健康是 诊断 的别名，同样只允许管理员
    out = _dispatch(plugin, FakeEvent(admin=False), ["健康"])
    assert "权限不足" in out[0][1]


def test_member_search_is_gated_before_embedder_warmup(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    assert plugin._embedder is None
    out = _dispatch(plugin, FakeEvent(admin=False), ["搜索", "开心"])
    assert "embedder 尚未预热" in out[0][1]


def test_member_status_is_allowed(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    out = _dispatch(plugin, FakeEvent(admin=False), ["状态"])
    assert out and "状态" in str(out[0][1])
