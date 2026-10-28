"""命令清单单一来源的守卫测试。

README 命令表 / metadata.yaml help / main.py 分发表 / /vm 帮助 都由
core/command_spec.py 生成；这里校验四者一致，文档改了就红。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from core.command_spec import (
    ADMIN_ONLY_SUBCOMMANDS,
    COMMAND_HANDLERS,
    COMMANDS,
    DEFAULT_COMMAND,
    MEMBER_COLD_LOAD_GATED_SUBCOMMANDS,
    render_readme_table,
    render_vm_help,
)
from core.version import VERSION

ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    path = ROOT / "scripts" / "gen_command_docs.py"
    spec = importlib.util.spec_from_file_location("gen_command_docs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_readme_and_metadata_are_generated_from_spec():
    gen = _load_generator()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    assert gen.rendered_readme(readme) == readme, "README 命令表/版本号已漂移，请运行 scripts/gen_command_docs.py"
    assert gen.rendered_metadata(metadata) == metadata, "metadata help/version 已漂移，请运行 scripts/gen_command_docs.py"
    assert render_readme_table() in readme
    assert gen._metadata_help_block() in metadata


def test_every_spec_handler_exists(main_module):
    for spec in COMMANDS:
        assert callable(getattr(main_module.VectorMemePlugin, spec.handler, None)), spec.handler


def test_dispatch_table_covers_dispatchable_commands():
    for spec in COMMANDS:
        if not spec.dispatchable:
            assert spec.name not in COMMAND_HANDLERS
            continue
        assert COMMAND_HANDLERS[spec.name] == spec.handler
        for alias in spec.aliases:
            assert COMMAND_HANDLERS[alias] == spec.handler


def test_command_names_and_aliases_are_unique():
    seen: dict[str, str] = {}
    for spec in COMMANDS:
        for token in (spec.name, *spec.aliases):
            assert token not in seen, f"{token} 同时属于 {seen.get(token)} 与 {spec.name}"
            seen[token] = spec.name


def test_default_command_is_dispatchable():
    assert DEFAULT_COMMAND in COMMAND_HANDLERS
    assert COMMAND_HANDLERS[DEFAULT_COMMAND] == "_cmd_status"


def test_vm_help_lists_every_visible_command():
    help_text = render_vm_help()
    for spec in COMMANDS:
        if spec.hidden:
            assert spec.name not in help_text
        else:
            assert spec.help_usage in help_text
    assert help_text.startswith("[vector_meme] 用法：")


def test_admin_and_gated_sets_derive_from_spec():
    assert "预热" in ADMIN_ONLY_SUBCOMMANDS
    assert "健康检查" in ADMIN_ONLY_SUBCOMMANDS
    assert "状态" not in ADMIN_ONLY_SUBCOMMANDS
    assert MEMBER_COLD_LOAD_GATED_SUBCOMMANDS == {"搜索", "解释", "为什么"}


def test_readme_table_marks_admin_rows_with_lock():
    table = render_readme_table()
    for spec in COMMANDS:
        row = next(line for line in table.splitlines() if spec.readme_usage in line)
        assert ("🔒" in row) is spec.admin_only, spec.name


def test_version_single_source_matches_metadata_and_readme():
    import re

    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^version:\s*{re.escape(VERSION)}\s*$", metadata, re.MULTILINE)
    assert f"当前版本 **{VERSION}**" in readme
    assert f"## [{VERSION}]" in changelog
