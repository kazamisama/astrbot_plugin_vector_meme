"""命令清单与用法文档的唯一来源。

main.py 的 `_dispatch_command`、`/vm 帮助`、README 命令表、metadata.yaml 的
help 文本全部由本模块生成，避免同一份命令清单在四处各维护一遍。

修改命令时只改这里，然后运行：

    python scripts/gen_command_docs.py

它会重写 README.md 的命令表区域和 metadata.yaml 的 help 块；
tests/test_command_spec.py 会在 CI 里校验文档没有漂移。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SECTION_BASIC = "basic"
SECTION_DIAG = "diag"
SECTION_GOVERN = "govern"

SECTION_TITLES = {
    SECTION_BASIC: "基础命令（触发词：表情向量 / vmem / vm）",
    SECTION_DIAG: "诊断与维护",
    SECTION_GOVERN: "分类与治理",
}

#: 不带子命令时执行的命令
DEFAULT_COMMAND = "状态"


@dataclass(frozen=True)
class CommandSpec:
    """一条子命令的完整描述。"""

    name: str
    handler: str
    summary: str
    doc: str = ""
    usage: str = ""
    aliases: tuple[str, ...] = ()
    admin_only: bool = False
    gated: bool = False
    section: str = SECTION_BASIC
    #: False 表示只用于生成文档、不进入命令分发表（如 "caption 导出"）
    dispatchable: bool = True
    #: README 表格里显示的用法；默认由 name + usage 生成
    doc_usage: str | None = None
    hidden: bool = False

    @property
    def doc_text(self) -> str:
        return self.doc or self.summary

    @property
    def readme_usage(self) -> str:
        if self.doc_usage is not None:
            return self.doc_usage
        usage = f" {self.usage}" if self.usage else ""
        return f"`/vm {self.name}{usage}`"

    @property
    def plain_usage(self) -> str:
        """metadata 表格用：不带别名后缀。"""
        usage = f" {self.usage}" if self.usage else ""
        return f"表情向量 {self.name}{usage}"

    @property
    def help_usage(self) -> str:
        """`/vm 帮助` 用：带别名后缀。"""
        if self.aliases:
            return f"{self.plain_usage}（别名：{'、'.join(self.aliases)}）"
        return self.plain_usage


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="状态",
        handler="_cmd_status",
        summary="查看索引状态",
        doc="查看索引状态，不触发模型加载",
    ),
    CommandSpec(
        name="预热",
        handler="_cmd_prewarm",
        summary="预加载 embedder，检查模型可用性",
        doc="只加载 embedder，提前检查模型可用性",
        admin_only=True,
    ),
    CommandSpec(
        name="索引",
        handler="_cmd_index",
        summary="扫描目录并增量建立索引",
        doc="扫描目录建索引，增量更新；目录仅限 `meme_dir` 及其子目录",
        usage="[目录]",
        admin_only=True,
    ),
    CommandSpec(
        name="重建",
        handler="_cmd_rebuild",
        summary="清空并重建整个索引",
        doc="在临时库构建成功后替换旧库；先备份 `.bak.v071`，失败保留旧库与备份",
        admin_only=True,
    ),
    CommandSpec(
        name="列表",
        handler="_cmd_list",
        summary="列出指定标签下的表情或整体统计",
        doc="列出指定标签下的表情或整体标签分布",
        usage="[标签]",
    ),
    CommandSpec(
        name="标签",
        handler="_cmd_tags",
        summary="列出所有已注册标签",
        doc="列出已注册标签",
    ),
    CommandSpec(
        name="搜索",
        handler="_cmd_search",
        summary="用文本搜索最匹配的表情",
        doc="文本检索测试，显示 final/raw 分数",
        usage="<文本> [--tag <标签>]",
        gated=True,
    ),
    CommandSpec(
        name="解释",
        handler="_cmd_explain",
        summary="查看选图原因和打分细节",
        doc="展示 tag bonus、重复惩罚、随机扰动等选图原因",
        usage="<文本> [--tag <标签>]",
        aliases=("为什么",),
        gated=True,
        section=SECTION_DIAG,
    ),
    CommandSpec(
        name="最近使用",
        handler="_cmd_recent",
        summary="查看最近发送过的表情",
        usage="[数量]",
        aliases=("最近",),
        section=SECTION_DIAG,
    ),
    CommandSpec(
        name="诊断",
        handler="_cmd_health",
        summary="检查 DB / FAISS / 文件系统一致性",
        doc="检查 DB、FAISS、文件系统一致性；`--deep` 额外比对文件内容 hash",
        usage="[--deep]",
        aliases=("健康检查", "健康"),
        admin_only=True,
        section=SECTION_DIAG,
        doc_usage="`/vm 诊断` / `/vm 健康检查 [--deep]`",
    ),
    CommandSpec(
        name="修复",
        handler="_cmd_repair",
        summary="清理缺失文件记录、补注册 tag、刷新 prompt",
        doc="清理缺失文件记录、补注册 tag、压缩孤儿向量、刷新 prompt",
        admin_only=True,
        section=SECTION_DIAG,
    ),
    CommandSpec(
        name="自动分类",
        handler="_cmd_auto_classify",
        summary="基于 prototype + KNN 预测标签",
        doc="预测标签；默认 dry-run，加 `apply` 才写入数据库",
        usage="<id> [apply]",
        aliases=("分类",),
        admin_only=True,
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="重标注",
        handler="_cmd_relabel",
        summary="手动修改数据库主 tag，不移动文件",
        usage="<id> <新tag>",
        admin_only=True,
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="评测",
        handler="_cmd_eval",
        summary="跑小型分类评测集",
        doc="跑小型分类评测集，输出 top1/top3 准确率",
        usage="<eval.json>",
        admin_only=True,
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="标签规范",
        handler="_cmd_tag_schema",
        summary="查看 `tag_schema.json` 标签说明",
        aliases=("tag_schema",),
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="caption",
        handler="_cmd_caption",
        summary="为无 caption 的表情批量生成视觉语义描述",
        doc="为无 caption 的表情批量生成视觉语义描述，自动复用 `captions.json`",
        usage="[limit]",
        admin_only=True,
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="caption 导出",
        handler="_cmd_caption",
        summary="把 DB 中的 caption 导出到 memes/captions.json",
        doc="把 DB 中的 caption 导出到 `memes/captions.json`",
        admin_only=True,
        dispatchable=False,
        section=SECTION_GOVERN,
    ),
    CommandSpec(
        name="删除",
        handler="_cmd_delete",
        summary="从索引中移除某条",
        doc="从索引移除某条记录",
        usage="<id>",
        admin_only=True,
    ),
    CommandSpec(
        name="去重",
        handler="_cmd_dedup",
        summary="检测/禁用近重复表情变体（每组保留 1 张）",
        doc="检测近重复表情变体（默认 preview）；apply 时每组保留 1 张、其余禁用，undo 可恢复",
        usage="[preview/apply/undo] [阈值]",
        admin_only=True,
    ),
    CommandSpec(
        name="刷新提示",
        handler="_cmd_reload_prompt",
        summary="重新注入表情标签 prompt 到全局人格",
        admin_only=True,
        section=SECTION_DIAG,
    ),
    CommandSpec(
        name="帮助",
        handler="_cmd_help",
        summary="显示命令列表",
    ),
)


def _iter_dispatchable():
    return (spec for spec in COMMANDS if spec.dispatchable)


#: 子命令（含别名）-> 处理方法名
COMMAND_HANDLERS: dict[str, str] = {}
for _spec in _iter_dispatchable():
    COMMAND_HANDLERS[_spec.name] = _spec.handler
    for _alias in _spec.aliases:
        COMMAND_HANDLERS[_alias] = _spec.handler

#: 需要管理员权限的子命令（含别名）
ADMIN_ONLY_SUBCOMMANDS: frozenset[str] = frozenset(
    name
    for name, spec in (
        [(s.name, s) for s in COMMANDS]
        + [(a, s) for s in COMMANDS for a in s.aliases]
    )
    if spec.admin_only and name in COMMAND_HANDLERS
)

#: 普通成员可用、但不允许触发 embedder 冷启动的子命令（含别名）
MEMBER_COLD_LOAD_GATED_SUBCOMMANDS: frozenset[str] = frozenset(
    name
    for name, spec in (
        [(s.name, s) for s in COMMANDS]
        + [(a, s) for s in COMMANDS for a in s.aliases]
    )
    if spec.gated and name in COMMAND_HANDLERS
)


def _display_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def render_vm_help(plugin_name: str = "vector_meme") -> str:
    """`/vm 帮助` 的输出文本。"""
    lines = [f"[{plugin_name}] 用法："]
    for spec in COMMANDS:
        if spec.hidden:
            continue
        lines.append(f"  {spec.help_usage}")
    return "\n".join(lines)


def render_readme_table() -> str:
    """README「命令」一节的表格（含表头，不含 HTML 标记）。"""
    lines = ["| 命令 | 说明 |", "|---|---|"]
    for spec in COMMANDS:
        if spec.hidden:
            continue
        lock = " 🔒" if spec.admin_only else ""
        lines.append(f"| {spec.readme_usage}{lock} | {spec.doc_text} |")
    return "\n".join(lines)


def render_metadata_help() -> str:
    """metadata.yaml 的 help 文本（写回文件时统一加两格缩进）。"""
    admin_names = "、".join(
        spec.name for spec in COMMANDS if spec.admin_only and spec.dispatchable
    )
    gated_names = "、".join(spec.name for spec in COMMANDS if spec.gated)

    body: list[str] = ["智能表情包 vector_meme", ""]
    body.append(SECTION_TITLES[SECTION_BASIC] + "：")
    body.append(f"  注：{admin_names} 等管理命令仅管理员可用；")
    body.append(f"      普通成员使用 {gated_names} 前需由管理员先 /vm 预热。")

    for section in (SECTION_BASIC, SECTION_DIAG, SECTION_GOVERN):
        rows = [s for s in COMMANDS if s.section == section and not s.hidden]
        if not rows:
            continue
        if section != SECTION_BASIC:
            body.append("")
            body.append(SECTION_TITLES[section] + "：")
        width = max(_display_width(spec.plain_usage) for spec in rows)
        for spec in rows:
            body.append(f"  {_pad(spec.plain_usage, width)} - {spec.summary}")

    body += [
        "",
        "LLM 集成：",
        "  在回复中输出 %%happy%% 这类占位符，插件会移除占位符并追加匹配图片。",
        "  配合 XML 结构化输出插件时，也可输出 <sticker>happy</sticker> 标记，",
        "  由 XML 插件经 search_sticker_for_external() 转发取图。",
        "  默认不使用 &&tag&&，避免和 meme_manager 冲突。",
    ]
    return "\n".join(body) + "\n"
