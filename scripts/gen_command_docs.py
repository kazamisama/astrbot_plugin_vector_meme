"""按唯一来源重写 README 命令表 / metadata.yaml help / 版本号。

唯一来源：
- 命令清单与用法：core/command_spec.py
- 版本号：core/version.py

用法：

    python scripts/gen_command_docs.py          # 写入
    python scripts/gen_command_docs.py --check  # 只检查是否漂移（CI 用，漂移返回 1）

README 的命令表必须被下面两行标记包住，生成器只重写标记之间的内容：

    <!-- commands:start -->
    ...
    <!-- commands:end -->
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.command_spec import render_metadata_help, render_readme_table  # noqa: E402
from core.version import VERSION  # noqa: E402

README = ROOT / "README.md"
METADATA = ROOT / "metadata.yaml"

README_START = "<!-- commands:start -->"
README_END = "<!-- commands:end -->"
README_REGION_RE = re.compile(
    re.escape(README_START) + r".*?" + re.escape(README_END),
    re.DOTALL,
)
README_VERSION_RE = re.compile(r"当前版本 \*\*[^*]+\*\*")
METADATA_HELP_RE = re.compile(r"^help: \|\n(?:  [^\n]*\n|\n)*", re.MULTILINE)
METADATA_VERSION_RE = re.compile(r"^version: .*$", re.MULTILINE)

README_NOTE = "<!-- 此表由 scripts/gen_command_docs.py 从 core/command_spec.py 生成，勿手改 -->"


def _readme_region() -> str:
    return "\n".join([README_START, README_NOTE, render_readme_table(), README_END])


def _metadata_help_block() -> str:
    lines = render_metadata_help().splitlines()
    body = "\n".join(("  " + line) if line else "" for line in lines)
    return f"help: |\n{body}\n"


def rendered_readme(text: str) -> str:
    if not README_REGION_RE.search(text):
        raise SystemExit(
            f"README.md 缺少命令表标记：请先补上 {README_START} / {README_END}"
        )
    text = README_REGION_RE.sub(lambda _m: _readme_region(), text, count=1)
    text = README_VERSION_RE.sub(f"当前版本 **{VERSION}**", text, count=1)
    return text


def rendered_metadata(text: str) -> str:
    if not METADATA_HELP_RE.search(text):
        raise SystemExit("metadata.yaml 缺少 help: | 块")
    text = METADATA_HELP_RE.sub(lambda _m: _metadata_help_block(), text, count=1)
    text = METADATA_VERSION_RE.sub(f"version: {VERSION}", text, count=1)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 README 命令表与 metadata help")
    parser.add_argument("--check", action="store_true", help="只检查，不写文件")
    args = parser.parse_args(argv)

    targets = [
        (README, "README.md", rendered_readme),
        (METADATA, "metadata.yaml", rendered_metadata),
    ]
    drifted: list[str] = []
    for path, label, render in targets:
        old = path.read_text(encoding="utf-8")
        new = render(old)
        if old == new:
            continue
        drifted.append(label)
        if not args.check:
            path.write_text(new, encoding="utf-8", newline="\n")

    if args.check:
        if drifted:
            print("文档与命令清单不一致，请运行 python scripts/gen_command_docs.py：" + "、".join(drifted))
            return 1
        print("文档与命令清单一致。")
        return 0

    print("已更新：" + ("、".join(drifted) if drifted else "无变化"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
