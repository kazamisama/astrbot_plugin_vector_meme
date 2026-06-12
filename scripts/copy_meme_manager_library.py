"""把 meme_manager 的 memes 子目录复制到 vector_meme 的独立目录。

不修改源目录，纯复制，重复跑安全（目标已存在则跳过）。

用法：
    python copy_meme_manager_library.py
    python copy_meme_manager_library.py --source <other_dir> --target <other_dir>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = Path(r"C:\Users\chiriu\.astrbot\data\plugins\meme_manager\memes")
DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "memes"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def copy_library(source: Path, target: Path, overwrite: bool = False) -> dict:
    source = Path(source)
    target = Path(target)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source}")

    target.mkdir(parents=True, exist_ok=True)

    stats = {
        "copied_dirs": [],
        "skipped_dirs": [],
        "empty_dirs": [],
        "total_files": 0,
        "total_bytes": 0,
    }

    for sub in sorted(source.iterdir()):
        if not sub.is_dir():
            continue
        files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not files:
            stats["empty_dirs"].append(sub.name)
            continue
        dest_sub = target / sub.name
        if dest_sub.exists() and not overwrite:
            stats["skipped_dirs"].append(sub.name)
            # 统计已有数量
            stats["total_files"] += len(list(dest_sub.iterdir()))
            continue
        if dest_sub.exists() and overwrite:
            shutil.rmtree(dest_sub)
        shutil.copytree(sub, dest_sub)
        stats["copied_dirs"].append(sub.name)
        stats["total_files"] += len(files)
        stats["total_bytes"] += sum(f.stat().st_size for f in files)

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的目标子目录")
    args = p.parse_args()

    print(f"源: {args.source}")
    print(f"目标: {args.target}")
    print(f"覆盖模式: {args.overwrite}")
    print()

    stats = copy_library(args.source, args.target, overwrite=args.overwrite)

    print("=== 复制结果 ===")
    print(f"复制: {len(stats['copied_dirs'])} 个目录 ({', '.join(stats['copied_dirs']) or '无'})")
    print(f"跳过（已存在）: {len(stats['skipped_dirs'])} 个目录 ({', '.join(stats['skipped_dirs']) or '无'})")
    print(f"跳过（空目录）: {len(stats['empty_dirs'])} 个目录 ({', '.join(stats['empty_dirs']) or '无'})")
    print(f"图片总数: {stats['total_files']}")
    print(f"总大小: {stats['total_bytes'] / 1024 / 1024:.1f} MB")
    print()
    print(f"目标路径: {args.target}")


if __name__ == "__main__":
    main()
