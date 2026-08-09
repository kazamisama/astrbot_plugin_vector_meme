# -*- coding: utf-8 -*-
"""迁移 0.4.0 -> 0.4.1：平铺配置 -> 分组配置。

0.4.1 将 _conf_schema.json 改为顶层 object 分组（参考 engram 插件）。
AstrBot 检测到 schema 变化后会用默认值重组 config.json，旧平铺 key 会被丢弃；
在 AstrBot 首次加载新 schema 之前运行本脚本，可保留旧的非默认配置值。

用法：
    python migrate_config_v041.py [配置文件路径]
    不带参数时自动定位插件配置目录下的 astrbot_plugin_vector_meme_config.json。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
SCHEMA_FILE = PLUGIN_DIR / "_conf_schema.json"

def default_config_path() -> Path:
    env = os.environ.get("ASTRBOT_DATA_PATH")
    if env:
        return Path(env) / "data" / "config" / "astrbot_plugin_vector_meme_config.json"
    return Path.home() / ".astrbot" / "data" / "config" / "astrbot_plugin_vector_meme_config.json"


CONFIG_FILE = default_config_path()


def load_schema() -> dict:
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        return json.load(f)


def is_nested(conf: dict) -> bool:
    """已是分组结构：所有顶层值都是 dict 且含叶子配置。"""
    return bool(conf) and all(isinstance(v, dict) for v in conf.values())


def migrate(conf: dict, schema: dict) -> dict:
    grouped: dict = {group: {} for group in schema}
    for group, group_schema in schema.items():
        keys = set(group_schema["items"].keys())
        for k, v in conf.items():
            if k in keys:
                grouped[group][k] = v
    return grouped


def main() -> int:
    if not SCHEMA_FILE.is_file():
        print(f"schema 不存在: {SCHEMA_FILE}")
        return 1
    if not CONFIG_FILE.is_file():
        print(f"配置文件不存在: {CONFIG_FILE}")
        return 1
    # AstrBot 写入的配置文件可能带 UTF-8 BOM，用 utf-8-sig 读取
    conf = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
    if is_nested(conf):
        print("配置已是分组结构，无需迁移。")
        return 0
    schema = load_schema()
    migrated = migrate(conf, schema)
    backup = CONFIG_FILE.with_suffix(".json.bak_v041")
    backup.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    CONFIG_FILE.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    kept = sum(1 for g in migrated.values() for _ in g)
    print(f"迁移完成：保留 {kept} 个配置值，原文件备份到 {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
