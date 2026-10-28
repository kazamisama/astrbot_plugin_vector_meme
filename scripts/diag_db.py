"""只读诊断：打印 vector_meme 运行库的结构、向量映射与使用分布。

用法：

    python scripts/diag_db.py                      # 默认运行库
    python scripts/diag_db.py --db path/memes.db   # 指定库
    python scripts/diag_db.py --json               # 机器可读

数据根目录默认 `~/.astrbot/data`，可用环境变量 `ASTRBOT_DATA` 覆盖。
SQLite 以 `mode=ro` 打开，不会写入任何数据；索引文件只做存在性/大小检查。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


def default_db_path() -> Path:
    data_root = Path(os.environ.get("ASTRBOT_DATA", Path.home() / ".astrbot" / "data"))
    return data_root / "plugin_data" / "vector_meme" / "memes.db"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = "file:" + db_path.as_posix() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        str(r["name"])
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]


def _count(conn: sqlite3.Connection, table: str) -> int | None:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return None


def _vector_stats(conn: sqlite3.Connection, column: str) -> dict:
    try:
        row = conn.execute(
            f"SELECT COUNT({column}) AS n, COUNT(DISTINCT {column}) AS distinct_n, "
            f"MIN({column}) AS min_id, MAX({column}) AS max_id FROM memes"
        ).fetchone()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    return {
        "rows_with_id": int(row["n"]),
        "distinct_ids": int(row["distinct_n"]),
        "min_id": row["min_id"],
        "max_id": row["max_id"],
    }


def inspect(db_path: Path, *, logs: int = 20) -> dict:
    """读取一个 vector_meme 运行库，返回结构化诊断结果（只读）。"""
    db_path = db_path.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")
    index_path = db_path.with_suffix(".faiss")

    conn = _connect_ro(db_path)
    try:
        tables = _table_names(conn)
        counts = {
            t: _count(conn, t)
            for t in tables
            if t in {"memes", "tags", "search_log", "index_log"}
        }

        meta: dict[str, str] = {}
        if "meta" in tables:
            for row in conn.execute("SELECT key, value FROM meta"):
                meta[str(row["key"])] = str(row["value"])

        tags: list[dict] = []
        if "memes" in tables:
            for row in conn.execute(
                "SELECT tag, COUNT(*) AS n, SUM(CASE WHEN disabled != 0 THEN 1 ELSE 0 END) AS disabled "
                "FROM memes GROUP BY tag ORDER BY n DESC, tag"
            ):
                tags.append({"tag": row["tag"], "count": int(row["n"]), "disabled": int(row["disabled"] or 0)})

        recent: list[dict] = []
        if "search_log" in tables and logs > 0:
            for row in conn.execute(
                "SELECT * FROM search_log ORDER BY id DESC LIMIT ?", (int(logs),)
            ):
                recent.append(dict(row))

        return {
            "db": str(db_path),
            "db_bytes": db_path.stat().st_size,
            "index": str(index_path),
            "index_bytes": index_path.stat().st_size if index_path.exists() else None,
            "tables": tables,
            "counts": counts,
            "meta": meta,
            "vector_id": _vector_stats(conn, "vector_id"),
            "caption_vector_id": _vector_stats(conn, "caption_vector_id"),
            "tags": tags,
            "recent_searches": recent,
        }
    finally:
        conn.close()


def _print_human(info: dict) -> None:
    index_size = info["index_bytes"]
    print(f"db    {info['db']} ({info['db_bytes']} bytes)")
    print(f"index {info['index']} ({index_size if index_size is not None else 'MISSING'})")
    print("tables:", ", ".join(info["tables"]))
    print("counts:", ", ".join(f"{k}={v}" for k, v in info["counts"].items()))
    if info["meta"]:
        print("meta:")
        for k, v in sorted(info["meta"].items()):
            print(f"  {k} = {v}")
    print("image vectors:", info["vector_id"])
    print("caption vectors:", info["caption_vector_id"])
    print("per-tag counts:")
    for row in info["tags"]:
        suffix = f" (disabled {row['disabled']})" if row["disabled"] else ""
        print(f"  {row['tag']:<12} {row['count']:>5}{suffix}")
    if info["recent_searches"]:
        print(f"recent searches (last {len(info['recent_searches'])}):")
        for row in info["recent_searches"]:
            print("  " + json.dumps(row, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vector_meme 运行库只读诊断")
    parser.add_argument("--db", type=Path, default=None, help="memes.db 路径")
    parser.add_argument("--logs", type=int, default=20, help="打印最近多少条检索日志（0 关闭）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    db_path = args.db or default_db_path()
    try:
        info = inspect(db_path, logs=args.logs)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        _print_human(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
