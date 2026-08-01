"""SQLite 元数据 + FAISS 向量存储。

设计要点：
- SQLite 存元数据、标签、统计
- FAISS 存向量，vector_id 与 SQLite 主键解耦
- 文件 hash 用于去重和增量更新
- 标签 / 子标签 / 描述都作为元数据，向量只管视觉特征
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS memes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    file_name TEXT NOT NULL,
    tag TEXT NOT NULL,
    sub_tags TEXT,
    description TEXT,
    caption TEXT,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    vector_id INTEGER NOT NULL,
    caption_vector_id INTEGER,
    usage_count INTEGER DEFAULT 0,
    last_used_at REAL,
    disabled INTEGER DEFAULT 0,
    created_at REAL DEFAULT (strftime('%s','now')),
    updated_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_memes_tag ON memes(tag);
CREATE INDEX IF NOT EXISTS idx_memes_hash ON memes(file_hash);
CREATE INDEX IF NOT EXISTS idx_memes_vector_id ON memes(vector_id);
CREATE INDEX IF NOT EXISTS idx_memes_last_used ON memes(last_used_at);

CREATE TABLE IF NOT EXISTS tags (
    name TEXT PRIMARY KEY,
    description TEXT,
    category TEXT,
    color TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    tag_filter TEXT,
    topk_ids TEXT,
    selected_id INTEGER,
    similarity REAL,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS index_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT,
    file_path TEXT,
    status TEXT,
    message TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);
"""


def compute_file_hash(path: Path) -> str:
    """SHA1 文件 hash（用于去重）。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MemeDatabase:
    """元数据 + 向量存储管理器。"""

    def __init__(self, db_path: Path, index_path: Path | None = None, dim: int = 512):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path) if index_path else self.db_path.with_suffix(".faiss")
        self.dim = dim
        self._lock = threading.RLock()
        self._init_db()
        self._index = self._load_or_create_index()
        # 如果索引里有向量但库里没有，优先信任库（启动时会做一次同步）

    # ---------- DB ----------

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {row[1] for row in c.execute('PRAGMA table_info(memes)')}
            if 'caption' not in cols:
                c.execute('ALTER TABLE memes ADD COLUMN caption TEXT')
            if 'caption_vector_id' not in cols:
                c.execute('ALTER TABLE memes ADD COLUMN caption_vector_id INTEGER')

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            c = sqlite3.connect(self.db_path)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    # ---------- FAISS ----------

    def _load_or_create_index(self):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss 未安装，请 pip install faiss-cpu")
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                if idx.d != self.dim:
                    # 维度不匹配，重建
                    idx = faiss.IndexFlatIP(self.dim)
                return idx
            except Exception:
                pass
        # 用内积（IP）+ 归一化向量 = 余弦相似度
        return faiss.IndexFlatIP(self.dim)

    def save_index(self) -> None:
        if FAISS_AVAILABLE and self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, str(self.index_path))

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def add_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """添加向量到 FAISS，返回对应的 vector_id 列表。"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = vectors.astype("float32")
        # 归一化（用内积模拟余弦）
        faiss.normalize_L2(vectors)
        with self._lock:
            start_id = self._index.ntotal
            self._index.add(vectors)
            return np.arange(start_id, start_id + vectors.shape[0], dtype=np.int64)

    def search(self, query: np.ndarray, topk: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """返回 (相似度, vector_id)。"""
        query = query.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        sims, ids = self._index.search(query, topk)
        return sims[0], ids[0]

    # ---------- Memes CRUD ----------

    def upsert_meme(
        self,
        file_path: str,
        file_hash: str,
        tag: str,
        vector_id: int,
        file_name: str | None = None,
        sub_tags: list[str] | None = None,
        description: str | None = None,
        width: int | None = None,
        height: int | None = None,
        file_size: int | None = None,
    ) -> int:
        """插入或更新表情，返回 meme id。如果 hash 未变则跳过。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT id, file_hash FROM memes WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            if row is not None:
                if row["file_hash"] == file_hash:
                    # 内容没变，只更新 tag / vector_id 等元数据
                    c.execute(
                        """UPDATE memes SET
                            tag = ?, sub_tags = ?, description = ?,
                            vector_id = ?, file_name = ?, updated_at = ?
                           WHERE id = ?""",
                        (
                            tag,
                            json.dumps(sub_tags or [], ensure_ascii=False),
                            description,
                            vector_id,
                            file_name or Path(file_path).name,
                            time.time(),
                            row["id"],
                        ),
                    )
                    return row["id"]
                # hash 变了，需要重新建向量
                c.execute("DELETE FROM memes WHERE id = ?", (row["id"],))
                # 旧 vector_id 作废（faiss 不支持物理删除，做标记即可）

            cur = c.execute(
                """INSERT INTO memes
                    (file_path, file_hash, file_name, tag, sub_tags, description,
                     width, height, file_size, vector_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_path,
                    file_hash,
                    file_name or Path(file_path).name,
                    tag,
                    json.dumps(sub_tags or [], ensure_ascii=False),
                    description,
                    width,
                    height,
                    file_size,
                    int(vector_id),
                ),
            )
            return cur.lastrowid

    def set_meme_caption(
        self,
        meme_id: int,
        caption: str | None,
        caption_vector_id: int | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                'UPDATE memes SET caption = ?, caption_vector_id = ? WHERE id = ?',
                (caption, caption_vector_id, int(meme_id)),
            )

    def memes_without_caption(self, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                'SELECT id, file_path FROM memes WHERE caption IS NULL OR length(caption) = 0 ORDER BY id ASC LIMIT ?',
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]

    def remove_meme(self, meme_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM memes WHERE id = ?", (meme_id,))
            return cur.rowcount > 0

    def remove_meme_by_path(self, file_path: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM memes WHERE file_path = ?", (file_path,))
            return cur.rowcount > 0

    def get_meme(self, meme_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memes WHERE id = ?", (meme_id,)).fetchone()
            return dict(row) if row else None

    def get_meme_by_vector_id(self, vector_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memes WHERE vector_id = ?", (vector_id,)).fetchone()
            return dict(row) if row else None

    def get_meme_by_path(self, file_path: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memes WHERE file_path = ?", (file_path,)).fetchone()
            return dict(row) if row else None

    def list_memes(
        self,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> list[dict]:
        sql = "SELECT * FROM memes WHERE 1=1"
        params: list[Any] = []
        if tag:
            sql += " AND tag = ?"
            params.append(tag)
        if not include_disabled:
            sql += " AND disabled = 0"
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count_by_tag(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT tag, COUNT(*) AS n FROM memes
                   WHERE disabled = 0 GROUP BY tag ORDER BY n DESC"""
            ).fetchall()
            return {r["tag"]: r["n"] for r in rows}

    def total_count(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM memes WHERE disabled = 0").fetchone()
            return row["n"] if row else 0

    def mark_used(self, meme_id: int) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE memes
                   SET usage_count = usage_count + 1, last_used_at = ?
                   WHERE id = ?""",
                (time.time(), meme_id),
            )

    def set_disabled(self, meme_id: int, disabled: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE memes SET disabled = ? WHERE id = ?", (int(disabled), meme_id))

    def relabel_meme(self, meme_id: int, new_tag: str, keep_old_as_subtag: bool = True) -> tuple[bool, str | None, str | None]:
        """重标注 meme 的主 tag，不移动文件。

        Returns:
            (ok, old_tag, new_tag)
        """
        new_tag = (new_tag or "").strip()
        if not new_tag:
            return False, None, None
        with self._conn() as c:
            row = c.execute("SELECT id, tag, sub_tags FROM memes WHERE id = ?", (meme_id,)).fetchone()
            if not row:
                return False, None, None
            old_tag = row["tag"]
            try:
                sub_tags = json.loads(row["sub_tags"] or "[]")
                if not isinstance(sub_tags, list):
                    sub_tags = []
            except Exception:
                sub_tags = []
            if keep_old_as_subtag and old_tag and old_tag != new_tag and old_tag not in sub_tags:
                sub_tags.append(old_tag)
            c.execute(
                """UPDATE memes
                   SET tag = ?, sub_tags = ?, updated_at = ?
                   WHERE id = ?""",
                (new_tag, json.dumps(sub_tags, ensure_ascii=False), time.time(), meme_id),
            )
            # 顺手保证 tag 表存在
            c.execute(
                """INSERT INTO tags (name, description, category)
                   VALUES (?, ?, ?)
                   ON CONFLICT(name) DO NOTHING""",
                (new_tag, "", ""),
            )
            return True, old_tag, new_tag

    def get_recently_used_ids(self, limit: int = 50) -> set[int]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id FROM memes WHERE last_used_at IS NOT NULL
                   ORDER BY last_used_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return {r["id"] for r in rows}

    def list_recently_used(self, limit: int = 10) -> list[dict]:
        """最近实际发送过的表情。"""
        with self._conn() as c:
            rows = c.execute(
                """SELECT id, file_name, file_path, tag, usage_count, last_used_at
                   FROM memes
                   WHERE last_used_at IS NOT NULL AND disabled = 0
                   ORDER BY last_used_at DESC LIMIT ?""",
                (max(int(limit), 1),),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_search_logs(self, limit: int = 10) -> list[dict]:
        """最近检索日志，用于解释和诊断。"""
        with self._conn() as c:
            rows = c.execute(
                """SELECT id, query_text, tag_filter, topk_ids, selected_id, similarity, created_at
                   FROM search_log
                   ORDER BY id DESC LIMIT ?""",
                (max(int(limit), 1),),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- Tags ----------

    def upsert_tag(self, name: str, description: str = "", category: str = "", color: str = "") -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO tags (name, description, category, color)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                     description = excluded.description,
                     category = excluded.category,
                     color = excluded.color""",
                (name, description, category, color),
            )

    def list_tags(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM tags ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # ---------- Logs ----------

    def log_search(self, query_text: str, tag_filter: str | None, topk_ids: list[int], selected_id: int | None, similarity: float | None) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO search_log
                    (query_text, tag_filter, topk_ids, selected_id, similarity)
                   VALUES (?, ?, ?, ?, ?)""",
                (query_text, tag_filter, json.dumps(topk_ids), selected_id, similarity),
            )

    def log_index_action(self, action: str, file_path: str, status: str, message: str = "") -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO index_log (action, file_path, status, message)
                   VALUES (?, ?, ?, ?)""",
                (action, file_path, status, message),
            )

    # ---------- 维护 ----------

    def rebuild_index_from_db(self) -> int:
        """用库里所有有效 meme 重新同步 vector_id -> id 映射。
        当 faiss index 重建后用此方法重建映射。
        """
        with self._conn() as c:
            # 找出 vector_id 重复 / 缺失的情况
            rows = c.execute("SELECT id, vector_id FROM memes WHERE disabled = 0 ORDER BY id").fetchall()
            for i, row in enumerate(rows):
                c.execute("UPDATE memes SET vector_id = ? WHERE id = ?", (i, row["id"]))
            return len(rows)

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) AS n FROM memes WHERE disabled = 0").fetchone()["n"]
            disabled = c.execute("SELECT COUNT(*) AS n FROM memes WHERE disabled = 1").fetchone()["n"]
            tags = c.execute("SELECT COUNT(DISTINCT tag) AS n FROM memes WHERE disabled = 0").fetchone()["n"]
        return {
            "total": total,
            "disabled": disabled,
            "tags": tags,
            "index_size": self.index_size,
            "dim": self.dim,
        }

    def health_check(self, root: str | Path | None = None) -> dict:
        """索引健康检查：DB/FAISS/文件系统一致性。"""
        with self._conn() as c:
            rows = c.execute("SELECT id, file_path, file_hash, tag, vector_id, disabled FROM memes").fetchall()
            tag_rows = c.execute("SELECT name FROM tags").fetchall()
            duplicate_hash_rows = c.execute(
                """SELECT file_hash, COUNT(*) AS n FROM memes
                   WHERE disabled = 0 GROUP BY file_hash HAVING n > 1"""
            ).fetchall()
        active = [dict(r) for r in rows if int(r["disabled"] or 0) == 0]
        known_tags = {r["name"] for r in tag_rows}
        used_tags = {r["tag"] for r in active}
        missing_files = []
        orphan_outside_root = []
        bad_vector_ids = []
        root_path = Path(root).resolve() if root else None
        for r in active:
            p = Path(r["file_path"])
            if not p.exists():
                missing_files.append({"id": r["id"], "file_path": r["file_path"]})
            if root_path is not None:
                try:
                    p.resolve().relative_to(root_path)
                except Exception:
                    orphan_outside_root.append({"id": r["id"], "file_path": r["file_path"]})
            try:
                vid = int(r["vector_id"])
            except Exception:
                vid = -1
            if vid < 0 or vid >= self.index_size:
                bad_vector_ids.append({"id": r["id"], "vector_id": r["vector_id"]})
        duplicate_hashes = [{"file_hash": r["file_hash"], "count": r["n"]} for r in duplicate_hash_rows]
        unregistered_tags = sorted(t for t in used_tags if t not in known_tags)
        index_mismatch = self.index_size < len(active)
        ok = not (missing_files or bad_vector_ids or index_mismatch)
        return {
            "ok": ok,
            "total": len(active),
            "disabled": len(rows) - len(active),
            "tag_count": len(used_tags),
            "registered_tag_count": len(known_tags),
            "index_size": self.index_size,
            "dim": self.dim,
            "index_mismatch": index_mismatch,
            "missing_files": missing_files,
            "missing_files_count": len(missing_files),
            "bad_vector_ids": bad_vector_ids,
            "bad_vector_ids_count": len(bad_vector_ids),
            "duplicate_hashes": duplicate_hashes,
            "duplicate_hash_count": len(duplicate_hashes),
            "unregistered_tags": unregistered_tags,
            "orphan_outside_root": orphan_outside_root,
            "orphan_outside_root_count": len(orphan_outside_root),
        }
