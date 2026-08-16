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
import logging
import os
import shutil
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

logger = logging.getLogger(__name__)


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

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        self.index_corrupted = False
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
                    # 维度不匹配：采用文件实际维度，避免建空索引导致检索全空
                    self.dim = idx.d
                return idx
            except Exception as exc:
                logger.error("FAISS index load failed for %s: %s", self.index_path, exc)
                self.index_corrupted = True
                return faiss.IndexFlatIP(self.dim)
        # 用内积（IP）+ 归一化向量 = 余弦相似度
        return faiss.IndexFlatIP(self.dim)

    def save_index(self) -> None:
        """持久化 FAISS 索引。

        空索引也写盘（ntotal=0），否则重建空库时临时索引文件缺失、
        compact 到 0 后旧孤儿向量会在重启时复活。写盘采用临时文件 +
        os.replace，避免写一半损坏正式索引。
        """
        if not FAISS_AVAILABLE or self._index is None:
            return
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.index_path.with_name(self.index_path.name + ".tmp")
            faiss.write_index(self._index, str(tmp_path))
            os.replace(tmp_path, self.index_path)

    def reload_index(self) -> None:
        """从磁盘重新加载 FAISS 索引；文件缺失时重建空索引。"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss 未安装，请 pip install faiss-cpu")
        with self._lock:
            if self.index_path.exists():
                try:
                    idx = faiss.read_index(str(self.index_path))
                    self.dim = idx.d
                    self._index = idx
                    self.index_corrupted = False
                    return
                except Exception as exc:
                    logger.error("FAISS index reload failed for %s: %s", self.index_path, exc)
                    self._index = faiss.IndexFlatIP(self.dim)
                    self.index_corrupted = True
                    return
            self._index = faiss.IndexFlatIP(self.dim)
            self.index_corrupted = False

    def replace_index_in_place(self, index: Any, dim: int) -> None:
        """用已构建好的新索引原子替换内存索引（调用方需已替换磁盘文件）。"""
        with self._lock:
            self._index = index
            self.dim = int(dim)
            self.index_corrupted = False

    def backup_files(self, suffix: str = ".bak.v071") -> list[tuple[Path, Path, bool]]:
        """备份 SQLite 与 FAISS 文件。

        Returns:
            [(original_path, backup_path, existed_before), ...]
        """
        pairs: list[tuple[Path, Path, bool]] = []
        for p in (self.db_path, self.index_path):
            bak = p.with_name(p.name + suffix)
            existed = p.exists()
            if not existed:
                pairs.append((p, bak, False))
                continue
            try:
                if p == self.db_path:
                    with self._lock:
                        src = sqlite3.connect(self.db_path)
                        dst = sqlite3.connect(bak)
                        try:
                            src.backup(dst)
                        finally:
                            src.close()
                            dst.close()
                else:
                    with self._lock:
                        shutil.copy2(p, bak)
                pairs.append((p, bak, True))
            except Exception:
                logger.exception("backup failed: %s -> %s", p, bak)
                for _, created_bak, _ in pairs:
                    created_bak.unlink(missing_ok=True)
                bak.unlink(missing_ok=True)
                # 备份失败视为致命错误：不能让调用方在无备份的情况下继续替换
                raise
        return pairs

    @property
    def index_size(self) -> int:
        with self._lock:
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

    def reset_index(self) -> None:
        """用当前维度新建空索引（带锁），用于重建。"""
        with self._lock:
            self.index_corrupted = False
            self._index = faiss.IndexFlatIP(self.dim)

    def search(self, query: np.ndarray, topk: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """返回 (相似度, vector_id)。"""
        query = query.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        with self._lock:
            sims, ids = self._index.search(query, topk)
        return sims[0], ids[0]

    def search_index(
        self,
        query: np.ndarray,
        topk: int,
        ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """带锁检索；ids 非空时用 IDSelectorBatch 限定候选集。"""
        query = query.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        with self._lock:
            if ids is not None and len(ids):
                sel = faiss.IDSelectorBatch(ids)
                params = faiss.SearchParameters()
                params.sel = sel
                sims, found = self._index.search(query, min(int(topk), len(ids)), params=params)
            else:
                sims, found = self._index.search(query, topk)
        return sims[0], found[0]

    def reconstruct(self, vector_id: int) -> np.ndarray:
        """按 vector_id 取回向量（带锁）。"""
        with self._lock:
            return self._index.reconstruct(int(vector_id))

    def reconstruct_all(self) -> np.ndarray:
        """取回索引全部向量（带锁）。"""
        with self._lock:
            return self._index.reconstruct_n(0, self._index.ntotal)

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

    def copy_runtime_state_from(self, source: "MemeDatabase") -> int:
        """从旧库复制运行时状态（使用统计/禁用标记/日志），用于重建保留历史。"""
        with source._conn() as c:
            meme_rows = c.execute(
                "SELECT file_path, usage_count, last_used_at, disabled FROM memes"
            ).fetchall()
            search_rows = c.execute(
                "SELECT query_text, tag_filter, topk_ids, selected_id, similarity, "
                "created_at FROM search_log"
            ).fetchall()
            index_rows = c.execute(
                "SELECT action, file_path, status, message, created_at FROM index_log"
            ).fetchall()

        copied = 0
        with self._conn() as c:
            for r in meme_rows:
                cur = c.execute(
                    "UPDATE memes SET usage_count = ?, last_used_at = ?, disabled = ? "
                    "WHERE file_path = ?",
                    (r["usage_count"], r["last_used_at"], r["disabled"], r["file_path"]),
                )
                copied += cur.rowcount
            c.executemany(
                "INSERT INTO search_log "
                "(query_text, tag_filter, topk_ids, selected_id, similarity, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        r["query_text"],
                        r["tag_filter"],
                        r["topk_ids"],
                        r["selected_id"],
                        r["similarity"],
                        r["created_at"],
                    )
                    for r in search_rows
                ],
            )
            c.executemany(
                "INSERT INTO index_log (action, file_path, status, message, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r["action"],
                        r["file_path"],
                        r["status"],
                        r["message"],
                        r["created_at"],
                    )
                    for r in index_rows
                ],
            )
        return copied

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

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

    def list_candidate_vector_ids(self, tag: str | None = None) -> list[tuple[int, int]]:
        sql = "SELECT id, vector_id FROM memes WHERE disabled = 0"
        params: list[Any] = []
        if tag:
            sql += " AND tag = ?"
            params.append(tag)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        max_size = self.index_size
        out: list[tuple[int, int]] = []
        for r in rows:
            try:
                vid = int(r["vector_id"])
            except Exception:
                continue
            if 0 <= vid < max_size:
                out.append((vid, int(r["id"])))
        return out

    def list_caption_vector_ids(self, tag: str | None = None) -> list[tuple[int, int]]:
        sql = "SELECT id, caption_vector_id FROM memes WHERE disabled = 0"
        params: list[Any] = []
        if tag:
            sql += " AND tag = ?"
            params.append(tag)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        max_size = self.index_size
        out: list[tuple[int, int]] = []
        for r in rows:
            try:
                cvid = int(r["caption_vector_id"]) if r["caption_vector_id"] is not None else -1
            except Exception:
                continue
            if 0 <= cvid < max_size:
                out.append((cvid, int(r["id"])))
        return out

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

    def total_row_count(self) -> int:
        """包含禁用行的总记录数，用于判断是否有旧向量需要保护。"""
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM memes").fetchone()
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

    def get_tag(self, name: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tags WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

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

    def live_vector_count(self) -> int:
        """DB 中仍被引用（含禁用行）的向量槽数量，用于孤儿判断。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT vector_id, caption_vector_id FROM memes"
            ).fetchall()
        max_size = self.index_size
        count = 0
        for r in rows:
            for col in ("vector_id", "caption_vector_id"):
                try:
                    vid = int(r[col]) if r[col] is not None else -1
                except Exception:
                    vid = -1
                if 0 <= vid < max_size:
                    count += 1
        return count

    def compact_index(self) -> dict:
        """重建 FAISS，移除孤儿向量并重映射 vector_id / caption_vector_id。

        保留 disabled 行仍被引用的向量，避免重新启用后失效。
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, vector_id, caption_vector_id FROM memes ORDER BY id"
            ).fetchall()
        max_size = self.index_size
        image_rows: list[sqlite3.Row] = []
        caption_rows: list[sqlite3.Row] = []
        for r in rows:
            try:
                vid = int(r["vector_id"])
            except Exception:
                vid = -1
            if 0 <= vid < max_size:
                image_rows.append(r)
            try:
                cvid = int(r["caption_vector_id"]) if r["caption_vector_id"] is not None else -1
            except Exception:
                cvid = -1
            if 0 <= cvid < max_size:
                caption_rows.append(r)

        image_vectors: list[np.ndarray] = []
        caption_vectors: list[np.ndarray] = []
        with self._lock:
            for r in image_rows:
                image_vectors.append(self._index.reconstruct(int(r["vector_id"])))
            for r in caption_rows:
                caption_vectors.append(self._index.reconstruct(int(r["caption_vector_id"])))
            new_index = faiss.IndexFlatIP(self.dim)
            if image_vectors:
                arr = np.stack(image_vectors).astype("float32")
                faiss.normalize_L2(arr)
                new_index.add(arr)
            if caption_vectors:
                arr = np.stack(caption_vectors).astype("float32")
                faiss.normalize_L2(arr)
                new_index.add(arr)

            with self._conn() as c:
                for i, r in enumerate(image_rows):
                    c.execute("UPDATE memes SET vector_id = ? WHERE id = ?", (i, int(r["id"])))
                for j, r in enumerate(caption_rows):
                    c.execute(
                        "UPDATE memes SET caption_vector_id = ? WHERE id = ?",
                        (len(image_vectors) + j, int(r["id"])),
                    )
            self._index = new_index

        self.index_corrupted = False
        self.save_index()
        return {
            "index_size": self.index_size,
            "image_vectors": len(image_vectors),
            "caption_vectors": len(caption_vectors),
        }

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
            "index_corrupted": self.index_corrupted,
        }

    def health_check(
        self, root: str | Path | None = None, verify_hashes: bool = False
    ) -> dict:
        """索引健康检查：DB/FAISS/文件系统一致性。"""
        with self._conn() as c:
            rows = c.execute("SELECT id, file_path, file_hash, tag, vector_id, caption_vector_id, disabled FROM memes").fetchall()
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
        bad_caption_vector_ids = []
        stale_hash_files = []
        root_path = Path(root).resolve() if root else None
        for r in active:
            p = Path(r["file_path"])
            if not p.exists():
                missing_files.append({"id": r["id"], "file_path": r["file_path"]})
            elif verify_hashes:
                try:
                    current_hash = compute_file_hash(p)
                except Exception:
                    current_hash = ""
                if current_hash != r["file_hash"]:
                    stale_hash_files.append(
                        {
                            "id": r["id"],
                            "file_path": r["file_path"],
                            "db_hash": r["file_hash"],
                            "disk_hash": current_hash,
                        }
                    )
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
            if r["caption_vector_id"] is not None:
                try:
                    cvid = int(r["caption_vector_id"])
                except Exception:
                    cvid = -1
                if cvid < 0 or cvid >= self.index_size:
                    bad_caption_vector_ids.append({"id": r["id"], "caption_vector_id": r["caption_vector_id"]})
        duplicate_hashes = [{"file_hash": r["file_hash"], "count": r["n"]} for r in duplicate_hash_rows]
        unregistered_tags = sorted(t for t in used_tags if t not in known_tags)
        index_mismatch = self.index_size < len(active)
        max_size = self.index_size
        live_vectors = 0
        for r in rows:
            for col in ("vector_id", "caption_vector_id"):
                try:
                    vid = int(r[col]) if r[col] is not None else -1
                except Exception:
                    vid = -1
                if 0 <= vid < max_size:
                    live_vectors += 1
        orphan_vector_count = max(max_size - live_vectors, 0)
        ok = not (
            missing_files
            or bad_vector_ids
            or bad_caption_vector_ids
            or index_mismatch
            or self.index_corrupted
            or (verify_hashes and stale_hash_files)
        )
        return {
            "ok": ok,
            "total": len(active),
            "disabled": len(rows) - len(active),
            "tag_count": len(used_tags),
            "registered_tag_count": len(known_tags),
            "index_size": self.index_size,
            "dim": self.dim,
            "index_corrupted": self.index_corrupted,
            "index_mismatch": index_mismatch,
            "orphan_vector_count": orphan_vector_count,
            "live_vector_count": live_vectors,
            "missing_files": missing_files,
            "missing_files_count": len(missing_files),
            "bad_vector_ids": bad_vector_ids,
            "bad_vector_ids_count": len(bad_vector_ids),
            "bad_caption_vector_ids": bad_caption_vector_ids,
            "bad_caption_vector_ids_count": len(bad_caption_vector_ids),
            "duplicate_hashes": duplicate_hashes,
            "duplicate_hash_count": len(duplicate_hashes),
            "unregistered_tags": unregistered_tags,
            "orphan_outside_root": orphan_outside_root,
            "orphan_outside_root_count": len(orphan_outside_root),
            "stale_hash_files": stale_hash_files,
            "stale_hash_count": len(stale_hash_files),
            "verify_hashes": verify_hashes,
        }
