"""近重复表情簇检测与去重。

针对"库里同一张表情的不同画质/尺寸/截图版本在嵌入空间余弦 >= threshold
挤成一簇，导致不同语义向量都命中同一簇代表图"的问题：
- cluster_near_duplicates(): 基于 FAISS 向量做余弦聚类，输出簇与簇内成员
- apply_dedup(): 每组保留一张代表图（使用次数最多、其次 id 最小），其余置为
  disabled 并在 sub_tags 打 dedup:<rep_id> 标记
- undo_dedup(): 撤销 apply，重新启用并清除标记
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .database import MemeDatabase

logger = logging.getLogger(__name__)

DEDUP_TAG_PREFIX = "dedup:"


@dataclass
class DupGroup:
    rep_id: int
    rep_path: str
    rep_tag: str
    rep_usage: int
    threshold: float
    members: list[dict] = field(default_factory=list)  # meme rows（含代表图）

    @property
    def member_ids(self) -> list[int]:
        return [int(m["id"]) for m in self.members]

    @property
    def dup_ids(self) -> list[int]:
        return [int(m["id"]) for m in self.members if int(m["id"]) != int(self.rep_id)]


def _active_vector_rows(db: MemeDatabase) -> tuple[list[dict], np.ndarray]:
    """返回 (rows, normalized_vectors)，仅含有效 vector_id 的启用行。"""
    rows = []
    vecs = []
    if db.index_size <= 0:
        return rows, np.empty((0, db.dim), dtype="float32")
    all_vecs = db.reconstruct_all()
    for r in db.list_memes(limit=10_000_000):
        try:
            vid = int(r["vector_id"])
        except Exception:
            continue
        if 0 <= vid < db.index_size:
            rows.append(r)
            vecs.append(all_vecs[vid])
    if not vecs:
        return rows, np.empty((0, db.dim), dtype="float32")
    arr = np.stack(vecs, axis=0).astype("float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return rows, arr


def cluster_near_duplicates(
    db: MemeDatabase,
    threshold: float = 0.95,
    min_group: int = 2,
    max_groups: int = 200,
) -> list[DupGroup]:
    """返回近重复簇，每个簇至少 min_group 个成员；按代表图使用次数降序。"""
    rows, vecs = _active_vector_rows(db)
    if len(rows) < 2 or vecs.size == 0:
        return []
    threshold = max(min(float(threshold), 1.0), 0.0)
    S = vecs @ vecs.T
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 邻接对（>= threshold）做并查集；阈值为 0 时退化为一整簇
    for i in range(n):
        for j in range(i + 1, n):
            if float(S[i, j]) >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    groups: list[DupGroup] = []
    for idxs in clusters.values():
        if len(idxs) < int(min_group):
            continue
        members = [dict(rows[i]) for i in idxs]
        rep = max(members, key=lambda m: (int(m.get("usage_count", 0) or 0), -int(m["id"])))
        groups.append(DupGroup(
            rep_id=int(rep["id"]),
            rep_path=str(rep["file_path"]),
            rep_tag=str(rep["tag"]),
            rep_usage=int(rep.get("usage_count", 0) or 0),
            threshold=threshold,
            members=members,
        ))
    groups.sort(key=lambda g: (-g.rep_usage, g.rep_id))
    return groups[:max(int(max_groups), 1)]


def apply_dedup(
    db: MemeDatabase,
    threshold: float = 0.95,
    min_group: int = 2,
    dry_run: bool = True,
) -> dict:
    """禁用每簇除代表外的成员；dry_run=True 只统计不改库。

    Returns:
        {"dry_run": bool, "groups": int, "kept": [ids], "disabled": [ids],
         "clusters": [{rep_id, rep_file, tag, dup_count, disabled, kept}]}
    """
    groups = cluster_near_duplicates(db, threshold=threshold, min_group=min_group)
    to_disable: list[int] = []
    kept: list[int] = []
    clusters_info = []
    for g in groups:
        g_disable = g.dup_ids
        to_disable.extend(g_disable)
        kept.append(g.rep_id)
        clusters_info.append({
            "rep_id": g.rep_id,
            "rep_file": g.rep_path,
            "tag": g.rep_tag,
            "dup_count": len(g_disable),
            "disabled": g_disable,
            "kept": [g.rep_id],
        })
    stats = {
        "dry_run": bool(dry_run),
        "threshold": float(threshold),
        "groups": len(groups),
        "kept": kept,
        "disabled": to_disable,
        "clusters": clusters_info,
    }
    if dry_run or not to_disable:
        return stats

    for g in groups:
        for m in g.members:
            mid = int(m["id"])
            if mid == int(g.rep_id):
                continue
            db.set_disabled(mid, True)
            db.append_meme_subtag(mid, f"{DEDUP_TAG_PREFIX}{g.rep_id}")
            logger.info("dedup: disable meme #%s (rep #%s)", mid, g.rep_id)
    return stats


def _has_dedup_marker(sub_tags) -> bool:
    try:
        tags = json.loads(sub_tags or "[]")
    except Exception:
        return False
    if not isinstance(tags, list):
        return False
    return any(str(t).startswith(DEDUP_TAG_PREFIX) for t in tags)


def undo_dedup(db: MemeDatabase, remove_marker: bool = True) -> dict:
    """重新启用所有带 dedup:<rep> 标记的成员，并（可选）清除标记。"""
    reenabled: list[int] = []
    with db._conn() as c:  # noqa: SLF001
        rows = c.execute(
            "SELECT id, sub_tags FROM memes WHERE disabled = 1"
        ).fetchall()
        for r in rows:
            if not _has_dedup_marker(dict(r).get("sub_tags")):
                continue
            mid = int(r["id"])
            db.set_disabled(mid, False)
            if remove_marker:
                try:
                    tags = json.loads(r["sub_tags"] or "[]")
                    tags = [t for t in tags if not str(t).startswith(DEDUP_TAG_PREFIX)]
                except Exception:
                    tags = []
                with db._conn() as c2:  # noqa: SLF001
                    c2.execute(
                        "UPDATE memes SET sub_tags = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(tags, ensure_ascii=False), time.time(), mid),
                    )
            reenabled.append(mid)
    return {"reenabled": reenabled, "count": len(reenabled)}
