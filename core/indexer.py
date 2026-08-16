"""表情包索引器。

职责：
- 扫描目录中的图片
- 增量更新（基于文件 hash）
- 调用 embedder 生成向量
- 写入数据库

子目录约定：
- 开启 use_subdir_as_tag 时，子目录名作为 tag
- 例：memes/happy/001.png → tag=happy
- 例：memes/001.png → tag=default_tag
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from .database import MemeDatabase, compute_file_hash
from .embedder import BaseEmbedder

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def tag_description(item: dict) -> str:
    """把 schema 单条定义拼成 DB description（persona 注入用）。"""
    parts = [str(item.get("meaning") or "")]
    visual = item.get("visual_features") or []
    exclude = item.get("exclude") or []
    if visual:
        parts.append("画面特征：" + "、".join(str(v) for v in visual))
    if exclude:
        parts.append("排除：" + "、".join(str(e) for e in exclude))
    return "；".join(p for p in parts if p)


def tag_meta(tag: str, schema: dict) -> dict:
    """从 schema 取 tag 的 (description, category, color)，缺省为空串。"""
    item = schema.get(tag) or {}
    if not isinstance(item, dict):
        return {"description": "", "category": "", "color": ""}
    return {
        "description": tag_description(item),
        "category": str(item.get("category") or ""),
        "color": str(item.get("color") or ""),
    }


def iter_image_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    """遍历目录里的图片文件。"""
    root = Path(root)
    if not root.exists():
        return
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                yield p


def _read_image_meta(path: Path) -> tuple[int, int, int] | None:
    """返回 (width, height, file_size)，失败返回 None。"""
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w, h, path.stat().st_size
    except (UnidentifiedImageError, OSError):
        return None


def _embed_one(embedder: BaseEmbedder, path: Path) -> "object | None":
    """对单张图嵌入。GIF 取第一帧。"""
    try:
        with Image.open(path) as img:
            if getattr(img, "is_animated", False):
                img.seek(0)
            img = ImageOps.exif_transpose(img)
            return embedder.embed_image(img.convert("RGB"))
    except NotImplementedError:
        # api 后端：图片不可直接编码，主向量留空（由调用方按 skipped 统计，检索走 caption 路）
        raise
    except Exception as e:
        logger.warning("嵌入失败 %s: %s", path, e)
        return None


class IndexProgress:
    """索引进度回调包装。"""

    def __init__(self, on_progress: Callable[[int, int, Path], None] | None = None):
        self.on_progress = on_progress
        self.total = 0
        self.done = 0
        self.added = 0
        self.updated = 0
        self.skipped = 0
        self.failed = 0

    def __call__(self, file: Path):
        if self.on_progress:
            self.on_progress(self.done, self.total, file)


class MemeIndexer:
    """扫描目录、生成向量、写入数据库。"""

    def __init__(
        self,
        db: MemeDatabase,
        embedder: BaseEmbedder,
        use_subdir_as_tag: bool = True,
        default_tag: str = "misc",
        tag_schema: dict | None = None,
    ):
        self.db = db
        self.embedder = embedder
        self.use_subdir_as_tag = use_subdir_as_tag
        self.default_tag = default_tag
        self.tag_schema = tag_schema or {}

    def _resolve_tag(self, file_path: Path, root: Path) -> str:
        if not self.use_subdir_as_tag:
            return self.default_tag
        try:
            rel = file_path.relative_to(root)
        except ValueError:
            return self.default_tag
        parts = rel.parts
        if len(parts) <= 1:
            return self.default_tag
        return parts[0]

    def index_directory(
        self,
        root: str | Path,
        recursive: bool = True,
        batch_size: int = 16,
        progress: IndexProgress | None = None,
        tag_root: str | Path | None = None,
    ) -> IndexProgress:
        """索引一个目录（增量）。

        tag_root 用于解析子目录 tag；缺省等于 root。传入更大的 tag_root
        时，可以扫描 root 子目录但保持与原库一致的 tag 归属。
        """
        # 统一存绝对路径，避免 cwd 变化后 DB 里的相对路径失效
        root = Path(root).resolve()
        tag_root = Path(tag_root).resolve() if tag_root is not None else root
        progress = progress or IndexProgress()

        files = list(iter_image_files(root, recursive=recursive))
        progress.total = len(files)
        if not files:
            return progress

        # 收集已存在记录的 hash（一次 SQL），做增量判重用
        with self.db._conn() as c:  # noqa: SLF001 - 受控内部访问
            rows = c.execute("SELECT file_path, file_hash FROM memes").fetchall()
            existing = {r["file_path"]: r["file_hash"] for r in rows}

        # 区分需要嵌入 vs 跳过
        to_embed: list[Path] = []
        file_meta: dict[Path, tuple[str, int, int, int, int]] = {}
        # (file_hash, width, height, file_size, vector_id_placeholder)

        for p in files:
            p_str = str(p)
            meta = _read_image_meta(p)
            if meta is None:
                progress.failed += 1
                self.db.log_index_action("add", p_str, "failed", "无法读取图片")
                continue
            w, h, size = meta
            fhash = compute_file_hash(p)
            file_meta[p] = (fhash, w, h, size, -1)
            if p_str in existing and existing[p_str] == fhash:
                progress.skipped += 1
                progress.done += 1
                if progress.on_progress:
                    progress.on_progress(progress.done, progress.total, p)
                continue
            to_embed.append(p)

        # 批嵌入
        vectors_batch: list = []
        meta_batch: list[Path] = []

        def flush():
            if not vectors_batch:
                return
            arr = __import__("numpy").stack(vectors_batch, axis=0)
            v_ids = self.db.add_vectors(arr)
            for path, vid in zip(meta_batch, v_ids):
                fhash, w, h, size, _ = file_meta[path]
                tag = self._resolve_tag(path, root)
                self.db.upsert_tag(tag, **tag_meta(tag, self.tag_schema))
                is_update = str(path) in existing
                self.db.upsert_meme(
                    file_path=str(path),
                    file_hash=fhash,
                    tag=tag,
                    vector_id=int(vid),
                    file_name=path.name,
                    width=w,
                    height=h,
                    file_size=size,
                )
                if is_update:
                    progress.updated += 1
                    self.db.log_index_action("update", str(path), "success")
                else:
                    progress.added += 1
                    self.db.log_index_action("add", str(path), "success")
            vectors_batch.clear()
            meta_batch.clear()

        def upsert_without_vector(path: Path):
            """api 后端：图片不可编码，仍入库 meme 记录（vector_id=-1），caption 向量由 enrich 阶段建立。"""
            fhash, w, h, size, _ = file_meta[path]
            tag = self._resolve_tag(path, tag_root)
            self.db.upsert_tag(tag, **tag_meta(tag, self.tag_schema))
            is_update = str(path) in existing
            self.db.upsert_meme(
                file_path=str(path),
                file_hash=fhash,
                tag=tag,
                vector_id=-1,
                file_name=path.name,
                width=w,
                height=h,
                file_size=size,
            )
            if is_update:
                progress.updated += 1
                self.db.log_index_action("update", str(path), "success")
            else:
                progress.added += 1
                self.db.log_index_action("add", str(path), "success")

        for p in to_embed:
            try:
                vec = _embed_one(self.embedder, p)
            except NotImplementedError:
                # api 后端：图片不可直接编码，主向量留空（vector_id=-1），检索走 caption 路
                upsert_without_vector(p)
                progress.done += 1
                if progress.on_progress:
                    progress.on_progress(progress.done, progress.total, p)
                continue
            if vec is None:
                progress.failed += 1
                self.db.log_index_action("add", str(p), "failed", "嵌入失败")
                progress.done += 1
                if progress.on_progress:
                    progress.on_progress(progress.done, progress.total, p)
                continue
            vectors_batch.append(vec)
            meta_batch.append(p)
            progress.done += 1
            if progress.on_progress:
                progress.on_progress(progress.done, progress.total, p)
            if len(vectors_batch) >= batch_size:
                flush()

        flush()
        self.db.save_index()
        return progress

    def remove_missing(self, root: str | Path) -> int:
        """从数据库中移除 root 下已经不存在的文件。"""
        root = Path(root).resolve()
        with self.db._conn() as c:  # noqa: SLF001
            rows = c.execute("SELECT id, file_path FROM memes").fetchall()
        removed = 0
        for r in rows:
            p = Path(r["file_path"])
            try:
                p.relative_to(root)
            except ValueError:
                continue  # 不在 root 范围内，不处理
            if not p.exists():
                self.db.remove_meme(r["id"])
                self.db.log_index_action("remove", r["file_path"], "success", "文件已删除")
                removed += 1
        return removed
