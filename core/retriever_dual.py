"""Dual-path retriever: image-text (existing CLIP) + caption-text (vision LLM).

Extends MemeRetriever without touching its internals: the base retrieve() call
still performs the image-text search, then this wrapper adds a second search
against stored caption vectors (CLIP text encoding of LLM-generated captions)
and fuses both rankings with per-path min-max normalization.

Instances are selected in main.py when enable_vision_caption is on; otherwise
the plain MemeRetriever is used, so existing behaviour is unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .retriever import MemeHit, MemeRetriever


class DualRetriever(MemeRetriever):
    """MemeRetriever with caption-vector fusion for semantic retrieval."""

    def __init__(
        self,
        db,
        embedder,
        caption_weight: float = 0.6,
        anti_repeat_window: int = 20,
        candidate_pool_size: int = 12,
        random_jitter: float = 0.015,
    ):
        super().__init__(
            db,
            embedder,
            anti_repeat_window=anti_repeat_window,
            candidate_pool_size=candidate_pool_size,
            random_jitter=random_jitter,
        )
        self.caption_weight = max(min(float(caption_weight), 1.0), 0.0)

    def _caption_candidates(self, tag: str | None) -> list[tuple[int, int]]:
        """Return [(caption_vector_id, meme_id)] filtered to valid FAISS ids."""
        rows = self.db.list_memes(tag=tag, limit=10_000_000)
        max_size = self.db.index_size
        result: list[tuple[int, int]] = []
        for row in rows:
            try:
                cvid = int(row['caption_vector_id'])
            except Exception:
                continue
            if cvid >= 0 and cvid < max_size:
                result.append((cvid, int(row['id'])))
        return result

    def _search_caption_path(
        self,
        query_vector: np.ndarray,
        candidates: list[tuple[int, int]],
        topk: int,
    ) -> dict[int, float]:
        """Search caption vectors, return {meme_id: cosine_similarity}."""
        if not candidates:
            return {}
        import faiss

        cap_ids = np.array([c[0] for c in candidates], dtype=np.int64)
        cvid_to_mid = {int(v): m for v, m in zip(cap_ids.tolist(), [c[1] for c in candidates])}
        sel = faiss.IDSelectorBatch(cap_ids)
        params = faiss.SearchParameters()
        params.sel = sel
        q = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        search_k = min(max(int(topk), self.candidate_pool_size), len(candidates))
        sims, found = self.db._index.search(q, search_k, params=params)  # noqa: SLF001
        out: dict[int, float] = {}
        for sim, vid in zip(sims[0], found[0]):
            if int(vid) < 0:
                continue
            mid = cvid_to_mid.get(int(vid))
            if mid is not None:
                out[mid] = float(sim)
        return out

    def _caption_hits(
        self,
        cap_scores: dict[int, float],
        topk: int,
        fallback_used: bool = False,
    ) -> list:
        """纯 caption 路（api 后端）：按 caption 分数构造 MemeHit 列表。"""
        order = sorted(cap_scores, key=cap_scores.get, reverse=True)[:max(int(topk), 0)]
        if not order:
            return []
        with self.db._conn() as c:  # noqa: SLF001
            placeholders = ",".join("?" * len(order))
            rows = c.execute(
                f"SELECT id, file_path, tag, usage_count, last_used_at, description "
                f"FROM memes WHERE id IN ({placeholders})",
                list(order),
            ).fetchall()
            meme_rows = {int(r["id"]): dict(r) for r in rows}
        hits = []
        for mid in order:
            row = meme_rows.get(mid)
            if row is None:
                continue
            raw = float(cap_scores[mid])
            hits.append(MemeHit(
                meme_id=mid,
                file_path=row["file_path"],
                tag=row["tag"],
                similarity=raw,
                raw_similarity=raw,
                usage_count=row.get("usage_count", 0) or 0,
                last_used_at=row.get("last_used_at"),
                description=row.get("description"),
                fallback_used=fallback_used,
            ))
        return hits

    @staticmethod
    def _minmax(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        lo = min(scores.values())
        hi = max(scores.values())
        if hi - lo < 1e-9:
            return {mid: 1.0 for mid in scores}
        return {mid: (v - lo) / (hi - lo) for mid, v in scores.items()}

    def retrieve(
        self,
        text: str,
        tag: str | None = None,
        topk: int = 5,
        anti_repeat: bool = True,
        fallback_to_all_tags: bool = True,
    ) -> Any:
        base = super().retrieve(
            text=text,
            tag=tag,
            topk=topk,
            anti_repeat=anti_repeat,
            fallback_to_all_tags=fallback_to_all_tags,
        )
        if self.caption_weight <= 0:
            return base

        # caption 候选集与图片路保持同一 tag 语义（fallback 同规则）
        cap_candidates = self._caption_candidates(tag)
        cap_fallback = False
        if not cap_candidates and fallback_to_all_tags:
            cap_candidates = self._caption_candidates(None)
            cap_fallback = True
        if not cap_candidates:
            return base

        query_vector = self.embedder.embed_text(self._build_query(text, tag))
        cap_scores = self._search_caption_path(query_vector, cap_candidates, topk)
        if not cap_scores:
            return base

        if not base.hits:
            # api 后端：图片路无向量，检索结果完全来自 caption 路
            base.hits = self._caption_hits(cap_scores, topk, fallback_used=cap_fallback)
            base.used_fallback = cap_fallback
            if base.hits:
                self.db.log_search(
                    query_text=text,
                    tag_filter=base.original_tag,
                    topk_ids=[h.meme_id for h in base.hits],
                    selected_id=base.hits[0].meme_id,
                    similarity=base.hits[0].similarity,
                )
            return base

        w_cap = self.caption_weight
        w_img = 1.0 - w_cap
        img_scores = {
            h.meme_id: (h.raw_similarity if h.raw_similarity is not None else h.similarity)
            for h in base.hits
        }
        norm_img = self._minmax(img_scores)
        norm_cap = self._minmax(cap_scores)
        fused: dict[int, float] = {}
        for h in base.hits:
            mid = h.meme_id
            img_s = norm_img.get(mid)
            cap_s = norm_cap.get(mid)
            if img_s is None:
                continue
            cap_part = 0.0 if cap_s is None else w_cap * float(cap_s)
            fused[mid] = w_img * float(img_s) + cap_part

        # 按融合分稳定排序；分数相同按原图片路分兜底
        order = sorted(
            base.hits,
            key=lambda h: (fused.get(h.meme_id, 0.0), img_scores.get(h.meme_id, 0.0)),
            reverse=True,
        )[:max(int(topk), 0)]
        for h in order:
            mid = h.meme_id
            h.image_similarity = float(img_scores.get(mid, h.similarity))
            h.caption_similarity = cap_scores.get(mid)
            fused_score = fused.get(mid)
            if fused_score is not None:
                h.similarity = fused_score
                h.raw_similarity = fused_score
        base.hits = order
        return base