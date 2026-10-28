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
        **kwargs: Any,
    ):
        super().__init__(
            db,
            embedder,
            anti_repeat_window=anti_repeat_window,
            candidate_pool_size=candidate_pool_size,
            random_jitter=random_jitter,
            **kwargs,
        )
        self.caption_weight = max(min(float(caption_weight), 1.0), 0.0)

    def _caption_candidates(self, tag: str | None) -> list[tuple[int, int]]:
        """Return [(caption_vector_id, meme_id)] filtered to valid FAISS ids."""
        return self.db.list_caption_vector_ids(tag=tag)

    def _search_caption_path(
        self,
        query_vector: np.ndarray,
        candidates: list[tuple[int, int]],
        topk: int,
    ) -> dict[int, float]:
        """Search caption vectors, return {meme_id: cosine_similarity}."""
        return dict(self._search_meme_scores(query_vector, candidates, topk))

    def _caption_hits(
        self,
        cap_scores: dict[int, float],
        topk: int,
        fallback_used: bool = False,
    ) -> list:
        """纯 caption 路（api 后端）：按 caption 分数构造 MemeHit 列表。"""
        order = sorted(cap_scores, key=cap_scores.get, reverse=True)[:max(int(topk), 0)]
        scores = [(mid, float(cap_scores[mid])) for mid in order]
        return self._build_hits(
            scores, self._load_meme_rows(order), fallback_used=fallback_used
        )

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
        rerank: bool = True,
        query_vector: np.ndarray | None = None,
        expand_pool: bool | None = None,
    ) -> Any:
        """与基类签名保持一致，query_vector 非空时直接复用，不再重复调用 embedder。

        回归背景：v0.8.0 给 pick()/pick_multiple() 加了 query_vector 直传，但本方法
        当时漏了同名参数。api 后端恒走 DualRetriever，于是每次自动表情都抛
        TypeError: DualRetriever.retrieve() got an unexpected keyword argument
        'query_vector'，被 on_decorating_result 的兜底 except 吞掉——内容检索、
        反重复、加权采样整条路径静默失效，只剩外部 <sticker> 直查在出图。
        """
        internal_topk = max(int(topk), self.candidate_pool_size)
        query_text = self._build_query(text, tag)
        if query_vector is None:
            query_vector = self.embedder.embed_text(query_text)
        else:
            query_vector = np.asarray(query_vector, dtype="float32").reshape(-1)
        base = super().retrieve(
            text=text,
            tag=tag,
            topk=internal_topk,
            anti_repeat=anti_repeat,
            fallback_to_all_tags=fallback_to_all_tags,
            query_vector=query_vector,
            rerank=rerank,
            expand_pool=expand_pool,
        )
        if self.caption_weight <= 0:
            return base

        expanded_tags: list[str] = list(base.expanded_tags or [])

        # caption 候选同样走 tag 过滤 -> 回退全库 -> 小池扩展（与图片路径共享实现）
        cap_candidates, cap_fallback, cap_expanded = self._resolve_candidates(
            tag,
            self.db.list_caption_vector_ids,
            query_vector,
            fallback_to_all_tags=fallback_to_all_tags,
            rerank=True,
            expand_pool=expand_pool,
        )
        for t in cap_expanded or []:
            if t not in expanded_tags:
                expanded_tags.append(t)
        if not cap_candidates:
            return base

        cap_scores = self._search_caption_path(query_vector, cap_candidates, internal_topk)
        if not cap_scores:
            return base

        caption_hits = self._caption_hits(cap_scores, internal_topk, fallback_used=cap_fallback)
        if not base.hits:
            if rerank:
                base.hits = self._rerank(
                    caption_hits,
                    anti_repeat=anti_repeat,
                    requested_tag=base.original_tag,
                    fallback_used=cap_fallback,
                    vectors_by_meme_id=self._vectors_by_meme_id([h.meme_id for h in caption_hits]),
                    expanded_tags=expanded_tags or None,
                )[:max(int(topk), 0)]
            else:
                caption_hits.sort(
                    key=lambda h: h.raw_similarity
                    if h.raw_similarity is not None
                    else h.similarity,
                    reverse=True,
                )
                base.hits = caption_hits[:max(int(topk), 0)]
            base.used_fallback = cap_fallback
            base.expanded_tags = expanded_tags or None
            self._log_search(text, base.original_tag, base.hits)
            return base

        w_cap = self.caption_weight
        w_img = 1.0 - w_cap
        base_by_id = {h.meme_id: h for h in base.hits}
        cap_by_id = {h.meme_id: h for h in caption_hits}
        all_ids = set(base_by_id) | set(cap_by_id)
        img_scores = {
            mid: (base_by_id[mid].raw_similarity if base_by_id[mid].raw_similarity is not None else base_by_id[mid].similarity)
            for mid in all_ids
            if mid in base_by_id
        }
        cap_scores_by_id = {
            mid: (cap_by_id[mid].raw_similarity if cap_by_id[mid].raw_similarity is not None else cap_by_id[mid].similarity)
            for mid in all_ids
            if mid in cap_by_id
        }
        norm_img = self._minmax(img_scores)
        norm_cap = self._minmax(cap_scores_by_id)

        fused_hits = []
        for mid in all_ids:
            source = base_by_id.get(mid) or cap_by_id.get(mid)
            img_s = norm_img.get(mid, 0.0)
            cap_s = norm_cap.get(mid, 0.0)
            fused_raw = w_img * float(img_s) + w_cap * float(cap_s)
            fused_hits.append(MemeHit(
                meme_id=mid,
                file_path=source.file_path,
                tag=source.tag,
                similarity=fused_raw,
                usage_count=source.usage_count,
                last_used_at=source.last_used_at,
                description=source.description,
                raw_similarity=fused_raw,
                fallback_used=(source.fallback_used or cap_fallback),
                rank_before_rerank=source.rank_before_rerank,
                image_similarity=img_scores.get(mid),
                caption_similarity=cap_scores_by_id.get(mid),
            ))

        if rerank:
            base.hits = self._rerank(
                fused_hits,
                anti_repeat=anti_repeat,
                requested_tag=base.original_tag,
                fallback_used=(base.used_fallback or cap_fallback),
                vectors_by_meme_id=self._vectors_by_meme_id([h.meme_id for h in fused_hits]),
                expanded_tags=expanded_tags or None,
            )[:max(int(topk), 0)]
        else:
            fused_hits.sort(key=lambda h: h.similarity, reverse=True)
            base.hits = fused_hits[:max(int(topk), 0)]
        base.expanded_tags = expanded_tags or None
        return base
