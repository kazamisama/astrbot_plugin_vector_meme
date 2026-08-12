"""两阶段检索器。

阶段 1（粗筛）：按 tag 过滤候选集
阶段 2（精排）：用 embedder 把 query 文本向量化，在候选集中找最相似

反重复：通过最近使用窗口、使用频次和时间衰减降低重复概率。
分类：基于已标注图片向量构建 tag prototype，并结合 KNN 邻居投票。
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .database import MemeDatabase
from .embedder import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class MemeHit:
    meme_id: int
    file_path: str
    tag: str
    similarity: float
    usage_count: int = 0
    last_used_at: float | None = None
    description: str | None = None
    raw_similarity: float | None = None
    tag_bonus: float = 0.0
    repeat_penalty: float = 0.0
    usage_penalty: float = 0.0
    random_jitter: float = 0.0
    fallback_used: bool = False
    rank_before_rerank: int | None = None
    image_similarity: float | None = None
    caption_similarity: float | None = None

    @property
    def name(self) -> str:
        return Path(self.file_path).name

    @property
    def explain_lines(self) -> list[str]:
        lines = [
            f"id=#{self.meme_id} file={self.name}",
            f"tag={self.tag}",
            f"raw_similarity={(self.raw_similarity if self.raw_similarity is not None else self.similarity):.3f}",
            f"final_score={self.similarity:.3f}",
        ]
        if self.tag_bonus:
            lines.append(f"tag_bonus=+{self.tag_bonus:.3f}")
        if self.repeat_penalty:
            lines.append(f"repeat_penalty=-{self.repeat_penalty:.3f}")
        if self.usage_penalty:
            lines.append(f"usage_penalty=-{self.usage_penalty:.3f}")
        if self.random_jitter:
            sign = "+" if self.random_jitter >= 0 else ""
            lines.append(f"random_jitter={sign}{self.random_jitter:.3f}")
        if self.fallback_used:
            lines.append("fallback=目标 tag 无候选，已回退全库")
        if self.rank_before_rerank is not None:
            lines.append(f"rank_before_rerank={self.rank_before_rerank}")
        if self.last_used_at:
            elapsed = max(time.time() - self.last_used_at, 0)
            lines.append(f"last_used={elapsed:.0f}s ago")
        if self.usage_count:
            lines.append(f"usage_count={self.usage_count}")
        return lines


@dataclass
class RetrievalResult:
    query_text: str
    tag_filter: str | None
    hits: list[MemeHit] = field(default_factory=list)
    used_fallback: bool = False
    original_tag: str | None = None

    def __bool__(self) -> bool:
        return bool(self.hits)

    def top(self) -> MemeHit | None:
        return self.hits[0] if self.hits else None


class MemeRetriever:
    """检索、采样、解释与 prototype/KNN 分类。"""

    def __init__(
        self,
        db: MemeDatabase,
        embedder: BaseEmbedder,
        anti_repeat_window: int = 20,
        candidate_pool_size: int = 12,
        random_jitter: float = 0.015,
    ):
        self.db = db
        self.embedder = embedder
        self.anti_repeat_window = max(int(anti_repeat_window), 0)
        self.candidate_pool_size = max(int(candidate_pool_size), 1)
        self.random_jitter = max(float(random_jitter), 0.0)
        # 默认开启加权采样；可通过 pick*() 参数显式覆盖
        self._stochastic_default = True

    def _candidate_vector_ids(self, tag: str | None) -> list[tuple[int, int]]:
        """返回 [(vector_id, meme_id)]，自动过滤越界或无效 vector_id。"""
        return self.db.list_candidate_vector_ids(tag=tag)

    def _build_query(self, text: str, tag: str | None) -> str:
        """构造向量查询文本。

        tag 不混入 query：tag 负责限定候选块，回复文本负责挑具体图片。
        """
        return text

    def _rerank(
        self,
        candidates: list[MemeHit],
        anti_repeat: bool = True,
        requested_tag: str | None = None,
        fallback_used: bool = False,
    ) -> list[MemeHit]:
        """混合重排序：相似度 + tag 加权 - 最近/高频惩罚 + 小随机扰动。"""
        if not candidates:
            return candidates

        now = time.time()
        recent_ids = self.db.get_recently_used_ids(self.anti_repeat_window) if anti_repeat and self.anti_repeat_window else set()
        adjusted: list[MemeHit] = []
        for rank, h in enumerate(candidates, start=1):
            raw = h.raw_similarity if h.raw_similarity is not None else h.similarity
            repeat_penalty = 0.0
            usage_penalty = 0.0
            tag_bonus = 0.0

            if requested_tag and h.tag == requested_tag:
                tag_bonus += 0.035

            if anti_repeat:
                if h.meme_id in recent_ids:
                    repeat_penalty += 0.35
                if h.last_used_at:
                    elapsed = now - h.last_used_at
                    if elapsed < 3600:
                        repeat_penalty += 0.05 * (1 - elapsed / 3600)
                        if elapsed < 600:
                            repeat_penalty += 0.15 * (1 - elapsed / 600)
                if h.usage_count > 0:
                    usage_penalty += min(0.025 * h.usage_count, 0.15)

            jitter = random.uniform(-self.random_jitter, self.random_jitter) if self.random_jitter else 0.0
            score = max(raw + tag_bonus - repeat_penalty - usage_penalty + jitter, 0.0)
            adjusted.append(MemeHit(
                meme_id=h.meme_id,
                file_path=h.file_path,
                tag=h.tag,
                similarity=score,
                usage_count=h.usage_count,
                last_used_at=h.last_used_at,
                description=h.description,
                raw_similarity=raw,
                tag_bonus=tag_bonus,
                repeat_penalty=repeat_penalty,
                usage_penalty=usage_penalty,
                random_jitter=jitter,
                fallback_used=fallback_used,
                rank_before_rerank=rank,
            ))
        adjusted.sort(key=lambda x: x.similarity, reverse=True)
        return adjusted

    def retrieve(
        self,
        text: str,
        tag: str | None = None,
        topk: int = 5,
        anti_repeat: bool = True,
        fallback_to_all_tags: bool = True,
        query_vector: np.ndarray | None = None,
    ) -> RetrievalResult:
        """用文本检索最匹配的表情。"""
        query_text = self._build_query(text, tag)
        qvec = query_vector if query_vector is not None else self.embedder.embed_text(query_text)

        candidates = self._candidate_vector_ids(tag)
        used_fallback = False
        original_tag = tag
        if not candidates and fallback_to_all_tags:
            candidates = self._candidate_vector_ids(None)
            used_fallback = True
            tag = None

        if not candidates or self.db.index_size <= 0:
            return RetrievalResult(query_text=query_text, tag_filter=tag, used_fallback=used_fallback, original_tag=original_tag)

        vec_ids = np.array([c[0] for c in candidates], dtype=np.int64)

        # IDSelectorBatch 只做候选集结果过滤；IndexFlatIP 仍是全量扫描，
        # 但避免 reconstruct_n(全量) + 临时索引重建的额外开销。
        q = qvec.reshape(1, -1).astype("float32")
        search_k = min(max(int(topk), self.candidate_pool_size), len(candidates))
        sims, found_vids = self.db.search_index(q, search_k, ids=vec_ids)

        # 只加载 FAISS 命中的 meme 行：命中数 ≤ search_k，避免全量预取和超长 IN 子句
        found_set = {int(v) for v in found_vids if int(v) >= 0}
        vid_to_mid: dict[int, int] = {}
        for vid, mid in candidates:
            if int(vid) in found_set:
                vid_to_mid[int(vid)] = mid
        found_mids = list(vid_to_mid.values())
        meme_rows: dict[int, dict] = {}
        if found_mids:
            with self.db._conn() as c:  # noqa: SLF001
                placeholders = ",".join("?" * len(found_mids))
                rows = c.execute(
                    f"SELECT id, file_path, tag, usage_count, last_used_at, description "
                    f"FROM memes WHERE id IN ({placeholders})",
                    found_mids,
                ).fetchall()
                for r in rows:
                    meme_rows[int(r["id"])] = dict(r)

        hits: list[MemeHit] = []
        for vid, sim in zip(found_vids, sims):
            if int(vid) < 0:
                continue
            mid = vid_to_mid.get(int(vid))
            if mid is None:
                continue
            row = meme_rows.get(mid)
            if row is None:
                continue
            raw = float(sim)
            hits.append(MemeHit(
                meme_id=mid,
                file_path=row["file_path"],
                tag=row["tag"],
                similarity=raw,
                raw_similarity=raw,
                usage_count=row.get("usage_count", 0) or 0,
                last_used_at=row.get("last_used_at"),
                description=row.get("description"),
                fallback_used=used_fallback,
            ))

        hits = self._rerank(
            hits,
            anti_repeat=anti_repeat,
            requested_tag=original_tag,
            fallback_used=used_fallback,
        )[:max(int(topk), 0)]

        result = RetrievalResult(
            query_text=query_text,
            tag_filter=tag,
            hits=hits,
            used_fallback=used_fallback,
            original_tag=original_tag,
        )
        if used_fallback:
            result.query_text += f" [fallback: no memes in tag '{original_tag}']"

        if hits:
            self.db.log_search(
                query_text=text,
                tag_filter=original_tag,
                topk_ids=[h.meme_id for h in hits],
                selected_id=hits[0].meme_id,
                similarity=hits[0].similarity,
            )
        return result

    def _weighted_pick(self, hits: list[MemeHit]) -> MemeHit | None:
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        weights = [max(h.similarity, 0.001) for h in hits]
        total = sum(weights)
        if total <= 0:
            return random.choice(hits)
        return random.choices(hits, weights=weights, k=1)[0]

    def pick(
        self,
        text: str,
        tag: str | None = None,
        anti_repeat: bool = True,
        fallback_to_all_tags: bool = True,
        selection_pool_size: int | None = None,
        stochastic: bool = True,
    ) -> MemeHit | None:
        """从候选池中选择一张并标记已用。"""
        pool = max(int(selection_pool_size or self.candidate_pool_size), 1)
        result = self.retrieve(
            text=text,
            tag=tag,
            topk=pool,
            anti_repeat=anti_repeat,
            fallback_to_all_tags=fallback_to_all_tags,
        )
        hit = self._weighted_pick(result.hits) if stochastic else result.top()
        if hit:
            self.db.mark_used(hit.meme_id)
        return hit

    def pick_multiple(
        self,
        text: str,
        tag: str | None = None,
        n: int = 2,
        anti_repeat: bool = True,
        fallback_to_all_tags: bool = True,
        stochastic: bool | None = None,
    ) -> list[MemeHit]:
        """取 top-n 张（标记已用）。

        stochastic=None 时沿用 retriever 实例的默认行为；显式传 False 可强制按相似度最高选。
        """
        do_stochastic = self._stochastic_default if stochastic is None else bool(stochastic)
        result = self.retrieve(
            text=text,
            tag=tag,
            topk=max(n, self.candidate_pool_size),
            anti_repeat=anti_repeat,
            fallback_to_all_tags=fallback_to_all_tags,
        )
        chosen: list[MemeHit] = []
        pool = list(result.hits)
        while pool and len(chosen) < n:
            if do_stochastic:
                hit = self._weighted_pick(pool)
            else:
                hit = pool[0]
            if hit is None:
                break
            self.db.mark_used(hit.meme_id)
            chosen.append(hit)
            pool = [h for h in pool if h.meme_id != hit.meme_id]
        return chosen

    # ---------- Prototype / KNN classification ----------

    def _all_active_vectors(self, exclude_meme_id: int | None = None) -> tuple[list[dict], np.ndarray]:
        rows = []
        vecs = []
        if self.db.index_size <= 0:
            return rows, np.empty((0, self.db.dim), dtype="float32")
        all_vecs = self.db.reconstruct_all()
        for row in self.db.list_memes(limit=10_000_000):
            if exclude_meme_id is not None and int(row["id"]) == int(exclude_meme_id):
                continue
            try:
                vid = int(row["vector_id"])
            except Exception:
                continue
            if 0 <= vid < self.db.index_size:
                rows.append(row)
                vecs.append(all_vecs[vid])
        if not vecs:
            return rows, np.empty((0, self.db.dim), dtype="float32")
        arr = np.stack(vecs, axis=0).astype("float32")
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return rows, arr / norm

    def classify_vector(
        self,
        vector: np.ndarray,
        exclude_meme_id: int | None = None,
        topk: int = 5,
        knn_k: int = 12,
        prototype_min_samples: int = 2,
    ) -> list[dict]:
        """基于 tag prototype + KNN 对图片向量分类。"""
        rows, vecs = self._all_active_vectors(exclude_meme_id=exclude_meme_id)
        if not rows or vecs.size == 0:
            return []

        q = vector.astype("float32").reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = vecs @ q

        knn_k = max(1, min(int(knn_k), len(rows)))
        nn_idx = np.argsort(sims)[::-1][:knn_k]
        knn_scores: dict[str, list[float]] = {}
        for idx in nn_idx:
            tag = str(rows[int(idx)]["tag"])
            knn_scores.setdefault(tag, []).append(float(sims[int(idx)]))

        tag_vecs: dict[str, list[np.ndarray]] = {}
        for row, vec in zip(rows, vecs):
            tag_vecs.setdefault(str(row["tag"]), []).append(vec)

        results = []
        for tag, vectors in tag_vecs.items():
            count = len(vectors)
            if count < prototype_min_samples:
                proto_sim = None
            else:
                proto = np.mean(np.stack(vectors, axis=0), axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-9)
                proto_sim = float(proto @ q)
            neighbor_values = knn_scores.get(tag, [])
            knn_score = float(np.mean(neighbor_values)) if neighbor_values else -1.0
            neighbor_count = len(neighbor_values)
            if proto_sim is None:
                final_score = knn_score
            elif neighbor_values:
                final_score = 0.7 * proto_sim + 0.3 * knn_score
            else:
                final_score = proto_sim - 0.05
            results.append({
                "tag": tag,
                "score": float(final_score),
                "prototype_similarity": proto_sim,
                "knn_score": knn_score if neighbor_values else None,
                "neighbor_count": neighbor_count,
                "sample_count": count,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max(int(topk), 1)]

    def classify_meme(
        self,
        meme_id: int,
        topk: int = 5,
        knn_k: int = 12,
        prototype_min_samples: int = 2,
    ) -> list[dict]:
        row = self.db.get_meme(meme_id)
        if not row:
            return []
        vid = int(row["vector_id"])
        if vid < 0 or vid >= self.db.index_size:
            return []
        vector = self.db.reconstruct(vid)
        return self.classify_vector(
            vector,
            exclude_meme_id=meme_id,
            topk=topk,
            knn_k=knn_k,
            prototype_min_samples=prototype_min_samples,
        )
