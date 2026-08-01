"""Vision caption generation for semantic-enriched retrieval.

Uses an AstrBot LLM provider (vision-capable chat model) to turn a meme image
into a Chinese semantic caption plus English keywords. The caption text is then
encoded with the CLIP text encoder and stored as a caption vector, enabling a
dual-path retrieval: image-text (existing) + caption-text (new).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CAPTION_SYSTEM_PROMPT = (
    '你是中文互联网表情包语义分析员。'
    '你的任务不是给图片写普通图注，而是还原这张图作为聊天回复时真正传达的意思，'
    '让人能够按对话情境准确搜索到它。'
)

CAPTION_PROMPT = (
    '请分析这张表情包图片，只输出 JSON，不要输出其他内容，格式为：\n'
    '{"caption": "中文语义描述（40字以内，还原它作为聊天回复时的含义）", '
    '"en_keywords": "3到6个英文逗号分隔关键词，描述图片主体与情绪"}'
)


class CaptionGenerator:
    """Wrap AstrBot context.llm_generate for meme captioning.

    Every failure (no context / no provider / bad response / exception) is
    reported as None so the caller can safely fall back to pure CLIP search.
    """

    def __init__(self, context: Any, provider_id: str = ''):
        self.context = context
        self.provider_id = (provider_id or '').strip()

    def available(self) -> bool:
        if self.context is None or not callable(getattr(self.context, 'llm_generate', None)):
            return False
        return bool(self.provider_id)

    async def caption_image(self, image_path: Path | str) -> dict | None:
        """Return {'caption': str, 'en_keywords': str} or None on any failure."""
        if not self.available():
            return None
        try:
            response = await self.context.llm_generate(
                chat_provider_id=self.provider_id,
                prompt=CAPTION_PROMPT,
                image_urls=[str(image_path)],
                system_prompt=CAPTION_SYSTEM_PROMPT,
                temperature=0,
                max_tokens=300,
            )
            payload = self._parse_json(self._extract_text(response))
            if not payload:
                return None
            caption = str(payload.get('caption') or '').strip()
            keywords = str(payload.get('en_keywords') or '').strip()
            if not caption:
                return None
            return {'caption': caption, 'en_keywords': keywords}
        except Exception as exc:
            logger.warning('caption_image failed for %s: %s', image_path, exc)
            return None

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        completion = getattr(response, 'completion_text', None)
        if completion:
            return str(completion)
        chain = getattr(response, 'result_chain', None) or getattr(response, 'chain', None)
        if chain is not None:
            parts = []
            for comp in chain:
                if isinstance(comp, dict) and comp.get('type') == 'plain':
                    parts.append(str(comp.get('text') or ''))
                elif comp.__class__.__name__ == 'Plain' and hasattr(comp, 'text'):
                    parts.append(str(comp.text))
            if parts:
                return ''.join(parts)
        return str(response or '')

    def _parse_json(self, text: str) -> dict | None:
        text = (text or '').strip()
        if not text:
            return None
        if text.startswith('```'):
            lines = text.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith('```'):
                lines = lines[:-1]
            text = '\n'.join(lines).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if 0 <= start < end:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None
async def enrich_meme_captions(
    db,
    embedder,
    captioner,
    limit: int = 200,
    batch_size: int = 16,
    progress_cb=None,
) -> dict:
    """Generate captions for memes lacking one and store their caption vectors.

    Args:
        db: MemeDatabase instance (must expose memes_without_caption /
            add_vectors / set_meme_caption).
        embedder: BaseEmbedder used to encode caption text into vectors.
        captioner: CaptionGenerator instance.
        limit: maximum number of memes processed in one run.
        batch_size: vectors per FAISS batch write.
        progress_cb: optional callable(done, total).

    Returns:
        {'total': int, 'ok': int, 'failed': int}
    """
    import asyncio

    import numpy as np

    pending = db.memes_without_caption(limit=int(limit))
    total = len(pending)
    if total == 0:
        return {'total': 0, 'ok': 0, 'failed': 0}

    ok = 0
    failed = 0
    done = 0
    vector_batch: list = []
    meta_batch: list = []

    def flush():
        nonlocal vector_batch, meta_batch
        if not vector_batch:
            return
        arr = np.stack(vector_batch, axis=0)
        v_ids = db.add_vectors(arr)
        for (meme_id, caption_text), vid in zip(meta_batch, v_ids):
            db.set_meme_caption(int(meme_id), caption_text, int(vid))
        vector_batch = []
        meta_batch = []

    for row in pending:
        done += 1
        if progress_cb:
            progress_cb(done, total)
        result = await captioner.caption_image(row['file_path'])
        if not result:
            failed += 1
            continue
        caption_text = str(result.get('caption') or '').strip()
        keywords = str(result.get('en_keywords') or '').strip()
        if not caption_text:
            failed += 1
            continue
        if keywords:
            caption_text = caption_text + ' ' + keywords
        try:
            vector = await asyncio.to_thread(embedder.embed_text, caption_text)
        except Exception as exc:
            logger.warning('caption embed failed for %s: %s', row['file_path'], exc)
            failed += 1
            continue
        vector_batch.append(vector)
        meta_batch.append((row['id'], caption_text))
        if len(vector_batch) >= max(int(batch_size), 1):
            flush()
        ok += 1
    flush()
    return {'total': total, 'ok': ok, 'failed': failed}
