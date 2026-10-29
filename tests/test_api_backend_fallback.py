"""api 后端（图片向量不落库）下，分类/评测/去重复用 caption 向量。"""
from __future__ import annotations

import json

import numpy as np

from helpers import FakeEvent, make_plugin, run_async


def _plain(out):
    return [item[1] for item in out if isinstance(item, tuple) and item[0] == "plain"]


def _seed_api_like(plugin):
    emb = plugin._embedder
    db = plugin._db
    texts = [
        ("happy", "dup caption"),
        ("happy", "dup caption"),
        ("happy", "happy other"),
        ("sad", "sad one"),
    ]
    cap_ids = db.add_vectors(
        np.stack([emb.embed_text(text) for _, text in texts])
    )
    ids = []
    for i, (tag, text) in enumerate(texts):
        mid = db.upsert_meme(
            str(plugin.meme_dir / f"{i}.png"), f"h{i}", tag, -1
        )
        db.set_meme_caption(mid, text, int(cap_ids[i]))
        ids.append(mid)
    db.save_index()
    return ids


def test_api_backend_auto_classify_eval_and_dedup(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert run_async  # keep import used in older harness
    import asyncio

    async def setup():
        await plugin._ensure_ready()
        return _seed_api_like(plugin)

    ids = asyncio.run(setup())
    target = ids[0]

    classify_out = _plain(
        run_async(plugin._cmd_auto_classify(FakeEvent(), [str(target)]))
    )
    assert not any("没有足够样本" in text for text in classify_out)
    assert any("建议tag: happy" in text for text in classify_out)

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps([{"id": target, "tag": "happy"}]), encoding="utf-8"
    )
    eval_out = _plain(run_async(plugin._cmd_eval(FakeEvent(), [str(eval_path)])))
    assert any("分类评测结果" in text for text in eval_out)
    assert any("样本数: 1" in text for text in eval_out)

    dedup_out = _plain(
        run_async(plugin._cmd_dedup(FakeEvent(), ["preview", "0.999"]))
    )
    assert any("近重复簇" in text for text in dedup_out)
