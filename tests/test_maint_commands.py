"""维护命令测试：/vm 去重 与 /vm 修复。

这两条命令都直接改数据库（禁用/启用、删行、压缩向量），此前没有任何测试。
"""
from __future__ import annotations

import numpy as np

from helpers import (
    FakeEvent,
    make_plugin,
    run_async as _run,
    texts as _texts,
    write_png as _write_png,
)


def _seed_duplicate(*, tmp_path, main_module, count=3, tag="happy"):
    """在库里放 count 条互相完全相同的向量（余弦 1.0），模拟近重复变体。"""
    plugin = make_plugin(tmp_path, main_module)
    db = plugin._db
    base = np.ones(db.dim, dtype="float32")
    vids = db.add_vectors(np.stack([base] * count))
    ids = []
    for i, vid in enumerate(vids):
        path = plugin.meme_dir / tag / f"dup{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-real-png")
        ids.append(db.upsert_meme(str(path), f"h{i}", tag, int(vid), file_name=path.name))
    db.save_index()
    return plugin, ids


# ---------- /vm 去重 ----------


def test_dedup_preview_does_not_load_embedder_or_touch_db(tmp_path, main_module):
    plugin, ids = _seed_duplicate(tmp_path=tmp_path, main_module=main_module)
    out = _run(plugin._cmd_dedup(FakeEvent(), []))
    texts = _texts(out)

    assert plugin._embedder is None, "去重不应触发 embedder 冷启动"
    assert "近重复簇" in texts[0]
    assert "确认无误后执行" in texts[-1]
    assert any(f"#{ids[0]}" in line for line in texts[1:])
    assert plugin._db.stats()["disabled"] == 0


def test_dedup_apply_then_undo_roundtrip(tmp_path, main_module):
    plugin, ids = _seed_duplicate(tmp_path=tmp_path, main_module=main_module)

    texts = _texts(_run(plugin._cmd_dedup(FakeEvent(), ["apply"])))
    assert "去重完成" in texts[-1]
    assert "禁用 2 张" in texts[-1]
    assert plugin._db.stats()["disabled"] == 2

    disabled_ids = [mid for mid in ids if plugin._db.get_meme(mid)["disabled"]]
    assert len(disabled_ids) == 2
    for mid in disabled_ids:
        assert "dedup:" in (plugin._db.get_meme(mid)["sub_tags"] or "")

    texts = _texts(_run(plugin._cmd_dedup(FakeEvent(), ["undo"])))
    assert "已恢复 2 张" in texts[-1]
    assert plugin._db.stats()["disabled"] == 0
    for mid in disabled_ids:
        assert "dedup:" not in (plugin._db.get_meme(mid)["sub_tags"] or "")


def test_dedup_empty_db_short_circuits(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    texts = _texts(_run(plugin._cmd_dedup(FakeEvent(), [])))
    assert "库为空" in texts[0]
    assert plugin._embedder is None


def test_dedup_accepts_custom_threshold(tmp_path, main_module):
    plugin, _ = _seed_duplicate(tmp_path=tmp_path, main_module=main_module)
    texts = _texts(_run(plugin._cmd_dedup(FakeEvent(), ["0.99"])))
    assert "阈值 0.990" in texts[0]


# ---------- /vm 修复 ----------


def test_repair_removes_missing_files_and_compacts_orphans(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    meme_dir = plugin.meme_dir
    for tag, name in (("happy", "a"), ("happy", "b"), ("sad", "c")):
        _write_png(meme_dir / tag / f"{name}.png")
    _run(plugin._cmd_index(FakeEvent(), []))
    assert plugin._db.total_count() == 3

    # 删掉一张图，再塞 60 个没有任何行引用的孤儿向量
    (meme_dir / "happy" / "b.png").unlink()
    orphans = np.random.default_rng(0).random((60, plugin._db.dim), dtype="float32")
    plugin._db.add_vectors(orphans)
    assert plugin._db.index_size == 63

    texts = _texts(_run(plugin._cmd_repair(FakeEvent(), [])))
    assert "修复完成" in texts[-1]
    assert "清理缺失文件记录: 1" in texts[-1]
    assert "orphan compaction" in texts[-1]
    assert plugin._db.total_count() == 2
    assert plugin._db.index_size == 2


def test_repair_without_orphans_reports_no_compaction(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    _write_png(plugin.meme_dir / "happy" / "a.png")
    _run(plugin._cmd_index(FakeEvent(), []))

    texts = _texts(_run(plugin._cmd_repair(FakeEvent(), [])))
    assert "清理缺失文件记录: 0" in texts[-1]
    assert "no compaction needed" in texts[-1]


def test_repair_refuses_when_index_corrupted(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    _write_png(plugin.meme_dir / "happy" / "a.png")
    _run(plugin._cmd_index(FakeEvent(), []))
    plugin._db.index_corrupted = True

    texts = _texts(_run(plugin._cmd_repair(FakeEvent(), [])))
    assert "FAISS 索引已损坏" in texts[0]
    assert plugin._db.total_count() == 1
