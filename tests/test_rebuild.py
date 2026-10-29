"""索引/重建命令的端到端测试。

覆盖 0.8.2 之前完全没有测试的三条关键路径：
- 增量索引：首次新增、二次全部跳过
- 重建成功：临时库替换旧库、使用记录保留、备份删除
- 重建失败：备份失败 / 构建失败 / 替换失败 三种回滚，旧库必须完好且备份保留
"""
from __future__ import annotations

import asyncio

from PIL import Image

BACKUP_SUFFIX = ".bak.v071"


class FakeEvent:
    def __init__(self, admin: bool = True):
        self._admin = admin
        self.sent = []

    def is_admin(self) -> bool:
        return self._admin

    @staticmethod
    def plain_result(text: str):
        return ("plain", text)

    async def send(self, result) -> None:
        self.sent.append(result)


def _make_plugin(tmp_path, main_module):
    class FakeProviderManager:
        def __init__(self):
            self.personas = []

    class FakeContext:
        def __init__(self):
            self.provider_manager = FakeProviderManager()

    meme_dir = tmp_path / "memes"
    meme_dir.mkdir(exist_ok=True)
    plugin = main_module.VectorMemePlugin(
        FakeContext(),
        {
            "data_dir": str(tmp_path / "data"),
            "meme_dir": str(meme_dir),
            "embedder_backend": "dummy",
            "enable_prompt_injection": False,
            "enable_vision_caption": False,
            "enable_stochastic_selection": False,
        },
    )
    return plugin, meme_dir


def _write_png(path, color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path


def _run(gen):
    async def collect():
        return [item async for item in gen]

    return asyncio.run(collect())


def _texts(out) -> list[str]:
    return [item[1] if isinstance(item, tuple) else str(item) for item in out]


def _backups(plugin) -> list:
    db = plugin._db
    return [
        db.db_path.with_name(db.db_path.name + BACKUP_SUFFIX),
        db.index_path.with_name(db.index_path.name + BACKUP_SUFFIX),
    ]


class _BrokenIndexer:
    def index_directory(self, *args, **kwargs):
        raise RuntimeError("simulated index failure")


# ---------- 增量索引 ----------


def test_cmd_index_adds_then_skips(tmp_path, main_module):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)
    _write_png(meme_dir / "happy" / "a.png")
    _write_png(meme_dir / "sad" / "b.png", (0, 0, 255))

    texts = _texts(_run(plugin._cmd_index(FakeEvent(), [])))
    assert "索引完成" in texts[-1]
    assert "总计: 2" in texts[-1]
    assert plugin._db.total_count() == 2
    assert plugin._db.count_by_tag() == {"happy": 1, "sad": 1}

    texts = _texts(_run(plugin._cmd_index(FakeEvent(), [])))
    assert "跳过: 2" in texts[-1]


def test_cmd_index_rejects_missing_and_outside_dirs(tmp_path, main_module):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)

    missing = meme_dir / "nope"
    assert "目录不存在" in _texts(_run(plugin._cmd_index(FakeEvent(), [str(missing)])))[0]

    outside = tmp_path / "outside"
    _write_png(outside / "x.png")
    texts = _texts(_run(plugin._cmd_index(FakeEvent(), [str(outside)])))
    assert "无权索引该目录" in texts[0]


# ---------- 重建成功 ----------


def test_rebuild_swaps_library_and_preserves_usage(tmp_path, main_module):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)
    _write_png(meme_dir / "happy" / "a.png")
    _write_png(meme_dir / "happy" / "b.png", (0, 255, 0))
    _run(plugin._cmd_index(FakeEvent(), []))

    rows = plugin._db.list_memes()
    used_id = int(rows[0]["id"])
    plugin._db.mark_used(used_id)

    texts = _texts(_run(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建完成" in texts[-1]
    assert "共处理 2 张" in texts[-1]

    assert plugin._db.total_count() == 2
    assert plugin._db.get_meme(used_id)["usage_count"] == 1
    assert all(not path.exists() for path in _backups(plugin))
    assert list(plugin.data_dir.glob("vector_meme_rebuild_*")) == []
    assert plugin._indexer.db is plugin._db


# ---------- 重建失败回滚 ----------


def test_rebuild_backup_failure_leaves_original_untouched(tmp_path, main_module, monkeypatch):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)
    _write_png(meme_dir / "happy" / "a.png")
    _run(plugin._cmd_index(FakeEvent(), []))
    old_db = plugin._db
    before = old_db.total_count()

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(old_db, "backup_files", _boom)

    texts = _texts(_run(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建备份失败，原库未受影响。" in texts[-1]
    assert plugin._db is old_db
    assert old_db.total_count() == before
    assert all(not path.exists() for path in _backups(plugin))


def test_rebuild_index_failure_rolls_back_and_keeps_backup(tmp_path, main_module, monkeypatch):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)
    _write_png(meme_dir / "happy" / "a.png")
    _run(plugin._cmd_index(FakeEvent(), []))
    old_db = plugin._db
    before = old_db.total_count()

    monkeypatch.setattr(plugin, "_make_indexer", lambda db: _BrokenIndexer())

    texts = _texts(_run(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建失败，已回滚原库。备份保留:" in texts[-1]
    assert plugin._db is old_db
    assert old_db.total_count() == before
    assert all(path.exists() for path in _backups(plugin))
    assert list(plugin.data_dir.glob("vector_meme_rebuild_*")) == []


def test_rebuild_swap_failure_restores_backup(tmp_path, main_module, monkeypatch):
    plugin, meme_dir = _make_plugin(tmp_path, main_module)
    _write_png(meme_dir / "happy" / "a.png")
    _run(plugin._cmd_index(FakeEvent(), []))
    old_db = plugin._db
    before = old_db.total_count()

    real_replace = main_module.os.replace

    def _fail_on_index(src, dst, *args, **kwargs):
        if str(dst).endswith(".faiss"):
            raise OSError("simulated replace failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(main_module.os, "replace", _fail_on_index)

    texts = _texts(_run(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建失败，已回滚原库。备份保留:" in texts[-1]
    assert plugin._db is old_db
    assert old_db.total_count() == before
    assert all(path.exists() for path in _backups(plugin))
    assert list(plugin.data_dir.glob("vector_meme_rebuild_*")) == []


def test_progress_reporter_throttles_and_reports_completion(tmp_path, main_module):
    plugin, _ = _make_plugin(tmp_path, main_module)

    async def scenario():
        event = FakeEvent()
        async with plugin._progress_reporter(
            event, log_tag="test", text_prefix="⏳ 测试进度"
        ) as on_progress:
            for count in range(1, 21):
                on_progress(count, 20, tmp_path / f"{count}.png")
            for _ in range(50):
                if len(event.sent) >= 3:
                    break
                await asyncio.sleep(0.02)
        return event

    event = asyncio.run(scenario())
    texts = [item["text"] for item in event.sent]
    assert len(texts) == 3
    assert "1/20" in texts[0]
    assert "11/20" in texts[1]
    assert "20/20" in texts[2]
