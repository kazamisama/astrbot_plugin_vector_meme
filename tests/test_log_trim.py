"""日志裁剪：search_log / index_log 的保留上限与插件启动、/vm 修复 的自动清理。"""
from __future__ import annotations

# 必须走包内路径：main.py 通过 vector_meme.core.database 导入，
# 顶层的 core.database 是另一个模块实例，monkeypatch 不到插件实际使用的常量。
import vector_meme.core.database as database_module
from helpers import FakeEvent, make_plugin, run_async as _run, texts as _texts
from vector_meme.core.database import INDEX_LOG_KEEP, SEARCH_LOG_KEEP, MemeDatabase


def _make_db(tmp_path, dim=8):
    return MemeDatabase(
        db_path=tmp_path / "memes.db",
        index_path=tmp_path / "memes.faiss",
        dim=dim,
    )


def _add_search(db, count):
    for i in range(count):
        db.log_search(f"q{i}", None, [], None, 0.5)


def _add_index(db, count):
    for i in range(count):
        db.log_index_action("add", f"/m/{i}.png", "success", "")


def _count(db, table):
    with db._conn() as c:
        return c.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def _index_paths(db):
    with db._conn() as c:
        rows = c.execute("SELECT file_path FROM index_log ORDER BY id").fetchall()
    return [r["file_path"] for r in rows]


# ---------- MemeDatabase.trim_logs ----------


def test_trim_keeps_newest_rows_and_reports_removed(tmp_path):
    db = _make_db(tmp_path)
    _add_search(db, 5)
    _add_index(db, 7)

    removed = db.trim_logs(search_keep=2, index_keep=3)

    assert removed == {"search_log": 3, "index_log": 4}
    assert _count(db, "search_log") == 2
    assert [r["query_text"] for r in db.list_search_logs(10)] == ["q4", "q3"]
    assert _index_paths(db) == ["/m/4.png", "/m/5.png", "/m/6.png"]


def test_trim_defaults_come_from_module_constants(tmp_path, monkeypatch):
    monkeypatch.setattr(database_module, "SEARCH_LOG_KEEP", 1)
    monkeypatch.setattr(database_module, "INDEX_LOG_KEEP", 2)
    db = _make_db(tmp_path)
    _add_search(db, 3)
    _add_index(db, 3)

    assert db.trim_logs() == {"search_log": 2, "index_log": 1}
    assert _count(db, "search_log") == 1
    assert _count(db, "index_log") == 2


def test_trim_zero_keep_deletes_all_and_under_limit_is_noop(tmp_path):
    db = _make_db(tmp_path)
    _add_search(db, 2)
    _add_index(db, 0)
    assert db.trim_logs(search_keep=5, index_keep=5) == {"search_log": 0, "index_log": 0}
    assert db.trim_logs(search_keep=0, index_keep=0) == {"search_log": 2, "index_log": 0}
    assert _count(db, "search_log") == 0


def test_trim_negative_keep_is_clamped_to_zero(tmp_path):
    db = _make_db(tmp_path)
    _add_search(db, 2)
    assert db.trim_logs(search_keep=-3, index_keep=-1)["search_log"] == 2
    assert _count(db, "search_log") == 0


def test_module_constants_are_positive():
    assert SEARCH_LOG_KEEP > 0
    assert INDEX_LOG_KEEP > 0


# ---------- 插件启动与 /vm 修复 ----------


def test_startup_trims_oversized_logs(tmp_path, main_module, monkeypatch):
    monkeypatch.setattr(database_module, "SEARCH_LOG_KEEP", 2)
    monkeypatch.setattr(database_module, "INDEX_LOG_KEEP", 1)

    data_dir = tmp_path / "data"
    seed = MemeDatabase(data_dir / "memes.db", data_dir / "memes.faiss", dim=512)
    _add_search(seed, 5)
    _add_index(seed, 4)

    plugin = make_plugin(tmp_path, main_module)

    assert _count(plugin._db, "search_log") == 2
    assert _count(plugin._db, "index_log") == 1
    assert [r["query_text"] for r in plugin._db.list_search_logs(5)] == ["q4", "q3"]


def test_startup_trim_failure_does_not_break_plugin_init(tmp_path, main_module, monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(database_module.MemeDatabase, "trim_logs", boom)
    plugin = make_plugin(tmp_path, main_module)
    assert plugin._db is not None


def test_repair_trims_logs_and_reports_counts(tmp_path, main_module, monkeypatch):
    monkeypatch.setattr(database_module, "SEARCH_LOG_KEEP", 2)
    monkeypatch.setattr(database_module, "INDEX_LOG_KEEP", 1)
    plugin = make_plugin(tmp_path, main_module)
    _add_search(plugin._db, 5)
    _add_index(plugin._db, 4)

    texts = _texts(_run(plugin._cmd_repair(FakeEvent(), [])))

    assert "- 裁剪日志: search_log 3 / index_log 3" in texts[-1]
    assert _count(plugin._db, "search_log") == 2
    assert _count(plugin._db, "index_log") == 1
