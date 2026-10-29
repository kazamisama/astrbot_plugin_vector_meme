"""命令处理的错误分支与边界输入。"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from helpers import FakeEvent, make_plugin, run_async, texts

from core.database import MemeDatabase
from core.embedder import DummyEmbedder
from core.retriever import MemeHit, RetrievalResult


def _plain_texts(out):
    result = []
    for item in out:
        if isinstance(item, tuple) and item[0] == "plain":
            result.append(item[1])
        elif isinstance(item, tuple) and item[0] == "chain":
            result.extend(
                component.text
                for component in item[1]
                if getattr(component, "text", None)
            )
    return result


def test_text_command_variants_and_vm_command(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    for text in (
        "",
        "表情向量",
        "表情向量 状态",
        "vmem",
        "vmem 状态",
        "vm",
        "vm 状态",
        "/vm 状态",
    ):
        event = FakeEvent()
        event.message_str = text
        out = run_async(plugin.vector_meme_text_command(event))
        if text == "":
            assert out == []
        else:
            assert out

    out = texts(run_async(plugin.vm_command(FakeEvent(), "状态")))
    assert "状态" in out[-1]


def test_db_uninitialized_branches(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin._db = None
    cases = [
        (plugin._cmd_list, []),
        (plugin._cmd_tags, []),
        (plugin._cmd_recent, []),
        (plugin._cmd_health, []),
        (plugin._cmd_relabel, ["1", "x"]),
        (plugin._cmd_delete, ["1"]),
        (plugin._cmd_status, []),
    ]
    for handler, rest in cases:
        out = texts(run_async(handler(FakeEvent(), rest)))
        assert "数据库未初始化" in out[0]


def test_search_explain_result_details(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)

    class EmptyRetriever:
        def retrieve(self, **kwargs):
            return RetrievalResult("q", None, hits=[])

    async def empty_ready(event, action):
        return (EmptyRetriever(), None, None)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", empty_ready)
    assert "未找到匹配" in _plain_texts(
        run_async(plugin._cmd_search(FakeEvent(), ["x"]))
    )[0]
    assert "没有可解释的候选结果" in _plain_texts(
        run_async(plugin._cmd_explain(FakeEvent(), ["x"]))
    )[0]

    class DetailRetriever:
        def retrieve(self, **kwargs):
            hit = MemeHit(
                1,
                "a.png",
                "happy",
                0.5,
                raw_similarity=0.4,
                tag_bonus=0.03,
                repeat_penalty=0.01,
                usage_penalty=0.02,
                random_jitter=0.005,
            )
            return RetrievalResult(
                "q", None, hits=[hit], expanded_tags=["sad"], pool_size=3
            )

    async def detail_ready(event, action):
        return (DetailRetriever(), None, None)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", detail_ready)
    search = " ".join(
        _plain_texts(run_async(plugin._cmd_search(FakeEvent(), ["x"])))
    )
    assert "扩展tag: sad" in search
    explain = " ".join(
        _plain_texts(run_async(plugin._cmd_explain(FakeEvent(), ["x"])))
    )
    assert "小池扩展 tag" in explain
    assert "tag +" in explain and "repeat -" in explain
    assert "usage -" in explain and "jitter" in explain


def test_eval_error_branches_and_confusion(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert "读取失败" in texts(
        run_async(plugin._cmd_eval(FakeEvent(), [str(bad_json)]))
    )[0]

    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({"items": 1}), encoding="utf-8")
    assert "格式错误" in texts(
        run_async(plugin._cmd_eval(FakeEvent(), [str(wrong_shape)]))
    )[0]

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            [
                {"id": 1, "tag": "happy"},
                {"id": 2},
                "junk",
            ]
        ),
        encoding="utf-8",
    )

    class Retriever:
        def classify_meme(self, meme_id, **kwargs):
            return [{"tag": "sad"}, {"tag": "happy"}]

    async def ready(event, action):
        return (Retriever(), None, None)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", ready)
    out = texts(run_async(plugin._cmd_eval(FakeEvent(), [str(eval_path)])))[0]
    assert "主要混淆" in out
    assert "happy -> sad" in out

    class NoPreds:
        def classify_meme(self, meme_id, **kwargs):
            return []

    async def no_preds_ready(event, action):
        return (NoPreds(), None, None)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", no_preds_ready)
    empty = texts(run_async(plugin._cmd_eval(FakeEvent(), [str(eval_path)])))[0]
    assert "没有可评测样本" in empty


def test_relabel_delete_and_reload_prompt_edges(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)
    assert "用法" in texts(run_async(plugin._cmd_relabel(FakeEvent(), [])))[0]
    assert "用法" in texts(run_async(plugin._cmd_delete(FakeEvent(), [])))[0]
    assert "重标注失败" in texts(
        run_async(plugin._cmd_relabel(FakeEvent(), ["9999", "happy"]))
    )[0]

    def boom():
        raise RuntimeError("inject failed")

    monkeypatch.setattr(plugin, "_inject_into_personas", boom)
    assert "重载失败" in texts(
        run_async(plugin._cmd_reload_prompt(FakeEvent(), []))
    )[0]


def test_tag_schema_missing_and_read_failure(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)

    missing_root = tmp_path / "no-schema"
    missing_root.mkdir()
    monkeypatch.setattr(main_module, "__file__", str(missing_root / "main.py"))
    assert "不存在" in texts(run_async(plugin._cmd_tag_schema(FakeEvent(), [])))[0]

    bad_root = tmp_path / "bad-schema"
    bad_root.mkdir()
    (bad_root / "tag_schema.json").write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(main_module, "__file__", str(bad_root / "main.py"))
    assert "读取失败" in texts(run_async(plugin._cmd_tag_schema(FakeEvent(), [])))[0]

    flat_root = tmp_path / "flat-schema"
    flat_root.mkdir()
    (flat_root / "tag_schema.json").write_text(
        json.dumps({"tags": {"x": "plain"}}), encoding="utf-8"
    )
    monkeypatch.setattr(main_module, "__file__", str(flat_root / "main.py"))
    assert "x: plain" in texts(run_async(plugin._cmd_tag_schema(FakeEvent(), [])))[0]


def test_auto_classify_error_and_apply_skips(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)
    assert "用法" in texts(run_async(plugin._cmd_auto_classify(FakeEvent(), [])))[0]

    async def not_ready(event, action):
        return None

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", not_ready)
    assert "embedder 未就绪" in texts(
        run_async(plugin._cmd_auto_classify(FakeEvent(), ["1"]))
    )[0]

    db = plugin._db
    mid = db.upsert_meme("x.png", "h", "happy", -1)
    predictions = []

    class Retriever:
        def classify_meme(self, meme_id, **kwargs):
            return predictions

    async def ready(event, action):
        return (Retriever(), None, db)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", ready)
    assert "id 不存在" in texts(
        run_async(plugin._cmd_auto_classify(FakeEvent(), ["9999"]))
    )[0]

    assert "没有足够样本" in texts(
        run_async(plugin._cmd_auto_classify(FakeEvent(), [str(mid)]))
    )[0]

    predictions[:] = [
        {
            "tag": "sad",
            "score": 0.05,
            "prototype_similarity": 0.05,
            "knn_score": 0.05,
            "neighbor_count": 1,
            "sample_count": 1,
        }
    ]
    weak = texts(run_async(plugin._cmd_auto_classify(FakeEvent(), [str(mid), "apply"])))[0]
    assert "未执行" in weak

    predictions[:] = [
        {
            "tag": "happy",
            "score": 0.9,
            "prototype_similarity": 0.9,
            "knn_score": 0.9,
            "neighbor_count": 1,
            "sample_count": 1,
        }
    ]
    same = texts(run_async(plugin._cmd_auto_classify(FakeEvent(), [str(mid), "apply"])))[0]
    assert "无需修改" in same


def test_caption_command_error_branches(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )

    async def not_ready(event, action):
        return None

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", not_ready)
    assert "embedder not ready" in texts(
        run_async(plugin._cmd_caption(FakeEvent(), []))
    )[0]

    class Unavailable:
        def available(self):
            return False

    async def ready(event, action):
        return (object(), None, plugin._db)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", ready)
    monkeypatch.setattr(
        main_module, "__file__", str(tmp_path / "main.py")
    )
    plugin.meme_dir.mkdir(parents=True, exist_ok=True)
    (plugin.meme_dir / "captions.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(plugin, "_ensure_captioner", lambda: Unavailable())
    assert "unavailable" in texts(run_async(plugin._cmd_caption(FakeEvent(), [])))[0]

    (plugin.meme_dir / "captions.json").write_text(
        json.dumps({"a.png": "cap"}), encoding="utf-8"
    )
    assert "limit 必须" in texts(
        run_async(plugin._cmd_caption(FakeEvent(), ["abc"]))
    )[0]
    assert "limit 必须" in texts(
        run_async(plugin._cmd_caption(FakeEvent(), ["0"]))
    )[0]


def test_enrich_captions_after_index_branches(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    monkeypatch.setattr(main_module, "__file__", str(tmp_path / "main.py"))

    class Unavailable:
        def available(self):
            return False

    monkeypatch.setattr(plugin, "_ensure_captioner", lambda: Unavailable())
    assert asyncio.run(plugin._enrich_captions_after_index()) == {}

    caption_file = tmp_path / "memes" / "captions.json"
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    caption_file.write_text(
        json.dumps({"happy/a.png": "外部 caption"}), encoding="utf-8"
    )
    plugin._db.upsert_meme(
        str(plugin.meme_dir / "happy" / "a.png"), "h", "happy", -1
    )
    plugin._embedder = DummyEmbedder(plugin._db.dim)

    async def enrich_fail(*args, **kwargs):
        raise RuntimeError("enrich failed")

    monkeypatch.setattr(main_module, "enrich_meme_captions", enrich_fail)
    assert asyncio.run(plugin._enrich_captions_after_index()) == {}


class _CloseBoom:
    dim = 8

    @property
    def fingerprint(self):
        return "boom:8"

    def close(self):
        raise RuntimeError("close failed")


def test_prepare_library_close_failures_and_empty_dim_rebuild(
    tmp_path, main_module
):
    plugin = make_plugin(tmp_path, main_module)
    plugin._embedder = _CloseBoom()
    db = MemeDatabase(tmp_path / "m.db", tmp_path / "m.faiss", dim=16)
    vids = db.add_vectors(np.ones((1, 16), dtype="float32"))
    db.upsert_meme("old.png", "h", "happy", int(vids[0]))
    db.set_meta("embedder_fingerprint", "old:8")
    plugin._db = db
    with pytest.raises(RuntimeError, match="指纹已变化"):
        plugin._prepare_library_for_embedder()

    plugin._embedder = _CloseBoom()
    db.set_meta("embedder_fingerprint", "boom:8")
    with pytest.raises(RuntimeError, match="维度"):
        plugin._prepare_library_for_embedder()

    empty = MemeDatabase(tmp_path / "empty.db", tmp_path / "empty.faiss", dim=16)
    empty.index_path.write_bytes(b"stale")
    plugin._embedder = DummyEmbedder(8)
    plugin._db = empty
    plugin._prepare_library_for_embedder()
    assert plugin._db.dim == 8
    assert plugin._db.index_path != empty.index_path
