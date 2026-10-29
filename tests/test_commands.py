"""只读/维护子命令与文本入口的端到端测试。"""
from __future__ import annotations

import asyncio
import json

from helpers import FakeEvent, make_plugin, run_async, texts


def _seed(plugin, tag="happy", count=2):
    """加载 dummy embedder 并写入同 tag 的若干条记录，返回 meme id。"""

    async def run():
        ready = await plugin._ensure_ready()
        assert ready is not None
        db = plugin._db
        embedder = plugin._embedder
        ids = []
        for i in range(count):
            vecs = db.add_vectors(embedder.embed_text(f"{tag} example {i}"))
            path = plugin.meme_dir / tag / f"{i}.png"
            ids.append(
                db.upsert_meme(
                    str(path), f"h{i}", tag, int(vecs[0]), file_name=path.name
                )
            )
        db.save_index()
        return ids

    return asyncio.run(run())


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


def test_list_and_tags_commands(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert "索引为空" in texts(run_async(plugin._cmd_list(FakeEvent(), [])))[0]
    assert "尚未注册标签" in texts(run_async(plugin._cmd_tags(FakeEvent(), [])))[0]

    _seed(plugin)
    plugin._db.upsert_tag("happy", "开心", "emotion", "#fff")
    listed = texts(run_async(plugin._cmd_list(FakeEvent(), ["happy"])))[0]
    assert "[happy] 共 2 条" in listed
    distribution = texts(run_async(plugin._cmd_list(FakeEvent(), [])))[0]
    assert "happy: 2" in distribution
    tags = texts(run_async(plugin._cmd_tags(FakeEvent(), [])))[0]
    assert "happy (emotion) - 开心" in tags


def test_recent_and_health_commands(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert "还没有最近使用记录" in texts(run_async(plugin._cmd_recent(FakeEvent(), [])))[0]

    ids = _seed(plugin)
    plugin._db.mark_used(ids[0])
    recent = texts(run_async(plugin._cmd_recent(FakeEvent(), ["5"])))[0]
    assert "happy" in recent and "使用 1 次" in recent

    health = texts(run_async(plugin._cmd_health(FakeEvent(), ["--deep"])))[0]
    assert "健康检查" in health
    assert "文件缺失: 2" in health
    assert "缺失文件示例" in health


def test_search_and_explain_commands(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    _seed(plugin)

    search = " ".join(_plain_texts(run_async(plugin._cmd_search(FakeEvent(), ["happy"]))))
    assert "搜索: happy" in search
    assert "final=" in search

    explain = " ".join(
        _plain_texts(run_async(plugin._cmd_explain(FakeEvent(), ["happy"])))
    )
    assert "选择解释" in explain
    assert "top candidates" in explain


def test_relabel_and_delete_commands(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    ids = _seed(plugin)

    relabeled = texts(run_async(plugin._cmd_relabel(FakeEvent(), [str(ids[0]), "sad"])))
    assert "已重标注" in relabeled[-1]
    assert plugin._db.get_meme(ids[0])["tag"] == "sad"

    deleted = texts(run_async(plugin._cmd_delete(FakeEvent(), [str(ids[0])])))
    assert "已删除" in deleted[-1]
    assert plugin._db.get_meme(ids[0]) is None
    assert "id 不存在" in texts(run_async(plugin._cmd_delete(FakeEvent(), [str(ids[0])])))[-1]


def test_tag_schema_and_reload_prompt_commands(tmp_path, main_module):
    plugin = make_plugin(
        tmp_path,
        main_module,
        personas=[{"prompt": "base"}],
        config={"enable_prompt_injection": True},
    )
    schema = texts(run_async(plugin._cmd_tag_schema(FakeEvent(), [])))[0]
    assert "tag schema" in schema
    assert "个规范 tag" in schema

    reloaded = texts(run_async(plugin._cmd_reload_prompt(FakeEvent(), [])))[0]
    assert "已重新注入 prompt" in reloaded

    disabled = make_plugin(tmp_path / "disabled", main_module)
    assert "enable_prompt_injection=False" in texts(
        run_async(disabled._cmd_reload_prompt(FakeEvent(), []))
    )[0]


def test_prewarm_command_loads_embedder_once(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    first = texts(run_async(plugin._cmd_prewarm(FakeEvent(), [])))[-1]
    assert "embedder 就绪" in first
    second = texts(run_async(plugin._cmd_prewarm(FakeEvent(), [])))[-1]
    assert "embedder 已就绪" in second


def test_caption_command_exports_and_reuses_external_captions(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    monkeypatch.setattr(main_module, "__file__", str(tmp_path / "main.py"))
    ids = _seed(plugin, count=1)

    captions_path = tmp_path / "memes" / "captions.json"
    captions_path.write_text(
        json.dumps({"happy/0.png": "外部 caption"}, ensure_ascii=False),
        encoding="utf-8",
    )
    done = texts(run_async(plugin._cmd_caption(FakeEvent(), ["5"])))[-1]
    assert "caption done" in done
    assert plugin._db.get_meme(ids[0])["caption"] == "外部 caption"

    exported = texts(run_async(plugin._cmd_caption(FakeEvent(), ["导出"])))[-1]
    assert "已导出 1 条" in exported
    assert json.loads(captions_path.read_text(encoding="utf-8")) == {
        "happy/0.png": "外部 caption"
    }

    disabled = make_plugin(tmp_path / "plain", main_module)
    assert "vision caption disabled" in texts(
        run_async(disabled._cmd_caption(FakeEvent(), []))
    )[0]


def test_auto_classify_preview_and_apply(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert "用法" in texts(run_async(plugin._cmd_auto_classify(FakeEvent(), [])))[0]
    ids = _seed(plugin, count=1)

    def _pred(tag, score):
        return {
            "tag": tag,
            "score": score,
            "prototype_similarity": score,
            "knn_score": score,
            "neighbor_count": 3,
            "sample_count": 4,
        }

    plugin._retriever.classify_meme = lambda *args, **kwargs: [
        _pred("happy", 0.5),
        _pred("sad", 0.1),
    ]
    preview = texts(
        run_async(plugin._cmd_auto_classify(FakeEvent(), [str(ids[0])]))
    )[0]
    assert "dry-run" in preview
    assert plugin._db.get_meme(ids[0])["tag"] == "happy"

    plugin._retriever.classify_meme = lambda *args, **kwargs: [_pred("sad", 0.9)]
    applied = texts(
        run_async(plugin._cmd_auto_classify(FakeEvent(), [str(ids[0]), "apply"]))
    )[0]
    assert "已修改数据库 tag" in applied
    assert plugin._db.get_meme(ids[0])["tag"] == "sad"


def test_eval_command_reports_top1_and_failures(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    ids = _seed(plugin, count=1)
    assert "用法" in texts(run_async(plugin._cmd_eval(FakeEvent(), [])))[0]
    assert "不存在" in texts(
        run_async(plugin._cmd_eval(FakeEvent(), [str(tmp_path / "missing.json")]))
    )[0]

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            [
                {"id": ids[0], "tag": "happy"},
                {"id": 9999, "tag": "sad"},
                "junk",
            ]
        ),
        encoding="utf-8",
    )

    def fake_classify(meme_id, **kwargs):
        if meme_id == ids[0]:
            return [{"tag": "happy"}, {"tag": "sad"}]
        raise RuntimeError("missing")

    plugin._retriever.classify_meme = fake_classify
    result = texts(run_async(plugin._cmd_eval(FakeEvent(), [str(eval_path)])))[0]
    assert "top1: 1/1" in result
    assert "失败 1" in result


def test_text_command_entry_and_terminate(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    event = FakeEvent()
    event.message_str = "vm 帮助"
    helped = texts(run_async(plugin.vector_meme_text_command(event)))
    assert helped and "状态" in helped[-1]

    ignored = FakeEvent()
    ignored.message_str = "hello"
    assert run_async(plugin.vector_meme_text_command(ignored)) == []

    _seed(plugin)
    asyncio.run(plugin.terminate())
    assert plugin._embedder is not None