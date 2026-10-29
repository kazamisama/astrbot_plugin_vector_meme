"""剩余未覆盖分支：初始化、发送、索引/重建守卫与钩子边界。"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from helpers import FakeEvent, make_plugin, run_async, texts, write_png

from core.embedder import DummyEmbedder


def test_module_probe_and_strip_helpers(main_module):
    assert main_module._module_available("numpy")
    assert not main_module._module_available("\0bad")
    assert main_module.strip_tags_outside_stickers("") == ""
    assert (
        main_module.strip_tags_outside_stickers("<sticker>%%happy%%</sticker>")
        == "<sticker>%%happy%%</sticker>"
    )
    assert main_module.strip_tags_outside_stickers(
        "&&happy&& :sad:", allow_legacy=True
    ).strip() == ""


def test_init_flattens_nested_config_and_default_data_dir(
    tmp_path, main_module, monkeypatch
):
    import astrbot.core.utils.astrbot_path as astrbot_path

    monkeypatch.setattr(
        astrbot_path,
        "get_astrbot_plugin_data_path",
        lambda: str(tmp_path / "plugin_data"),
    )

    class ProviderManager:
        personas = []

    class Context:
        provider_manager = ProviderManager()
        sent_messages = []

    plugin = main_module.VectorMemePlugin(
        Context(),
        {
            "basic": {"embedder_backend": "dummy"},
            "model": {"device": "cpu"},
        },
    )
    assert plugin.config["embedder_backend"] == "dummy"
    assert plugin.config["device"] == "cpu"
    assert plugin.data_dir == tmp_path / "plugin_data" / "vector_meme"


def test_send_via_event_swallows_and_prewarm_failure(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(tmp_path, main_module)

    class BadEvent(FakeEvent):
        async def send(self, result):
            raise RuntimeError("send failed")

    asyncio.run(plugin._send_via_event(BadEvent(), "hello"))

    async def boom(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(plugin, "_with_embedder_heartbeat", boom)
    out = texts(run_async(plugin._cmd_prewarm(FakeEvent(), [])))
    assert "预热失败" in out[-1]


def test_ensure_ready_none_and_api_caption_weight(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)
    plugin._prepare_library_for_embedder()

    async def fake_ensure():
        return SimpleNamespace()

    monkeypatch.setattr(plugin, "_ensure_embedder", fake_ensure)
    plugin._retriever = None
    assert asyncio.run(plugin._ensure_ready()) is None

    plugin._backend_name = "api"
    plugin.caption_enabled = True
    retriever = plugin._make_retriever(plugin._db)
    assert retriever.caption_weight == 1.0


def test_ensure_embedder_applies_env_overrides(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path,
        main_module,
        backend="open_clip",
        config={
            "hf_endpoint": "https://hf.example",
            "open_clip_cache_dir": str(tmp_path / "cache"),
        },
    )
    captured = {}
    before = os.environ.get("HF_ENDPOINT")

    def fake_create(backend, **kwargs):
        captured["backend"] = backend
        captured["hf"] = os.environ.get("HF_HOME")
        captured["endpoint"] = os.environ.get("HF_ENDPOINT")
        return DummyEmbedder(8)

    monkeypatch.setattr(
        main_module.EmbedderFactory, "create", staticmethod(fake_create)
    )
    embedder = asyncio.run(plugin._ensure_embedder(prepare_library=False))
    assert embedder.dim == 8
    assert captured["backend"] == "open_clip"
    assert captured["endpoint"] == "https://hf.example"
    assert captured["hf"] == str(tmp_path / "cache")
    assert os.environ.get("HF_ENDPOINT") == before


def test_external_caption_invalid_path_and_export_skips(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(tmp_path, main_module)
    monkeypatch.setattr(main_module, "__file__", str(tmp_path / "main.py"))
    caption_dir = tmp_path / "memes"
    caption_dir.mkdir(parents=True, exist_ok=True)
    (caption_dir / "captions.json").write_text(
        json.dumps({"\u0000bad": "cap", "ok.png": " "}), encoding="utf-8"
    )
    assert plugin._load_external_captions() == {}

    db = plugin._db
    inside = db.upsert_meme(
        str(plugin.meme_dir / "happy" / "a.png"), "a", "happy", -1
    )
    outside = db.upsert_meme(str(tmp_path / "outside" / "b.png"), "b", "happy", -1)
    db.upsert_meme(str(plugin.meme_dir / "happy" / "c.png"), "c", "happy", -1)
    with db._conn() as c:
        c.execute("UPDATE memes SET caption = 'cap' WHERE id = ?", (inside,))
        c.execute("UPDATE memes SET caption = 'x' WHERE id = ?", (outside,))
    assert plugin._export_captions() == 1


def test_sync_tag_registry_empty_non_dict_and_stale(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)
    db = plugin._db
    monkeypatch.setattr(plugin, "_load_tag_schema", lambda: {})
    assert plugin._sync_tag_registry(db) == 0

    monkeypatch.setattr(
        plugin,
        "_load_tag_schema",
        lambda: {"happy": {"meaning": "开心"}, "bad": "not-a-dict"},
    )
    db.upsert_tag("stale")
    db.upsert_meme("a.png", "a", "happy", -1)
    db.upsert_meme("b.png", "b", "extra", -1)
    assert plugin._sync_tag_registry(db) == 1
    assert db.get_tag("happy") is not None
    assert db.get_tag("extra") is not None
    assert db.get_tag("bad") is None


def test_index_guards_and_caption_result(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    png = write_png(plugin.meme_dir / "happy" / "a.png")
    assert "不是目录" in texts(
        run_async(plugin._cmd_index(FakeEvent(), [str(png)]))
    )[0]

    async def not_ready(*args, **kwargs):
        return None

    monkeypatch.setattr(plugin, "_with_embedder_heartbeat", not_ready)
    assert "embedder 未就绪" in texts(
        run_async(plugin._cmd_index(FakeEvent(), [str(plugin.meme_dir)]))
    )[-1]

    ready = asyncio.run(plugin._ensure_ready())

    async def ready_now(*args, **kwargs):
        return ready

    async def fake_enrich():
        return {"total": 1, "ok": 1, "failed": 0}

    monkeypatch.setattr(plugin, "_with_embedder_heartbeat", ready_now)
    monkeypatch.setattr(plugin, "_enrich_captions_after_index", fake_enrich)
    out = texts(run_async(plugin._cmd_index(FakeEvent(), [str(plugin.meme_dir)])))
    assert any("caption: total=1" in text for text in out)


def test_rebuild_missing_meme_dir_and_missing_db(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    plugin.meme_dir_setting = str(tmp_path / "missing")
    assert "meme_dir 未配置或不存在" in texts(
        run_async(plugin._cmd_rebuild(FakeEvent(), []))
    )[0]

    plugin2 = make_plugin(tmp_path / "second", main_module)
    plugin2._db = None
    out = texts(run_async(plugin2._cmd_rebuild(FakeEvent(), [])))
    assert "数据库未初始化" in out[-1]


def test_repair_guards_and_caption_result(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )

    async def not_ready(*args, **kwargs):
        return None

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", not_ready)
    assert "embedder 未就绪" in texts(
        run_async(plugin._cmd_repair(FakeEvent(), []))
    )[-1]

    db = plugin._db

    async def corrupt_after_ready(*args, **kwargs):
        db.index_corrupted = True
        return (None, None, db)

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", corrupt_after_ready)
    assert "FAISS 索引已损坏" in texts(
        run_async(plugin._cmd_repair(FakeEvent(), []))
    )[-1]

    db.index_corrupted = False
    ready = asyncio.run(plugin._ensure_ready())

    async def fresh_ready(*args, **kwargs):
        return ready

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", fresh_ready)

    async def fake_enrich():
        return {"total": 1, "ok": 1, "failed": 0}

    monkeypatch.setattr(plugin, "_enrich_captions_after_index", fake_enrich)
    out = texts(run_async(plugin._cmd_repair(FakeEvent(), [])))
    assert any("caption: total=1" in text for text in out)


def test_dedup_parse_fallback_and_no_groups(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    async def seed():
        await plugin._ensure_ready()
        vec = plugin._embedder.embed_text("only one").reshape(1, -1)
        vids = plugin._db.add_vectors(vec)
        plugin._db.upsert_meme("one.png", "h", "happy", int(vids[0]))
        plugin._db.save_index()

    asyncio.run(seed())
    out = texts(run_async(plugin._cmd_dedup(FakeEvent(), ["auto"])))
    assert "未发现近重复簇" in out[-1]

    out2 = texts(run_async(plugin._cmd_dedup(FakeEvent(), ["preview", "0.99"])))
    assert "未发现近重复簇" in out2[-1]


async def _seed(plugin, tag, text):
    ready = await plugin._ensure_ready()
    assert ready is not None
    vec = plugin._embedder.embed_text(text).reshape(1, -1)
    vids = plugin._db.add_vectors(vec)
    plugin._db.upsert_meme(
        str(plugin.meme_dir / f"{tag}.png"), f"h-{tag}", tag, int(vids[0])
    )
    plugin._db.save_index()


def test_llm_legacy_duplicate_and_cap(tmp_path, main_module):
    plugin = make_plugin(
        tmp_path,
        main_module,
        config={"allow_legacy_markup": True, "max_per_message": 2},
    )

    async def run():
        for tag in ("happy", "sad", "angry"):
            await _seed(plugin, tag, tag)
        event = FakeEvent()
        await plugin.on_llm_response(
            event, main_module.LLMResponse("&&happy&& :sad:")
        )
        legacy = event.get_extra("vector_meme_pending_tags")
        event2 = FakeEvent()
        await plugin.on_llm_response(
            event2,
            main_module.LLMResponse(
                "%%happy%% %%happy%% %%sad%% %%angry%%"
            ),
        )
        return legacy, event2.get_extra("vector_meme_pending_tags")

    legacy, capped = asyncio.run(run())
    assert legacy == ["happy", "sad"]
    assert capped == ["happy", "sad"]

    event = FakeEvent()
    asyncio.run(plugin.on_llm_response(event, main_module.LLMResponse("")))
    assert event.get_extra("vector_meme_pending_tags") is None


def test_decorating_chain_types_and_query_vector_edges(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(tmp_path, main_module, config={"query_text_source": "user"})

    class NoneResult:
        chain = None
        result_content_type = "text"

    class UnknownResult:
        chain = [123]
        result_content_type = "text"

    class RawChainResult:
        chain = 123
        result_content_type = "text"

    for result in (NoneResult(), UnknownResult(), RawChainResult()):
        event = FakeEvent()
        event.extras = {}
        event.get_result = lambda r=result: r
        asyncio.run(plugin.on_decorating_result(event))

    class SpyEmbedder:
        def __init__(self, inner):
            self.inner = inner
            self.texts = []

        def embed_text(self, text):
            self.texts.append(text)
            return self.inner.embed_text(text)

        @property
        def dim(self):
            return self.inner.dim

    async def run(source, reply, user):
        await _seed(plugin, "happy", "happy")
        spy = SpyEmbedder(plugin._embedder)
        plugin._embedder = spy
        plugin.config["query_text_source"] = source
        plugin._retriever.pick = lambda *args, **kwargs: None
        event = FakeEvent()
        event.extras = {
            "vector_meme_pending_tags": ["happy"],
            "vector_meme_query_text": reply,
            "vector_meme_user_text": user,
        }

        class Result:
            chain = [main_module.Plain("hi")]
            result_content_type = "text"

        event.get_result = lambda: Result()
        await plugin.on_decorating_result(event)
        return spy.texts

    assert asyncio.run(run("user", "reply", "")) == []
    assert asyncio.run(run("combined", "", "user only")) == ["user only"]


def test_decorating_and_after_send_outer_exceptions(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)

    async def boom_ready():
        raise RuntimeError("ready boom")

    monkeypatch.setattr(plugin, "_ensure_ready", boom_ready)
    event = FakeEvent()
    event.extras = {"vector_meme_pending_tags": ["happy"]}

    class Result:
        chain = [main_module.Plain("hi")]
        result_content_type = "text"

    event.get_result = lambda: Result()
    asyncio.run(plugin.on_decorating_result(event))

    async def boom_send(chain):
        raise RuntimeError("send boom")

    async def boom_context(umo, chain):
        raise RuntimeError("context boom")

    plugin.context.send_message = boom_context
    after = FakeEvent()
    after.send = boom_send
    after.unified_msg_origin = "umo"
    after.extras = {"vector_meme_pending_images": [object()]}
    asyncio.run(plugin.after_message_sent(after))


def test_external_ready_none_blank_tag_and_empty_tag_name(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(tmp_path, main_module)

    async def run():
        await _seed(plugin, "happy", "happy")

        async def not_ready():
            return None

        monkeypatch.setattr(plugin, "_ensure_ready", not_ready)
        return await plugin.search_sticker_for_external("happy")

    assert asyncio.run(run()) is None
    assert not plugin._is_valid_emotion_tag(" ")

    plugin._db.upsert_tag("")
    assert "- :" not in plugin._build_tag_list_string()


def test_terminate_swallows_cleanup_errors(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    def boom():
        raise RuntimeError("restore boom")

    def save_boom():
        raise RuntimeError("save boom")

    plugin._restore_personas = boom
    plugin._db.save_index = save_boom
    asyncio.run(plugin.terminate())


def test_backup_helpers_report_errors(tmp_path, main_module):
    directory = tmp_path / "dir"
    directory.mkdir()
    errors = main_module.VectorMemePlugin._restore_backup_files(
        [(directory, tmp_path / "bak", False)]
    )
    assert errors
    main_module.VectorMemePlugin._delete_backup_files(
        [(directory, directory, True)]
    )


def test_search_explain_usage_and_not_ready(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(tmp_path, main_module)
    assert "用法" in texts(run_async(plugin._cmd_search(FakeEvent(), [])))[0]
    assert "用法" in texts(run_async(plugin._cmd_explain(FakeEvent(), [])))[0]

    async def not_ready(event, action):
        return None

    monkeypatch.setattr(plugin, "_ensure_ready_with_heartbeat", not_ready)
    for rest in (["--tag", "happy"], ["x"]):
        assert "未就绪" in texts(run_async(plugin._cmd_search(FakeEvent(), rest)))[-1]
        assert "未就绪" in texts(run_async(plugin._cmd_explain(FakeEvent(), rest)))[-1]

    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps([{"id": 1, "tag": "happy"}]), encoding="utf-8")
    assert "未就绪" in texts(
        run_async(plugin._cmd_eval(FakeEvent(), [str(eval_path)]))
    )[-1]


def test_strip_legacy_inside_sticker_and_default_meme_dir(tmp_path, main_module):
    from pathlib import Path

    assert main_module.strip_tags_outside_stickers(
        "<sticker>x</sticker> &&happy&&", allow_legacy=True
    ).strip() == "<sticker>x</sticker>"
    plugin = make_plugin(tmp_path, main_module, config={"meme_dir": ""})
    assert plugin.meme_dir == Path(main_module.__file__).resolve().parent / "memes"


def test_enrich_captions_success_path(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )

    class Available:
        def available(self):
            return True

    async def fake_enrich(*args, **kwargs):
        return {"total": 1, "ok": 1, "failed": 0}

    monkeypatch.setattr(plugin, "_ensure_captioner", lambda: Available())
    monkeypatch.setattr(plugin, "_load_external_captions", lambda: {})
    monkeypatch.setattr(main_module, "enrich_meme_captions", fake_enrich)
    out = asyncio.run(plugin._enrich_captions_after_index())
    assert out["total"] == 1


def test_swap_and_restore_failure_paths(tmp_path, main_module, monkeypatch):
    import pytest

    plugin = make_plugin(tmp_path, main_module)
    old_db = plugin._db
    tmp_db = SimpleNamespace(
        db_path=tmp_path / "new.db",
        index_path=tmp_path / "new.faiss",
    )
    monkeypatch.setattr(plugin, "_restore_backup_files", lambda paths: ["restore failed"])

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(main_module.os, "replace", boom)
    with pytest.raises(OSError):
        plugin._swap_rebuilt_db(old_db, tmp_db, [])

    def restore_boom(paths):
        raise RuntimeError("restore boom")

    monkeypatch.setattr(plugin, "_restore_backup_files", restore_boom)
    plugin._restore_failed_rebuild(old_db, old_db.dim, [])


def test_rebuild_caption_enrich_success(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    write_png(plugin.meme_dir / "happy" / "a.png")

    class Available:
        def available(self):
            return True

    async def fake_enrich(*args, **kwargs):
        return {"total": 1, "ok": 1, "failed": 0}

    monkeypatch.setattr(plugin, "_ensure_captioner", lambda: Available())
    monkeypatch.setattr(plugin, "_load_external_captions", lambda: {})
    monkeypatch.setattr(plugin, "_collect_db_captions", lambda db: {})
    monkeypatch.setattr(main_module, "enrich_meme_captions", fake_enrich)
    out = texts(run_async(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建完成" in out[-1]


def test_rebuild_caption_unavailable_branch(tmp_path, main_module, monkeypatch):
    plugin = make_plugin(
        tmp_path, main_module, config={"enable_vision_caption": True}
    )
    write_png(plugin.meme_dir / "happy" / "a.png")

    class Unavailable:
        def available(self):
            return False

    monkeypatch.setattr(plugin, "_ensure_captioner", lambda: Unavailable())
    monkeypatch.setattr(plugin, "_load_external_captions", lambda: {})

    monkeypatch.setattr(plugin, "_collect_db_captions", lambda db: {})
    out = texts(run_async(plugin._cmd_rebuild(FakeEvent(), [])))
    assert "重建完成" in out[-1]
