"""LLM 钩子、外部入口与 tag 过滤的边界分支。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from helpers import FakeEvent, make_plugin, run_async, texts


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


async def _seed(plugin, tag="happy", count=2):
    ready = await plugin._ensure_ready()
    assert ready is not None
    paths = []
    for i in range(count):
        path = plugin.meme_dir / f"{tag}_{i}.png"
        vecs = plugin._db.add_vectors(plugin._embedder.embed_text(f"{tag} {i}"))
        plugin._db.upsert_meme(
            str(path), f"h{i}", tag, int(vecs[0]), file_name=path.name
        )
        paths.append(str(path.resolve()))
    plugin._db.save_index()
    return paths


def test_llm_response_edges(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    event = FakeEvent()
    asyncio.run(plugin.on_llm_response(event, None))
    assert event.get_extra("vector_meme_pending_tags") is None

    async def run():
        plugin2 = make_plugin(tmp_path / "second", main_module)
        await _seed(plugin2)
        response = main_module.LLMResponse("")
        response.result_chain = main_module.MessageChain(
            [main_module.Plain("hi %%happy%% %%nope%%")]
        )
        event = FakeEvent()
        await plugin2.on_llm_response(event, response)
        return event, response

    event, response = asyncio.run(run())
    assert event.get_extra("vector_meme_pending_tags") == ["happy"]
    assert response.completion_text == "hi"


def test_llm_response_filters_invalid_tags_and_zero_cap(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module, config={"max_per_message": 0})

    async def run():
        await _seed(plugin)
        response = main_module.LLMResponse(
            "%%happy%% %%123%% %%!!!%% %%" + "x" * 31 + "%%"
        )
        event = FakeEvent()
        await plugin.on_llm_response(event, response)
        return event, response

    event, response = asyncio.run(run())
    assert event.get_extra("vector_meme_pending_tags") is None
    assert "happy" not in response.completion_text


def test_llm_response_swallows_hook_exception(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    async def run():
        await _seed(plugin)
        event = FakeEvent()

        def boom(*args, **kwargs):
            raise RuntimeError("extra failed")

        event.set_extra = boom
        await plugin.on_llm_response(event, main_module.LLMResponse("%%happy%%"))

    asyncio.run(run())


def test_decorating_result_none_and_string_chain(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)

    class NoneEvent(FakeEvent):
        def get_result(self):
            return None

    asyncio.run(plugin.on_decorating_result(NoneEvent()))

    class StringResult:
        chain = "hi %%happy%%"
        result_content_type = "text"

    class Event(FakeEvent):
        def get_result(self):
            return StringResult()

    asyncio.run(plugin.on_decorating_result(Event()))


def test_compute_query_vector_sources(tmp_path, main_module):
    def run_case(index, source, reply, user, api_backend=False):
        plugin = make_plugin(
            tmp_path / f"case{index}",
            main_module,
            config={"query_text_source": source},
        )

        async def run():
            await _seed(plugin)
            spy = SpyEmbedder(plugin._embedder)
            plugin._embedder = spy
            if api_backend:
                plugin._backend_name = "api"
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

        return asyncio.run(run())

    assert run_case(1, "user", "reply q", "user q") == ["user q"]
    assert run_case(2, "combined", "reply q", "user q") == ["user q", "reply q"]
    assert run_case(3, "reply", "reply q", "user q") == []
    assert run_case(4, "combined", "reply q", "user q", api_backend=True) == ["user q"]


def test_decorating_result_isolates_image_and_stream_failures(
    tmp_path, main_module, monkeypatch
):
    plugin = make_plugin(tmp_path, main_module)
    asyncio.run(_seed(plugin))

    def _boom(cls, path):
        raise RuntimeError("image boom")

    monkeypatch.setattr(
        main_module.ImgComp, "fromFileSystem", classmethod(_boom)
    )
    event = FakeEvent()
    event.extras = {
        "vector_meme_pending_tags": ["happy"],
        "vector_meme_query_text": "happy",
        "vector_meme_user_text": "",
    }
    class Result:
        chain = [main_module.Plain("hi")]
        result_content_type = "text"
    event.get_result = lambda: Result()
    asyncio.run(plugin.on_decorating_result(event))
    assert event.get_extra("vector_meme_pending_tags") is None

    monkeypatch.undo()
    streaming = FakeEvent()
    streaming.extras = {
        "vector_meme_pending_tags": ["happy"],
        "vector_meme_query_text": "happy",
        "vector_meme_user_text": "",
    }
    class StreamResult:
        chain = [main_module.Plain("hi")]
        result_content_type = main_module.ResultContentType.STREAMING_FINISH
    streaming.get_result = lambda: StreamResult()

    async def fail_send(chain):
        raise RuntimeError("send boom")

    streaming.send = fail_send
    asyncio.run(plugin.on_decorating_result(streaming))


def test_external_search_invalid_unknown_and_no_hits(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert asyncio.run(plugin.search_sticker_for_external("")) is None
    assert asyncio.run(plugin.search_sticker_for_external("123")) is None
    assert asyncio.run(plugin.search_sticker_for_external("!!!")) is None
    assert plugin._embedder is None

    async def run():
        await _seed(plugin)
        class Empty:
            hits = []
        plugin._retriever.retrieve = lambda **kwargs: Empty()
        return await plugin.search_sticker_for_external("happy")

    assert asyncio.run(run()) is None


def test_external_search_stochastic_and_write_failure(tmp_path, main_module):
    plugin = make_plugin(
        tmp_path, main_module, config={"external_stochastic": True}
    )

    async def run():
        paths = await _seed(plugin)

        def boom(_meme_id):
            raise RuntimeError("db locked")

        plugin._db.mark_used = boom
        return paths, await plugin.search_sticker_for_external("happy")

    paths, got = asyncio.run(run())
    assert got in {str(Path(path).resolve()) for path in paths}


def test_tag_validation_known_tags_and_prompt_helpers(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    assert not plugin._is_valid_emotion_tag("")
    assert not plugin._is_valid_emotion_tag("123")
    assert not plugin._is_valid_emotion_tag("!!!")
    assert not plugin._is_valid_emotion_tag("x" * 31)
    assert not plugin._is_valid_emotion_tag("!")
    assert not plugin._is_valid_emotion_tag("ф")
    assert plugin._is_valid_emotion_tag("a")
    assert plugin._is_valid_emotion_tag("开心")
    assert plugin._known_tags() == set()
    assert plugin._build_tag_list_string() == ""

    class BrokenDB:
        def count_by_tag(self):
            raise RuntimeError("boom")

        def list_tags(self):
            raise RuntimeError("boom")

    plugin._db = BrokenDB()
    assert plugin._known_tags() == set()
    assert plugin._build_tag_list_string() == ""

    plugin._db = None
    assert plugin._known_tags() == set()
    assert plugin._build_tag_list_string() == ""

    no_prompt = make_plugin(
        tmp_path / "no-prompt",
        main_module,
        config={"enable_prompt_injection": False},
    )
    assert no_prompt._build_prompt_block() == ""


def test_dispatch_default_and_admin_fallbacks(tmp_path, main_module):
    plugin = make_plugin(tmp_path, main_module)
    out = texts(run_async(plugin._dispatch_command(FakeEvent(), [])))
    assert "状态" in out[-1]

    def boom(event, rest):
        raise RuntimeError("handler boom")

    plugin._cmd_status = boom
    errors = texts(run_async(plugin._dispatch_command(FakeEvent(), ["状态"])))
    assert "出错" in errors[0]

    class RoleEvent:
        role = "admin"

    assert plugin._event_is_admin(RoleEvent())

    class RaisingEvent:
        def is_admin(self):
            raise RuntimeError("no admin api")

    assert not plugin._event_is_admin(RaisingEvent())
