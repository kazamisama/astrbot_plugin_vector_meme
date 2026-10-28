"""LLM 钩子与外部取图入口的端到端测试。

这三条路径在 0.8.1 之前没有任何测试覆盖：
- on_decorating_result：0.8.1 的 TypeError 就是在这里被兜底 except 吞掉的
- after_message_sent：非混合模式与流式输出的补发路径
- search_sticker_for_external：外部插件取图的契约（永不抛异常）
"""
from __future__ import annotations

import asyncio

import numpy as np


# ---------- 公共构造 ----------


def _make_plugin(tmp_path, main_module, **extra_config):
    class FakeProviderManager:
        def __init__(self):
            self.personas = []

    class FakeContext:
        def __init__(self):
            self.provider_manager = FakeProviderManager()
            self.sent_messages = []

        async def send_message(self, umo, chain):
            self.sent_messages.append((umo, chain))

    meme_dir = tmp_path / "memes"
    meme_dir.mkdir(exist_ok=True)
    config = {
        "data_dir": str(tmp_path / "data"),
        "meme_dir": str(meme_dir),
        "embedder_backend": "dummy",
        "enable_prompt_injection": False,
        # 关掉随机采样，断言才可复现
        "enable_stochastic_selection": False,
    }
    config.update(extra_config)
    return main_module.VectorMemePlugin(FakeContext(), config)


async def _seed(plugin, tag="happy", texts=("happy", "joy")):
    """加载 embedder 并写入同 tag 的若干张图；返回解析后的绝对路径列表。"""
    ready = await plugin._ensure_ready()
    assert ready is not None
    emb = plugin._embedder
    db = plugin._db
    paths = []
    for i, text in enumerate(texts):
        path = plugin.meme_dir / f"{tag}_{i}.png"
        vids = db.add_vectors(emb.embed_text(text))
        db.upsert_meme(str(path), f"h{i}", tag, int(vids[0]), file_name=path.name)
        paths.append(str(path.resolve()))
    db.save_index()
    return paths


class FakeResult:
    def __init__(self, chain, content_type="text"):
        self.chain = chain
        self.result_content_type = content_type


class DecoratingEvent:
    """on_decorating_result 需要的 event 形状。"""

    def __init__(self, result, tags=None, reply="", user=""):
        self.result = result
        self.extras = {
            "vector_meme_pending_tags": tags,
            "vector_meme_query_text": reply,
            "vector_meme_user_text": user,
        }
        self.sent = []

    def get_result(self):
        return self.result

    def get_extra(self, key):
        return self.extras.get(key)

    def set_extra(self, key, value):
        self.extras[key] = value

    async def send(self, chain):
        self.sent.append(chain)


def _plain_texts(chain):
    """从 list / MessageChain 里取出 Plain 文本。"""
    items = chain.chain if hasattr(chain, "chain") else chain
    return [
        c.text for c in items
        if getattr(c, "text", None) is not None and not isinstance(c, dict)
    ]


# ---------- on_decorating_result ----------


def test_decorating_mixes_image_into_chain(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        paths = await _seed(plugin)
        result = FakeResult([main_module.Plain("你好 %%happy%%")])
        event = DecoratingEvent(result, tags=["happy"], reply="你好")
        await plugin.on_decorating_result(event)
        return paths, result, event

    paths, result, event = asyncio.run(_run())
    assert _plain_texts(result.chain) == ["你好 "]
    tail = result.chain[-1]
    assert isinstance(tail, dict) and tail["path"] in paths
    assert event.extras["vector_meme_pending_tags"] is None


def test_decorating_streaming_sends_images_directly(tmp_path, main_module):
    """流式收尾：图片不能追加进 chain（不会被发送），必须直接补发。"""
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        paths = await _seed(plugin)
        chain = [main_module.Plain("流式 %%happy%%")]
        result = FakeResult(
            chain, content_type=main_module.ResultContentType.STREAMING_FINISH
        )
        event = DecoratingEvent(result, tags=["happy"], reply="流式")
        await plugin.on_decorating_result(event)
        return paths, result, event

    paths, result, event = asyncio.run(_run())
    # 链只被清洗，不追加图片
    assert _plain_texts(result.chain) == ["流式 "]
    assert len(result.chain) == 1
    # 图片通过 event.send 直接发出
    assert len(event.sent) == 1
    sent_chain = event.sent[0]
    assert isinstance(sent_chain, main_module.MessageChain)
    assert sent_chain.chain[0]["path"] in paths


def test_decorating_non_mixed_defers_images_to_after_send(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module, enable_mixed_message=False)

    async def _run():
        paths = await _seed(plugin)
        result = FakeResult([main_module.Plain("hi %%happy%%")])
        event = DecoratingEvent(result, tags=["happy"], reply="hi")
        await plugin.on_decorating_result(event)
        return paths, result, event

    paths, result, event = asyncio.run(_run())
    assert _plain_texts(result.chain) == ["hi "]
    assert len(result.chain) == 1
    pending = event.extras["vector_meme_pending_images"]
    assert isinstance(pending, list) and len(pending) == 1
    assert pending[0]["path"] in paths


def test_decorating_cleans_placeholders_without_tags(tmp_path, main_module):
    """没有待发送 tag 时也要清理残留占位符，并且不选图。"""
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        result = FakeResult([main_module.Plain("hi %%happy%%")])
        event = DecoratingEvent(result, tags=None)
        await plugin.on_decorating_result(event)
        return result, event

    result, event = asyncio.run(_run())
    assert _plain_texts(result.chain) == ["hi "]
    assert not event.sent
    assert "vector_meme_pending_images" not in event.extras


def test_decorating_skips_when_retriever_not_ready(tmp_path, main_module, monkeypatch):
    plugin = _make_plugin(tmp_path, main_module)

    async def _not_ready():
        return None

    monkeypatch.setattr(plugin, "_ensure_ready", _not_ready)

    result = FakeResult([main_module.Plain("hi %%happy%%")])
    event = DecoratingEvent(result, tags=["happy"], reply="hi")
    asyncio.run(plugin.on_decorating_result(event))
    assert _plain_texts(result.chain) == ["hi "]
    assert len(result.chain) == 1
    assert not event.sent


def test_decorating_supports_message_chain_result(tmp_path, main_module):
    """result.chain 可能是 MessageChain，需要原地替换其内部 chain。"""
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        paths = await _seed(plugin)
        wrapper = main_module.MessageChain([main_module.Plain("hi %%happy%%")])
        result = FakeResult(wrapper)
        event = DecoratingEvent(result, tags=["happy"], reply="hi")
        await plugin.on_decorating_result(event)
        return paths, result

    paths, result = asyncio.run(_run())
    assert isinstance(result.chain, main_module.MessageChain)
    assert _plain_texts(result.chain.chain) == ["hi "]
    assert result.chain.chain[-1]["path"] in paths


# ---------- after_message_sent ----------


def _pending_event(images, send_impl=None, umo="umo://test"):
    class Ev:
        def __init__(self):
            self.extras = {"vector_meme_pending_images": images}
            self.unified_msg_origin = umo

        def get_extra(self, key):
            return self.extras.get(key)

        def set_extra(self, key, value):
            self.extras[key] = value

    ev = Ev()
    if send_impl is not None:
        ev.send = send_impl
    return ev


def test_after_send_uses_event_send(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    sent = []

    async def _send(chain):
        sent.append(chain)

    ev = _pending_event([{"path": "a.png"}, {"path": "b.png"}], _send)
    asyncio.run(plugin.after_message_sent(ev))
    assert len(sent) == 2
    assert all(isinstance(c, main_module.MessageChain) for c in sent)
    assert ev.extras["vector_meme_pending_images"] is None
    assert plugin.context.sent_messages == []


def test_after_send_falls_back_to_context(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)

    async def _send(_chain):
        raise RuntimeError("platform send failed")

    ev = _pending_event([{"path": "a.png"}], _send, umo="umo://fallback")
    asyncio.run(plugin.after_message_sent(ev))
    assert len(plugin.context.sent_messages) == 1
    umo, chain = plugin.context.sent_messages[0]
    assert umo == "umo://fallback"
    assert isinstance(chain, main_module.MessageChain)


def test_after_send_without_umo_does_not_raise(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)

    class Ev:
        def __init__(self):
            self.extras = {"vector_meme_pending_images": [{"path": "a.png"}]}

        def get_extra(self, key):
            return self.extras.get(key)

        def set_extra(self, key, value):
            self.extras[key] = value

    ev = Ev()
    asyncio.run(plugin.after_message_sent(ev))
    assert ev.extras["vector_meme_pending_images"] is None


def test_after_send_noop_without_pending(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)
    ev = _pending_event([])
    asyncio.run(plugin.after_message_sent(ev))
    assert plugin.context.sent_messages == []


# ---------- search_sticker_for_external ----------


def test_external_rejects_empty_and_unknown_without_loading(tmp_path, main_module):
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        empty = await plugin.search_sticker_for_external("")
        unknown = await plugin.search_sticker_for_external("nope")
        return empty, unknown

    empty, unknown = asyncio.run(_run())
    assert empty is None
    assert unknown is None
    assert plugin._embedder is None  # 连模型都没加载


def test_external_accepts_sticker_placeholder(tmp_path, main_module):
    """<sticker>%%happy%%</sticker> 的混合写法要能归一化。"""
    plugin = _make_plugin(tmp_path, main_module)

    async def _run():
        paths = await _seed(plugin)
        got = await plugin.search_sticker_for_external("%%happy%%")
        return paths, got

    paths, got = asyncio.run(_run())
    assert got in paths


def test_external_returns_none_when_not_ready(tmp_path, main_module, monkeypatch):
    plugin = _make_plugin(tmp_path, main_module)

    async def _not_ready():
        return None

    monkeypatch.setattr(plugin, "_ensure_ready", _not_ready)
    assert asyncio.run(plugin.search_sticker_for_external("happy")) is None


def test_external_never_raises_on_search_error(tmp_path, main_module, monkeypatch):
    """契约：检索内部炸掉也只返回 None，不向调用方抛异常。"""
    plugin = _make_plugin(tmp_path, main_module)

    async def _seed_then_break():
        paths = await _seed(plugin)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("faiss exploded")

        monkeypatch.setattr(plugin._retriever, "retrieve", _boom)
        return paths

    paths = asyncio.run(_seed_then_break())
    assert paths
    assert asyncio.run(plugin.search_sticker_for_external("happy")) is None
