"""pytest 共享配置：注册 astrbot 最小桩，供 main.py 测试导入。"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _decorator(*_args, **_kwargs):
    def wrap(fn):
        return fn
    return wrap


class GreedyStr(str):
    pass


class MessageChain:
    def __init__(self, chain=None):
        self.chain = chain if chain is not None else []


class MessageEventResult:
    def message(self, text):
        return {"text": str(text)}


class ResultContentType:
    STREAMING_FINISH = "streaming_finish"
    TEXT = "text"


class Plain:
    def __init__(self, text=""):
        self.text = text


class Image:
    @classmethod
    def fromFileSystem(cls, path):
        return {"path": str(path)}


class LLMResponse:
    def __init__(self, completion_text=""):
        self.completion_text = completion_text


class AstrMessageEvent:
    pass


class Context:
    pass


class Star:
    def __init__(self, context):
        self.context = context


def _install_astrbot_stubs():
    if "astrbot" in sys.modules:
        return
    _make_module("astrbot")
    api = _make_module("astrbot.api")
    api.logger = logging.getLogger("astrbot.stub")
    _make_module("astrbot.api.all")
    _make_module(
        "astrbot.api.event",
        AstrMessageEvent=AstrMessageEvent,
        filter=types.SimpleNamespace(
            EventMessageType=types.SimpleNamespace(ALL="all"),
            command=_decorator,
            event_message_type=_decorator,
            on_llm_response=_decorator,
            on_decorating_result=_decorator,
            after_message_sent=_decorator,
        ),
    )
    _make_module("astrbot.api.message_components", Image=Image, Plain=Plain)
    _make_module("astrbot.api.provider", LLMResponse=LLMResponse)
    _make_module("astrbot.api.star", Context=Context, Star=Star, register=_decorator)
    _make_module("astrbot.core")
    _make_module("astrbot.core.message")
    _make_module(
        "astrbot.core.message.message_event_result",
        MessageChain=MessageChain,
        ResultContentType=ResultContentType,
        MessageEventResult=MessageEventResult,
    )
    _make_module("astrbot.core.star")
    _make_module("astrbot.core.star.filter")
    _make_module("astrbot.core.star.filter.command", GreedyStr=GreedyStr)
    _make_module("astrbot.core.utils")
    _make_module(
        "astrbot.core.utils.astrbot_path",
        get_astrbot_plugin_data_path=lambda: str(ROOT / ".cache"),
    )
    pkg = types.ModuleType("vector_meme")
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = "vector_meme"
    sys.modules["vector_meme"] = pkg


_install_astrbot_stubs()


def load_main_module():
    """以 vector_meme.main 形式加载 main.py，使相对导入 .core 可用。"""
    if "vector_meme.main" in sys.modules:
        return sys.modules["vector_meme.main"]
    spec = importlib.util.spec_from_file_location("vector_meme.main", ROOT / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vector_meme.main"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def main_module():
    return load_main_module()
