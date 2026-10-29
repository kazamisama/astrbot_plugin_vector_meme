"""测试共享构造：插件实例、事件对象与异步执行辅助。"""
from __future__ import annotations

import asyncio

from PIL import Image


class FakeEvent:
    """AstrBot 事件的最小形状，覆盖命令、钩子与维护命令用到的接口。"""

    def __init__(self, admin: bool = True):
        self._admin = admin
        self.sent = []
        self.stopped = False
        self.extras = {}

    def is_admin(self) -> bool:
        return self._admin

    @staticmethod
    def plain_result(text: str):
        return ("plain", text)

    @staticmethod
    def chain_result(chain):
        return ("chain", chain)

    def stop_event(self) -> None:
        self.stopped = True

    def set_extra(self, key, value) -> None:
        self.extras[key] = value

    def get_extra(self, key):
        return self.extras.get(key)

    async def send(self, result) -> None:
        self.sent.append(result)


def make_plugin(
    tmp_path,
    main_module,
    *,
    backend: str = "dummy",
    personas=None,
    providers=None,
    getter=None,
    config: dict | None = None,
):
    """构造插件；providers/getter 只用于 auto 后端探测。"""

    class FakeProviderManager:
        def __init__(self):
            self.personas = [] if personas is None else personas

    class FakeContext:
        def __init__(self):
            self.provider_manager = FakeProviderManager()
            if getter is not None:
                self.get_all_embedding_providers = getter
            elif providers is not None:
                self.get_all_embedding_providers = lambda: providers
            self.sent_messages = []

        async def send_message(self, umo, chain):
            self.sent_messages.append((umo, chain))

    meme_dir = tmp_path / "memes"
    meme_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "data_dir": str(tmp_path / "data"),
        "meme_dir": str(meme_dir),
        "embedder_backend": backend,
        "enable_prompt_injection": False,
        "enable_vision_caption": False,
        "enable_stochastic_selection": False,
    }
    if config:
        values.update(config)
    return main_module.VectorMemePlugin(FakeContext(), values)


async def collect_async(agen):
    return [item async for item in agen]


def run_async(gen):
    return asyncio.run(collect_async(gen))


def texts(out) -> list[str]:
    return [item[1] if isinstance(item, tuple) else str(item) for item in out]


def write_png(path, color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path
