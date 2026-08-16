"""嵌入模型抽象层。

使用工厂模式，默认提供：
- OpenCLIPEmbedder: OpenCLIP ViT-B/32 之类的模型
- DummyEmbedder: 随机向量，仅用于测试和单元测试

自定义：
继承 BaseEmbedder 实现 embed_image / embed_text / dim，然后在
EmbedderFactory.register("my_backend", MyEmbedder) 注册。
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image, ImageOps


class BaseEmbedder(ABC):
    """所有嵌入器的基类。"""

    @abstractmethod
    def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        """输入图片或图片路径，返回 1D numpy 向量。"""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """输入文本，返回 1D numpy 向量（与图片向量同空间）。"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""
        ...

    @property
    def fingerprint(self) -> str:
        """嵌入空间指纹。模型/权重/提供商变化后必须变化。"""
        return f"{self.__class__.__name__}:{self.dim}"

    def _load_image(self, source: Image.Image | str | Path) -> Image.Image:
        if isinstance(source, Image.Image):
            img = ImageOps.exif_transpose(source).convert("RGB")
        else:
            with Image.open(source) as opened:
                img = ImageOps.exif_transpose(opened).convert("RGB")
        return img

    def close(self) -> None:
        """释放资源（GPU 显存等）。"""
        pass


# -------------------- Dummy --------------------

class DummyEmbedder(BaseEmbedder):
    """随机向量。用于不下载模型的快速测试。"""

    def __init__(self, dim: int = 512, seed: int | None = None):
        self._dim = dim
        self._rng = np.random.default_rng(seed)
        # 缓存 image_hash -> vector，让同一张图多次调用结果稳定
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            key = hashlib.md5(Path(image).read_bytes()).hexdigest()
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            key = hashlib.md5(buf.getvalue()).hexdigest()
        else:
            key = str(id(image))
        with self._lock:
            if key not in self._cache:
                v = self._rng.standard_normal(self._dim).astype("float32")
                v /= np.linalg.norm(v) + 1e-9
                self._cache[key] = v
            return self._cache[key].copy()

    def embed_text(self, text: str) -> np.ndarray:
        # 用 text 的 hash 当种子
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self._dim).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v


# -------------------- OpenCLIP --------------------

class OpenCLIPEmbedder(BaseEmbedder):
    """OpenCLIP 嵌入器。

    说明：
    - `pretrained="openai"` 时强制走 OpenAI 官方 URL 下载，避免 OpenCLIP 3.x 默认优先访问 HF Hub 导致卡住。
    - 其他权重仍走 open_clip 原生加载逻辑。
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        cache_dir: str | None = None,
    ):
        # 实例级锁：避免不同实例之间互相阻塞（原来用类属性，多实例会共享一把锁）
        self._lock = threading.Lock()
        try:
            import open_clip  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未安装 open_clip_torch，请先 pip install open_clip_torch"
            ) from e

        self._device = device
        self._model_name = model_name
        self._pretrained = pretrained
        self._cache_dir = cache_dir
        self._checkpoint_path: str | None = None
        self._checkpoint_fingerprint: str | None = None
        self._model_state_digest: str | None = None

        # 本地权重：强制跳过 HF / OpenAI 下载。
        if pretrained and Path(str(pretrained)).exists():
            checkpoint_path = str(Path(str(pretrained)).resolve())
            self._checkpoint_path = checkpoint_path
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
                device=device,
            )
            # PyTorch 2.6+ 默认 weights_only=True，TorchScript 存档会报错，需显式关闭
            open_clip.load_checkpoint(self._model, checkpoint_path, device=device, weights_only=False)
        elif pretrained == "openai":
            cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
            url = open_clip.get_pretrained_url(model_name, pretrained)
            if not url:
                raise RuntimeError(f"{model_name}/{pretrained} 没有可用 URL")
            checkpoint_path = open_clip.download_pretrained_from_url(url, cache_dir=cache_dir)
            self._checkpoint_path = checkpoint_path
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
                device=device,
                force_quick_gelu=bool(cfg.get("quick_gelu", False)),
                image_mean=tuple(cfg.get("mean")) if cfg.get("mean") else None,
                image_std=tuple(cfg.get("std")) if cfg.get("std") else None,
                image_interpolation=cfg.get("interpolation"),
                image_resize_mode=cfg.get("resize_mode"),
            )
            open_clip.load_checkpoint(self._model, checkpoint_path, device=device, weights_only=False)
        else:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device, cache_dir=cache_dir
            )
        self._tokenize = open_clip.get_tokenizer(model_name)
        self._model.eval()

        # 获取维度
        with self._lock:
            token = self._tokenize(["test"]).to(device)
            with self._no_grad():
                feat = self._model.encode_text(token)
            self._dim = int(feat.shape[-1])

        # 在 executor 线程内提前计算权重指纹，避免之后阻塞事件循环
        if self._checkpoint_path:
            self._checkpoint_hash()
        else:
            self._compute_model_state_digest()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        parts = ["open_clip", self._model_name, str(self._pretrained), str(self._dim)]
        if self._checkpoint_path:
            parts.append(self._checkpoint_hash())
        else:
            parts.append(self._compute_model_state_digest())
        return ":".join(parts)

    def _checkpoint_hash(self) -> str:
        """checkpoint 内容指纹：大小 + SHA1 前 16 位。"""
        if self._checkpoint_fingerprint is not None or not self._checkpoint_path:
            return self._checkpoint_fingerprint or self._checkpoint_path or ""
        try:
            path = Path(self._checkpoint_path)
            h = hashlib.sha1()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            self._checkpoint_fingerprint = f"{path.stat().st_size}:{h.hexdigest()[:16]}"
        except OSError:
            self._checkpoint_fingerprint = self._checkpoint_path
        return self._checkpoint_fingerprint

    def _compute_model_state_digest(self) -> str:
        """对已加载权重做内容摘要，覆盖无法拿到 checkpoint 路径的 pretrained 来源。"""
        if self._model_state_digest is not None:
            return self._model_state_digest
        try:
            torch = self._torch()
            state_dict = self._model.state_dict()
            h = hashlib.sha1()
            for name in sorted(state_dict.keys()):
                tensor = state_dict[name]
                if not torch.is_tensor(tensor):
                    tensor = torch.as_tensor(tensor)
                tensor = tensor.detach().cpu()
                h.update(name.encode("utf-8"))
                h.update(b"|")
                h.update(str(tensor.dtype).encode("utf-8"))
                h.update(b"|")
                h.update(str(tuple(tensor.shape)).encode("utf-8"))
                h.update(b"|")
                flat = tensor.reshape(-1)
                if not flat.is_contiguous():
                    flat = flat.contiguous()
                try:
                    arr = flat.numpy()
                except TypeError:
                    arr = flat.float().numpy()
                step = max(1, (1 << 20) // max(arr.itemsize, 1))
                for start in range(0, arr.size, step):
                    h.update(arr[start:start + step].tobytes())
                h.update(bytes([10]))
            self._model_state_digest = h.hexdigest()[:20]
        except Exception as exc:
            logger.warning("OpenCLIP model state digest failed: %s", exc)
            self._model_state_digest = f"state-unavailable:{self._dim}"
        return self._model_state_digest

    def _no_grad(self):
        import torch
        return torch.no_grad()

    def _torch(self):
        import torch
        return torch

    def embed_image(self, image) -> np.ndarray:
        torch = self._torch()
        img = self._load_image(image)
        x = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32").flatten()

    def embed_text(self, text: str) -> np.ndarray:
        torch = self._torch()
        token = self._tokenize([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(token)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32").flatten()

    def close(self) -> None:
        try:
            del self._model
            torch = self._torch()
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass


# -------------------- API (AstrBot Embedding 提供商) --------------------

class APIEmbedder(BaseEmbedder):
    """通过 AstrBot 已配置的 Embedding 提供商编码文本。

    - embed_text：调用 provider.get_embedding(text)（兼容 embedding/embed/encode 别名）
    - embed_image：纯文本 embedding 无图片通道，抛 NotImplementedError；
      索引侧应改用 caption 文本向量（api 后端强制开启 vision caption）。
    """

    _METHOD_NAMES = ("get_embedding", "embedding", "embed", "encode")

    def __init__(
        self,
        context: Any | None = None,
        provider_id: str = "",
        timeout: float = 60.0,
    ):
        self._context = context
        self._provider_id = provider_id or ""
        self._timeout = float(timeout)
        self._provider = None
        self._dim: int | None = None
        # 自建专用事件循环线程，避免调用方线程与 loop 线程冲突导致死锁
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(target=self._loop_runner, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(timeout=5.0)
        if self._loop is None:
            raise RuntimeError("api embedder 事件循环启动失败")
        try:
            self._probe()
        except Exception:
            self.close()
            raise

    def _loop_runner(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self._timeout)

    def _resolve_provider(self):
        context = self._context
        if context is None:
            raise RuntimeError("api embedder 需要 AstrBot context（插件实例）")
        if self._provider_id:
            getter = getattr(context, "get_provider_by_id", None)
            if getter:
                try:
                    prov = getter(self._provider_id)
                    if asyncio.iscoroutine(prov):
                        prov = self._run_async(prov)
                    if prov:
                        return prov
                except Exception as e:
                    logger.warning("按 provider_id 获取 embedding provider 失败: %s", e)
        getter = getattr(context, "get_all_embedding_providers", None)
        if getter:
            try:
                provs = getter()
                if asyncio.iscoroutine(provs):
                    provs = self._run_async(provs)
                for prov in (provs or []):
                    if prov:
                        return prov
            except Exception as e:
                logger.warning("获取 embedding providers 失败: %s", e)
        raise RuntimeError(
            "未找到可用的 AstrBot Embedding 提供商：请先在 AstrBot「服务提供商」配置，"
            "并在插件配置 embedding_provider_id 指定"
        )

    def _call(self, text: str) -> list[float]:
        prov = self._provider
        if prov is None:
            raise RuntimeError("api embedder provider 未初始化")
        for name in self._METHOD_NAMES:
            fn = getattr(prov, name, None)
            if fn is None:
                continue
            try:
                out = fn(text)
                if asyncio.iscoroutine(out):
                    out = self._run_async(out)
            except Exception as e:
                logger.warning("embedding provider %s 调用失败: %s", name, e)
                continue
            if isinstance(out, list) and out and all(
                isinstance(x, (int, float)) for x in out
            ):
                return [float(x) for x in out]
            if isinstance(out, np.ndarray):
                arr = out.reshape(-1)
                if arr.size and np.issubdtype(arr.dtype, np.number):
                    return [float(x) for x in arr]
        raise RuntimeError("embedding provider 返回了无法解析的向量")

    def _probe(self):
        self._provider = self._resolve_provider()
        vec = self._call("astrbot")
        if not vec:
            raise RuntimeError("embedding provider 返回空向量")
        self._dim = len(vec)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("api embedder 尚未初始化维度")
        return self._dim

    @property
    def fingerprint(self) -> str:
        prov = self._provider
        provider_name = type(prov).__name__ if prov is not None else "none"
        model = ""
        if prov is not None:
            for attr in ("model", "model_name", "embedding_model", "name"):
                value = getattr(prov, attr, None)
                if value:
                    model = str(value)
                    break
        return ":".join(("api", self._provider_id, provider_name, model, str(self._dim)))

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._call(text)
        return np.asarray(vec, dtype="float32")

    def embed_image(self, image) -> np.ndarray:
        raise NotImplementedError(
            "api embedder 不支持图片编码；请使用 caption 文本向量（开启 vision caption）"
        )

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2.0)
        self._loop = None

# -------------------- Factory --------------------

class EmbedderFactory:
    _registry: dict[str, Type[BaseEmbedder]] = {
        "dummy": DummyEmbedder,
        "open_clip": OpenCLIPEmbedder,
        "api": APIEmbedder,
    }

    @classmethod
    def register(cls, name: str, embedder_cls: Type[BaseEmbedder]) -> None:
        cls._registry[name] = embedder_cls

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def create(cls, backend: str, **kwargs) -> BaseEmbedder:
        if backend not in cls._registry:
            raise ValueError(
                f"未知 embedder 后端: {backend}，可选: {cls.available()}"
            )
        return cls._registry[backend](**kwargs)
