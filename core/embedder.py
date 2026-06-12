"""嵌入模型抽象层。

使用工厂模式，默认提供：
- OpenCLIPEmbedder: OpenCLIP ViT-B/32 之类的模型
- DummyEmbedder: 随机向量，仅用于测试和单元测试

自定义：
继承 BaseEmbedder 实现 embed_image / embed_text / dim，然后在
EmbedderFactory.register("my_backend", MyEmbedder) 注册。
"""
from __future__ import annotations

import hashlib
import io
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import numpy as np
from PIL import Image


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

    def _load_image(self, source: Image.Image | str | Path) -> Image.Image:
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
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

        # 本地权重：强制跳过 HF / OpenAI 下载。
        if pretrained and Path(str(pretrained)).exists():
            checkpoint_path = str(Path(str(pretrained)).resolve())
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
                device=device,
            )
            open_clip.load_checkpoint(self._model, checkpoint_path, device=device)
        elif pretrained == "openai":
            cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
            url = open_clip.get_pretrained_url(model_name, pretrained)
            if not url:
                raise RuntimeError(f"{model_name}/{pretrained} 没有可用 URL")
            checkpoint_path = open_clip.download_pretrained_from_url(url, cache_dir=cache_dir)
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
            open_clip.load_checkpoint(self._model, checkpoint_path, device=device)
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

    @property
    def dim(self) -> int:
        return self._dim

    _lock = threading.Lock()

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


# -------------------- Factory --------------------

class EmbedderFactory:
    _registry: dict[str, Type[BaseEmbedder]] = {
        "dummy": DummyEmbedder,
        "open_clip": OpenCLIPEmbedder,
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
