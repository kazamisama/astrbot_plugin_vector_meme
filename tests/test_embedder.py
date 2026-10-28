"""embedder 后端测试：Dummy / OpenCLIP（假模块）/ API（假 provider）/ 工厂。

不需要安装 torch 或 open_clip_torch：OpenCLIP 分支通过注入假模块覆盖。
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
import types

import numpy as np
import pytest
from PIL import Image

from core.embedder import (
    APIEmbedder,
    BaseEmbedder,
    DummyEmbedder,
    EmbedderFactory,
    OpenCLIPEmbedder,
)


# ---------- Dummy ----------


def test_dummy_is_deterministic_and_caches_images(tmp_path):
    emb = DummyEmbedder(dim=16, seed=1)
    assert emb.dim == 16
    assert emb.fingerprint == "DummyEmbedder:16"

    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img)

    first = emb.embed_image(img)
    second = emb.embed_image(img)
    assert first.shape == (16,)
    assert np.array_equal(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5

    other = tmp_path / "b.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(other)
    assert not np.array_equal(first, emb.embed_image(other))

    assert np.array_equal(emb.embed_text("你好"), emb.embed_text("你好"))
    assert not np.array_equal(emb.embed_text("你好"), emb.embed_text("再见"))
    emb.close()
    assert np.array_equal(emb.embed_text("你好"), emb.embed_text("你好"))


def test_dummy_accepts_pil_image():
    emb = DummyEmbedder(dim=8)
    vec = emb.embed_image(Image.new("RGB", (4, 4), (1, 2, 3)))
    assert vec.shape == (8,)
    assert np.array_equal(vec, emb.embed_image(Image.new("RGB", (4, 4), (1, 2, 3))))


def test_base_embedder_load_image_normalizes(tmp_path):
    class Echo(BaseEmbedder):
        def embed_image(self, image):
            return np.zeros(1, dtype="float32")

        def embed_text(self, text):
            return np.zeros(1, dtype="float32")

        @property
        def dim(self):
            return 1

    emb = Echo()
    img = Image.new("RGBA", (5, 5), (1, 2, 3, 4))
    loaded = emb._load_image(img)
    assert loaded.mode == "RGB"
    assert loaded.size == (5, 5)
    assert emb.fingerprint == "Echo:1"


# ---------- OpenCLIP（假模块） ----------


class FakeTensor:
    """把 numpy 包一层，模拟 torch.Tensor 在 embedder 里用到的那几个方法。"""

    def __init__(self, data):
        self._data = np.asarray(data, dtype="float32")

    @property
    def shape(self):
        return self._data.shape

    def to(self, device):
        return self

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self._data, axis))

    def norm(self, dim=-1, keepdim=True):
        return FakeTensor(np.linalg.norm(self._data, axis=dim, keepdims=keepdim))

    def __truediv__(self, other):
        value = other._data if isinstance(other, FakeTensor) else other
        return FakeTensor(self._data / value)

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def astype(self, dtype):
        return self._data.astype(dtype)

    def flatten(self):
        return self._data.flatten()


class FakeModel:
    def __init__(self, digest_error=False):
        self.eval_called = False
        self.loaded = None
        self._digest_error = digest_error

    def eval(self):
        self.eval_called = True

    def encode_text(self, tokens):
        return FakeTensor([[3.0, 4.0]])

    def encode_image(self, batch):
        return FakeTensor([[2.0, 0.0, 0.0, 0.0]])

    def state_dict(self):
        if self._digest_error:
            raise RuntimeError("no state")
        return {}


def _install_fake_clip(monkeypatch, tmp_path, model=None, ckpt_bytes=b"abc"):
    model = model or FakeModel()
    created = {}
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(ckpt_bytes)

    mod = types.ModuleType("open_clip")

    def get_pretrained_cfg(name, pretrained):
        return {
            "quick_gelu": True,
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic",
            "resize_mode": "shortest",
        }

    def create_model_and_transforms(name, pretrained=None, device="cpu", **kwargs):
        created["name"] = name
        created["pretrained"] = pretrained
        created["kwargs"] = kwargs
        return model, None, (lambda img: FakeTensor(np.zeros((3, 4, 4))))

    def load_checkpoint(m, path, device="cpu", weights_only=True):
        m.loaded = (str(path), weights_only)

    def download_pretrained_from_url(url, cache_dir=None):
        created["url"] = url
        created["cache_dir"] = cache_dir
        return str(ckpt)

    mod.get_pretrained_cfg = get_pretrained_cfg
    mod.get_pretrained_url = lambda name, pretrained: "https://example.test/w.pt"
    mod.download_pretrained_from_url = download_pretrained_from_url
    mod.create_model_and_transforms = create_model_and_transforms
    mod.load_checkpoint = load_checkpoint
    mod.get_tokenizer = lambda name: (lambda texts: FakeTensor([[1.0, 2.0]]))

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: contextlib.nullcontext()
    fake_torch.is_tensor = lambda obj: False
    fake_torch.as_tensor = lambda obj: obj
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    monkeypatch.setitem(sys.modules, "open_clip", mod)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return created, model, ckpt


def test_openclip_requires_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(RuntimeError, match="requirements-openclip"):
        OpenCLIPEmbedder()


def test_openclip_openai_branch_downloads_and_embeds(monkeypatch, tmp_path):
    created, model, ckpt = _install_fake_clip(monkeypatch, tmp_path)
    emb = OpenCLIPEmbedder(model_name="ViT-B-32", pretrained="openai", cache_dir=str(tmp_path / "cache"))

    assert created["name"] == "ViT-B-32"
    assert created["pretrained"] is None
    assert created["url"] == "https://example.test/w.pt"
    assert created["cache_dir"] == str(tmp_path / "cache")
    assert model.eval_called
    assert model.loaded[0] == str(ckpt.resolve())
    assert model.loaded[1] is False
    assert emb.dim == 2

    expected_hash = hashlib.sha1(b"abc").hexdigest()[:16]
    assert emb.fingerprint == f"open_clip:ViT-B-32:openai:2:3:{expected_hash}"
    assert emb.fingerprint == emb.fingerprint  # 缓存后仍然一致

    vec = emb.embed_image(Image.new("RGB", (4, 4), (0, 0, 0)))
    assert np.allclose(vec, [1.0, 0.0, 0.0, 0.0])
    text_vec = emb.embed_text("hello")
    assert np.allclose(text_vec, [0.6, 0.8])

    emb.close()
    assert not hasattr(emb, "_model")


def test_openclip_local_checkpoint_skips_download(monkeypatch, tmp_path):
    created, model, ckpt = _install_fake_clip(monkeypatch, tmp_path, ckpt_bytes=b"local-ckpt")
    emb = OpenCLIPEmbedder(pretrained=str(ckpt))

    assert created["pretrained"] is None
    assert "url" not in created
    assert model.loaded[0] == str(ckpt.resolve())
    assert model.loaded[1] is False
    expected_hash = hashlib.sha1(b"local-ckpt").hexdigest()[:16]
    assert emb.fingerprint.endswith(f"10:{expected_hash}")


def test_openclip_missing_url_raises(monkeypatch, tmp_path):
    _install_fake_clip(monkeypatch, tmp_path)
    sys.modules["open_clip"].get_pretrained_url = lambda name, pretrained: None
    with pytest.raises(RuntimeError, match="没有可用 URL"):
        OpenCLIPEmbedder(pretrained="openai")


def test_openclip_passthrough_uses_model_state_digest(monkeypatch, tmp_path):
    _install_fake_clip(monkeypatch, tmp_path)
    emb = OpenCLIPEmbedder(pretrained="custom-weights")
    empty_digest = hashlib.sha1(b"").hexdigest()[:20]
    assert emb.fingerprint == f"open_clip:ViT-B-32:custom-weights:2:{empty_digest}"


def test_openclip_digest_failure_degrades_gracefully(monkeypatch, tmp_path):
    _install_fake_clip(monkeypatch, tmp_path, model=FakeModel(digest_error=True))
    emb = OpenCLIPEmbedder(pretrained="custom-weights")
    assert emb.fingerprint.endswith("state-unavailable:2")


def test_openclip_checkpoint_hash_falls_back_on_missing_file(tmp_path):
    emb = OpenCLIPEmbedder.__new__(OpenCLIPEmbedder)
    emb._checkpoint_path = str(tmp_path / "nope.pt")
    emb._checkpoint_fingerprint = None
    assert emb._checkpoint_hash() == str(tmp_path / "nope.pt")


# ---------- API ----------


class SyncProvider:
    model = "text-embedding-3"

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


class AliasProvider:
    def encode(self, text):
        return np.array([1.0, 2.0], dtype="float32")


class AsyncProvider:
    async def get_embedding(self, text):
        return [0.5, 0.5]


class BadProvider:
    def get_embedding(self, text):
        return "oops"


class RaisingProvider:
    def get_embedding(self, text):
        raise RuntimeError("boom")


class FakeContext:
    def __init__(self, providers=None, by_id=None):
        self.providers = providers or []
        self.by_id = by_id or {}

    def get_all_embedding_providers(self):
        return self.providers

    def get_provider_by_id(self, provider_id):
        return self.by_id.get(provider_id)


def _api(context, provider_id=""):
    emb = APIEmbedder(context, provider_id=provider_id, timeout=5.0)
    return emb


def test_api_resolves_first_available_provider():
    emb = _api(FakeContext([SyncProvider()]))
    try:
        assert emb.dim == 3
        assert emb.fingerprint == "api::SyncProvider:text-embedding-3:3"
        vec = emb.embed_text("hi")
        assert vec.dtype == np.float32
        assert np.allclose(vec, [0.1, 0.2, 0.3])
        with pytest.raises(NotImplementedError):
            emb.embed_image("x.png")
    finally:
        emb.close()


def test_api_prefers_explicit_provider_id():
    chosen = AliasProvider()
    emb = _api(FakeContext([SyncProvider()], by_id={"p1": chosen}), provider_id="p1")
    try:
        assert emb.dim == 2
        assert emb.fingerprint == "api:p1:AliasProvider::2"
    finally:
        emb.close()


def test_api_supports_async_provider_and_getters():
    class AsyncContext:
        async def get_provider_by_id(self, provider_id):
            return AsyncProvider()

        async def get_all_embedding_providers(self):
            return []

    emb = _api(AsyncContext(), provider_id="p1")
    try:
        assert emb.dim == 2
        assert np.allclose(emb.embed_text("x"), [0.5, 0.5])
    finally:
        emb.close()


def test_api_falls_back_when_id_lookup_misses():
    emb = _api(FakeContext([AliasProvider()], by_id={}), provider_id="missing")
    try:
        assert emb.fingerprint.startswith("api:missing:AliasProvider:")
    finally:
        emb.close()


def test_api_requires_context():
    with pytest.raises(RuntimeError, match="需要 AstrBot context"):
        APIEmbedder(None)


def test_api_reports_missing_provider_with_hint():
    with pytest.raises(RuntimeError, match="服务提供商"):
        APIEmbedder(FakeContext([]))


def test_api_wraps_unparseable_output():
    with pytest.raises(RuntimeError, match="无法解析"):
        APIEmbedder(FakeContext([BadProvider()]))


def test_api_wraps_provider_errors():
    with pytest.raises(RuntimeError, match="无法解析"):
        APIEmbedder(FakeContext([RaisingProvider()]))


def test_api_tolerates_provider_lookup_exception():
    class BrokenContext:
        def get_provider_by_id(self, provider_id):
            raise RuntimeError("nope")

        def get_all_embedding_providers(self):
            return [SyncProvider()]

    emb = _api(BrokenContext(), provider_id="x")
    try:
        assert emb.dim == 3
    finally:
        emb.close()


def test_api_close_is_idempotent():
    emb = _api(FakeContext([SyncProvider()]))
    emb.close()
    emb.close()
    assert emb._loop is None


# ---------- 工厂 ----------


def test_factory_creates_and_lists_backends():
    available = EmbedderFactory.available()
    assert {"dummy", "open_clip", "api"} <= set(available)
    assert isinstance(EmbedderFactory.create("dummy", dim=8), DummyEmbedder)
    with pytest.raises(ValueError, match="未知 embedder 后端"):
        EmbedderFactory.create("nope")


def test_factory_register_custom_backend():
    class Custom(DummyEmbedder):
        pass

    EmbedderFactory.register("custom_test", Custom)
    try:
        assert isinstance(EmbedderFactory.create("custom_test", dim=4), Custom)
        assert "custom_test" in EmbedderFactory.available()
    finally:
        EmbedderFactory._registry.pop("custom_test", None)
