"""视觉 caption 生成与 enrich_meme_captions 的单元测试。

全部使用假 context / 假 llm_generate，不触网、不需要视觉模型。
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
from core.captioner import (
    CAPTION_PROMPT,
    CAPTION_SYSTEM_PROMPT,
    CaptionGenerator,
    enrich_meme_captions,
)


class FakeResponse:
    def __init__(self, text: str):
        self.completion_text = text


class Plain:
    """模拟 AstrBot 消息链组件（类名必须叫 Plain 才走分支）。"""

    def __init__(self, text: str):
        self.text = text


def _run(coro):
    return asyncio.run(coro)


# ---------- available / caption_image ----------


def test_available_requires_context_and_provider():
    assert not CaptionGenerator(None).available()
    assert not CaptionGenerator(object()).available()
    assert not CaptionGenerator(type("C", (), {"llm_generate": lambda self, **kw: None})(), "  ").available()
    ctx = type("C", (), {"llm_generate": lambda self, **kw: None})()
    assert CaptionGenerator(ctx, "qwen-vl").available()


def test_caption_image_success_forwards_arguments(tmp_path):
    calls = {}

    class Ctx:
        async def llm_generate(self, **kwargs):
            calls.update(kwargs)
            return FakeResponse(json.dumps({"caption": "无语", "en_keywords": "speechless"}))

    gen = CaptionGenerator(Ctx(), "my-vl")
    image = tmp_path / "a.png"
    result = _run(gen.caption_image(image))

    assert result == {"caption": "无语", "en_keywords": "speechless"}
    assert calls["chat_provider_id"] == "my-vl"
    assert calls["prompt"] == CAPTION_PROMPT
    assert calls["system_prompt"] == CAPTION_SYSTEM_PROMPT
    assert calls["image_urls"] == [str(image)]
    assert calls["temperature"] == 0
    assert calls["max_tokens"] == 300


def test_caption_image_accepts_plain_string_response():
    class Ctx:
        async def llm_generate(self, **kwargs):
            return '{"caption": "好耶", "en_keywords": ""}'

    result = _run(CaptionGenerator(Ctx(), "p").caption_image("x.png"))
    assert result == {"caption": "好耶", "en_keywords": ""}


def test_caption_image_extracts_plain_from_result_chain():
    class Ctx:
        async def llm_generate(self, **kwargs):
            resp = type("R", (), {})()
            resp.result_chain = [
                {"type": "image", "url": "x"},
                {"type": "plain", "text": '{"caption": "哈哈", '},
                {"type": "plain", "text": '"en_keywords": "lol"}'},
            ]
            return resp

    result = _run(CaptionGenerator(Ctx(), "p").caption_image("x.png"))
    assert result == {"caption": "哈哈", "en_keywords": "lol"}


def test_caption_image_extracts_plain_component_objects():
    class Ctx:
        async def llm_generate(self, **kwargs):
            resp = type("R", (), {})()
            resp.chain = [Plain('{"caption": "溜了", "en_keywords": "run"}')]
            return resp

    result = _run(CaptionGenerator(Ctx(), "p").caption_image("x.png"))
    assert result == {"caption": "溜了", "en_keywords": "run"}


def test_caption_image_unwraps_code_fence():
    fence = chr(96) * 3

    class Ctx:
        async def llm_generate(self, **kwargs):
            body = '{"caption": "懂了", "en_keywords": "got it"}'
            return FakeResponse(fence + "json\n" + body + "\n" + fence)

    result = _run(CaptionGenerator(Ctx(), "p").caption_image("x.png"))
    assert result == {"caption": "懂了", "en_keywords": "got it"}


def test_caption_image_extracts_json_from_surrounding_prose():
    class Ctx:
        async def llm_generate(self, **kwargs):
            return FakeResponse('分析如下：{"caption": "笑死", "en_keywords": "lmao"} 以上。')

    result = _run(CaptionGenerator(Ctx(), "p").caption_image("x.png"))
    assert result == {"caption": "笑死", "en_keywords": "lmao"}


def test_caption_image_returns_none_on_bad_payloads():
    class MissingCaption:
        async def llm_generate(self, **kwargs):
            return FakeResponse('{"en_keywords": "x"}')

    class NotJson:
        async def llm_generate(self, **kwargs):
            return FakeResponse("完全不是 JSON")

    class Raising:
        async def llm_generate(self, **kwargs):
            raise RuntimeError("provider down")

    assert _run(CaptionGenerator(MissingCaption(), "p").caption_image("x.png")) is None
    assert _run(CaptionGenerator(NotJson(), "p").caption_image("x.png")) is None
    assert _run(CaptionGenerator(Raising(), "p").caption_image("x.png")) is None


def test_parse_json_rejects_non_object():
    gen = CaptionGenerator(None)
    assert gen._parse_json("[1, 2, 3]") is None
    assert gen._parse_json("") is None
    assert gen._parse_json('{"caption": "ok"}') == {"caption": "ok"}


# ---------- enrich_meme_captions ----------


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.limit = None
        self.vector_batches = []
        self.captions = []

    def memes_without_caption(self, limit=200):
        self.limit = limit
        return list(self.rows)

    def add_vectors(self, arr):
        arr = np.asarray(arr)
        start = sum(batch.shape[0] for batch in self.vector_batches)
        self.vector_batches.append(arr)
        return np.arange(start, start + arr.shape[0])

    def set_meme_caption(self, meme_id, text, vector_id):
        self.captions.append((meme_id, text, int(vector_id)))


class FakeEmbedder:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def embed_text(self, text):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("embed down")
        return np.ones(4, dtype="float32")


class FakeCaptioner:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def caption_image(self, path):
        self.calls.append(path)
        return self.mapping.get(path)


def _rows(*pairs):
    return [{"id": i, "file_path": p} for i, p in pairs]


def test_enrich_empty_returns_zero_without_side_effects():
    db = FakeDB([])
    result = _run(enrich_meme_captions(db, FakeEmbedder(), FakeCaptioner({})))
    assert result == {"total": 0, "ok": 0, "failed": 0}
    assert db.vector_batches == []


def test_enrich_reuses_external_captions_and_skips_vision_model():
    db = FakeDB(_rows((1, "/m/a.png"), (2, "/m/b.png")))
    embedder = FakeEmbedder()
    captioner = FakeCaptioner({"/m/b.png": {"caption": "来自模型", "en_keywords": "model"}})

    result = _run(
        enrich_meme_captions(
            db, embedder, captioner, external_captions={"/m/a.png": "已固化文案"}
        )
    )

    assert result == {"total": 2, "ok": 2, "failed": 0}
    assert captioner.calls == ["/m/b.png"]
    assert db.captions == [(1, "已固化文案", 0), (2, "来自模型 model", 1)]
    assert embedder.calls == ["已固化文案", "来自模型 model"]


def test_enrich_appends_keywords_only_when_present():
    db = FakeDB(_rows((1, "/m/a.png"), (2, "/m/b.png")))
    captioner = FakeCaptioner(
        {
            "/m/a.png": {"caption": "大笑", "en_keywords": "laugh, happy"},
            "/m/b.png": {"caption": "无语", "en_keywords": ""},
        }
    )
    _run(enrich_meme_captions(db, FakeEmbedder(), captioner))
    assert [c[1] for c in db.captions] == ["大笑 laugh, happy", "无语"]


def test_enrich_counts_failures_and_continues():
    db = FakeDB(_rows((1, "/m/a.png"), (2, "/m/b.png"), (3, "/m/c.png")))
    captioner = FakeCaptioner(
        {
            "/m/a.png": None,
            "/m/b.png": {"caption": "好", "en_keywords": ""},
            "/m/c.png": {"caption": "坏", "en_keywords": ""},
        }
    )
    embedder = FakeEmbedder(fail_on={"坏"})

    result = _run(enrich_meme_captions(db, embedder, captioner))

    assert result == {"total": 3, "ok": 1, "failed": 2}
    assert [c[0] for c in db.captions] == [2]


def test_enrich_flushes_in_batches_and_assigns_contiguous_ids():
    rows = _rows(*[(i, f"/m/{i}.png") for i in range(1, 6)])
    db = FakeDB(rows)
    captioner = FakeCaptioner({f"/m/{i}.png": {"caption": f"c{i}", "en_keywords": ""} for i in range(1, 6)})

    result = _run(enrich_meme_captions(db, FakeEmbedder(), captioner, batch_size=2))

    assert result == {"total": 5, "ok": 5, "failed": 0}
    assert [batch.shape[0] for batch in db.vector_batches] == [2, 2, 1]
    assert [c[2] for c in db.captions] == [0, 1, 2, 3, 4]


def test_enrich_clamps_limit_and_reports_progress():
    db = FakeDB(_rows((1, "/m/a.png"), (2, "/m/b.png")))
    progress = []
    captioner = FakeCaptioner({"/m/a.png": {"caption": "a"}, "/m/b.png": {"caption": "b"}})

    _run(
        enrich_meme_captions(
            db,
            FakeEmbedder(),
            captioner,
            limit=-5,
            progress_cb=lambda d, t: progress.append((d, t)),
        )
    )

    assert db.limit == 0
    assert progress == [(1, 2), (2, 2)]
