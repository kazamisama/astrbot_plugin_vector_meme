"""vector_meme 插件主入口。

提供：
- 索引管理命令
- 文本搜索命令
- LLM 回复后处理钩子（替换 &&tag&& 占位符）
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from pathlib import Path

from astrbot.api import logger
from astrbot.api.all import *  # noqa: F403
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image as ImgComp
from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register
from astrbot.core.message.message_event_result import MessageChain

from .core import BaseEmbedder, EmbedderFactory, MemeDatabase, MemeIndexer, MemeRetriever
from .core.indexer import IndexProgress

PLUGIN_NAME = "vector_meme"

# 默认占位符格式：%%<tag>%%
# 可选兼容旧格式：&&<tag>&& 或 :<tag>:（见 allow_legacy_markup 配置）
VM_TAG_PATTERN = re.compile(r"%%\s*([a-zA-Z0-9_\-一-鿿]+)\s*%%")
LEGACY_TAG_PATTERN = re.compile(r"&&\s*([a-zA-Z0-9_\-一-鿿]+)\s*&&|:\s*([a-zA-Z0-9_\-一-鿿]+)\s*:")

# 默认注入到全局人格的 prompt 模板
# 拼装规则：prompt_head + 标签列表 + prompt_tail_1 + max_n + prompt_tail_2
DEFAULT_PROMPT_HEAD = (
    "\n\n[vector_meme 表情系统]\n"
    "你可以通过在回复中插入 `%%tag%%` 标记来表达情绪/发送表情包。\n"
    "tag 必须使用下面列表中的有效标签，否则插件无法识别。\n\n"
    "[可用表情标签]\n"
)
DEFAULT_PROMPT_TAIL_1 = (
    "\n\n[使用规则]\n"
    "每次回复最多使用 "
)
DEFAULT_PROMPT_TAIL_2 = (
    " 个标签。\n"
    "- 仅在情绪/场景真正需要时插入，避免为用而用\n"
    "- 标签应自然融入文本上下文\n"
    "- 严禁使用列表外标签；如需表达列表外的情绪，选择最接近的"
)

@register(
    PLUGIN_NAME,
    "chiriu & 橘雪莉",
    "基于向量检索的智能表情包插件",
    "0.1.0",
)
class VectorMemePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

        # ---------- 解析配置 ----------
        data_dir = Path(self.config.get("data_dir") or "") if self.config.get("data_dir") else None
        if data_dir is None:
            # 默认放 plugin_data 下
            from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
            data_dir = Path(get_astrbot_plugin_data_path()) / "vector_meme"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.meme_dir_setting = self.config.get("meme_dir") or ""
        backend = self.config.get("embedder_backend", "open_clip")
        self._backend_name = backend
        self._embedder: BaseEmbedder | None = None

        # ---------- 延迟加载 embedder ----------
        self._embedder_lock = asyncio.Lock()
        self._db: MemeDatabase | None = None
        self._indexer: MemeIndexer | None = None
        self._retriever: MemeRetriever | None = None
        self._indexer_lock = asyncio.Lock()

        self._init_storage()

        # ---------- 备份 persona 用于 prompt 注入还原 ----------
        # 必须在第一次注入之前 deep copy，保留每个 persona 的原始 prompt。
        # 即使 enable_prompt_injection 关闭也要备份，方便后续开启。
        personas_now = self.context.provider_manager.personas or []
        self._persona_backup: list[dict] = copy.deepcopy(personas_now)
        self._sys_prompt_add: str = ""
        self._inject_into_personas()  # 首次尝试注入（DB 可能为空）

    # ---------- 初始化 ----------

    def _init_storage(self) -> None:
        """先以 dummy 维度建库，等 embedder 准备好再对齐。"""
        # 临时 dim，等真 embedder 加载后会重置
        temp_dim = int(self.config.get("_temp_dim", 512))
        self._db = MemeDatabase(
            db_path=self.data_dir / "memes.db",
            index_path=self.data_dir / "memes.faiss",
            dim=temp_dim,
        )

    async def _ensure_embedder(self) -> BaseEmbedder:
        """懒加载嵌入器（首次调用才下载/加载模型）。"""
        if self._embedder is not None:
            return self._embedder
        async with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder
            loop = asyncio.get_event_loop()

            def _create():
                return EmbedderFactory.create(
                    self._backend_name,
                    **self._embedder_kwargs(),
                )

            self._embedder = await loop.run_in_executor(None, _create)
            # 如果维度不匹配，需要重建 db
            if self._db is not None and self._db.dim != self._embedder.dim:
                logger.info(
                    "embedder 维度 %d 与现有库维度 %d 不一致，重建库",
                    self._embedder.dim, self._db.dim,
                )
                self._db = MemeDatabase(
                    db_path=self.data_dir / "memes.db",
                    index_path=self.data_dir / "memes.faiss",
                    dim=self._embedder.dim,
                )
            self._indexer = MemeIndexer(
                self._db, self._embedder,
                use_subdir_as_tag=bool(self.config.get("use_subdir_as_tag", True)),
                default_tag=self.config.get("default_tag", "misc"),
            )
            self._retriever = MemeRetriever(
                self._db,
                self._embedder,
                anti_repeat_window=int(self.config.get("anti_repeat_window", 20)),
                candidate_pool_size=int(self.config.get("selection_pool_size", 12)),
                random_jitter=float(self.config.get("selection_random_jitter", 0.015)),
            )
            return self._embedder

    def _embedder_kwargs(self) -> dict:
        if self._backend_name == "open_clip":
            hf_endpoint = (self.config.get("hf_endpoint") or "").strip()
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
            cache_dir = (self.config.get("open_clip_cache_dir") or "").strip()
            if not cache_dir:
                cache_dir = str(Path(__file__).resolve().parent / "models" / "cache")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
            os.environ.setdefault("TORCH_HOME", str(Path(cache_dir) / "torch"))
            local_weights = (self.config.get("open_clip_local_weights") or "").strip().strip('\"').strip("'")
            if local_weights:
                local_path = Path(local_weights).expanduser()
                if not local_path.exists():
                    raise FileNotFoundError(f"open_clip_local_weights 指向的文件不存在: {local_path}")
                pretrained = str(local_path)
                logger.info(f"[{PLUGIN_NAME}] 使用本地 OpenCLIP 权重: {pretrained}")
            else:
                pretrained = self.config.get("open_clip_pretrained", "openai")
                logger.info(f"[{PLUGIN_NAME}] 未设置本地权重，使用在线 pretrained tag: {pretrained}")
            return {
                "model_name": self.config.get("open_clip_model", "ViT-B-32"),
                "pretrained": pretrained,
                "device": self.config.get("device", "cpu"),
                "cache_dir": cache_dir,
            }
        return {}

    async def _ensure_ready(self) -> tuple[MemeRetriever, MemeIndexer, MemeDatabase] | None:
        await self._ensure_embedder()
        if self._retriever is None or self._indexer is None or self._db is None:
            return None
        return self._retriever, self._indexer, self._db

    # ---------- 路径与自动索引 ----------

    @property
    def meme_dir(self) -> Path:
        """meme_dir 配置可能为空，运行时根据插件自带目录解析。"""
        if self.meme_dir_setting:
            return Path(self.meme_dir_setting)
        # 留空：使用插件自带的 memes/ 目录
        plugin_root = Path(__file__).resolve().parent
        return plugin_root / "memes"

    async def _auto_index(self):
        """保留自动索引方法，但不挂启动钩子，避免 AstrBot 版本差异。"""
        ready = await self._ensure_ready()
        if not ready:
            return
        _, indexer, _ = ready
        async with self._indexer_lock:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: indexer.index_directory(self.meme_dir)
            )
        # 标签列表可能已变化，重新注入
        self._inject_into_personas()
        logger.info(f"[{PLUGIN_NAME}] 自动索引完成")

    # ---------- 命令 ----------

    async def _dispatch_command(self, event: AstrMessageEvent, args: list[str]):
        """统一命令分发。支持 @filter.command 和普通文本监听共用。"""
        if not args:
            handler = self._cmd_status
            rest = []
        else:
            sub = args[0]
            rest = list(args[1:])
            handler = {
                "状态": self._cmd_status,
                "预热": self._cmd_prewarm,
                "索引": self._cmd_index,
                "重建": self._cmd_rebuild,
                "列表": self._cmd_list,
                "标签": self._cmd_tags,
                "搜索": self._cmd_search,
                "解释": self._cmd_explain,
                "为什么": self._cmd_explain,
                "最近使用": self._cmd_recent,
                "最近": self._cmd_recent,
                "诊断": self._cmd_health,
                "健康检查": self._cmd_health,
                "健康": self._cmd_health,
                "修复": self._cmd_repair,
                "自动分类": self._cmd_auto_classify,
                "分类": self._cmd_auto_classify,
                "评测": self._cmd_eval,
                "标签规范": self._cmd_tag_schema,
                "tag_schema": self._cmd_tag_schema,
                "重标注": self._cmd_relabel,
                "删除": self._cmd_delete,
                "刷新提示": self._cmd_reload_prompt,
                "帮助": self._cmd_help,
            }.get(sub, self._cmd_help)
        try:
            async for result in handler(event, rest):
                yield result
            event.stop_event()
        except Exception as e:
            logger.exception("vector_meme 命令出错")
            yield event.plain_result(f"[{PLUGIN_NAME}] 出错: {e}")
            event.stop_event()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=999)
    async def vector_meme_text_command(self, event: AstrMessageEvent):
        """无前缀文本命令监听：表情向量 ... / vmem ...。"""
        text = (event.message_str or "").strip()
        if not text:
            return
        # 兼容用户不带 / 的情况，也兼容前面带一个 / 的情况
        if text.startswith("/"):
            text = text[1:].strip()
        if text == "表情向量":
            args = []
        elif text.startswith("表情向量 "):
            args = text.split()[1:]
        elif text == "vmem":
            args = []
        elif text.startswith("vmem "):
            args = text.split()[1:]
        else:
            return
        async for result in self._dispatch_command(event, args):
            yield result

    @filter.command("表情向量", alias={"vmem"})
    async def vm_command(self, event: AstrMessageEvent, *args: str):
        """vector_meme 统一入口。"""
        async for result in self._dispatch_command(event, list(args)):
            yield result

    async def _cmd_prewarm(self, event: AstrMessageEvent, rest: list[str]):
        """预热 embedder：只下载/加载模型，不进入索引流程。"""
        if self._embedder is not None:
            yield event.plain_result(
                f"embedder 已就绪（{self._backend_name}）"
            )
            return
        from astrbot.core.message.message_event_result import MessageEventResult

        async def _send(text):
            try:
                await event.send(MessageEventResult().message(text))
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] 后台发消息失败: {e}")

        done = asyncio.Event()

        async def _heartbeat():
            t = 0
            await _send(f"⏳ 预热 {self._backend_name} 启动...")
            while not done.is_set():
                await asyncio.sleep(5)
                t += 5
                if done.is_set():
                    break
                await _send(f"⏳ 预热 {self._backend_name} 仍进行中... ({t}s)")

        hb_task = asyncio.create_task(_heartbeat())
        try:
            await self._ensure_embedder()
        except Exception as e:
            logger.exception("embedder 预热失败")
            yield event.plain_result(f"❌ embedder 预热失败: {e}")
            return
        finally:
            done.set()
            try:
                await asyncio.wait_for(hb_task, timeout=3.0)
            except asyncio.TimeoutError:
                hb_task.cancel()
        yield event.plain_result(
            f"✅ embedder 就绪：{self._backend_name} (dim={self._embedder.dim})"
        )

    async def _cmd_status(self, event: AstrMessageEvent, rest: list[str] | None = None):
        # 状态命令不加载 embedder，避免第一次 /vm 状态 就下载 OpenCLIP 模型
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        s = db.stats()
        msg = (
            f"[{PLUGIN_NAME}] 索引状态\n"
            f"- 总数: {s['total']}（禁用: {s['disabled']}）\n"
            f"- 标签数: {s['tags']}\n"
            f"- FAISS 向量数: {s['index_size']}\n"
            f"- 向量维度: {s['dim']}\n"
            f"- 后端: {self._backend_name}\n"
            f"- embedder: {'已加载' if self._embedder is not None else '未加载'}\n"
            f"- meme_dir: {self.meme_dir}\n"
            f"- data_dir: {self.data_dir}\n"
        )
        yield event.plain_result(msg)

    async def _cmd_index(self, event: AstrMessageEvent, rest: list[str]):
        path = Path(rest[0]) if rest else self.meme_dir
        if not path or not path.exists():
            yield event.plain_result(f"目录不存在: {path}")
            return
        yield event.plain_result(
            f"开始索引 {path} ...\n后端: {self._backend_name}\n"
            f"如果是 open_clip，首次加载/下载模型可能需要较久。"
        )

        # 心跳：embedder 加载阶段（可能下载模型，可能拉很久）
        embedder_done = asyncio.Event()
        umo = event.unified_msg_origin
        bot = getattr(event, "bot", None)

        async def _send(text):
            if bot is not None and umo:
                from astrbot.core.message.message_event_result import MessageEventResult
                chain = MessageEventResult().message(text)
                try:
                    await bot.send(umo, chain)
                except Exception as e:
                    logger.warning(f"[{PLUGIN_NAME}] 后台发消息失败: {e}")
            else:
                # 退路：依旧 yield（虽然很可能到不了用户）
                try:
                    event.send(event.plain_result(text))
                except Exception:
                    pass

        async def _embedder_heartbeat():
            t = 0
            await _send("⏳ 正在加载 embedder...")
            while not embedder_done.is_set():
                await asyncio.sleep(5)
                t += 5
                if embedder_done.is_set():
                    break
                await _send(f"⏳ 仍在加载 embedder... ({t}s)")
        heartbeat_task = asyncio.create_task(_embedder_heartbeat())
        try:
            ready = await self._ensure_ready()
        finally:
            embedder_done.set()
            try:
                await asyncio.wait_for(heartbeat_task, timeout=3.0)
            except asyncio.TimeoutError:
                heartbeat_task.cancel()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        _, indexer, _ = ready

        progress_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _on_progress(done, total, fp):
            logger.info(f"[index] {done}/{total} {fp}")
            try:
                progress_q.put_nowait((done, total, str(fp)))
            except asyncio.QueueFull:
                pass

        from astrbot.core.message.message_event_result import MessageEventResult

        async def _send_progress(text):
            try:
                await event.send(MessageEventResult().message(text))
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] progress send failed: {e}")

        async def _progress_reporter():
            last_done = -10
            while True:
                try:
                    done, total, fp = await asyncio.wait_for(progress_q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if reporter_done.is_set():
                        break
                    continue
                if done - last_done < 10 and done != total:
                    continue
                last_done = done
                await _send_progress(
                    f"⏳ 索引进度: {done}/{total}\n最近: {Path(fp).name}"
                )
                if done >= total:
                    break

        reporter_done = asyncio.Event()
        reporter_task = asyncio.create_task(_progress_reporter())
        try:
            async with self._indexer_lock:
                progress = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: indexer.index_directory(path, progress=IndexProgress(_on_progress)),
                )
        finally:
            reporter_done.set()
            try:
                await asyncio.wait_for(reporter_task, timeout=3.0)
            except asyncio.TimeoutError:
                reporter_task.cancel()
        # 索引完成 → 标签列表可能变化 → 重新注入 prompt
        self._inject_into_personas()
        yield event.plain_result(
            f"索引完成\n"
            f"- 新增: {progress.added}\n"
            f"- 更新: {progress.updated}\n"
            f"- 跳过: {progress.skipped}\n"
            f"- 失败: {progress.failed}\n"
            f"- 总计: {progress.total}"
        )

    async def _cmd_rebuild(self, event: AstrMessageEvent, rest: list[str]):
        path = self.meme_dir
        if not path or not path.exists():
            yield event.plain_result(f"meme_dir 未配置或不存在: {path}")
            return
        yield event.plain_result(
            f"开始重建索引 {path} ...\n后端: {self._backend_name}\n"
            f"如果是 open_clip，首次加载/下载模型可能需要较久。"
        )

        embedder_done = asyncio.Event()
        umo = event.unified_msg_origin
        bot = getattr(event, "bot", None)

        async def _send(text):
            if bot is not None and umo:
                from astrbot.core.message.message_event_result import MessageEventResult
                chain = MessageEventResult().message(text)
                try:
                    await bot.send(umo, chain)
                except Exception as e:
                    logger.warning(f"[{PLUGIN_NAME}] 后台发消息失败: {e}")
            else:
                # 退路：依旧 yield（虽然很可能到不了用户）
                try:
                    event.send(event.plain_result(text))
                except Exception:
                    pass

        async def _embedder_heartbeat():
            t = 0
            await _send("⏳ 正在加载 embedder...")
            while not embedder_done.is_set():
                await asyncio.sleep(5)
                t += 5
                if embedder_done.is_set():
                    break
                await _send(f"⏳ 仍在加载 embedder... ({t}s)")
        heartbeat_task = asyncio.create_task(_embedder_heartbeat())
        try:
            ready = await self._ensure_ready()
        finally:
            embedder_done.set()
            try:
                await asyncio.wait_for(heartbeat_task, timeout=3.0)
            except asyncio.TimeoutError:
                heartbeat_task.cancel()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        _, indexer, _ = ready

        progress_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _on_progress(done, total, fp):
            logger.info(f"[rebuild] {done}/{total} {fp}")
            try:
                progress_q.put_nowait((done, total, str(fp)))
            except asyncio.QueueFull:
                pass

        from astrbot.core.message.message_event_result import MessageEventResult

        async def _send_progress(text):
            try:
                await event.send(MessageEventResult().message(text))
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] progress send failed: {e}")

        async def _progress_reporter():
            last_done = -10
            while True:
                try:
                    done, total, fp = await asyncio.wait_for(progress_q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if reporter_done.is_set():
                        break
                    continue
                if done - last_done < 10 and done != total:
                    continue
                last_done = done
                await _send_progress(
                    f"⏳ 重建进度: {done}/{total}\n最近: {Path(fp).name}"
                )
                if done >= total:
                    break

        reporter_done = asyncio.Event()
        reporter_task = asyncio.create_task(_progress_reporter())
        try:
            async with self._indexer_lock:
                with indexer.db._conn() as c:  # noqa: SLF001
                    c.execute("DELETE FROM memes")
                if indexer.db.index_path.exists():
                    indexer.db.index_path.unlink()
                indexer.db._index = type(indexer.db._index)(indexer.db.dim)  # noqa: SLF001
                progress = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: indexer.index_directory(path, progress=IndexProgress(_on_progress))
                )
        finally:
            reporter_done.set()
            try:
                await asyncio.wait_for(reporter_task, timeout=3.0)
            except asyncio.TimeoutError:
                reporter_task.cancel()
        # 重建完成 → 标签列表已变 → 重新注入
        self._inject_into_personas()
        yield event.plain_result(
            f"重建完成，共处理 {progress.total} 张（新增 {progress.added}）"
        )

    async def _cmd_list(self, event: AstrMessageEvent, rest: list[str]):
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        if rest:
            tag = rest[0]
            rows = db.list_memes(tag=tag, limit=50)
            lines = [f"[{tag}] 共 {len(rows)} 条（最多显示 50）"]
            for r in rows:
                lines.append(f"  #{r['id']} {r['file_name']} (使用 {r['usage_count']} 次)")
            yield event.plain_result("\n".join(lines))
        else:
            counts = db.count_by_tag()
            if not counts:
                yield event.plain_result("索引为空")
                return
            lines = ["标签分布："]
            for tag, n in counts.items():
                lines.append(f"  {tag}: {n}")
            yield event.plain_result("\n".join(lines))

    async def _cmd_tags(self, event: AstrMessageEvent, rest: list[str]):
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        tags = db.list_tags()
        if not tags:
            yield event.plain_result("尚未注册标签（标签由索引时子目录自动生成）")
            return
        lines = [f"{t['name']} ({t.get('category') or '未分类'}) - {t.get('description') or ''}" for t in tags]
        yield event.plain_result("\n".join(lines))

    async def _cmd_search(self, event: AstrMessageEvent, rest: list[str]):
        if not rest:
            yield event.plain_result("用法: /vm 搜索 <文本> [标签]")
            return
        text = " ".join(rest)
        tag = None
        # 简易解析：最后一个参数如果是已知 tag 就当 tag
        if rest[-1] in {t for t, _ in (self._db.count_by_tag() if self._db else {}).items()}:
            tag = rest[-1]
            text = " ".join(rest[:-1])
        ready = await self._ensure_ready()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        retriever, _, _ = ready
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: retriever.retrieve(text=text, tag=tag, topk=5)
        )
        if not result:
            yield event.plain_result("未找到匹配")
            return
        chain = [Plain(f"搜索: {text}\n标签: {tag or '全库'}\n匹配:\n")]
        for h in result.hits:
            raw = h.raw_similarity if h.raw_similarity is not None else h.similarity
            chain.append(Plain(f"  #{h.meme_id} final={h.similarity:.3f} raw={raw:.3f} {h.name}\n"))
        yield event.chain_result(chain)
        for h in result.hits[:3]:
            yield event.chain_result([
                ImgComp.fromFileSystem(h.file_path),
            ])

    async def _cmd_explain(self, event: AstrMessageEvent, rest: list[str]):
        """解释一次检索为什么会选中某张图。"""
        if not rest:
            yield event.plain_result("用法: 表情向量 解释 <文本> [标签]")
            return
        text = " ".join(rest)
        tag = None
        counts = self._db.count_by_tag() if self._db else {}
        if rest[-1] in counts:
            tag = rest[-1]
            text = " ".join(rest[:-1]) or tag
        ready = await self._ensure_ready()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        retriever, _, _ = ready
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: retriever.retrieve(
                text=text,
                tag=tag,
                topk=int(self.config.get("selection_pool_size", 12)),
                anti_repeat=True,
            ),
        )
        if not result:
            yield event.plain_result("没有可解释的候选结果")
            return
        lines = [
            "选择解释：",
            f"- query: {text}",
            f"- tag_filter: {tag or '全库'}",
            f"- fallback: {'yes' if result.used_fallback else 'no'}",
            "- top candidates:",
        ]
        for i, h in enumerate(result.hits[:5], start=1):
            lines.append(f"  {i}. #{h.meme_id} {h.name} [{h.tag}] final={h.similarity:.3f} raw={(h.raw_similarity if h.raw_similarity is not None else h.similarity):.3f}")
            detail = []
            if h.tag_bonus:
                detail.append(f"tag +{h.tag_bonus:.3f}")
            if h.repeat_penalty:
                detail.append(f"repeat -{h.repeat_penalty:.3f}")
            if h.usage_penalty:
                detail.append(f"usage -{h.usage_penalty:.3f}")
            if h.random_jitter:
                detail.append(f"jitter {h.random_jitter:+.3f}")
            if detail:
                lines.append("     " + ", ".join(detail))
        yield event.plain_result("\n".join(lines))
        top = result.top()
        if top:
            yield event.chain_result([ImgComp.fromFileSystem(top.file_path)])

    async def _cmd_recent(self, event: AstrMessageEvent, rest: list[str]):
        """展示最近使用的表情。"""
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        limit = 10
        if rest and rest[0].isdigit():
            limit = min(max(int(rest[0]), 1), 30)
        rows = db.list_recently_used(limit=limit)
        if not rows:
            yield event.plain_result("还没有最近使用记录")
            return
        import time as _time
        now = _time.time()
        lines = [f"最近使用（{len(rows)} 条）："]
        for r in rows:
            elapsed = now - float(r.get("last_used_at") or now)
            lines.append(f"  #{r['id']} [{r['tag']}] {r['file_name']} - {elapsed:.0f}s 前，使用 {r['usage_count']} 次")
        yield event.plain_result("\n".join(lines))

    async def _cmd_health(self, event: AstrMessageEvent, rest: list[str]):
        """健康检查 / 诊断。"""
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        h = db.health_check(root=self.meme_dir)
        lines = [
            f"[{PLUGIN_NAME}] 健康检查: {'OK' if h['ok'] else 'WARN'}",
            f"- DB有效表情: {h['total']}（禁用 {h['disabled']}）",
            f"- FAISS向量数: {h['index_size']} / dim={h['dim']}",
            f"- 标签: 使用中 {h['tag_count']} / 注册 {h['registered_tag_count']}",
            f"- 索引数量不足: {'yes' if h['index_mismatch'] else 'no'}",
            f"- 文件缺失: {h['missing_files_count']}",
            f"- vector_id异常: {h['bad_vector_ids_count']}",
            f"- 重复hash组: {h['duplicate_hash_count']}",
            f"- root外孤儿记录: {h['orphan_outside_root_count']}",
        ]
        if h["unregistered_tags"]:
            lines.append("- 未注册tag: " + ", ".join(h["unregistered_tags"][:20]))
        for key, title in (("missing_files", "缺失文件"), ("bad_vector_ids", "异常vector"), ("orphan_outside_root", "root外记录")):
            items = h[key][:5]
            if items:
                lines.append(f"- {title}示例:")
                for item in items:
                    lines.append(f"  {item}")
        yield event.plain_result("\n".join(lines))

    async def _cmd_repair(self, event: AstrMessageEvent, rest: list[str]):
        """轻量修复：清理缺失文件记录、注册缺失 tag、刷新 prompt。"""
        ready = await self._ensure_ready()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        _, indexer, db = ready
        removed = await asyncio.get_event_loop().run_in_executor(
            None, lambda: indexer.remove_missing(self.meme_dir)
        )
        counts = db.count_by_tag()
        for tag in counts:
            db.upsert_tag(tag)
        self._inject_into_personas()
        yield event.plain_result(
            f"修复完成\n- 清理缺失文件记录: {removed}\n- 注册/确认 tag: {len(counts)}\n- 已刷新 prompt"
        )

    async def _cmd_auto_classify(self, event: AstrMessageEvent, rest: list[str]):
        """prototype + KNN 自动分类。默认 dry-run；加 apply 才改数据库 tag。"""
        if not rest or not rest[0].isdigit():
            yield event.plain_result("用法: 表情向量 自动分类 <id> [apply]\n说明: 默认只预览，不移动文件；apply 只修改数据库 tag。")
            return
        meme_id = int(rest[0])
        apply = any(x.lower() in {"apply", "确认", "执行"} for x in rest[1:])
        threshold = float(self.config.get("auto_classify_threshold", 0.18))
        margin = float(self.config.get("auto_classify_margin", 0.03))
        ready = await self._ensure_ready()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        retriever, _, db = ready
        row = db.get_meme(meme_id)
        if not row:
            yield event.plain_result(f"id 不存在: {meme_id}")
            return
        loop = asyncio.get_event_loop()
        preds = await loop.run_in_executor(
            None,
            lambda: retriever.classify_meme(
                meme_id,
                topk=5,
                knn_k=int(self.config.get("auto_classify_knn_k", 12)),
                prototype_min_samples=int(self.config.get("prototype_min_samples", 2)),
            ),
        )
        if not preds:
            yield event.plain_result("没有足够样本进行 prototype/KNN 分类")
            return
        best = preds[0]
        second_score = preds[1]["score"] if len(preds) > 1 else -999.0
        confident = best["score"] >= threshold and (best["score"] - second_score) >= margin
        lines = [
            f"自动分类 #{meme_id} {row['file_name']}",
            f"当前tag: {row['tag']}",
            f"建议tag: {best['tag']} score={best['score']:.3f} confident={'yes' if confident else 'no'}",
            "候选：",
        ]
        for p in preds:
            proto = "-" if p["prototype_similarity"] is None else f"{p['prototype_similarity']:.3f}"
            knn = "-" if p["knn_score"] is None else f"{p['knn_score']:.3f}"
            lines.append(
                f"  {p['tag']}: score={p['score']:.3f}, proto={proto}, knn={knn}, neighbors={p['neighbor_count']}, samples={p['sample_count']}"
            )
        if apply:
            if not confident:
                lines.append("未执行：置信度或 margin 不足。若要强制，请用 重标注 <id> <tag>。")
            elif best["tag"] == row["tag"]:
                lines.append("无需修改：建议 tag 与当前 tag 相同。")
            else:
                ok, old, new = db.relabel_meme(meme_id, best["tag"], keep_old_as_subtag=True)
                lines.append(f"已修改数据库 tag: {old} -> {new}" if ok else "修改失败")
                self._inject_into_personas()
        else:
            lines.append("dry-run：加 apply 才会修改数据库 tag，不会移动文件。")
        yield event.plain_result("\n".join(lines))
        yield event.chain_result([ImgComp.fromFileSystem(row["file_path"])])

    async def _cmd_tag_schema(self, event: AstrMessageEvent, rest: list[str]):
        """展示 tag schema 治理文件路径和简要内容。"""
        schema_path = Path(__file__).resolve().parent / "tag_schema.json"
        if not schema_path.exists():
            yield event.plain_result(f"tag schema 不存在: {schema_path}")
            return
        try:
            data = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as e:
            yield event.plain_result(f"tag schema 读取失败: {e}")
            return
        tags = data.get("tags", data)
        lines = [f"tag schema: {schema_path}", f"共 {len(tags)} 个规范 tag"]
        for name, item in list(tags.items())[:20]:
            if isinstance(item, dict):
                lines.append(f"  {name}: {item.get('meaning') or item.get('description') or ''}")
            else:
                lines.append(f"  {name}: {item}")
        yield event.plain_result("\n".join(lines))

    async def _cmd_eval(self, event: AstrMessageEvent, rest: list[str]):
        """跑一个小型分类评测集，输出 top1/top3 准确率。"""
        if not rest:
            yield event.plain_result("用法: 表情向量 评测 <eval.json>\n格式: [{\"id\": 1, \"tag\": \"happy\"}] 或 {\"items\": [...]}")
            return
        path = Path(" ".join(rest)).expanduser()
        if not path.exists():
            yield event.plain_result(f"评测集不存在: {path}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            yield event.plain_result(f"评测集读取失败: {e}")
            return
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            yield event.plain_result("评测集格式错误：需要 list 或 {items: list}")
            return
        ready = await self._ensure_ready()
        if not ready:
            yield event.plain_result("embedder 未就绪")
            return
        retriever, _, _ = ready

        def _run_eval():
            total = 0
            top1 = 0
            top3 = 0
            confused: dict[tuple[str, str], int] = {}
            failed = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                meme_id = item.get("id") or item.get("meme_id")
                expected = item.get("tag") or item.get("expected")
                if meme_id is None or not expected:
                    continue
                try:
                    preds = retriever.classify_meme(int(meme_id), topk=5)
                except Exception:
                    failed += 1
                    continue
                if not preds:
                    failed += 1
                    continue
                total += 1
                pred_tags = [p["tag"] for p in preds]
                if pred_tags[0] == expected:
                    top1 += 1
                else:
                    confused[(str(expected), str(pred_tags[0]))] = confused.get((str(expected), str(pred_tags[0])), 0) + 1
                if expected in pred_tags[:3]:
                    top3 += 1
            return total, top1, top3, failed, confused

        total, top1, top3, failed, confused = await asyncio.get_event_loop().run_in_executor(None, _run_eval)
        if total <= 0:
            yield event.plain_result(f"没有可评测样本（失败 {failed}）")
            return
        lines = [
            "分类评测结果：",
            f"- 样本数: {total}（失败 {failed}）",
            f"- top1: {top1}/{total} = {top1 / total:.1%}",
            f"- top3: {top3}/{total} = {top3 / total:.1%}",
        ]
        if confused:
            lines.append("- 主要混淆:")
            for (expected, predicted), n in sorted(confused.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  {expected} -> {predicted}: {n}")
        yield event.plain_result("\n".join(lines))

    async def _cmd_relabel(self, event: AstrMessageEvent, rest: list[str]):
        """手动重标注主 tag，只改数据库，不移动文件。"""
        if len(rest) < 2 or not rest[0].isdigit():
            yield event.plain_result("用法: 表情向量 重标注 <id> <新tag>")
            return
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        meme_id = int(rest[0])
        new_tag = rest[1].strip()
        ok, old, new = db.relabel_meme(meme_id, new_tag, keep_old_as_subtag=True)
        if not ok:
            yield event.plain_result(f"重标注失败: id={meme_id}")
            return
        self._inject_into_personas()
        yield event.plain_result(f"已重标注 #{meme_id}: {old} -> {new}（只修改数据库，不移动文件）")

    async def _cmd_delete(self, event: AstrMessageEvent, rest: list[str]):
        if not rest or not rest[0].isdigit():
            yield event.plain_result("用法: /vm 删除 <id>")
            return
        db = self._db
        if db is None:
            yield event.plain_result("数据库未初始化")
            return
        ok = db.remove_meme(int(rest[0]))
        yield event.plain_result("已删除" if ok else "id 不存在")

    async def _cmd_reload_prompt(self, event: AstrMessageEvent, rest: list[str]):
        """手动重新注入表情标签 prompt 到全局人格。"""
        try:
            self._inject_into_personas()
            tag_count = self._db.list_tags().__len__() if self._db else 0
            if self.config.get("enable_prompt_injection", True):
                yield event.plain_result(
                    f"[{PLUGIN_NAME}] 已重新注入 prompt（{tag_count} 个标签）"
                )
            else:
                yield event.plain_result(
                    f"[{PLUGIN_NAME}] enable_prompt_injection=False，仅清空已注入的 prompt"
                )
        except Exception as e:
            logger.exception(f"[{PLUGIN_NAME}] 重载 prompt 失败")
            yield event.plain_result(f"[{PLUGIN_NAME}] 重载失败: {e}")

    async def _cmd_help(self, event: AstrMessageEvent, rest: list[str] | None = None):
        yield event.plain_result(
            f"[{PLUGIN_NAME}] 用法：\n"
            "  表情向量 状态\n"
            "  表情向量 预热\n"
            "  表情向量 索引 [目录]\n"
            "  表情向量 重建\n"
            "  表情向量 列表 [标签]\n"
            "  表情向量 标签\n"
            "  表情向量 搜索 <文本> [标签]\n"
            "  表情向量 解释 <文本> [标签]\n"
            "  表情向量 最近使用 [数量]\n"
            "  表情向量 诊断 / 健康检查\n"
            "  表情向量 修复\n"
            "  表情向量 自动分类 <id> [apply]\n"
            "  表情向量 评测 <eval.json>\n"
            "  表情向量 标签规范\n"
            "  表情向量 重标注 <id> <新tag>\n"
            "  表情向量 删除 <id>\n"
            "  表情向量 刷新提示\n"
            "  表情向量 帮助"
        )

    # ---------- LLM 回复钩子 / 发送装饰 ----------

    @filter.on_llm_response(priority=99998)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """只负责提取 %%tag%%，清理文本，并把待发送 tag 存进 event.extra。

        不在这里直接 append 图片。真正选图/塞图放到 on_decorating_result，
        发送后补图放到 after_message_sent，和 meme_manager 的生命周期保持一致。
        """
        try:
            if not response:
                return

            text = getattr(response, "completion_text", None) or ""

            # completion_text 有时为空，兜底从 result_chain/chain 拼 Plain 文本
            if not text:
                comp = getattr(response, "result_chain", None) or getattr(response, "chain", None)
                chain = getattr(comp, "chain", comp) if comp is not None else None
                if chain:
                    parts = []
                    for c in chain:
                        if isinstance(c, Plain):
                            parts.append(c.text)
                    text = "".join(parts)

            if not text:
                return

            tags = VM_TAG_PATTERN.findall(text)
            if self.config.get("allow_legacy_markup", False):
                legacy = LEGACY_TAG_PATTERN.findall(text)
                tags.extend([a or b for a, b in legacy])

            # 去空、去重、限量；过滤掉数字引用/纯标点等非法 tag
            max_n = int(self.config.get("max_per_message", 2))
            seen = set()
            filtered_tags = []
            for t in tags:
                t = (t or "").strip()
                if not t or t in seen:
                    continue
                if not self._is_valid_emotion_tag(t):
                    logger.debug(f"[{PLUGIN_NAME}] 过滤非法 tag: {t!r}")
                    continue
                seen.add(t)
                filtered_tags.append(t)
                if len(filtered_tags) >= max_n:
                    break

            # 无 tag 也要清理残留非法 %%...%%，但不触发表情
            clean_text = VM_TAG_PATTERN.sub("", text)
            if self.config.get("allow_legacy_markup", False):
                clean_text = LEGACY_TAG_PATTERN.sub("", clean_text)
            clean_text = clean_text.strip()

            if hasattr(response, "completion_text"):
                response.completion_text = clean_text

            # 同步清理 result_chain/chain 里的 Plain，防止最终消息残留标记
            comp = getattr(response, "result_chain", None) or getattr(response, "chain", None)
            chain = getattr(comp, "chain", comp) if comp is not None else None
            if chain:
                for c in chain:
                    if isinstance(c, Plain):
                        c.text = VM_TAG_PATTERN.sub("", c.text)
                        if self.config.get("allow_legacy_markup", False):
                            c.text = LEGACY_TAG_PATTERN.sub("", c.text)

            if not filtered_tags:
                return

            event.set_extra("vector_meme_pending_tags", filtered_tags)
            event.set_extra("vector_meme_query_text", clean_text or text)
            logger.info(f"[{PLUGIN_NAME}] 捕获待发送表情标签: {filtered_tags}")
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] llm hook 出错")

    @filter.on_decorating_result(priority=99998)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """发送前根据 pending tags 做向量检索，把图片合入 result.chain 或挂到 pending。"""
        result = event.get_result()
        if not result:
            return

        tags = event.get_extra("vector_meme_pending_tags") or []
        query_text = event.get_extra("vector_meme_query_text") or ""
        # 截断 query_text：CLIP 对短句更敏感，长回复会稀释语义
        qmax = int(self.config.get("query_text_max_length", 80))
        if qmax > 0 and len(query_text) > qmax:
            query_text = query_text[:qmax]

        try:
            # 先清理最终消息链里可能残留的占位符
            original_chain = result.chain
            cleaned_components = []

            def _clean_text(s: str) -> str:
                s = VM_TAG_PATTERN.sub("", s)
                if self.config.get("allow_legacy_markup", False):
                    s = LEGACY_TAG_PATTERN.sub("", s)
                return s

            if original_chain:
                if isinstance(original_chain, str):
                    cleaned = _clean_text(original_chain)
                    if cleaned.strip():
                        cleaned_components.append(Plain(cleaned.strip()))
                elif isinstance(original_chain, MessageChain):
                    iterable = original_chain.chain
                    for component in iterable:
                        if isinstance(component, Plain):
                            cleaned = _clean_text(component.text)
                            if cleaned.strip():
                                cleaned_components.append(Plain(cleaned.strip()))
                        else:
                            cleaned_components.append(component)
                elif isinstance(original_chain, list):
                    for component in original_chain:
                        if isinstance(component, Plain):
                            cleaned = _clean_text(component.text)
                            if cleaned.strip():
                                cleaned_components.append(Plain(cleaned.strip()))
                        else:
                            cleaned_components.append(component)

            if not tags:
                if cleaned_components:
                    result.chain = cleaned_components
                return

            # 概率判定放在 decorating 阶段
            import random
            prob = int(self.config.get("trigger_probability", 80))
            if random.randint(1, 100) > prob:
                event.set_extra("vector_meme_pending_tags", None)
                if cleaned_components:
                    result.chain = cleaned_components
                return

            ready = await self._ensure_ready()
            if not ready:
                logger.warning(f"[{PLUGIN_NAME}] retriever 未就绪，跳过自动表情")
                if cleaned_components:
                    result.chain = cleaned_components
                return
            retriever, _, _ = ready

            loop = asyncio.get_event_loop()
            images = []
            for tag in tags:
                hit = await loop.run_in_executor(
                    None,
                    lambda t=tag: retriever.pick(
                        text=query_text or t,
                        tag=t,
                        anti_repeat=True,
                        fallback_to_all_tags=True,
                        selection_pool_size=int(self.config.get("selection_pool_size", 12)),
                        stochastic=bool(self.config.get("enable_stochastic_selection", True)),
                    ),
                )
                if hit:
                    try:
                        images.append(ImgComp.fromFileSystem(hit.file_path))
                        logger.info(f"[{PLUGIN_NAME}] 选中表情: tag={tag}, file={hit.file_path}, sim={hit.similarity:.3f}")
                    except Exception as e:
                        logger.warning(f"[{PLUGIN_NAME}] 构造图片组件失败 {hit.file_path}: {e}")

            event.set_extra("vector_meme_pending_tags", None)

            if not images:
                if cleaned_components:
                    result.chain = cleaned_components
                return

            # 默认混合图文；如关闭则 after_message_sent 补发
            enable_mixed = bool(self.config.get("enable_mixed_message", True))
            if enable_mixed:
                # 简单策略：图片追加到文本后。后续可做更复杂插入策略。
                result.chain = (cleaned_components if cleaned_components else []) + images
            else:
                result.chain = cleaned_components
                event.set_extra("vector_meme_pending_images", images)
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] decorating 出错")

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """发送后补发非混合模式下的图片。"""
        pending_images = event.get_extra("vector_meme_pending_images")
        try:
            if pending_images:
                for image in pending_images:
                    if event.get_platform_name() == "gewechat":
                        await event.send(MessageChain([image]))
                    else:
                        await self.context.send_message(
                            event.unified_msg_origin,
                            MessageChain([image]),
                        )
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] after_message_sent 补发表情失败")
        finally:
            event.set_extra("vector_meme_pending_images", None)

    def _is_valid_emotion_tag(self, tag: str) -> bool:
        """判断提取出的 tag 是不是一个合法表情标签。

        防御性过滤 LLM 输出里的数字引用、纯标点、句子片段等噪声，
        避免检索阶段把无意义字符串当作目标 tag 走 fallback 路径。
        """
        if not tag:
            return False
        t = tag.strip()
        if not t:
            return False
        # 纯数字引用，如 1、2024
        if re.fullmatch(r"\d+", t):
            return False
        # 纯标点/符号（含 _ 下划线单独也算）
        if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
            return False
        # 太长（>30）通常是句子片段
        if len(t) > 30:
            return False
        # 单字符且不是字母/中文（排除 -、.、: 这种）
        if len(t) == 1 and not re.match(r"[a-zA-Z\u4e00-\u9fff0-9]", t):
            return False
        return True

    # ---------- persona 注入 ----------

    def _inject_into_personas(self) -> None:
        """根据 DB 中现有 tag 列表，把 prompt 注入到每个 persona。

        幂等：每次调用先还原 persona，再重新注入。
        enable_prompt_injection=False 时只还原不注入。
        """
        personas = self.context.provider_manager.personas
        if not personas or not self._persona_backup:
            return

        # 1) 先把 persona 还原成原始 prompt（保证幂等 + 实现"关开关时清掉注入"）
        for persona, persona_backup in zip(personas, self._persona_backup):
            persona["prompt"] = persona_backup["prompt"]

        if not self.config.get("enable_prompt_injection", True):
            return

        # 2) 拼 prompt
        head = self.config.get("prompt_head") or DEFAULT_PROMPT_HEAD
        tail_1 = self.config.get("prompt_tail_1") or DEFAULT_PROMPT_TAIL_1
        tail_2 = self.config.get("prompt_tail_2") or DEFAULT_PROMPT_TAIL_2
        max_n = int(self.config.get("max_per_message", 2))

        tag_block = self._build_tag_list_string()
        self._sys_prompt_add = head + tag_block + tail_1 + str(max_n) + tail_2

        # 3) 重新注入
        for persona in personas:
            persona["prompt"] = persona["prompt"] + self._sys_prompt_add

        logger.debug(
            f"[{PLUGIN_NAME}] 已注入 prompt 到 {len(personas)} 个 persona，"
            f"tag 数 {tag_block.count(chr(10)) + 1 if tag_block else 0}"
        )

    def _restore_personas(self) -> None:
        """把 persona 还原成原始 prompt（不重新注入）。terminate 时清理用。"""
        personas = self.context.provider_manager.personas
        if not personas or not self._persona_backup:
            return
        for persona, persona_backup in zip(personas, self._persona_backup):
            persona["prompt"] = persona_backup["prompt"]
        logger.debug(f"[{PLUGIN_NAME}] 已还原 {len(personas)} 个 persona")

    def _build_tag_list_string(self) -> str:
        """从 DB 读 tag 列表，拼成 "- name: description" 形式的纯文本块。"""
        if self._db is None:
            return ""
        try:
            tags = self._db.list_tags()
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 读 tag 列表失败，跳过标签注入: {e}")
            return ""
        if not tags:
            return ""
        lines: list[str] = []
        for t in tags:
            name = (t.get("name") or "").strip()
            if not name:
                continue
            desc = (t.get("description") or "").strip()
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)

    def _strip_placeholders(self, comp: list, plain_indexes: list[int]) -> None:
        """兼容旧方法：从 Plain 组件里清掉占位符。"""
        for idx in plain_indexes:
            c = comp[idx]
            if isinstance(c, Plain):
                c.text = VM_TAG_PATTERN.sub("", c.text)
                if self.config.get("allow_legacy_markup", False):
                    c.text = LEGACY_TAG_PATTERN.sub("", c.text)

    async def terminate(self):
        try:
            self._restore_personas()
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] 还原 persona 失败")
        try:
            if self._embedder is not None:
                self._embedder.close()
            if self._db is not None:
                self._db.save_index()
        except Exception:
            pass
