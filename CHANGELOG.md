# 更新日志

## [0.6.8] - 2026-08-02

### 修复
- 修复 faiss 维度不匹配时启动加载成空索引、导致 api 后端重启后检索全空的问题：`_load_or_create_index` 改为采用文件实际维度
- 修复 `<sticker>%%tag%%</sticker>` 被 `on_llm_response` 提前剥离占位符、导致 XML 插件解析到空 `<sticker></sticker>` 静默丢弃的问题：vector_meme 不再提取/清理 `<sticker>` 块内的 `%%tag%%`
- `search_sticker_for_external` 兼容 `<sticker>%%tag%%</sticker>` 写法（`%%` 归一化），并补充命中/未命中日志

## [0.6.7] - 2026-08-02

### 修复
- `search_sticker_for_external`（XML 插件转发路径）：冷 embedder 改为触发一次懒加载，不再直接返回 None
- 外部检索超时 2s → 30s（对齐 api 后端单次 embedding 约 20s 的实际耗时），修复 `<sticker>` 标签被直发原始 XML 的问题
- 外部检索开启 `fallback_to_all_tags` 兜底，caption 检索 miss 时不再空手而归

## [0.6.6] - 2026-08-02

### 修复
- `/vm 重建` 不再被「embedder 维度与旧库不一致」保护拦截：重建先清空旧库再加载 embedder，修复「重建命令本身跑不了」的逻辑矛盾

## [0.6.5] - 2026-08-02

### 修复
- api 后端 `/vm 重建` 后补跑 caption 向量建立（复用 memes/captions.json），修复重建后 FAISS 向量数仍为 0、检索无结果的问题

## [0.6.4] - 2026-08-02

### 修复
- api 后端索引：图片不可编码时不再跳过，meme 记录照常入库（vector_id=-1），caption 向量由 enrich 阶段经 captions.json 建立
- 修复 api 后端 `/vm 重建` 后总数/标签/向量全为 0、检索无结果的问题

## [0.6.3] - 2026-08-02

### 修复
- metadata.yaml 补充 repo 字段，修复 AstrBot 面板无法更新插件（does not specify a repository URL）

## [0.6.2] - 2026-08-02

### 修复
- /vm 命令通道适配 AstrBot v4.25.5：handler 改用 GreedyStr 接收全部剩余参数，修复 `vm_command() got an unexpected keyword argument 'args'`
- 文本监听补充 `vm` 前缀（无参数 /vm 也可用）

## [0.6.1] - 2026-08-02

### 改进
- vision_provider_id 改为下拉选择（_special=select_provider），直接选已配置的对话模型提供商，不再手填 ID
- description 精简为「生成 caption 的视觉模型」

## [0.6.0] - 2026-08-02

### 新增
- caption 固化分发：新增 `/vm caption 导出`，把 DB 中已有 caption 导出为 `memes/captions.json`（key=相对路径，随表情库进 git）
- 索引/重建/修复时优先复用 `memes/captions.json` 中的 caption（`external_captions`），只编码向量、不调视觉模型
- 表情库固定时：本地跑一次 `/vm caption` + `/vm caption 导出` 推送后，客户端无需视觉模型即可建立 caption 向量
- `_enrich_captions_after_index`：vision provider 不可用但存在 captions.json 时仍可完成 caption 向量构建

## [0.5.0] - 2026-08-02

### 新增
- embedder_backend 新增 api 后端：调用 AstrBot 已配置的 Embedding 提供商（open_clip / api / dummy 三选一）
- 新增配置 embedding_provider_id：指定 AstrBot Embedding 提供商 ID，留空自动用第一个可用提供商
- api 后端强制开启 vision caption，检索走 caption 文本向量（纯文本检索），图片路向量留空

### 行为
- DualRetriever：图片路无向量（api 后端）时，检索结果完全来自 caption 路
- 索引时 api 后端跳过图片编码（NotImplementedError 静默），不再误报嵌入失败

### 限制
- api 后端下 /vm 自动分类（prototype/KNN）不可用（依赖图片向量）
- api 向量维度与 open_clip 不同，切换后需 /vm 重建索引

## [0.4.1] - 2026-08-02

### 配置面板分框
- _conf_schema.json 重写为 7 组 object 分组（基础路径与存储 / 模型与设备 / 语义增强 / 索引与标签 / 检索选择与反重复 / LLM 标签与发送 / 自动分类与评测），面板按卡片分组展示，参考 engram 插件
- 删除失效配置 auto_index_on_start（_auto_index 从未挂启动钩子）
- 移除 description 中的【分组名】前缀（由分组卡片承担）
- 配置读取兼容：插件启动时拍平分组配置，self.config.get(key) 读取方式不变

## [0.4.0] - 2026-08-01

### 标签体系 v2（tag 体系重做）
- 表情库替换：memes/ 由 22 目录（398 张）整体替换为 emoji 库 17 目录（191 张）
- 17 标签三分类：emotion（10）/ social_action（4）/ context_meme（3）
- tag_schema.json 重写 v2：每条含 category / meaning / visual_features / exclude / color
- moew 改名为 meow
- 索引/修复时按 tag_schema 同步 tags 表 description/category/color，persona 注入自动带出新定义
- /vm 修复、/vm 重建 自动清理 schema 外且无表情引用的残留 tag
- memes/ 表情库纳入 git 跟踪，便于迁移
- 修复：命令注册补充 vm alias，`/vm` 前缀命令可用（此前仅 `表情向量`/`vmem`）
- 修复：OpenCLIP 权重加载兼容 PyTorch 2.6+（load_checkpoint 显式 weights_only=False）
- 文档：README/metadata 更新（XML 结构化输出集成、memes/ 入库、17 标签三分类、caption 命令）
- 修复：/vm 索引、/vm 重建 后台进度消息改用 event.send，修复 v4.25.5 下 bot.send 参数不识别导致的发送失败

## [0.3.0] - 2026-08-01

### 新增
- 视觉 caption 双路融合检索（方案 A）：
  - 新增配置 `enable_vision_caption`（默认关闭）、`vision_provider_id`、`caption_score_weight`（默认 0.6）、`caption_batch_limit`（默认 500）
  - 新增 `/vm caption [limit]` 命令：为无 caption 的表情批量生成视觉 caption（走 AstrBot LLM 视觉通道）
  - 开启后检索按 caption 文本向量 + 图片向量双路融合排序，caption 强匹配可上位
  - 新增 `core/captioner.py`、`core/retriever_dual.py`
  - 数据库幂等迁移：`memes` 表新增 `caption`、`caption_vector_id` 列
- 依赖：`faiss-cpu`（已有）

### 行为变化
- 无 caption 的图按 caption 分数 0 参与融合（统一口径）

### 说明
- `enable_vision_caption` 默认关闭，未开启时行为与 0.2.0 一致