# 更新日志

## [0.7.4] - 2026-08-13

### 修复
- 管理命令（索引/重建/修复/caption/删除/重标注等）增加管理员权限校验，索引目录限制为 meme_dir 及其子目录
- 重建改为“临时库构建成功后原地替换”，失败自动回滚并保留 `.bak.v071`；不再先删旧索引
- FAISS 空索引也写盘，修复空库重建失败与 compact 到 0 后重启复活孤儿向量的问题
- DB 记录 embedder 指纹（backend/model/权重/dim），切换不兼容模型时强制提示重建
- LLM 占位符与外部 `<sticker>` 调用改为按 DB 真实 tag 白名单过滤，未知 tag 不再 fallback 发图
- `search_sticker_for_external` 关闭随机扰动，相同输入返回相同图片
- `/vm caption` 复用 `captions.json` 并拒绝非正整数 limit；重建自动迁移旧库 caption
- FAISS 损坏时 `/vm 修复` 明确指引使用 `/vm 重建`
- 冷启动搜索/解释/修复/分类/评测/caption 增加加载心跳；状态/健康检查输出补全
- 普通成员搜索/解释不再触发冷模型加载，需管理员先 `/vm 预热`
- `/vm 诊断 --deep` 支持比对文件内容 hash
- 重建保留 `usage_count` / `last_used_at` / `disabled` 以及检索、索引日志
- OpenCLIP 非本地 pretrained 权重增加模型参数内容摘要指纹
- 统一写操作互斥，覆盖索引/重建/修复/caption/删除/重标注，降低并发 DB/FAISS 不一致风险
- 重建改为临时库构建成功后替换，失败保持原库
- compact_index 原子化，避免索引替换与 DB 重映射之间的错位窗口
- DualRetriever 图片/caption 路径取并集；caption-only 结果也走反重复/使用次数惩罚
- FAISS 读取失败标记索引损坏，健康检查增加 caption_vector_id 校验
- 检索候选查询瘦身，降低大图库下的内存/IO 压力
- pytest 根目录仅收集 tests，避免 scripts/test_openclip_load.py 被误收集

## [0.7.3] - 2026-08-11

### 修复
- 修正 `search_sticker_for_external` 契约说明：30s 超时只覆盖检索部分，冷 embedder 懒加载发生在超时之前，首次调用可能更久

## [0.7.2] - 2026-08-09

### 变更
- 移除 `trigger_probability` 配置：LLM 输出标签后直接检索发送，不再整条消息随机丢弃；旧配置残留静默失效
- `_temp_dim` 隐藏配置内部化为模块常量，不再从配置文件读取
- 明确 `selection_random_jitter` 与 `enable_stochastic_selection` 的分工：前者只影响排序扰动，后者控制是否随机选图

## [0.7.1] - 2026-08-09

### 修复
- 插件注册版本与 metadata/CHANGELOG 对齐到 0.7.1
- `/vm 重建` 先备份旧库并先加载 embedder，加载/索引失败时保留恢复入口
- 空库 + 残留 FAISS 换模型维度时按新维度重建空索引
- FAISS add/search/reconstruct 统一加锁，新增带锁检索/重建接口
- 索引进度跨线程写入改为 call_soon_threadsafe
- 检索改为命中后取行，消除全量预取和超长 IN 子句上限
- persona 注入改为 marker 幂等块，移除一次性备份和位置 zip 依赖
- `/vm 修复` 自动压缩孤儿向量并重映射 vector_id / caption_vector_id
- 修复 migrate_config_v041.py CONFIG_FILE 未定义与 UTF-8 BOM 读取
- 索引路径统一存绝对路径；移除死代码
- 新增 pytest 核心测试（dev-only）

## [0.7.0] - 2026-08-02

### 新增
- 流式输出兼容：`on_decorating_result` 检测到 `STREAMING_FINISH`（流式收尾）时，不再把图片追加进 `result.chain`（流式路径不会发送该 chain），改为直接 `event.send` 补发，修复 webchat 等平台开启流式输出后表情包选中但始终发不出的问题

## [0.6.9] - 2026-08-02

### 修复
- 修复 `search_sticker_for_external` 解包 `_ensure_ready()` 返回值位置错误（`_, retriever, _` → `retriever, _, _`），导致 XML 插件 `<sticker>` 转发路径拿到 MemeIndexer 报 `AttributeError: 'MemeIndexer' object has no attribute 'retrieve'`、表情包始终发不出的问题

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
