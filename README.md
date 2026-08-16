# astrbot_plugin_vector_meme

基于向量检索的 AstrBot 智能表情包插件，当前版本 **0.7.4**。

插件用 CLIP/OpenCLIP 把表情图片编码为向量。LLM 只需要在回复里输出情绪/场景占位符（`%%happy%%`）或配合 XML 结构化输出插件输出 `<sticker>happy</sticker>` 标记，具体发送哪张图由向量相似度、标签权重、反重复策略和随机采样共同决定。

> 插件可独立运行，不依赖 `meme_manager`。如需复用现有图库，可以复制一份，或通过 `meme_dir` 配置直接指向 `meme_manager/memes`。

## 功能特性

- **向量检索选图**：回复文本与表情图片在同一向量空间比对，按相似度选图。
- **多嵌入后端**：`open_clip`（默认）、`api`（AstrBot Embedding 提供商）、`dummy`（测试链路）。
- **两阶段检索**：先按 tag 限定候选集，再用文本向量精排。
- **混合打分**：`raw_similarity + tag_bonus - repeat_penalty - usage_penalty + random_jitter`。
- **反重复与多样性**：近期/高频使用过的图自动降权，TopK 内按分数加权随机采样。
- **视觉 caption 双路融合**：可选开启 vision 模型生成语义描述，检索时融合 caption 文本向量与图片向量（`embedder_backend=api` 时强制开启）。
- **自动分类辅助**：基于 tag prototype 原型向量 + KNN 最近邻投票预测标签。
- **标签治理**：`tag_schema.json` 统一管理 17 个标签的类别、含义、视觉特征和排除项。
- **诊断与修复**：健康检查 DB / FAISS / 文件系统一致性；`/vm 修复` 清理缺失文件、注册标签、压缩孤儿向量。
- **重建安全**：`/vm 重建` 先备份旧库再加载 embedder，失败时保留原库与恢复入口。
- **XML 结构化输出联动**：可被 `astrbot_plugin_xml_structured_output` 通过 `search_sticker_for_external()` 按 `<sticker>` 标记取图。

## 快速开始

```bash
git clone https://github.com/kazamisama/astrbot_plugin_vector_meme.git
cd astrbot_plugin_vector_meme
pip install -r requirements.txt
```

把表情放进 `memes/` 目录，子目录名作为 tag：

```text
memes/
├── happy/
│   ├── 001.png
│   └── 002.jpg
├── sad/
└── shy/
```

然后在 AstrBot 里执行：

```text
/vm 索引
/vm 状态
/vm 搜索 有点害羞
```

> 只想先跑通命令链路时，把 `embedder_backend` 配成 `dummy`，不下载模型。

## 嵌入后端

| 后端 | 说明 |
|---|---|
| `open_clip`（默认） | 本地 OpenCLIP，图片 + 文本双模态向量 |
| `api` | 调用 AstrBot 已配置的 Embedding 提供商；图片不可直接编码，必须依赖 vision caption，检索走 caption 文本向量；`/vm 自动分类` 不可用 |
| `dummy` | 随机向量，仅用于测试链路 |

换模型架构后向量维度会变（如 ViT-B-32=512、ViT-L-14=768），需要 `/vm 重建`。

## 配置

`_conf_schema.json` 按用途分组，AstrBot 配置面板按卡片展示：

1. **基础路径与存储**：`meme_dir`、`data_dir`
2. **模型与设备**：`embedder_backend`、`embedding_provider_id`、设备、OpenCLIP 模型/权重/缓存、HF 镜像
3. **语义增强（vision caption）**：`enable_vision_caption`、`vision_provider_id`、caption 权重与批量
4. **索引与标签**：子目录 tag、默认 tag
5. **检索选择与反重复**：反重复窗口、采样池、随机扰动
6. **LLM 标签与发送**：占位符格式、prompt 注入、触发概率、发送数量
7. **自动分类与评测**：prototype/KNN 分类阈值与样本数

推荐初始配置：

```json
{
  "embedder_backend": "open_clip",
  "device": "cpu",
  "meme_dir": "",
  "data_dir": "",
  "allow_legacy_markup": false,
  "enable_prompt_injection": true,
  "max_per_message": 2,
  "enable_mixed_message": true,
  "query_text_max_length": 80,
  "anti_repeat_window": 20,
  "selection_pool_size": 12,
  "enable_stochastic_selection": true,
  "selection_random_jitter": 0.015
}
```

国内网络下载 OpenCLIP 权重不稳定时，可设置 HF 镜像或本地权重：

```json
{
  "hf_endpoint": "https://hf-mirror.com",
  "open_clip_local_weights": "C:\\path\\to\\open_clip_pytorch_model.bin"
}
```

## 图库

### 模式 A：插件独立图库（推荐）

把表情放入 `memes/`，子目录名即 tag。已有 `meme_manager` 图库时可复制一份：

```bash
python scripts/copy_meme_manager_library.py
```

复制不会影响源图库。

### 模式 B：直接指向 meme_manager 图库

```json
{
  "meme_dir": "C:\\Users\\chiriu\\.astrbot\\data\\plugins\\meme_manager\\memes"
}
```

优点是省空间；缺点是修改或删除图片会影响原图库。

### caption 固化分发

开启 vision caption 后，视觉模型生成 caption 会消耗调用成本。可以本地生成一次并固化：

```text
/vm caption
/vm caption 导出
```

导出内容写入 `memes/captions.json`，随图库一起分发。其他实例索引/重建时会自动复用固化 caption，只用自己的 embedder 编码 caption 向量，不再调用视觉模型。

## 命令

触发词：`表情向量`（别名 `vmem`、`vm`）。

> 表中 🔒 命令仅管理员可用。

| 命令 | 说明 |
|---|---|
| `/vm 状态` | 查看索引状态，不触发模型加载 |
| `/vm 预热` 🔒 | 只加载 embedder，提前检查模型可用性 |
| `/vm 索引 [目录]` 🔒 | 扫描目录建索引，增量更新；目录仅限 `meme_dir` 及其子目录 |
| `/vm 重建` 🔒 | 在临时库构建成功后替换旧库；先备份 `.bak.v071`，失败保留旧库与备份 |
| `/vm 列表 [标签]` | 列出指定标签下的表情或整体标签分布 |
| `/vm 标签` | 列出已注册标签 |
| `/vm 搜索 <文本> [--tag <标签>]` | 文本检索测试，显示 final/raw 分数 |
| `/vm 解释 <文本> [--tag <标签>]` | 展示 tag bonus、重复惩罚、随机扰动等选图原因 |
| `/vm 最近使用 [数量]` | 查看最近发送过的表情 |
| `/vm 诊断` / `/vm 健康检查 [--deep]` 🔒 | 检查 DB、FAISS、文件系统一致性；`--deep` 额外比对文件内容 hash |
| `/vm 修复` 🔒 | 清理缺失文件记录、补注册 tag、压缩孤儿向量、刷新 prompt |
| `/vm 自动分类 <id> [apply]` 🔒 | 预测标签；默认 dry-run，加 `apply` 才写入数据库 |
| `/vm 重标注 <id> <新tag>` 🔒 | 修改数据库主 tag，不移动文件 |
| `/vm 评测 <eval.json>` 🔒 | 跑小型分类评测集，输出 top1/top3 准确率 |
| `/vm 标签规范` | 查看 `tag_schema.json` 标签说明 |
| `/vm caption [limit]` 🔒 | 为无 caption 的表情批量生成视觉语义描述，自动复用 `captions.json` |
| `/vm caption 导出` 🔒 | 把 DB 中的 caption 导出到 `memes/captions.json` |
| `/vm 删除 <id>` 🔒 | 从索引移除某条记录 |
| `/vm 刷新提示` | 重新注入表情标签 prompt 到全局人格 |
| `/vm 帮助` | 显示命令列表 |

> `/vm 搜索` 只认显式 `--tag <name>`，不再用“最后一个词是 tag”的启发式，避免查询文本里恰好出现标签名时被误判。

## LLM 集成

### `%%tag%%` 占位符

插件会把可用标签列表和规则注入全局人格（marker 块幂等管理，重复刷新不会累积）。LLM 在回复中输出：

```text
今天心情很好 %%happy%%
```

插件移除占位符，并按回复文本检索一张匹配的 `happy` 表情。发送行为受这些配置影响：

- `max_per_message`：单条消息最多表情数
- `enable_mixed_message`：图文混合发送，或先文本后补图
- `query_text_max_length`：检索用文本截断长度，避免长回复稀释语义
- `allow_legacy_markup`：是否兼容 `&&tag&&` / `:tag:` 旧格式

### `<sticker>` 与 XML 结构化输出

配合 `astrbot_plugin_xml_structured_output` 时，LLM 还可以输出：

```xml
<sticker>happy</sticker>
```

- XML 插件通过 `search_sticker_for_external()` 调用本插件取图；
- vector_meme 不提取、不清理 `<sticker>...</sticker>` 块内的 `%%tag%%`，避免 XML 插件丢 sticker；
- 外部调用超时 30 秒，冷启动时触发一次 embedder 懒加载；
- 外部调用不写反重复池（不调用 `pick()`），高频调用不会污染内部去重窗口；
- 空 tag、加载失败、超时或无命中一律返回 `None`，不抛异常；
- 本插件的 `%%tag%%` 链路与 XML 链路可并存。

## 检索策略

一次自动选图流程：

1. 从 LLM 回复提取 `%%tag%%`（sticker 块除外）。
2. 按 tag 限定候选图片；该 tag 无候选时回退到全库并标记 `fallback`。
3. 用回复文本生成查询向量。
4. 在候选集内检索最相近的 TopK（FAISS 全量扫描 + IDSelector 过滤候选）。
5. 混合打分：相似度 + tag 加权 - 近期/高频惩罚 + 小随机扰动。
6. 在 TopK 内按分数加权随机采样（可关闭）。
7. 记录使用日志，短期避免重复。

关键配置：

- `selection_pool_size`：进入加权采样的候选池大小，推荐 8-16。
- `enable_stochastic_selection`：控制是否随机选图；关闭后固定选最高分。
- `selection_random_jitter`：仅影响排序扰动，不控制是否随机选图；0 关闭。
- `anti_repeat_window`：同一张图多少条消息内不重复使用。

## 标签治理与自动分类

`tag_schema.json` 内置 17 个标签：happy / laugh / heart / shy / angry / sad / cry / surprised / confused / baka / like / thanks / see / meow / morning / sleep / kfc。

每个标签定义类别（`emotion` / `social_action` / `context_meme`）、含义、视觉特征和排除项。自动分类只改数据库 tag，不移动文件，建议先 dry-run：

```text
/vm 自动分类 1
/vm 自动分类 1 apply
```

评测集格式：

```json
{
  "items": [
    {"id": 1, "tag": "happy"},
    {"id": 2, "tag": "shy"}
  ]
}
```

```text
/vm 评测 C:\\path\\to\\eval.json
```

## 数据与维护

### 运行数据

- 默认 `data_dir` 为 AstrBot 的 `plugin_data/vector_meme`，存放 `memes.db`（SQLite 元数据）和 `memes.faiss`（向量索引）。
- `.cache/` 是离线建索引脚本的输出目录，已 git 忽略。
- `models/` 存放 OpenCLIP/HuggingFace 权重与缓存，已 git 忽略。
- `memes/` 表情图库和 `memes/captions.json` 纳入 git 跟踪，便于跨机器迁移。

### 重建与备份

`/vm 重建` 顺序：

1. 把 `memes.db` / `memes.faiss` 备份为 `.bak.v071`（SQLite 走备份 API，FAISS 在锁内复制）。
2. 仅加载 embedder，不删除现有库。
3. 在临时目录构建完整新库（含旧 caption 迁移与 `captions.json` 复用）。
4. 构建成功后替换磁盘文件并原地切换内存索引；成功才删除备份，失败自动回滚并保留备份。
5. 重建会保留 `usage_count` / `last_used_at` / `disabled` 以及检索、索引日志。

> 🔒 标记的命令需要管理员权限；普通成员可使用状态/搜索/列表等只读命令。
> 普通成员的 `搜索` / `解释` 在 embedder 未预热时不会触发模型冷加载，需管理员先执行 `/vm 预热`。

### 孤儿向量

删除/更新表情不会立即回收 FAISS 向量，孤儿向量会累积。`/vm 修复` 在 `index_size` 比 DB 引用向量数多出 `max(50, 20% × 引用向量数)` 时自动压缩索引，重映射 `vector_id` / `caption_vector_id`；健康检查会显示 `orphan_vector_count`。

## 目录结构

```text
astrbot_plugin_vector_meme/
├── main.py                 # 插件入口：命令、LLM 钩子、persona 注入
├── metadata.yaml
├── _conf_schema.json
├── tag_schema.json
├── core/
│   ├── database.py         # SQLite + FAISS，含孤儿向量压缩
│   ├── embedder.py         # open_clip / api / dummy 工厂
│   ├── indexer.py          # 扫描、增量索引、绝对路径存储
│   ├── retriever.py        # 两阶段检索、混合打分、自动分类
│   ├── retriever_dual.py   # caption 双路融合检索
│   └── captioner.py        # 视觉 caption 生成
├── scripts/
│   ├── build_openclip_index.py
│   ├── copy_meme_manager_library.py
│   ├── migrate_config_v041.py
│   └── test_openclip_load.py
├── tests/                  # pytest 核心测试（dev-only）
├── utils/
└── memes/                  # 内置图库，子目录名即 tag
```

## 故障排查

### OpenCLIP 下载失败

```json
{
  "hf_endpoint": "https://hf-mirror.com"
}
```

仍失败时手动下载权重并配置 `open_clip_local_weights`。

### 重建或索引很慢

- 先用 `dummy` 后端确认命令链路；
- `/vm 预热` 单独验证模型加载；
- 有 CUDA 时把 `device` 设为 `cuda`；
- 后续更新走增量 `/vm 索引`，不必每次全量重建。

### 搜索结果不理想

- 检查图库子目录 tag 是否准确；
- 用 `/vm 解释` 查看打分细节；
- 调低 `selection_random_jitter` 或减小 `selection_pool_size`；
- 完善 `tag_schema.json` 里的标签说明；
- 开启 vision caption 双路融合提升语义匹配。

## 开发与测试

测试依赖仅用于开发，不进入插件运行时依赖：

```bash
pip install -r tests/requirements-dev.txt
python -m pytest tests -q
```

常用检查：

```bash
python -m py_compile main.py core/*.py utils/*.py tests/*.py
```

离线建索引（不启动 AstrBot）：

```bash
python scripts/build_openclip_index.py --meme-dir memes --data-dir .cache
```

## 更新日志

版本历史见 [CHANGELOG.md](CHANGELOG.md)。

## License

未指定许可证。发布前如需开源复用，请补充 LICENSE。
