# astrbot_plugin_vector_meme

基于向量检索的 AstrBot 智能表情包插件。插件用 CLIP/OpenCLIP 将图片编码为向量，LLM 只负责在回复中输出情绪或场景占位符（`%%happy%%`），或配合 XML 结构化输出插件输出 `<sticker>happy</sticker>` 标记；具体发送哪张图由向量相似度、标签权重、反重复策略和随机采样共同决定。

> 插件可独立运行，不依赖 `meme_manager`。如需复用现有图库，可以复制一份或通过配置直接指向 `meme_manager/memes`。

## 功能特性

- **向量检索选图**：根据回复文本和标签，在候选图库中找最匹配的表情。
- **OpenCLIP 后端**：支持 `open_clip`，也提供 `dummy` 后端用于测试链路。
- **两阶段检索**：先按 tag 粗筛，再用文本向量精排。
- **混合打分**：`raw_similarity + tag_bonus - repeat_penalty - usage_penalty + random_jitter`。
- **反重复与多样性**：近期/高频使用过的图自动降权，并支持 TopK 加权随机采样。
- **解释与诊断**：可查看为什么选中某张图，并检查 DB / FAISS / 文件系统一致性。
- **自动分类辅助**：基于 tag prototype 原型向量 + KNN 最近邻投票预测标签。
- **标签治理**：`tag_schema.json` 统一 17 个标签（emotion/social_action/context_meme 三分类）的含义、视觉特征和排除项。
- **视觉 caption 双路融合**：可选开启 vision 模型生成表情语义描述，检索时按 caption 文本向量 + 图片向量融合排序（0.3.0+）。
- **XML 结构化输出联动**：可被 `astrbot_plugin_xml_structured_output` 通过 `search_sticker_for_external()` 调用，按 `<sticker>` 标记取图（0.3.0+）。

## 目录结构

```text
astrbot_plugin_vector_meme/
├── metadata.yaml
├── _conf_schema.json
├── requirements.txt
├── main.py
├── tag_schema.json
├── core/
│   ├── database.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── retriever_dual.py   # 0.3.0+ 双路融合检索
│   └── captioner.py        # 0.3.0+ 视觉 caption
├── scripts/
│   ├── build_openclip_index.py
│   ├── copy_meme_manager_library.py
│   └── test_openclip_load.py
└── utils/
    └── image_utils.py
```

以下目录默认不会提交到仓库：

```text
models/    # OpenCLIP/HuggingFace 模型权重和缓存
data/      # SQLite / FAISS 运行数据
```

> 0.4.0 起表情图库 `memes/` 已纳入 git 跟踪，便于跨机器迁移。

## 安装

在 AstrBot 插件目录中克隆：

```bash
git clone https://github.com/kazamisama/astrbot_plugin_vector_meme.git
```

安装依赖：

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install faiss-cpu Pillow numpy
pip install open_clip_torch torch
```

如果只是想先跑通命令和索引流程，可以在配置中把 `embedder_backend` 改为 `dummy`。

### 三种嵌入后端（embedder_backend）

- `open_clip`（默认）：本地 OpenCLIP，图片+文本双模态语义向量
- `api`：调用 AstrBot 已配置的 Embedding 提供商（文本 embedding）。**限制**：图片无法直接编码，必须搭配 vision caption（强制开启），检索走 caption 文本向量；切换后需 `/vm 重建`（向量维度不同）；`/vm 自动分类` 不可用
- `dummy`：随机向量，仅测试链路

## 配置分组

`_conf_schema.json` 已按用途分块（面板按卡片分组展示）：

1. **基础路径与存储**：`meme_dir`、`data_dir`
2. **模型与设备**：`embedder_backend`、`embedding_provider_id`、设备、OpenCLIP 模型/权重/缓存、HF 镜像
3. **语义增强（vision caption）**：`enable_vision_caption`、`vision_provider_id`、caption 权重与批量
4. **索引与标签**：子目录 tag、默认 tag
5. **检索选择与反重复**：反重复窗口、采样池、随机扰动
6. **LLM 标签与发送**：占位符、prompt 注入、触发概率、发送数量
7. **自动分类与评测**：prototype/KNN 分类阈值和样本数

推荐初始配置：

```json
{
  "embedder_backend": "open_clip",
  "device": "cpu",
  "meme_dir": "",
  "allow_legacy_markup": false,
  "trigger_probability": 80,
  "max_per_message": 2,
  "anti_repeat_window": 20,
  "selection_pool_size": 12,
  "enable_stochastic_selection": true
}
```

国内网络下载 OpenCLIP 权重不稳定时，可以使用：

```json
{
  "hf_endpoint": "https://hf-mirror.com",
  "open_clip_cache_dir": "",
  "open_clip_local_weights": "C:\\path\\to\\open_clip_pytorch_model.bin"
}
```

## 准备图库

### 模式 A：使用插件独立图库（推荐）

把表情放到插件目录的 `memes/` 下，子目录名作为 tag：

```text
memes/
├── happy/
│   ├── 001.png
│   └── 002.jpg
├── sad/
└── shy/
```

如果你已经有 `meme_manager` 图库，可以复制一份：

```bash
python scripts/copy_meme_manager_library.py
```

这种方式不会影响原本的 `meme_manager` 图库。

### 模式 B：直接指向 meme_manager 图库

在配置中设置：

```json
{
  "meme_dir": "C:\\Users\\chiriu\\.astrbot\\data\\plugins\\meme_manager\\memes"
}
```

优点是省空间；缺点是修改或删除图片会影响原图库。

## 基本使用

索引图库：

```text
/vm 索引
```

查看状态：

```text
/vm 状态
```

测试搜索：

```text
/vm 搜索 有点害羞
/vm 搜索 --tag shy 有点害羞
```

> 0.2.0 起：`/vm 搜索` 不再用"最后一个词当 tag"的启发式，避免文本里恰好出现 tag 名时被误判。需要按 tag 限定候选集时显式加 `--tag <name>`。

LLM 在回复中输出：

```text
今天心情很好 %%happy%%
```

插件会移除 `%%happy%%` 占位符，并追加一张匹配的 `happy` 表情。

## 命令

| 命令 | 说明 |
|---|---|
| `/vm 状态` | 查看索引状态，不触发模型加载 |
| `/vm 预热` | 只加载 embedder，便于提前检查模型 |
| `/vm 索引 [目录]` | 扫描目录建索引，增量更新 |
| `/vm 重建` | 清空并重建整个索引 |
| `/vm 列表 [标签]` | 列出表情或标签分布 |
| `/vm 标签` | 列出已注册标签 |
| `/vm 搜索 <文本> [--tag <标签>]` | 文本检索测试，显示 final/raw 分数；可选 `--tag` 限定候选集 |
| `/vm 解释 <文本> [--tag <标签>]` | 展示 tag bonus、重复惩罚、随机扰动等选图原因 |
| `/vm 最近使用 [数量]` | 查看最近发送过的表情 |
| `/vm 诊断` / `/vm 健康检查` | 检查 DB、FAISS、文件系统一致性 |
| `/vm 修复` | 清理缺失文件记录、补注册 tag、刷新 prompt |
| `/vm 自动分类 <id> [apply]` | 预测标签；默认只预览，加 `apply` 才写入数据库 |
| `/vm 重标注 <id> <新tag>` | 手动修改数据库主 tag，不移动文件 |
| `/vm 评测 <eval.json>` | 跑小型分类评测集，输出 top1/top3 准确率 |
| `/vm 标签规范` | 查看 `tag_schema.json` 标签治理说明 |
| `/vm caption [limit]` | 为无 caption 的表情批量生成视觉语义描述（需开启 vision 配置） |
| `/vm 删除 <id>` | 从索引移除某条记录 |

> 命令触发词：`表情向量`（alias：`vmem`、`vm`）。如果 AstrBot 命令前缀不是 `/`，请按你的实例配置调整。

## 与 XML 结构化输出插件集成

配合 `astrbot_plugin_xml_structured_output`（雪莉人设 XML 输出）时，LLM 除了输出 `%%tag%%` 占位符，还可以在 XML 中输出 `<sticker>tag</sticker>` 标记：

- XML 插件解析后通过 `search_sticker_for_external()` 调 vector_meme 取图；
- 该 API 2 秒超时、不写反重复池、不触发懒加载，专为高频外部调用设计；
- vector_meme 自身的 `%%tag%%` 链路不受影响，两条路径可并存；
- 可用 XML 插件配置 `sticker.enable_vector_meme_forward`（默认开启）控制转发。

## 检索策略说明

一次自动选图大致分为：

1. 从 LLM 回复中提取 `%%tag%%`。
2. 使用 tag 过滤候选图片。
3. 将回复文本编码为文本向量。
4. 计算文本向量和图片向量相似度。
5. 根据 tag、近期使用、历史使用次数和随机扰动做混合打分。
6. 在 TopK 候选中按分数加权采样。
7. 记录使用日志，避免短时间内重复。

关键配置：

- `query_text_max_length`：截断过长回复，避免语义被稀释。
- `selection_pool_size`：进入加权随机采样的候选池大小。
- `enable_stochastic_selection`：是否启用随机采样；关闭后固定选最高分。
- `selection_random_jitter`：给相近分数候选一点扰动。
- `anti_repeat_window`：同一张图在多少条消息内不重复。

## 标签治理与自动分类

`tag_schema.json`（v2）定义每个 tag 的：

- `category`：三分类（`emotion` / `social_action` / `context_meme`）；
- `meaning`：含义与群氛围匹配提示（emotion 类）；
- `visual_features`：视觉特征；
- `exclude`：排除项。

0.4.0 起内置 17 个标签（emoji 图库）：happy / laugh / heart / shy / angry / sad / cry / surprised / confused / baka / like / thanks / see / meow / morning / sleep / kfc。

自动分类不会移动图片文件，只会在 `apply` 时更新数据库中的 tag。建议先 dry-run：

```text
/vm 自动分类 1
```

确认可信后再写入：

```text
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

运行：

```text
/vm 评测 C:\\path\\to\\eval.json
```

## 数据与隐私

`.gitignore` 默认排除：

- `models/`：模型权重和缓存；
- `data/`、`*.db`、`*.faiss`：运行数据库和向量索引；
- `*.log`、`__pycache__/`、`.codegraph/` 等本地缓存。

0.4.0 起 `memes/` 表情图库纳入 git 跟踪（17 标签、191 张），方便跨机器迁移；模型权重和运行数据仍不上传。

## 故障排查

### OpenCLIP 下载失败

优先设置：

```json
{
  "hf_endpoint": "https://hf-mirror.com"
}
```

如果仍失败，可以手动下载权重，然后设置 `open_clip_local_weights`。

### 重建索引耗时很长

OpenCLIP 在 CPU 上处理大量图片会比较慢。建议：

- 先用 `dummy` 后端确认命令链路；
- 使用 `/vm 预热` 单独测试模型加载；
- 有 CUDA 环境时把 `device` 设置为 `cuda`；
- 大图库首次索引时耐心等待，后续使用增量索引。

### 搜索结果不理想

可以尝试：

- 检查图片目录 tag 是否准确；
- 用 `/vm 解释 <文本> [标签]` 看打分细节；
- 调低 `selection_random_jitter`；
- 减小 `selection_pool_size`；
- 补充 `tag_schema.json` 中的标签说明。

## 开发

常用检查：

```bash
python -m py_compile main.py core/*.py utils/*.py
```

建议提交前确认不会加入本地大文件：

```bash
git status --ignored
```

## License

未指定许可证。发布前如需开源复用，请补充 LICENSE。
