# 更新日志

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