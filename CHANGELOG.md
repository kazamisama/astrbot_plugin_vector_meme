# 更新日志

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