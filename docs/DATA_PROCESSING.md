# 数据处理与全流程对接说明

本文件详细说明 `scripts/prepare_data.py` 的数据处理逻辑、与训练阶段的对接方式、配置项解释与兼容性，以及常见问题排查。

## 数据集与目标

- SFT：Mxode/Chinese-Instruct → 产出 `messages` 格式的对话数据
- DPO：llamafactory/DPO-En-Zh-20k → 产出 `prompt/chosen/rejected` 三元组

## 运行模式

- 本地默认 dry-run：`python -m scripts.prepare_data --dry-run true`
- 远程真实执行：`python -m scripts.prepare_data --dry-run false`

## 流水线步骤

1) 解析配置与覆盖
- 读取 `configs/data.yaml`，并应用命令行覆盖（如 `--stage`、`--sft-mix`、`--pref-mix` 等）。
- 兼容旧版：若顶层存在 `pref`，会视为 `preference`。

2) 静态自检
- 检查 `data_raw/<dataset_key>/` 目录是否存在；
- 使用内置伪样本进行解析与语言比例估计校验；
- 构建 `MixedBucketSampler`（长度分桶 + 中文占比 + 混合权重 + 课程式权重）。

3) 转换与过滤
- SFT（Chinese-Instruct）字段映射：`instruction`+`input` → user；`output|response` → assistant → `messages`；
- DPO 字段映射：优先使用 `prompt/chosen/rejected`，并对常见别名做兼容（如 `chosen_response` 等）。
- 过滤规则：`min_tokens`/`max_tokens`、`min_cn_ratio`、`max_repetition`（基于 n-gram 重复率）。
- 去重：按拼接文本的 SHA256 哈希 `hash` 去重。

4) 采样与写出
- 使用 `MixedBucketSampler.plan` 在分桶与权重约束下选择样本；
- 写出 JSONL：
  - SFT：`data_proc/sft_train.jsonl`、`sft_val.jsonl`、`sft_test.jsonl`
  - PREF：`data_proc/pref_train.jsonl`、`pref_val.jsonl`、`pref_test.jsonl`

## 与训练的契合

- `scripts/train_sft.py` 默认读取 `messages` 字段；若未设置 `dataset_text_field`，脚本内部会将 `messages` 拼接成文本。
- `scripts/train_dpo.py` 读取 `prompt/chosen/rejected` 三字段，直接对接偏好对齐数据。
- 默认文件名与数据模式与准备脚本完全一致，无需额外改动。

## 数据加载逻辑

- SFT：优先使用本地 `data_raw/mxode_chinese_instruct/*.jsonl|*.json`，否则从 Hugging Face 拉取 `Mxode/Chinese-Instruct`；
- DPO：从 Hugging Face 拉取 `llamafactory/DPO-En-Zh-20k`；
- 真实运行需安装 `datasets`，dry-run 模式则不需要。

## 关键配置项

- `general.stage`: `all|sft|pref`
- `general.min_cn_ratio`: 中文占比阈值（默认 0.3）
- `general.allow_english_fallback`: 中文不足时是否回退到英文（默认 false）
- `general.{min_tokens,max_tokens,max_repetition}`: 长度与重复度过滤
- `sft.max_samples`、`sft.mix`: SFT 抽样数与源权重
- `preference.max_samples`、`preference.mix`: 偏好抽样数与源权重
- `length_buckets`: 长度分桶定义

命令行权重覆盖：

```bash
python -m scripts.prepare_data \
  --config configs/data.yaml \
  --stage all \
  --sft-mix mxode_chinese_instruct=1.0 \
  --pref-mix dpo_en_zh_20k=1.0 \
  --dry-run false
```

## 常见问题

- 中文样本不足：若 `allow_english_fallback=false`，抽样会按中文池裁剪；可提高中文占比或允许回退。
- 样本过长/过短：调整 `min_tokens`/`max_tokens`。
- 重复度较高：调低 `max_repetition` 或采用更强的清洗策略。
- 本地数据覆盖：把 JSON 放到 `data_raw/mxode_chinese_instruct/` 即可覆盖远程。

## 验证与 Dry-run

- 本地执行 `python -m scripts.prepare_data --dry-run true` 查看采样统计、输出路径与日志；
- 真实运行请在远程环境安装依赖并设置 `--dry-run false`。
