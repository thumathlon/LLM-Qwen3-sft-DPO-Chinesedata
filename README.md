﻿# 项目说明：数据处理 + SFT + DPO/ORPO 完整流程
链结束提供从数据准备、指令微调（SFT）、到偏好对齐（DPO/ORPO）的完整训练流程。所有脚本默认以 dry-run 方式运行（不下载/不写入），在远程 GPU 环境请传入 `--dry-run false` 才会执行真实处理与训练。
- 数据集：
  - SFT：Mxode/Chinese-Instruct
  - DPO：llamafactory/DPO-En-Zh-20k
- 主要输出：
  - `data_proc/sft_{train,val,test}.jsonl`（字段：`messages`）
  - `data_proc/pref_{train,val,test}.jsonl`（字段：`prompt`、`chosen`、`rejected`）
## 目录结构
- `scripts/`：数据准备、配置、训练、评估、导出等脚本
- `configs/`：示例 YAML 配置，090/5090 兼容
- `docs/`：文档与说明，新增 `docs/DATA_PROCESSING.md` 详细数据处理
- `data_raw/`、`data_proc/`、`outputs/`：原始处理数据与模型输出（本地默认为空）
- `tests/`：单元测试入口（默认跳过）

## 环境准备
```bash
pip install -r requirements.txt
# 可选：配置 accelerate 分布式
# accelerate config
```

如需从 Hugging Face 抓取数据，请确保网络与认证（如需 `huggingface-cli login`）。
## 数据下载与加载
- SFT（Chinese-Instruct）：
  - 优先使用本地文件，将 `*.jsonl|*.json` 放入 `data_raw/mxode_chinese_instruct/`
  - 若本地不存在，则使用 Hugging Face `Mxode/Chinese-Instruct` 并缓存到该目录；
- DPO（DPO-En-Zh-20k）：
  - 默认从 Hugging Face 抓取 `llamafactory/DPO-En-Zh-20k` 并缓存到 `data_raw/dpo_en_zh_20k/`
注意：本地 dry-run 不会下载/写入。真实执行需在远程环境并安装 `datasets`。
## 数据处理（scripts/prepare_data.py）
- 运行方式：
  - 本地验证（dry-run）：
    ```bash
    python -m scripts.prepare_data --config configs/data.yaml --dry-run true
    ```
  - 远程实际执行：
    ```bash
    python -m scripts.prepare_data --config configs/data.yaml --dry-run false
    ```
- 过滤与质量：
  - 长度阈值：`general.min_tokens`/`max_tokens`
  - 中文占比：`general.min_cn_ratio`（可选 `general.allow_english_fallback`）
  - 重复率：`general.max_repetition`（基于 n-gram 重复率）
- 采样与去重：
  - `MixedBucketSampler` 同时考虑中文占比、长度分桶、数据源权重、阶段式权重（可选）
  - 去重基于连接文本的 `hash`（SHA256）
- 输出：
  - SFT：`data_proc/sft_train.jsonl`、`sft_val.jsonl`、`sft_test.jsonl`（字段：`messages`）
  - PREF：`data_proc/pref_train.jsonl`、`pref_val.jsonl`、`pref_test.jsonl`（字段：`prompt`、`chosen`、`rejected`）
CLI 参数示例：
```bash
python -m scripts.prepare_data \
  --config configs/data.yaml \
  --stage all \
  --sft-mix mxode_chinese_instruct=1.0 \
  --pref-mix dpo_en_zh_20k=1.0 \
  --allow-english-fallback \
  --dry-run false
```

兼容性：若使用旧版 `pref` 配置键，脚本将视作 `preference`。
## 训练（SFT）
- 命令：
```bash
accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false
```
- 数据接口：默认读取 `data_proc/sft_{train,val}.jsonl` 中的 `messages` 字段（若未指定 `dataset_text_field` 将自动推断为文本）
- 关键超参数（configs/sft.yaml）：
  - `model.base_model`: 例如 `Qwen/Qwen2.5-7B-Instruct`
  - model.max_seq_length: 4096（与数据处理上限一致，可按显存调整）
  - `model.packing`: true（高效率优化）
  - `lora.enable|r|alpha|dropout|target_modules`
  - `qlora.enable|quant_dtype|double_quant`（量化训练时启用）
  - `training.epochs|per_device_train_batch_size|gradient_accumulation_steps`
  - `training.learning_rate|lr_scheduler_type|warmup_ratio|weight_decay|max_grad_norm`
  - `training.bf16|tf32|dataloader_num_workers`

## 训练（DPO 或 ORPO）
- DPO 命令：
```bash
accelerate launch -m scripts.train_dpo --config configs/dpo.yaml --dry-run false
```
- ORPO 命令（如需）：
```bash
accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false
```
- 数据接口：`data_proc/pref_{train,val}.jsonl`，字段 `prompt`、`chosen`、`rejected`
- SFT → DPO：`configs/dpo.yaml -> model.base_model` 默认 `outputs/sft`（衔接 SFT 输出）
- 关键超参数（configs/dpo.yaml）：
  - `dpo.beta|reference_free|length_penalty|prefer_chinese_ratio`
  - `training.epochs|batch_size/accumulation|learning_rate|scheduler|warmup|bf16|tf32`

## 评估与导出
```bash
python -m scripts.eval_harness --config configs/eval.yaml --dry-run false
python -m scripts.export_lora   --config configs/sft.yaml  --dry-run false
```

## 常见问题（FAQ）
- 中文文本不足：若 `allow_english_fallback=false`，样本将被中文过滤截断，可提高中文占比阈值。允许英文回退或调整权重。
- 文本过长/过短：调整 `general.{min_tokens,max_tokens}`。
- 重复率过高：降低 `general.max_repetition` 或加强过滤。
- 本地数据优先：将 Chinese-Instruct JSON 放入 `data_raw/mxode_chinese_instruct/` 避免远程下载。
## 参考
- 数据处理详细文档：`docs/DATA_PROCESSING.md`
- 训练脚本关键实现：`scripts/train_sft.py`, `scripts/train_dpo.py`
- 采样与过滤逻辑：`scripts/utils_data.py`
