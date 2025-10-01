# LLM 中文教学场景微调与对齐方案

## 1. 目标与数据

- 任务：面向 K12 中文教学场景的问答/讲解模型，兼顾泛化能力。
- SFT 数据集：`Mxode/Chinese-Instruct`（instruction/input/output）。
- DPO 数据集：`llamafactory/DPO-En-Zh-20k`（prompt/chosen/rejected）。
- 数据存放：
  - `data_raw/mxode_chinese_instruct/`
  - `data_raw/dpo_en_zh_20k/`
  - 若需要离线运行，可将 JSON/JSONL 直接放入上述目录。

## 2. 清洗与规范化

- SFT：
  - 读取 instruction/input/output，合并为 user → assistant 两轮 `messages`。
  - 质量过滤：长度 [8, 2048]，字符级 4-gram 重复率 ≤0.6。
  - 中文占比 ≥30%，否则如果 `allow_english_fallback=false` 则丢弃。
  - SHA256 去重。
- DPO：
  - 兼容字段 `prompt`、`chosen`、`rejected` 的多种别名（`chosen_response` 等）。
  - 过滤与去重规则同 SFT。

## 3. 数据脚本（`scripts/prepare_data.py`）

- `SFT_SPECS` 与 `PREF_SPECS` 指向新数据集。
- 如果 `data_raw/<dataset>` 下有本地 JSON/JSONL，则优先使用本地文件。
- `MixedBucketSampler`：中文配额 70%，长度桶轮转；权重为 0 的数据集不会被加载。
- Dry-run 输出数据计划，真实运行生成 `data_proc/sft_*.jsonl` 与 `pref_*.jsonl`。

## 4. SFT 训练（`scripts/train_sft.py`）

- 基座：`Qwen/Qwen2.5-7B-Instruct`，LoRA r=16 α=16。
- 支持 QLoRA（4bit）以适配 24GB 显存。
- `formatting_func` 自动使用 tokenizer chat template 处理 `messages`。
- dry-run 提供采样计划、步数估计；真实运行保存到 `outputs/sft/`。

## 5. 偏好对齐（`scripts/train_orpo.py` / `scripts/train_dpo.py`）

- 数据：全部来自 `llamafactory/DPO-En-Zh-20k`。
- ORPO：β=0.2，length_penalty=0.02，prefer_chinese_ratio=0.7。
- DPO：β=0.2，reference_free 可选；惰性导入避免 dry-run OOM。
- 输出 LoRA 适配器保存到 `outputs/align/`。

## 6. 评测与导出

- `scripts/eval_harness.py`：默认加载 LoRA（`peft_adapter=outputs/align`），任务 `cmmlu, ceval, hellaswag, winogrande`。
- `scripts/export_lora.py`：支持导出适配器或合并权重。

## 7. 推荐流程（远程执行）

```bash
pip install -r requirements.txt
source scripts/setup_env.sh --dry-run false
python -m scripts.prepare_data --config configs/data.yaml --dry-run false
accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false
accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false
python -m scripts.eval_harness --config configs/eval.yaml --dry-run false
python -m scripts.export_lora --config configs/sft.yaml --dry-run false
```

## 8. 结构与目录

```
project_root/
  data_raw/
    mxode_chinese_instruct/
    dpo_en_zh_20k/
  data_proc/
    sft_train.jsonl
    sft_val.jsonl
    pref_train.jsonl
    pref_val.jsonl
  outputs/
    sft/
    align/
    eval/
    adapters/
  scripts/
  configs/
  docs/
```

## 9. 注意事项

- 使用 HF Token（`huggingface-cli login`）可避免频繁限流。
- 若要离线操作，先将 JSON/JSONL 拷贝到 `data_raw/<dataset>/` 后再运行脚本。
- 微调参数可在 `configs/sft.yaml` 和 `configs/orpo.yaml` 按 4090/5090 显存调节。
