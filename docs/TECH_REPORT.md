# 今晚变更摘要（Changelog）

- 切换数据源：SFT 采用 `Mxode/Chinese-Instruct`，偏好对齐采用 `llamafactory/DPO-En-Zh-20k`。
- `scripts/prepare_data.py` 支持本地 JSON/JSONL（`data_raw/mxode_chinese_instruct/`）优先加载，避免重复下载。
- 清洗与采样逻辑更新：保留中文占比 ≥0.3，字符级 n-gram 去重，权重为 0 的数据集将被跳过。
- 文档、配置与 README 均同步到新的 K12 中文教学场景。

---

## 1. 目标与约束

- 场景：K12 教育问答与课堂辅导，保持泛化能力。
- 资源：单卡 4090/5090，可在 4bit QLoRA 或 LoRA 条件下训练。
- 数据：全部来自公开数据集，无私有数据；允许本地缓存/离线下载。
- 训练阶段：SFT → 偏好对齐（ORPO/DPO）→ 评测 → 导出，保持灾难性遗忘可控。

## 2. 数据流水线与过滤

- 源数据：
  - `Mxode/Chinese-Instruct`（中文教学与知识问答，字段 `instruction`/`input`/`output`）。
  - `llamafactory/DPO-En-Zh-20k`（中英双语偏好对，字段 `prompt`/`chosen`/`rejected`）。
- 本地优先：将离线下载的 JSON/JSONL 文件放入 `data_raw/<dataset_key>/` 即可直接使用。
- 清洗规则：
  - `normalize_text` 做 Unicode NFKC、空白折叠。
  - 字符级 n-gram 重复度阈值 0.6。
  - Token 数 [8, 2048]。
  - 中文占比 ≥ 0.3（可用 `--allow-english-fallback` 放宽）。
  - SHA256 去重（按对话拼接文本计算）。
- 输出格式：
  - SFT：`messages` 数组，user/assistant 轮次，保存至 `data_proc/sft_{train,val,test}.jsonl`。
  - 偏好：`{prompt, chosen, rejected, source, hash}` 保存至 `data_proc/pref_{train,val,test}.jsonl`。

## 3. 混合与采样策略

- `MixedBucketSampler`：按长度桶与语种配额（中文 ≥70%）进行轮转采样。
- 权重控制：`configs/data.yaml -> sft.mix` / `preference.mix`，权重为 0 的数据集将跳过加载。
- 课程式混合（可选）：支持在 SFT 阶段按短 → 长指令权重切换。
- 采样器提供 WARN 日志，当中文样本量不足时自动缩量并提示。

## 4. 模型与 LoRA/QLoRA 配置

- 基座：`Qwen/Qwen2.5-7B-Instruct`（可换成 3B/14B 版本）。
- LoRA 设置：target modules = q/k/v/o + up/gate/down，r=16, α=16, dropout=0.05。
- QLoRA：可启用 NF4 量化（`qlora.enable: true`）以节省显存。
- 梯度策略：gradient checkpointing + 数据 packing，学习率 2e-4（SFT），warmup 3%。

## 5. 偏好对齐（ORPO/DPO）

- 数据：`llamafactory/DPO-En-Zh-20k`。
- ORPO：β=0.2，长度惩罚 0.02，prefer_chinese_ratio=0.7。
- DPO：β=0.2，reference_free 可选；默认 LoRA 细调。
- 训练时同样使用 MixedBucketSampler，确保中文优先并控制长度。

## 6. 指标与日志

- 训练：loss、learning_rate、grad_norm、tokens/s，默认写入 `outputs/<stage>/logs/`。
- 验证（若设定 eval 文件）：记录 eval_loss、生成长度、重复度（简化统计）。
- 可选后端：tensorboard/wandb；默认 `none`。

## 7. 评测基准

- `lm-eval-harness`：`cmmlu`、`ceval`、`hellaswag`、`winogrande`。
- `configs/eval.yaml` 默认使用 LoRA 适配器（`peft_adapter=outputs/align`）。如合并权重，清空 `peft_adapter` 并改 `base_model` 为合并目录。

## 8. 复现实验清单

1. `pip install -r requirements.txt`
2. `source scripts/setup_env.sh --dry-run false`
3. `python -m scripts.prepare_data --config configs/data.yaml --dry-run false`
4. `accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false`
5. `accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false`（或 `train_dpo`）
6. `python -m scripts.eval_harness --config configs/eval.yaml --dry-run false`
7. `python -m scripts.export_lora --config configs/sft.yaml --dry-run false`

## 9. 风险与合规

- 公共数据仍可能包含噪声或敏感内容，需结合人工抽查。
- Hugging Face 数据集有速率限制，建议使用 Token 登录或本地缓存。
- 偏好数据包含中英双语回答，若只关注中文，可在清洗阶段额外筛选。

## 10. 开放问题与替代方案

- 如需更细粒度的 K12 标签，可在 SFT 阶段基于 `subject` 字段（若提供）做筛选。
- 可追加指令增强数据（如 Math、Chinese composition）提升目标场景覆盖。
- 如果要支持完全离线训练，可进一步拓展 `scripts/prepare_data.py` 在偏好阶段也支持本地 JSON/JSONL 优先加载。

## 文件索引

- `scripts/prepare_data.py`：数据下载/清洗/采样（本地优先、权重跳过、质量过滤）。
- `scripts/train_sft.py`：LoRA/QLoRA SFT（带 formatting_func、采样器干跑）。
- `scripts/train_orpo.py`、`scripts/train_dpo.py`：偏好对齐训练（惰性导入，适配新数据）。
- `scripts/eval_harness.py`：lm-eval-harness 封装（支持 LoRA 适配器）。
- `scripts/export_lora.py`：导出 LoRA/合并权重。
- `configs/`：实验模板（4090/5090）。
- `docs/TECH_REPORT.md`：技术报告（本文件）。
