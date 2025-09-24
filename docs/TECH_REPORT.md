# 今晚变更摘要（Changelog）

- 新增数据准备、SFT、ORPO、DPO、评测与导出脚本，全部默认 `--dry-run true` 并符合《LLM_Student_Project_Report.md》流程。
- 引入 `scripts/utils_data.py` 提供语言占比 + 分桶采样器，训练与数据脚本复用，缓解灾难性遗忘。
- 新建配置模板、数据卡模板、测试样例、评测 README 以及 requirements/README/PR 文档骨架。

---

## 1. 目标与约束

- 对齐原始方案：仅使用公开许可数据（OASST1、COIG、COIG-PC-core、SHP、UltraFeedback-binarized、HH-RLHF、PKU-SafeRLHF）。
- 面向个人研究者：单卡（4090/5090）可完成 SFT + 偏好对齐 + 评测全流程。
- 本地阶段只编辑文件，不运行命令；所有脚本默认 dry-run 并显示操作计划。
- 环境约束：LoRA/QLoRA 微调，TRL + PEFT + Accelerate 生态；评测采用 lm-eval-harness；日志默认本地文件（可选 tensorboard/wandb）。

## 2. 数据流水线与过滤细节

- 下载计划：通过 `scripts/prepare_data.py`（读取 `configs/data.yaml`）在 dry-run 下执行目录/配置检查，远程执行后调用 `datasets.load_dataset`。
- 语言比例：`min_cn_ratio=0.3`，拒绝中文比例低于阈值样本；若设置 `--allow-english-fallback` 则英文补充最多 30%。
- 长度约束：使用 `LengthBucket` (0-256 / 257-1024 / 1025-2048 tokens)，上界 2048 防止超长序列。
- 文本归一化：NFKC 兼容写法（去全角空格、压缩空白）；对话拼接后取 SHA256 去重。
- 样本结构校验：脚本内置 OASST/COIG/偏好伪样例，dry-run 时执行 `run_inline_checks()`，确保 parser 稳定。
- 日志示例（注释形式）：记录原始数目、过滤后数目、中文比例与采样统计。
- 数据卡：`data_proc/DATA_CARD.md` 模板预填说明，远程执行后补写真实计数。

## 3. 混合采样与灾难性遗忘防护设计

- `scripts/utils_data.MixedBucketSampler`：
  - 目标语言占比（默认中文 ≥ 70%）。
  - 分桶采样：按源数据集与长度桶轮转取样，减少单一来源主导。
  - 课程式调度：配置中可提供阶段边界与权重（示例在 `configs/data.yaml`），在 SFT 训练脚本中导入使用。
- SFT/对齐脚本在加载 JSON 后使用相同采样器生成再训练计划，保证训练阶段与数据准备阶段策略一致。

## 4. 模型与 LoRA/QLoRA 配置理由

- 基座模型：`Qwen/Qwen2.5-7B-Instruct`（可替代为 Qwen3-4B-Instruct）。中文能力强、显存友好。
- LoRA：目标模块覆盖 q/k/v/o + up/gate/down，`r=16/alpha=16/dropout=0.05` 为 4090 起步值；5090 可提升至 r=32。
- QLoRA：在配置中预留 `qlora.enable`，激活后使用 NF4 + double quant，适合 24GB 环境。
- 梯度策略：勾选 gradient checkpointing、packing/streaming 默认关闭（amp-friendly）。

## 5. 训练与对齐损失函数 / 超参区间

- SFT：
  - `epochs=1.5`、`lr=2e-4`、`warmup_ratio=0.03`、`cosine` 调度、`max_grad_norm=1.0`，与原方案一致。
  - 训练脚本收集 loss、lr、tokens/s、梯度范数；dry-run 输出采样计划。
- ORPO：
  - `beta=0.2`，长度惩罚 `0.02`，中文优先比 `0.7`。
  - 基于 TRL `ORPOTrainer`，使用 pairwise collator。
- DPO：
  - `beta=0.2`，`reference_free=false` 默认；其余设置类似 ORPO。
- 两者均使用相同的 MixedBucketSampler 保障数据分布一致性。

## 6. 指标体系与日志结构

- 训练日志：`outputs/sft/logs/` 与 `outputs/align/logs/`。指标包括 loss、learning_rate、grad_norm、tokens/s；dry-run 阶段不写日志。
- 验证：自动记录 eval_loss（若配置 eval 数据），并保留生成样例的抽样（在 dry-run 中列出提示词）。
- 监控后端：`--log_backend` 支持 none/tensorboard/wandb（默认 none），保证本地无依赖。

## 7. 评测基准与解释

- `scripts/eval_harness.py` 包装 `lm-eval-harness`。
- 默认任务：`cmmlu`、`ceval`（可配置子集）、`hellaswag`、`winogrande`，覆盖中文学科、常识与英文推理。
- 输出：`results.json`（原始）、`results.csv`（扁平化指标表）。dry-run 仅打印计划。

## 8. 复现实验清单与资源预算

1. （远程）安装依赖：`pip install -r requirements.txt`。
2. `source scripts/setup_env.sh --dry-run false`（设置 HF 镜像与缓存）。
3. `python scripts/prepare_data.py --config configs/data.yaml --dry-run false`。
4. `accelerate launch scripts/train_sft.py --config configs/sft.yaml --dry-run false`。
5. `accelerate launch scripts/train_orpo.py --config configs/orpo.yaml --dry-run false` 或 DPO 版本。
6. `python scripts/eval_harness.py --config configs/eval.yaml --dry-run false`。
7. `python scripts/export_lora.py --config configs/sft.yaml --dry-run false`。

> 注：以下命令仅可在远程 GPU 环境执行，示例顺序与上方清单一致。

```bash
# (远程 autodl 执行示例；本地禁止执行)
pip install -r requirements.txt
source scripts/setup_env.sh --dry-run false
python scripts/prepare_data.py --config configs/data.yaml --dry-run false
accelerate launch scripts/train_sft.py  --config configs/sft.yaml  --dry-run false
accelerate launch scripts/train_orpo.py --config configs/orpo.yaml --dry-run false
# 或使用 DPO
accelerate launch scripts/train_dpo.py --config configs/dpo.yaml --dry-run false
python scripts/eval_harness.py --config configs/eval.yaml --dry-run false
python scripts/export_lora.py --config configs/sft.yaml --dry-run false
```

资源：
- 4090（24GB）：SFT 使用 LoRA + bf16 + bs=2/acc=16；ORPO/DPO 同规格。
- 5090（≥32GB）：可提升序列长度至 4096，并增大 batch size。

## 9. 风险与合规

- 公开数据仍可能携带噪声或敏感内容；建议评测后补充红队或人工筛查。
- Anthropic/PKU 数据集需遵守研究许可，禁止商用再分发。
- QLoRA 量化在部分驱动下可能不稳定；如遇精度问题可退回 fp16/bf16 模式。

## 10. 开放问题与替代方案

- Curriculum schedule 暂未在配置中启用默认值，如需动态调权需进一步验证学习曲线；此处维持最保守实现。
- 评测集未覆盖安全性指标（如 AdvBench）；后续可在 `configs/eval.yaml` 扩展任务。
- 训练脚本没有内置 early stopping，仅依赖步长指标；若需可在未来加上 callback。
- 采样器当前使用简单长度估计（字符÷4），若要更精细可引入 tokenizer-based 统计。

## 11. 文件索引与职责表

- `scripts/prepare_data.py`：数据下载、清洗、采样、数据卡模板生成。
- `scripts/utils_data.py`：语言检测、长度分桶、混合采样逻辑。
- `scripts/train_sft.py`：SFT 训练主脚本（LoRA/QLoRA）。
- `scripts/train_orpo.py`：ORPO 偏好对齐训练。
- `scripts/train_dpo.py`：DPO 偏好对齐训练。
- `scripts/eval_harness.py`：lm-eval-harness 包装与结果格式化。
- `scripts/export_lora.py`：LoRA 适配器导出/合并。
- `configs/*.yaml`：各阶段参数模板。
- `tests/`：pytest 静态样例（默认 skip）。
- `outputs/eval/README.md`：评测执行指南。
- `README.md`：整体工作流与排错清单。
- `requirements.txt`：依赖与版本约束（远程安装）。
