# ⚠️ 本仓库在本地仅允许编辑文件，所有命令须在远程（AutoDL 等）执行

本项目为公开数据驱动的 LLM 微调/对齐流水线骨架。设计原则：

- 本地（或无 GPU 环境）仅编辑代码与文档，不运行任何命令。
- 远程 GPU 主机按文档提供的命令顺序手动执行。
- 全部脚本默认 `--dry-run true`，需在远程显式传 `--dry-run false` 才会操作数据或模型。

## 目录概览

- `scripts/`：数据准备、训练、对齐、评测、导出脚本。
- `configs/`：4090/5090 友好的 YAML 配置模板。
- `docs/`：方案说明与技术报告。
- `data_raw/`、`data_proc/`、`outputs/`：远程执行时产生的目录（本地留空）。
- `tests/`：pytest 样例（均使用 `@pytest.mark.skip` 仅作静态校验）。

## 远程执行流程（示例）

> 注：需在远程执行；本地禁止运行以下命令

```bash
# (远程 autodl 执行示例；本地禁止执行)
pip install -r requirements.txt
source scripts/setup_env.sh --dry-run false
python scripts/prepare_data.py --config configs/data.yaml --dry-run false
accelerate launch scripts/train_sft.py  --config configs/sft.yaml  --dry-run false
accelerate launch scripts/train_orpo.py --config configs/orpo.yaml --dry-run false  # 或 train_dpo.py
python scripts/eval_harness.py --config configs/eval.yaml --dry-run false
python scripts/export_lora.py --config configs/sft.yaml --dry-run false
```

## 常见坑位与排错

- 显存不足：调低 `per_device_train_batch_size` 或启用 `qlora.enable`。
- 驱动/库冲突：在远程新建虚拟环境后执行 `pip install -r requirements.txt`。
- 数据缺失：确认 `data_proc/sft_*.jsonl` 与 `pref_*.jsonl` 是否由 `prepare_data.py` 生成。
- 评测超时：减少 `configs/eval.yaml` 中任务或降低 `lm_eval.batch_size`。
- 日志后端异常：若未安装 tensorboard/wandb，则保持 `log_backend=none`。

### 使用本地 COIG 数据，避免再次下载

- 将从网页或 hf-cli 下载的 COIG 文件（如 `*.jsonl`/`*.json`）放到项目目录 `data_raw/coig/` 下。
- 本项目已支持本地优先：若检测到 `data_raw/coig/*.jsonl|*.json`，`prepare_data` 会直接用本地文件，不再联网下载。
- 命令示例（只处理 COIG 的 SFT，并跳过 OASST）：
  - `python -m scripts.prepare_data --config configs/data.yaml --stage sft --sft-mix coig=1.0,oasst1=0.0 --dry-run false`
- 如仅想先跑 OASST：把 `coig=0.0`，后续网络可用或本地文件就绪后再把权重设回 >0 重跑 SFT。

### 数据质量与采样提示

- 质量过滤开关：`configs/data.yaml -> general.{min_tokens,max_tokens,max_repetition}`，中文重复度按字符 n-gram 计算。
- 语言占比：默认中文 ≥ 70%；若中文样本不足且 `allow_english_fallback=false`，采样器会缩量并给出 WARN 日志。
- 数据混合：通过 `sft.mix` 与 `preference.mix` 控制来源权重。

### 评测注意事项

- 适配器评测：`configs/eval.yaml -> model.peft_adapter` 指向 `outputs/align`；如评测合并权重，请清空该项并将 `base_model` 指向合并目录。
- 任务集合：默认 `cmmlu, ceval, hellaswag, winogrande`；如报 task 不存在，请对齐 `lm-eval` 版本或减少任务。

## 贡献与扩展

- 新增数据集或训练方法前，请更新 `docs/LLM_Student_Project_Report.md` 与 `docs/TECH_REPORT.md`。
- 所有脚本必须提供 `--dry-run` 开关，并在 `tests/` 目录补充静态导入测试。
