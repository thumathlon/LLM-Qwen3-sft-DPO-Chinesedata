# 面向个人研究者的开源 LLM 微调与对齐项目方案（仅公开数据 + 全自动处理）

版本：v1.0  ·  适配环境：单卡 4090/5090  ·  许可合规：是（仅公开数据）

---

## 目录

1. 项目目标与产出
2. 数据获取与处理
   2.1 数据集选择（SFT 与偏好/安全）
   2.2 下载策略与镜像
   2.3 目录结构与文件规范
   2.4 清洗与规范化流程
   2.5 规模与数据配比
3. 模型与训练设计
   3.1 基座模型与 LoRA 配置
   3.2 第一阶段：SFT 训练
   3.3 第二阶段：偏好对齐（ORPO / DPO）
   3.4 推理与 LoRA 导出
4. 评测与展示
   4.1 自动化评测
   4.2 人工小测面板
   4.3 安全基线检查
5. 执行路径与资源预算
6. 合规性与风险说明
7. 复现 Checklist

---

## 1. 项目目标与产出

- 主要目标：在单卡 GPU（4090/5090）上，使用开源中文能力强的基座模型完成端到端实验，掌握数据准备、SFT 微调与偏好对齐的完整流程。
- 关键约束：不自建任何偏好/对齐数据对；全流程仅使用公开数据集并自动处理。
- 最终产出：
  - 可复现的 LoRA 适配器权重（便于展示与复现）。
  - 完整的数据处理与训练报告（本文档，可直接放入简历/仓库）。

---

## 2. 数据获取与处理

### 2.1 数据集选择（SFT 与偏好/安全）

- 指令/对话（SFT）：
  - OpenAssistant/oasst1（Apache-2.0）：多轮对话树，含中英文，覆盖对话结构与角色分工。
  - BAAI/COIG（Apache-2.0）：高质量中文指令与写作类数据，增强中文任务执行与写作。

- 偏好/对齐（Pairwise/Comparisons）：全部公开、可直接使用，无需人工标注。
  - BAAI/COIG-PC-core（Apache-2.0）：中文偏好对（核心子集）。
  - Stanford/SHP（SHP）开源许可：问答偏好（英文为主，可混合少量提升泛化）。
  - OpenBMB/UltraFeedback（binarized 版本）：多维反馈二值化偏好对。
  - Anthropic HH-RLHF（有较宽松研究许可）：有害/有帮助偏好对，用于对齐风格与安全边界。
  - PKU-SafeRLHF（研究许可）：安全拒答相关偏好对（中文/英文混合）。

说明：可按中文占比优先，辅以英文偏好对提高稳健性；所有集均为公开数据源，不涉及自建。

### 2.2 下载策略与镜像

- 优先使用中国区镜像：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`。
- 如镜像/网络不可用：在可联网机器上预下载 `.jsonl/.parquet`，放入 `data_raw/` 对应子目录，后续脚本从本地加载。
- 推荐工具：`datasets`（Python），`huggingface_hub`。

### 2.3 目录结构与文件规范

```
project_root/
  data_raw/
    oasst1/               # 原始数据（HF cache 或本地导出）
    coig/
    coig_pc_core/
    shp/
    ultrafeedback/
    hh_rlhf/
    saferlhf/
  data_proc/
    sft_train.jsonl       # 统一 messages 格式
    sft_val.jsonl
    pref_train.jsonl      # 统一 {prompt, chosen, rejected}
    pref_val.jsonl
  outputs/
    sft/                  # SFT checkpoints & logs
    align/                # ORPO/DPO checkpoints & logs
    adapters/             # 导出的 LoRA 适配器
  scripts/                # 数据与训练脚本（可选）
  docs/
    LLM_Student_Project_Report.md
```

### 2.4 清洗与规范化流程

1) OASST-1 树重建 → 多轮对话：
- 原始为扁平消息（`message_id`, `parent_id`）。根据父子关系从根到叶生成对话路径。
- 角色映射：`prompter → user`，`assistant → assistant`；忽略 `system`/其他非必要角色或并入首轮。
- 输出 messages 标准格式：

```json
{"id":"o1-0001","source":"OASST1","messages":[
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."}
]}
```

2) COIG 规范化 → 两轮对话：
- 将 `instruction`→user，`output`→assistant，组成两轮。

3) 文本归一化：
- NFKC 规范化、空白折叠、合并空行、去 BOM 与控制字符。

4) 语言与长度过滤：
- 中文比例阈值：≥ 30%（基于汉字占比的启发式）。
- 长度阈值：去除过短（<8 字）与过长（>4096 字符）样本；控制多轮总 token 不超最大长度限制。

5) 精确去重：
- 将一条对话的所有轮次文本拼接后做 `SHA256`，哈希去重，保留首个出现样本。

6) 安全与内容过滤（轻量）：
- 关键词启发式（暴恐、露骨、个人隐私等）筛除；必要时叠加正则黑名单。

7) 划分与日志：
- 80/10/10 划分为 train/val/test（test 可选仅用于复核）。
- 记录来源、过滤规则、各步骤样本数与丢弃率，生成数据卡（data card）。

8) 偏好数据统一化：
- 目标格式为 pairwise：`{"prompt","chosen","rejected","source"}`。
- 适配各数据集字段名（如 `question/response_a/response_b/label` → 归并到 chosen/rejected）。
- 对中文偏好集优先保留；英文样本保留 20–40% 作为稳健性增强（可配置）。

### 2.5 规模与数据配比

- SFT：4–6 万对话样本（OASST-1 + COIG）。
- 偏好/对齐：6–12 万 pairwise（优先 COIG-PC-core + UltraFeedback-binarized；再混入 SHP/HH/SafeRLHF）。
- 语言混合：中文占比目标 ≥ 70%；其余为英文，以增强对齐稳健性与拒答风格学习。

---

## 3. 模型与训练设计

### 3.1 基座模型与 LoRA 配置

- 基座模型：Qwen3-4B-Instruct（如不可用，回退到 Qwen2.5-3B/7B-Instruct）。
- LoRA 目标模块：q/k/v/o + up/gate/down。
- 推荐配置：`r=16/32`，`alpha=16/32`，`dropout=0.05`，`bias=none`。
- 精度与显存：优先 BF16（更稳定）；显存紧张时启用 QLoRA（4bit `bnb_4bit_quant_type=fp4`）。

### 3.2 第一阶段：SFT 训练

- 训练框架：TRL（`SFTTrainer`）+ PEFT（LoRA/QLoRA）+ bitsandbytes。
- 最大序列长度：
  - 4090：`max_seq_len=2048`
  - 5090：`max_seq_len=4096`
- 批次与累积：
  - 4090：`per_device_train_batch_size=1–2`，`gradient_accumulation_steps=16–32`
  - 5090：`per_device_train_batch_size=2–4`，`gradient_accumulation_steps=16–32`
- 优化与策略：`lr=2e-4`，调度 `cosine`，`warmup_ratio=0.03`，`weight_decay=0.01`。
- 训练技巧：开启 gradient checkpointing、`flash_attention`（若可用）、packing（样本拼接提升吞吐）。
- 轮次：1–3 epoch（建议从 1–1.5 起步，观察验证损失与输出质量）。

示例（命令行伪代码）：

```
accelerate launch train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file data_proc/sft_train.jsonl \
  --validation_file data_proc/sft_val.jsonl \
  --lora_r 16 --lora_alpha 16 --lora_dropout 0.05 \
  --max_seq_length 2048 --bf16 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --num_train_epochs 1.5 --logging_steps 50 --save_steps 1000 \
  --output_dir outputs/sft
```

### 3.3 第二阶段：偏好对齐（ORPO / DPO）

- 目标：学习偏好次序与安全拒答风格，优化输出长度与可用性。
- 数据：仅使用公开偏好对（COIG-PC-core、UltraFeedback-binarized、SHP、HH-RLHF、PKU-SafeRLHF）。
- 方法选择：
  - ORPO：实现简单、稳定，单模型前向；推荐起步。
  - DPO：需参考模型对比（隐式 KL），可作为对照实验。
- 训练设置（推荐起点）：
  - ORPO：`orpo_alpha=0.1–0.3`，`lr=5e-5–1e-4`，`epoch=1`。
  - DPO：`beta=0.1–0.3`，`lr=5e-5–1e-4`，`epoch=1`。
  - 长度正则：对 `chosen/rejected` 长度差加入轻度约束，避免“越长越好”。
  - 混合比例：中文偏好集 : 英文偏好集 ≈ 7 : 3（可据验证集调整）。

示例（命令行伪代码）：

```
accelerate launch train_orpo.py \
  --model_name_or_path outputs/sft \
  --train_file data_proc/pref_train.jsonl \
  --validation_file data_proc/pref_val.jsonl \
  --max_seq_length 2048 --bf16 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 8e-5 --num_train_epochs 1 \
  --orpo_alpha 0.2 --logging_steps 50 --save_steps 1000 \
  --output_dir outputs/align
```

### 3.4 推理与 LoRA 导出

- 推理：使用基座 + LoRA 适配器进行生成；或将 LoRA 合并权重导出用于纯推理部署。
- 示例（伪代码）：

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "outputs/align")  # 或 outputs/sft

prompt = "写一段关于春天的短诗，4 行。"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 4. 评测与展示

### 4.1 自动化评测

- 工具：`lm-eval-harness`（可离线运行）
- 建议任务（中文优先）：`cmmlu` 子集、`ceval` 验证子集、`hellaswag`、`winogrande`；数学可选 `gsm8k` dev。
- 示例命令（伪代码）：

```
lm-eval --model hf-causal --model_args \
  pretrained=Qwen/Qwen2.5-7B-Instruct,peft=outputs/align,trust_remote_code=True \
  --tasks cmmlu,ceval-valid-lite,hellaswag \
  --batch_size 8 --device cuda:0 --output_path outputs/eval
```

### 4.2 人工小测面板

- 主题：一般问答 / 多轮对话 / 中文写作（记叙/说明/议论）/ 结构化输出（列表/JSON）。
- 指标：遵从度（0–5）、平均输出长度、格式正确率、失败类型归类（偏题、事实错误、冗长等）。
- 规模：20–30 条，每次训练后复用对比（SFT vs 对齐后）。

### 4.3 安全基线检查

- 采用 SafeRLHF/HH-RLHF 的红队提示子集，统计拒答率与越狱率。
- 简单规则层：加入推理时敏感词拒答（可选）以降低误触发风险。

---

## 5. 执行路径与资源预算

- 第 1 天：环境配置（`transformers`/`trl`/`peft`/`datasets`/`bitsandbytes`）→ 数据下载 → 原始数据验收。
- 第 2 天：数据清洗与统一（`data_proc/sft_*.jsonl`、`pref_*.jsonl`）→ 统计日志与数据卡。
- 第 3 天：SFT 训练（1–1.5 epoch）→ 保存 `outputs/sft`。
- 第 4 天：ORPO/DPO 训练（1 epoch）→ 保存 `outputs/align`。
- 第 5 天：评测（`lm-eval-harness` + 人工小测）→ 记录结果。
- 第 6 天：调参与复跑（可选：增大偏好数据比例或调 `alpha/beta`）。
- 第 7 天：导出 LoRA 适配器与整理报告。

显存参考：
- 4090（24GB）：QLoRA 4bit 更稳妥；`bs=1–2`，`acc=16–32`。
- 5090（更大显存/更快）：BF16 + LoRA；`bs=2–4`，`acc=16–32`。

---

## 6. 合规性与风险说明

- 许可：所列数据集均为公开许可（Apache/MIT/研究许可等），需遵循各仓库 LICENSE；避免二次分发可能受限的变体。
- 隐私：过滤明显的个人隐私字段；不采集非公开个人数据。
- 模型使用：遵循基座模型使用条款；导出的 LoRA 仅在研究与合规范围内使用。

---

## 7. 复现 Checklist

- [ ] 设置镜像与缓存：`HF_ENDPOINT`、`HF_HOME`。
- [ ] 下载并验收原始数据：`data_raw/` 各子目录存在且大小合理。
- [ ] 跑通清洗脚本，生成：`data_proc/sft_{train,val}.jsonl`、`pref_{train,val}.jsonl`。
- [ ] SFT 训练完成：`outputs/sft/` 有最佳 checkpoint；验证损失稳定。
- [ ] 对齐训练完成：`outputs/align/` 有最佳 checkpoint；偏好损失下降。
- [ ] 推理验收：示例提示在本地可生成、格式正确。
- [ ] 评测报告：`outputs/eval/` 与人工小测结果记录完整。

---

## 8. 数据准备脚本与操作说明

- **环境初始化**
  - （可选）`python -m venv .venv && source .venv/bin/activate`
  - `pip install -U datasets huggingface_hub`（后续训练阶段再补装 `transformers/trl/peft` 等）
  - `source scripts/setup_env.sh`
- **运行数据流水线**
  - `python scripts/prepare_data.py --stage all`
  - 仅跑指令微调数据：`python scripts/prepare_data.py --stage sft`
  - 仅跑偏好数据：`python scripts/prepare_data.py --stage pref`
- **常用参数（均有默认）**
  - `--sft-max-samples` / `--pref-max-samples`：控制采样规模（默认 6 万 / 12 万）
  - `--sft-mix` / `--pref-mix`：按 `key=weight` 设置数据配比（如 `oasst1=0.4,coig=0.6`）
  - `--min-cn-ratio`：中文字符比例阈值（默认 0.3），配合 `--allow-english-fallback` 调整英文占比
- **输出位置**
  - 指令数据：`data_proc/sft_{train,val,test}.jsonl`
  - 偏好数据：`data_proc/pref_{train,val,test}.jsonl`
  - 原始缓存：`data_raw/<dataset_key>/`
- **日志与复现**
  - 脚本默认输出过滤与采样统计，便于写入数据卡
  - 再次运行会重用缓存；若需全量重跑，删除 `data_raw/` 或指定新的 `--seed`

---

附注（与原始草案的关键改动）：
- 移除“自建小规模中文偏好对”的步骤，统一改为使用公开偏好数据（COIG-PC-core、SHP、UltraFeedback-binarized、HH-RLHF、PKU-SafeRLHF）。
- 所有数据处理环节均可脚本化与自动化，无需人工标注或人工筛选。
- 明确了中文优先的混合策略与长度正则，保证结果更可控、更易复现。
