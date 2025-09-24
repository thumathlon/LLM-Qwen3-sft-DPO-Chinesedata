+# 变更概览
+
+- 重写数据准备/训练/对齐/评测/导出脚本，均默认 `--dry-run true` 并与 `configs/*.yaml` 配置对齐。
+- 新增 `scripts/utils_data.py`（语言分桶采样、n-gram 重复率、哈希去重等工具）供各阶段共享。
+- 更新配置模板、技术报告、README、数据卡模板及 pytest 静态测试样例。
+- 增强 `setup_env.sh` 以支持 dry-run，确保远程执行前可审查计划。
+
+# 风险点
+
+- `MixedBucketSampler` 的分桶/权重逻辑需重点审查，确认中文占比和课程式配置是否符合预期。
+- 训练脚本的 LoRA/QLoRA 参数与日志管线依赖 transformers/trl 等版本，需在远程验证兼容性。
+- `prepare_data.py` 在非 dry-run 下会批量写入 `data_proc/`，建议先 dry-run 查看计划再执行。
+
+# 复现路径（远程执行）
+
+```
+# (远程 autodl 执行示例；本地禁止执行)
+pip install -r requirements.txt
+source scripts/setup_env.sh --dry-run false
+python scripts/prepare_data.py --config configs/data.yaml --dry-run false
+accelerate launch scripts/train_sft.py  --config configs/sft.yaml  --dry-run false
+accelerate launch scripts/train_orpo.py --config configs/orpo.yaml --dry-run false  # 或 train_dpo.py
+python scripts/eval_harness.py --config configs/eval.yaml --dry-run false
+python scripts/export_lora.py --config configs/sft.yaml --dry-run false
+```
+
+# 评审检查清单
+
+- [ ] dry-run 模式能否覆盖主要路径（数据准备/训练/评测/导出）。
+- [ ] `MixedBucketSampler` 是否满足中文 ≥70% + 长度桶轮转策略。
+- [ ] YAML 配置与脚本参数一致（含 `--log-backend`、LoRA/QLoRA 设置）。
+- [ ] 文档是否充分标注“仅远程执行”，避免本地误操作。

+# 变更统计
+
+- 预计影响文件：约 24 个
+- 估算 diff 规模：~2500 行新增 / ~450 行删除（基于编辑器统计）

