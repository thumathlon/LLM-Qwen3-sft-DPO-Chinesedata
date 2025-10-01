# ������ժҪ��Changelog��

- �л�����Դ��SFT ���� `Mxode/Chinese-Instruct`��ƫ�ö������ `llamafactory/DPO-En-Zh-20k`��
- `scripts/prepare_data.py` ֧�ֱ��� JSON/JSONL��`data_raw/mxode_chinese_instruct/`�����ȼ��أ������ظ����ء�
- ��ϴ������߼����£���������ռ�� ��0.3���ַ��� n-gram ȥ�أ�Ȩ��Ϊ 0 �����ݼ�����������
- �ĵ��������� README ��ͬ�����µ� K12 ���Ľ�ѧ������

---

## 1. Ŀ����Լ��

- ������K12 �����ʴ�����ø��������ַ���������
- ��Դ������ 4090/5090������ 4bit QLoRA �� LoRA ������ѵ����
- ���ݣ�ȫ�����Թ������ݼ�����˽�����ݣ������ػ���/�������ء�
- ѵ���׶Σ�SFT �� ƫ�ö��루ORPO/DPO���� ���� �� ���������������������ɿء�

## 2. ������ˮ�������

- Դ���ݣ�
  - `Mxode/Chinese-Instruct`�����Ľ�ѧ��֪ʶ�ʴ��ֶ� `instruction`/`input`/`output`����
  - `llamafactory/DPO-En-Zh-20k`����Ӣ˫��ƫ�öԣ��ֶ� `prompt`/`chosen`/`rejected`����
- �������ȣ����������ص� JSON/JSONL �ļ����� `data_raw/<dataset_key>/` ����ֱ��ʹ�á�
- ��ϴ����
  - `normalize_text` �� Unicode NFKC���հ��۵���
  - �ַ��� n-gram �ظ�����ֵ 0.6��
  - Token �� [8, 2048]��
  - ����ռ�� �� 0.3������ `--allow-english-fallback` �ſ���
  - SHA256 ȥ�أ����Ի�ƴ���ı����㣩��
- �����ʽ��
  - SFT��`messages` ���飬user/assistant �ִΣ������� `data_proc/sft_{train,val,test}.jsonl`��
  - ƫ�ã�`{prompt, chosen, rejected, source, hash}` ������ `data_proc/pref_{train,val,test}.jsonl`��

## 3. ������������

- `MixedBucketSampler`��������Ͱ������������ ��70%��������ת������
- Ȩ�ؿ��ƣ�`configs/data.yaml -> sft.mix` / `preference.mix`��Ȩ��Ϊ 0 �����ݼ����������ء�
- �γ�ʽ��ϣ���ѡ����֧���� SFT �׶ΰ��� �� ��ָ��Ȩ���л���
- �������ṩ WARN ��־������������������ʱ�Զ���������ʾ��

## 4. ģ���� LoRA/QLoRA ����

- ������`Qwen/Qwen2.5-7B-Instruct`���ɻ��� 3B/14B �汾����
- LoRA ���ã�target modules = q/k/v/o + up/gate/down��r=16, ��=16, dropout=0.05��
- QLoRA�������� NF4 ������`qlora.enable: true`���Խ�ʡ�Դ档
- �ݶȲ��ԣ�gradient checkpointing + ���� packing��ѧϰ�� 2e-4��SFT����warmup 3%��

## 5. ƫ�ö��루ORPO/DPO��

- ���ݣ�`llamafactory/DPO-En-Zh-20k`��
- ORPO����=0.2�����ȳͷ� 0.02��prefer_chinese_ratio=0.7��
- DPO����=0.2��reference_free ��ѡ��Ĭ�� LoRA ϸ����
- ѵ��ʱͬ��ʹ�� MixedBucketSampler��ȷ���������Ȳ����Ƴ��ȡ�

## 6. ָ������־

- ѵ����loss��learning_rate��grad_norm��tokens/s��Ĭ��д�� `outputs/<stage>/logs/`��
- ��֤�����趨 eval �ļ�������¼ eval_loss�����ɳ��ȡ��ظ��ȣ���ͳ�ƣ���
- ��ѡ��ˣ�tensorboard/wandb��Ĭ�� `none`��

## 7. �����׼

- `lm-eval-harness`��`cmmlu`��`ceval`��`hellaswag`��`winogrande`��
- `configs/eval.yaml` Ĭ��ʹ�� LoRA ��������`peft_adapter=outputs/align`������ϲ�Ȩ�أ���� `peft_adapter` ���� `base_model` Ϊ�ϲ�Ŀ¼��

## 8. ����ʵ���嵥

1. `pip install -r requirements.txt`
2. `source scripts/setup_env.sh --dry-run false`
3. `python -m scripts.prepare_data --config configs/data.yaml --dry-run false`
4. `accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false`
5. `accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false`���� `train_dpo`��
6. `python -m scripts.eval_harness --config configs/eval.yaml --dry-run false`
7. `python -m scripts.export_lora --config configs/sft.yaml --dry-run false`

## 9. ������Ϲ�

- ���������Կ��ܰ����������������ݣ������˹���顣
- Hugging Face ���ݼ����������ƣ�����ʹ�� Token ��¼�򱾵ػ��档
- ƫ�����ݰ�����Ӣ˫��ش���ֻ��ע���ģ�������ϴ�׶ζ���ɸѡ��

## 10. �����������������

- �����ϸ���ȵ� K12 ��ǩ������ SFT �׶λ��� `subject` �ֶΣ����ṩ����ɸѡ��
- ��׷��ָ����ǿ���ݣ��� Math��Chinese composition������Ŀ�곡�����ǡ�
- ���Ҫ֧����ȫ����ѵ�����ɽ�һ����չ `scripts/prepare_data.py` ��ƫ�ý׶�Ҳ֧�ֱ��� JSON/JSONL ���ȼ��ء�

## �ļ�����

- `scripts/prepare_data.py`����������/��ϴ/�������������ȡ�Ȩ���������������ˣ���
- `scripts/train_sft.py`��LoRA/QLoRA SFT���� formatting_func�����������ܣ���
- `scripts/train_orpo.py`��`scripts/train_dpo.py`��ƫ�ö���ѵ�������Ե��룬���������ݣ���
- `scripts/eval_harness.py`��lm-eval-harness ��װ��֧�� LoRA ����������
- `scripts/export_lora.py`������ LoRA/�ϲ�Ȩ�ء�
- `configs/`��ʵ��ģ�壨4090/5090����
- `docs/TECH_REPORT.md`���������棨���ļ�����
