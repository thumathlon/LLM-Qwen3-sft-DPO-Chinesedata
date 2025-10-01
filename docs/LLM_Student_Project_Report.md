# LLM ���Ľ�ѧ����΢������뷽��

## 1. Ŀ��������

- �������� K12 ���Ľ�ѧ�������ʴ�/����ģ�ͣ���˷���������
- SFT ���ݼ���`Mxode/Chinese-Instruct`��instruction/input/output����
- DPO ���ݼ���`llamafactory/DPO-En-Zh-20k`��prompt/chosen/rejected����
- ���ݴ�ţ�
  - `data_raw/mxode_chinese_instruct/`
  - `data_raw/dpo_en_zh_20k/`
  - ����Ҫ�������У��ɽ� JSON/JSONL ֱ�ӷ�������Ŀ¼��

## 2. ��ϴ��淶��

- SFT��
  - ��ȡ instruction/input/output���ϲ�Ϊ user �� assistant ���� `messages`��
  - �������ˣ����� [8, 2048]���ַ��� 4-gram �ظ��� ��0.6��
  - ����ռ�� ��30%��������� `allow_english_fallback=false` ������
  - SHA256 ȥ�ء�
- DPO��
  - �����ֶ� `prompt`��`chosen`��`rejected` �Ķ��ֱ�����`chosen_response` �ȣ���
  - ������ȥ�ع���ͬ SFT��

## 3. ���ݽű���`scripts/prepare_data.py`��

- `SFT_SPECS` �� `PREF_SPECS` ָ�������ݼ���
- ��� `data_raw/<dataset>` ���б��� JSON/JSONL��������ʹ�ñ����ļ���
- `MixedBucketSampler`��������� 70%������Ͱ��ת��Ȩ��Ϊ 0 �����ݼ����ᱻ���ء�
- Dry-run ������ݼƻ�����ʵ�������� `data_proc/sft_*.jsonl` �� `pref_*.jsonl`��

## 4. SFT ѵ����`scripts/train_sft.py`��

- ������`Qwen/Qwen2.5-7B-Instruct`��LoRA r=16 ��=16��
- ֧�� QLoRA��4bit�������� 24GB �Դ档
- `formatting_func` �Զ�ʹ�� tokenizer chat template ���� `messages`��
- dry-run �ṩ�����ƻ����������ƣ���ʵ���б��浽 `outputs/sft/`��

## 5. ƫ�ö��루`scripts/train_orpo.py` / `scripts/train_dpo.py`��

- ���ݣ�ȫ������ `llamafactory/DPO-En-Zh-20k`��
- ORPO����=0.2��length_penalty=0.02��prefer_chinese_ratio=0.7��
- DPO����=0.2��reference_free ��ѡ�����Ե������ dry-run OOM��
- ��� LoRA ���������浽 `outputs/align/`��

## 6. �����뵼��

- `scripts/eval_harness.py`��Ĭ�ϼ��� LoRA��`peft_adapter=outputs/align`�������� `cmmlu, ceval, hellaswag, winogrande`��
- `scripts/export_lora.py`��֧�ֵ�����������ϲ�Ȩ�ء�

## 7. �Ƽ����̣�Զ��ִ�У�

```bash
pip install -r requirements.txt
source scripts/setup_env.sh --dry-run false
python -m scripts.prepare_data --config configs/data.yaml --dry-run false
accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false
accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false
python -m scripts.eval_harness --config configs/eval.yaml --dry-run false
python -m scripts.export_lora --config configs/sft.yaml --dry-run false
```

## 8. �ṹ��Ŀ¼

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

## 9. ע������

- ʹ�� HF Token��`huggingface-cli login`���ɱ���Ƶ��������
- ��Ҫ���߲������Ƚ� JSON/JSONL ������ `data_raw/<dataset>/` �������нű���
- ΢���������� `configs/sft.yaml` �� `configs/orpo.yaml` �� 4090/5090 �Դ���ڡ�
