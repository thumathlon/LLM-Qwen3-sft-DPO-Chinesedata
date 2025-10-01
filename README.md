﻿# 椤圭洰璇存槑锛氭暟鎹鐞?+ SFT + DPO/ORPO 鍏ㄦ祦绋?
鏈粨搴撴彁渚涗粠鏁版嵁鍑嗗銆佺洃鐫ｅ井璋冿紙SFT锛夊埌鍋忓ソ瀵归綈锛圖PO/ORPO锛夌殑瀹屾暣娴佹按绾裤€傛墍鏈夎剼鏈粯璁や互 dry-run 鏂瑰紡杩愯锛堜笉涓嬭浇/涓嶅啓鍏ワ級锛屽湪杩滅▼ GPU 鐜灏?`--dry-run false` 鎵嶈繘琛岀湡瀹炲鐞嗕笌璁粌銆?
- 鏁版嵁闆嗭細
  - SFT锛歁xode/Chinese-Instruct
  - DPO锛歭lamafactory/DPO-En-Zh-20k
- 涓昏浜х墿锛?  - `data_proc/sft_{train,val,test}.jsonl`锛堝瓧娈碉細`messages`锛?  - `data_proc/pref_{train,val,test}.jsonl`锛堝瓧娈碉細`prompt`銆乣chosen`銆乣rejected`锛?
## 鐩綍缁撴瀯
- `scripts/`锛氭暟鎹噯澶囥€佽缁冦€佽瘎娴嬨€佸鍑虹瓑鑴氭湰
- `configs/`锛氱ず渚?YAML 閰嶇疆锛?090/5090 鍙嬪ソ锛?- `docs/`锛氭姤鍛婁笌璇存槑锛堟柊澧?`docs/DATA_PROCESSING.md` 璇﹁В鏁版嵁澶勭悊锛?- `data_raw/`銆乣data_proc/`銆乣outputs/`锛氬師濮?澶勭悊鏁版嵁涓庢ā鍨嬭緭鍑猴紙鏈湴榛樿涓虹┖锛?- `tests/`锛氳交閲忕骇瀵煎叆娴嬭瘯锛堥粯璁よ烦杩囷級

## 鐜鍑嗗
```bash
pip install -r requirements.txt
# 鍙€夛細閰嶇疆 accelerate 鍒嗗竷寮?# accelerate config
```

濡傞渶浠?Hugging Face 鎷夊彇鏁版嵁锛岀‘淇濆叿澶囩綉缁滀笌鍑瘉锛堝闇€ `huggingface-cli login`锛夈€?
## 鏁版嵁涓嬭浇涓庡姞杞介€昏緫
- SFT锛圕hinese-Instruct锛?  - 浼樺厛浣跨敤鏈湴鏂囦欢锛氬皢 `*.jsonl|*.json` 鏀惧叆 `data_raw/mxode_chinese_instruct/`锛?  - 鑻ユ湰鍦颁笉瀛樺湪锛屽垯浣跨敤 Hugging Face `Mxode/Chinese-Instruct` 骞剁紦瀛樺埌璇ョ洰褰曪紱
- DPO锛圖PO-En-Zh-20k锛?  - 榛樿浠?Hugging Face 鎷夊彇 `llamafactory/DPO-En-Zh-20k` 骞剁紦瀛樺埌 `data_raw/dpo_en_zh_20k/`锛?
娉細鏈湴 dry-run 涓嶄細涓嬭浇/鍐欏叆銆傜湡瀹炴墽琛岄渶鍦ㄨ繙绋嬬幆澧冨苟瀹夎 `datasets`銆?
## 鏁版嵁澶勭悊锛坰cripts/prepare_data.py锛?- 杩愯鏂瑰紡锛?  - 鏈湴棰勬紨锛坉ry-run锛夛細
    ```bash
    python -m scripts.prepare_data --config configs/data.yaml --dry-run true
    ```
  - 杩滅▼鐪熷疄鎵ц锛?    ```bash
    python -m scripts.prepare_data --config configs/data.yaml --dry-run false
    ```
- 杩囨护涓庢竻娲楋細
  - 闀垮害闃堝€硷細`general.min_tokens`/`max_tokens`
  - 涓枃鍗犳瘮锛歚general.min_cn_ratio`锛涘彲閫?`general.allow_english_fallback`
  - 閲嶅搴︼細`general.max_repetition`锛堝熀浜?n鈥慻ram 閲嶅鐜囷級
- 閲囨牱涓庡幓閲嶏細
  - `MixedBucketSampler` 鍚屾椂鑰冭檻涓枃鍗犳瘮銆侀暱搴﹀垎妗躲€佹暟鎹簮鏉冮噸銆佽绋嬪紡鏉冮噸锛堝彲閫夛級
  - 鍘婚噸鍩轰簬鎷兼帴鏂囨湰鐨?`hash`锛圫HA256锛?- 杈撳嚭锛?  - SFT锛歚data_proc/sft_train.jsonl`銆乣sft_val.jsonl`銆乣sft_test.jsonl`锛堝瓧娈碉細`messages`锛?  - PREF锛歚data_proc/pref_train.jsonl`銆乣pref_val.jsonl`銆乣pref_test.jsonl`锛堝瓧娈碉細`prompt`銆乣chosen`銆乣rejected`锛?
CLI 瑕嗙洊绀轰緥锛?```bash
python -m scripts.prepare_data \
  --config configs/data.yaml \
  --stage all \
  --sft-mix mxode_chinese_instruct=1.0 \
  --pref-mix dpo_en_zh_20k=1.0 \
  --allow-english-fallback \
  --dry-run false
```

鍏煎鎬э細鏃х増鑻ヤ娇鐢ㄩ《灞?`pref` 閰嶇疆閿紝鑴氭湰浼氬皢鍏惰涓?`preference`銆?
## 璁粌锛圫FT锛?- 鍛戒护锛?```bash
accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false
```
- 鏁版嵁鎺ュ彛锛氶粯璁よ鍙?`data_proc/sft_{train,val}.jsonl` 涓殑 `messages` 瀛楁锛堣嫢鏈寚瀹?`dataset_text_field` 浼氳嚜鍔ㄦ嫾鎺ヤ负鏂囨湰锛夈€?- 鍏抽敭瓒呭弬鏁帮紙configs/sft.yaml锛夛細
  - `model.base_model`: 渚嬪 `Qwen/Qwen2.5-7B-Instruct`
  - model.max_seq_length: 4096（与数据处理上限一致，可按显存调整）
  - `model.packing`: true锛堥珮鏁堟墦鍖咃級
  - `lora.enable|r|alpha|dropout|target_modules`
  - `qlora.enable|quant_dtype|double_quant`锛堟樉瀛樼揣寮犳椂鍚敤锛?  - `training.epochs|per_device_train_batch_size|gradient_accumulation_steps`
  - `training.learning_rate|lr_scheduler_type|warmup_ratio|weight_decay|max_grad_norm`
  - `training.bf16|tf32|dataloader_num_workers`

## 璁粌锛圖PO 鎴?ORPO锛?- DPO 鍛戒护锛?```bash
accelerate launch -m scripts.train_dpo --config configs/dpo.yaml --dry-run false
```
- ORPO 鍛戒护锛堝闇€锛夛細
```bash
accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false
```
- 鏁版嵁鎺ュ彛锛歚data_proc/pref_{train,val}.jsonl`锛屽瓧娈?`prompt`銆乣chosen`銆乣rejected`
- SFT 鈫?DPO锛歚configs/dpo.yaml -> model.base_model` 榛樿涓?`outputs/sft`锛堣鎺?SFT 杈撳嚭锛?- 鍏抽敭瓒呭弬鏁帮紙configs/dpo.yaml锛夛細
  - `dpo.beta|reference_free|length_penalty|prefer_chinese_ratio`
  - `training.epochs|batch_size/accumulation|learning_rate|scheduler|warmup|bf16|tf32`

## 璇勬祴涓庡鍑?```bash
python -m scripts.eval_harness --config configs/eval.yaml --dry-run false
python -m scripts.export_lora   --config configs/sft.yaml  --dry-run false
```

## 甯歌闂锛團AQ锛?- 涓枃鏍锋湰涓嶈冻锛氳嫢 `allow_english_fallback=false`锛屾娊鏍蜂細琚腑鏂囨睜鎴柇锛涘彲鎻愰珮涓枃鍗犳瘮銆佸厑璁歌嫳鏂囧洖閫€鎴栬皟鏉冮噸銆?- 鏍锋湰杩囬暱/杩囩煭锛氳皟鏁?`general.{min_tokens,max_tokens}`銆?- 閲嶅搴﹀亸楂橈細闄嶄綆 `general.max_repetition` 鎴栧姞寮烘竻娲椼€?- 鏈湴鏁版嵁瑕嗙洊锛氬皢 Chinese-Instruct JSON 鏀惧叆 `data_raw/mxode_chinese_instruct/` 鍙鐩栬繙绋嬩笅杞姐€?
## 鍙傝€?- 鏁版嵁澶勭悊璇︾粏閫昏緫锛歚docs/DATA_PROCESSING.md`
- 璁粌鑴氭湰鍏抽敭瀹炵幇锛歚scripts/train_sft.py`, `scripts/train_dpo.py`
- 閲囨牱涓庢竻娲楀伐鍏凤細`scripts/utils_data.py`
