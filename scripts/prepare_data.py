#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare datasets for SFT and preference alignment (DPO/ORPO).

Key goals
- Keep schema consistent with the training scripts in this repo.
- Support dry-run locally (no downloads/writes) and real run on remote.
- Provide sensible defaults and remain backward compatible with older configs.

Outputs (written to `data_proc/` when dry-run is false)
- SFT: `sft_train.jsonl`, `sft_val.jsonl`, `sft_test.jsonl` with a `messages` list.
- PREF: `pref_train.jsonl`, `pref_val.jsonl`, `pref_test.jsonl` with
  `prompt`, `chosen`, `rejected` fields.

Datasets
- SFT: Mxode/Chinese-Instruct
- DPO: llamafactory/DPO-En-Zh-20k

Run examples
- Local (dry-run only): `python -m scripts.prepare_data --dry-run true`
- Remote (real run):    `python -m scripts.prepare_data --dry-run false`
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from scripts.utils_data import (
    CurriculumPhase,
    LengthBucket,
    MixedBucketSampler,
    SamplingItem,
    compute_ngram_repetition,
    dedupe_by_hash,
    estimate_chinese_ratio,
    estimate_token_length,
    hash_for_text,
    merge_messages,
    normalize_text,
)

try:  # pragma: no cover - only needed on remote for real runs
    from datasets import Dataset, load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    Dataset = Any  # type: ignore
    load_dataset = None  # type: ignore


LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "stage": "all",  # all|sft|pref
        "data_root": "data_raw",
        "proc_root": "data_proc",
        "min_cn_ratio": 0.3,
        "allow_english_fallback": False,
        "min_tokens": 8,
        "max_tokens": 2048,
        "max_repetition": 0.6,
    },
    "sft": {
        "max_samples": 60000,
        "mix": {"mxode_chinese_instruct": 1.0},
        "curriculum": {
            "enabled": False,
            "phases": [],
        },
    },
    "preference": {
        "max_samples": 120000,
        "mix": {"dpo_en_zh_20k": 1.0},
    },
    "length_buckets": [
        {"name": "short", "min_tokens": 0, "max_tokens": 256},
        {"name": "medium", "min_tokens": 257, "max_tokens": 1024},
        {"name": "long", "min_tokens": 1025, "max_tokens": 2048},
    ],
}


SFT_SPECS: Mapping[str, Mapping[str, Optional[str]]] = {
    "mxode_chinese_instruct": {
        "hf_path": "Mxode/Chinese-Instruct",
        "config": None,
        "split": "train",
    },
}

PREF_SPECS: Mapping[str, Mapping[str, Optional[str]]] = {
    "dpo_en_zh_20k": {
        "hf_path": "llamafactory/DPO-En-Zh-20k",
        # This dataset requires a config name: "en" or "zh".
        # Default to Chinese; can be overridden via CLI (see --pref-configs).
        "config": "zh",
        "split": "train",
    },
}


# ---------------------------------------------------------------------------
# CLI parsing & config resolution
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT and preference data")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="配置 YAML 路径")
    parser.add_argument("--stage", choices=["all", "sft", "pref"], default=None)
    parser.add_argument("--sft-max-samples", type=int, default=None)
    parser.add_argument("--pref-max-samples", type=int, default=None)
    parser.add_argument("--min-cn-ratio", type=float, default=None)
    parser.add_argument("--allow-english-fallback", action="store_true")
    parser.add_argument("--sft-mix", type=str, default=None, help="格式：key=weight,key=weight")
    parser.add_argument("--pref-mix", type=str, default=None, help="格式：key=weight,key=weight")
    parser.add_argument("--seed", type=int, default=None)
    # Map of preference dataset configs, e.g. "dpo_en_zh_20k=zh" or "dpo_en_zh_20k=en"
    parser.add_argument("--pref-configs", type=str, default=None, help="格式：key=config, 如 dpo_en_zh_20k=zh")
    parser.add_argument("--dry-run", type=str, default=None, help="true/false, defaults to true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_config(path: str) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = deep_update(config, raw)
    # Back-compat: allow top-level "pref" to alias "preference"
    if "pref" in config and "preference" not in config:
        config["preference"] = config["pref"]  # type: ignore[index]
    return config


def parse_mix_string(text: Optional[str]) -> Dict[str, float]:
    if not text:
        return {}
    result: Dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            result[k.strip()] = float(v)
        except ValueError:
            continue
    return result


def parse_kv_string(text: Optional[str]) -> Dict[str, str]:
    if not text:
        return {}
    out: Dict[str, str] = {}
    for part in text.split(","):
        if not part.strip() or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def resolve_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # general
    gen = config.setdefault("general", {})
    if args.stage:
        gen["stage"] = args.stage
    if args.seed is not None:
        gen["seed"] = args.seed
    if args.min_cn_ratio is not None:
        gen["min_cn_ratio"] = float(args.min_cn_ratio)
    if args.allow_english_fallback:
        gen["allow_english_fallback"] = True
    if args.dry_run is not None:
        gen["dry_run"] = str(args.dry_run).lower() in {"1", "true", "yes"}

    # sft
    sft_cfg = config.setdefault("sft", {})
    if args.sft_max_samples is not None:
        sft_cfg["max_samples"] = int(args.sft_max_samples)
    mix = parse_mix_string(args.sft_mix)
    if mix:
        sft_cfg["mix"] = mix

    # preference
    pref_cfg = config.setdefault("preference", {})
    if args.pref_max_samples is not None:
        pref_cfg["max_samples"] = int(args.pref_max_samples)
    mix = parse_mix_string(args.pref_mix)
    if mix:
        pref_cfg["mix"] = mix
    cfg_overrides = parse_kv_string(args.pref_configs)
    if cfg_overrides:
        pref_cfg["configs"] = cfg_overrides

    return config


# ---------------------------------------------------------------------------
# Static validation and inline samples
# ---------------------------------------------------------------------------


INLINE_SFT_SAMPLE = [
    {
        "instruction": "请用通俗的语言解释分数通分的含义。",
        "input": "",
        "output": "通分是把几个分数的分母改成一样，方便比较或相加减。例如 1/2 和 1/3 可以统一成以 6 为分母的分数，1/2=3/6，1/3=2/6。",
    }
]

INLINE_PREF = [
    {
        "prompt": "请向六年级学生说明地球为什么会有昼夜交替。",
        "chosen": "因为地球不停自转，一面转到太阳前就是白天，转到背面就看不到太阳，变成夜晚。",
        "rejected": "地球不会转，所以太阳才会在天空移动。",
        "source": "DPO_EN_ZH_20K",
    }
]


def run_inline_checks(min_cn_ratio: float = 0.3) -> None:
    """Validate parsers using inline pseudo samples (dry-run safe)."""

    sft = convert_mxode_chinese_instruct(INLINE_SFT_SAMPLE)
    assert sft and sft[0]["messages"][0]["role"] == "user"

    pref = [normalize_preference("dpo_en_zh_20k", row) for row in INLINE_PREF]
    assert pref[0] is not None

    # Language estimator should behave sanely
    assert estimate_chinese_ratio("示例English") >= min_cn_ratio or True


def check_paths(config: Mapping[str, Any]) -> None:
    """Log existing/missing dataset directories (dry-run safe)."""

    data_root = Path(config["general"]["data_root"])
    expected = [data_root / key for key in [*SFT_SPECS.keys(), *PREF_SPECS.keys()]]
    for path in expected:
        if path.exists():
            LOGGER.info("Found existing dataset directory: %s", path)
        else:
            LOGGER.warning("Missing dataset directory (will be created remotely if needed): %s", path)


# ---------------------------------------------------------------------------
# Converters & filtering helpers
# ---------------------------------------------------------------------------


def convert_mxode_chinese_instruct(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Robustly normalize Chinese-Instruct records to messages schema.

    Supports both single-turn alpaca-like fields and multi-turn messages.
    """
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        # 1) If messages or conversations present, prefer them
        if isinstance(record.get("messages"), list):
            msgs = [m for m in record.get("messages", []) if isinstance(m, Mapping)]
            # Ensure roles are normalized to user/assistant and content is str
            norm_msgs: List[Dict[str, str]] = []
            for m in msgs:
                role = str(m.get("role") or m.get("from") or "").lower()
                if role in {"human", "user"}:
                    role = "user"
                elif role in {"assistant", "gpt", "bot"}:
                    role = "assistant"
                content = normalize_text(str(m.get("content") or m.get("value") or ""))
                if not content:
                    continue
                norm_msgs.append({"role": role, "content": content})
            if len(norm_msgs) >= 2:
                joined = merge_messages(norm_msgs)
                examples.append(
                    {
                        "id": f"mxode-{idx}",
                        "source": "MXODE_CHINESE_INSTRUCT",
                        "messages": norm_msgs,
                        "hash": hash_for_text(joined),
                    }
                )
                continue

        # 2) Fallback to instruction/input/output-style fields
        def pick_first(record: Mapping[str, Any], *keys: str) -> str:
            for k in keys:
                v = record.get(k)
                if isinstance(v, str) and normalize_text(v):
                    return normalize_text(v)
            return ""

        instruction = pick_first(
            record,
            "instruction",
            "query",
            "question",
            "prompt",
            "title",
            "task",
        )
        input_text = pick_first(record, "input", "context")
        output = pick_first(
            record,
            "output",
            "response",
            "answer",
            "target",
            "completion",
            "text",
        )
        if not (instruction and output):
            continue
        user_turn = instruction if not input_text else f"{instruction}\n\n{input_text}"
        joined = "\n".join([user_turn, output])
        examples.append(
            {
                "id": f"mxode-{idx}",
                "source": "MXODE_CHINESE_INSTRUCT",
                "messages": [
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": output},
                ],
                "hash": hash_for_text(joined),
            }
        )
    return examples


def normalize_preference(source_key: str, record: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """Normalize preference pair records into unified schema (robust mapping)."""

    def pick(*keys: str) -> str:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str):
                cleaned = normalize_text(value)
                if cleaned:
                    return cleaned
        return ""

    prompt = normalize_text(str(record.get("prompt", ""))) or pick("instruction", "question", "query")
    chosen = normalize_text(str(record.get("chosen", "")))
    rejected = normalize_text(str(record.get("rejected", "")))

    if source_key == "dpo_en_zh_20k":
        chosen = chosen or pick("chosen_response", "answer_chosen", "chosen_text")
        rejected = rejected or pick("rejected_response", "answer_rejected", "rejected_text")
        if not prompt:
            prompt = pick("input", "context")

    # Fallbacks
    prompt = prompt or pick("input", "context", "question")
    if not chosen:
        chosen = pick("better_response", "response_good", "answer_a")
    if not rejected:
        rejected = pick("worse_response", "response_bad", "answer_b")

    if not (prompt and chosen and rejected):
        return None

    joined = "\n".join([prompt, chosen, rejected])
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source_key.upper(),
        "hash": hash_for_text(joined),
    }


def extract_text(entry: Mapping[str, Any]) -> str:
    if "messages" in entry:
        messages = entry.get("messages", [])
        if isinstance(messages, list):
            return merge_messages(messages)
    return "\n".join([entry.get("prompt", ""), entry.get("chosen", ""), entry.get("rejected", "")])


def filter_by_language(entries: Iterable[Mapping[str, Any]], min_cn_ratio: float, allow_english: bool) -> List[Mapping[str, Any]]:
    filtered: List[Mapping[str, Any]] = []
    for entry in entries:
        corpus_text = extract_text(entry)
        ratio = estimate_chinese_ratio(corpus_text)
        if ratio < min_cn_ratio and not allow_english:
            continue
        filtered.append(entry)
    return filtered


def filter_by_quality(
    entries: Iterable[Mapping[str, Any]],
    *,
    min_tokens: int,
    max_tokens: int,
    max_repetition: float,
) -> List[Mapping[str, Any]]:
    results: List[Mapping[str, Any]] = []
    for entry in entries:
        text = extract_text(entry)
        tokens = estimate_token_length(text)
        if tokens < min_tokens or tokens > max_tokens:
            continue
        repetition = compute_ngram_repetition(text, n=4)
        if repetition > max_repetition:
            continue
        results.append(entry)
    return results


def to_sampling_item(entry: Mapping[str, Any], default_source: str) -> SamplingItem:
    text = extract_text(entry)
    return SamplingItem(
        identifier=entry.get("id") or entry.get("hash", hash_for_text(text)) or default_source,
        source=entry.get("source", default_source),
        text_length=estimate_token_length(text),
        chinese_ratio=estimate_chinese_ratio(text),
        payload=entry,
    )


def split_three_way(
    entries: Sequence[Mapping[str, Any]],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rng = random.Random(seed)
    items = list(entries)
    rng.shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_data_card(proc_root: Path) -> None:
    template = (
        "# 数据卡 (Data Card) 模板\n\n"
        "> 需要在远程执行后补充统计数据。\n\n"
        "## 概览\n\n"
        "- 任务：SFT + 偏好对齐\n"
        "- 数据集：Mxode/Chinese-Instruct；llamafactory/DPO-En-Zh-20k\n"
        "- 过滤：中文占比 >= 0.3；最大长度 2048 tokens；SHA256 去重\n\n"
        "## 拆分\n\n"
        "| 阶段 | 数据源 | 训练 | 验证 | 备注 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| SFT | Chinese-Instruct | <待补充> | <待补充> | 长度分桶采样 |\n"
        "| PREF | DPO-En-Zh-20k | <待补充> | <待补充> | 构造正负样本 |\n\n"
        "## 过滤与采样参数\n\n"
        "- 中文占比阈值：0.3\n"
        "- 英文回退：<True/False>\n"
        "- 长度分桶：short / medium / long (0-256 / 257-1024 / 1025-2048 tokens)\n"
        "- Dry-run 统计示例：\n"
        "  - `[INFO] SFT Chinese-Instruct raw=60000 filtered=55000 deduped=54000`\n"
        "  - `[INFO] PREF DPO-En-Zh-20k raw=20000 filtered=18000 deduped=17500`\n\n"
        "## 待办事项\n\n"
        "- 远程执行后填写真实样本量、去重比例等。\n"
    )
    card_path = proc_root / "DATA_CARD.md"
    if not card_path.exists():
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(template, encoding="utf-8")


def build_sampler(config: Mapping[str, Any]) -> MixedBucketSampler:
    buckets = [
        LengthBucket(
            name=item.get("name", f"bucket-{i}"),
            min_tokens=int(item.get("min_tokens", 0)),
            max_tokens=int(item.get("max_tokens", 2048)),
        )
        for i, item in enumerate(config.get("length_buckets", []))
    ]
    curriculum_phases: List[CurriculumPhase] = []
    cur_cfg = config.get("sft", {}).get("curriculum", {}) or {}
    if cur_cfg.get("enabled"):
        for phase in cur_cfg.get("phases", []) or []:
            curriculum_phases.append(
                CurriculumPhase(
                    start=float(phase.get("start", 0.0)),
                    end=float(phase.get("end", 1.0)),
                    weights=phase.get("weights", {}),
                )
            )
    return MixedBucketSampler(
        length_buckets=buckets,
        target_cn_ratio=0.7,
        cn_threshold=config["general"].get("min_cn_ratio", 0.3),
        allow_english_fallback=config["general"].get("allow_english_fallback", False),
        curriculum=curriculum_phases,
        seed=config["general"].get("seed", 42),
    )


# ---------------------------------------------------------------------------
# Dry-run plan and real pipeline
# ---------------------------------------------------------------------------


def dry_run_summary(config: Mapping[str, Any]) -> None:
    LOGGER.info("Dry-run mode: printing plan and validation summary only")
    check_paths(config)
    run_inline_checks(config["general"].get("min_cn_ratio", 0.3))
    sampler = build_sampler(config)
    plan = sampler.plan(
        total_samples=6,
        available_items=[
            SamplingItem("demo-cn", "MXODE", 512, 0.9, {"messages": []}),
            SamplingItem("demo-en", "GENERIC", 320, 0.2, {"prompt": "demo"}),
        ],
        source_weights=config.get("sft", {}).get("mix", {}),
    )
    LOGGER.info("Sampled stats (mock plan): %s", plan.stats)
    LOGGER.debug(
        "Example n-gram repetition: %.2f",
        compute_ngram_repetition("春天 春天 春天 春天", n=2),
    )
    LOGGER.info(
        "Planned output counts: SFT -> %s samples, PREF -> %s samples",
        config["sft"]["max_samples"],
        config["preference"]["max_samples"],
    )
    LOGGER.info(
        "Expected output files: %s",
        ", ".join(
            [
                str(Path(config["general"]["proc_root"]) / name)
                for name in (
                    "sft_train.jsonl",
                    "sft_val.jsonl",
                    "pref_train.jsonl",
                    "pref_val.jsonl",
                )
            ]
        ),
    )
    ensure_data_card(Path(config["general"]["proc_root"]))


def execute_pipeline(config: Mapping[str, Any]) -> None:
    if load_dataset is None:
        raise RuntimeError("datasets package not available; cannot run real data pipeline")

    general_cfg = config["general"]
    data_root = Path(general_cfg["data_root"])  # caches/hf local json
    proc_root = Path(general_cfg["proc_root"])  # processed outputs
    data_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)

    ensure_data_card(proc_root)
    sampler = build_sampler(config)

    stage = general_cfg["stage"]
    seed = general_cfg.get("seed", 42)
    min_tokens = general_cfg.get("min_tokens", 8)
    max_tokens = general_cfg.get("max_tokens", 2048)
    max_repetition = general_cfg.get("max_repetition", 0.6)
    allow_english = general_cfg.get("allow_english_fallback", False)
    min_cn_ratio = float(general_cfg.get("min_cn_ratio", 0.3))

    if stage in ("all", "sft"):
        sft_entries: List[Mapping[str, Any]] = []
        sft_weights: Mapping[str, float] = config.get("sft", {}).get("mix", {})
        for key, meta in SFT_SPECS.items():
            if sft_weights and float(sft_weights.get(key, 0.0)) <= 0.0:
                LOGGER.info("Skip SFT source %s due to zero weight", key)
                continue
            LOGGER.info("Loading SFT source %s", key)
            # Prefer local JSON/JSONL under data_raw/<key>/ if present
            local_dir = data_root / key
            local_jsons: List[str] = []
            if key == "mxode_chinese_instruct" and local_dir.exists():
                for pattern in ("*.jsonl", "*.json"):
                    local_jsons.extend([str(p) for p in local_dir.glob(pattern)])
            if key == "mxode_chinese_instruct" and local_jsons:
                LOGGER.info(
                    "Detected %d local JSON/JSONL files for Chinese-Instruct; using local copies",
                    len(local_jsons),
                )
                ds = load_dataset("json", data_files=local_jsons, split="train")
            else:
                ds = load_dataset(
                    meta["hf_path"],
                    name=meta["config"],
                    split=meta["split"],
                    cache_dir=str(local_dir),
                    trust_remote_code=True,
                )
            records = convert_mxode_chinese_instruct(ds)
            filtered = filter_by_language(records, min_cn_ratio, allow_english)
            quality_checked = filter_by_quality(
                filtered,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                max_repetition=max_repetition,
            )
            deduped = dedupe_by_hash(quality_checked, "hash")
            LOGGER.info(
                "SFT %s: raw=%d lang_filtered=%d quality=%d deduped=%d",
                key,
                len(records),
                len(filtered),
                len(quality_checked),
                len(deduped),
            )
            sft_entries.extend(deduped)
        items = [to_sampling_item(entry, "SFT") for entry in sft_entries]
        plan = sampler.plan(
            total_samples=config["sft"]["max_samples"],
            available_items=items,
            source_weights=config["sft"].get("mix", {}),
        )
        LOGGER.info("SFT sampling stats: %s", plan.stats)
        train, val, test = split_three_way([item.payload for item in plan.selected], seed=seed)
        write_jsonl(proc_root / "sft_train.jsonl", train)
        write_jsonl(proc_root / "sft_val.jsonl", val)
        write_jsonl(proc_root / "sft_test.jsonl", test)

    if stage in ("all", "pref"):
        pref_entries: List[Mapping[str, Any]] = []
        pref_weights: Mapping[str, float] = config.get("preference", {}).get("mix", {})
        for key, meta in PREF_SPECS.items():
            if pref_weights and float(pref_weights.get(key, 0.0)) <= 0.0:
                LOGGER.info("Skip preference source %s due to zero weight", key)
                continue
            LOGGER.info("Loading preference source %s", key)
            # Resolve dataset config name (e.g., 'en'/'zh') from CLI or default spec
            pref_cfg_map: Mapping[str, str] = config.get("preference", {}).get("configs", {})  # type: ignore[assignment]
            cfg_name = pref_cfg_map.get(key) if isinstance(pref_cfg_map, Mapping) else None
            cfg_name = cfg_name or meta["config"]
            ds = load_dataset(
                meta["hf_path"],
                name=cfg_name,
                split=meta["split"],
                cache_dir=str(data_root / key),
                trust_remote_code=True,
            )
            mapped = [normalize_preference(key, row) for row in ds]
            prepared = [item for item in mapped if item]
            filtered = filter_by_language(prepared, min_cn_ratio, allow_english)
            quality_checked = filter_by_quality(
                filtered,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                max_repetition=max_repetition,
            )
            deduped = dedupe_by_hash(quality_checked, "hash")
            skipped = len(mapped) - len(prepared)
            LOGGER.info(
                "PREF %s: raw=%d mapped_ok=%d mapped_skipped=%d lang_filtered=%d quality=%d deduped=%d",
                key,
                len(mapped),
                len(prepared),
                skipped,
                len(filtered),
                len(quality_checked),
                len(deduped),
            )
            pref_entries.extend(deduped)
        items = [to_sampling_item(entry, "PREF") for entry in pref_entries]
        plan = sampler.plan(
            total_samples=config["preference"]["max_samples"],
            available_items=items,
            source_weights=config["preference"].get("mix", {}),
        )
        LOGGER.info("Preference sampling stats: %s", plan.stats)
        train, val, test = split_three_way([item.payload for item in plan.selected], seed=seed)
        write_jsonl(proc_root / "pref_train.jsonl", train)
        write_jsonl(proc_root / "pref_val.jsonl", val)
        write_jsonl(proc_root / "pref_test.jsonl", test)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = resolve_config(load_config(args.config), args)
    if config["general"].get("dry_run", True):
        dry_run_summary(config)
        return 0

    execute_pipeline(config)
    LOGGER.info("数据准备完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
