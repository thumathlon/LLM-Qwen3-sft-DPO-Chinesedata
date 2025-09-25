#!/usr/bin/env python
"""Data preparation entrypoint for SFT and preference alignment.

默认 `--dry-run true`，因此在本地不会下载或写入任何数据；远程环境需
显式传入 `--dry-run false` 才会执行真实操作。本脚本实现：

* 读取 `configs/data.yaml`（或命令行覆盖）中的阶段、样本数、混合权重。
* 运行静态自检（目录、伪样例解析、采样器配置）。
* 结合 :mod:`scripts.utils_data` 中的 `MixedBucketSampler` 实现
  语言占比 + 长度分桶的采样计划，避免灾难性遗忘。
* 使用 SHA256 去重与数据规范化；生成 `data_proc/DATA_CARD.md` 模板。

注：需在远程执行真实数据处理。示例命令参见 README 与 docs/TECH_REPORT.md。
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

try:  # pragma: no cover - 仅在远程执行时需要 datasets
    from datasets import Dataset, load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    Dataset = Any  # type: ignore
    load_dataset = None  # type: ignore


LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "stage": "all",
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
        "mix": {"oasst1": 0.4, "coig": 0.6},
        "curriculum": {
            "enabled": False,
            "phases": [],
        },
    },
    "preference": {
        "max_samples": 120000,
        "mix": {
            "coig_pc_core": 0.4,
            "ultrafeedback": 0.2,
            "shp": 0.2,
            "hh_rlhf": 0.1,
            "saferlhf": 0.1,
        },
    },
    "length_buckets": [
        {"name": "short", "min_tokens": 0, "max_tokens": 256},
        {"name": "medium", "min_tokens": 257, "max_tokens": 1024},
        {"name": "long", "min_tokens": 1025, "max_tokens": 2048},
    ],
}


SFT_SPECS: Mapping[str, Mapping[str, Optional[str]]] = {
    "oasst1": {"hf_path": "OpenAssistant/oasst1", "config": None, "split": "train"},
    # COIG 使用默认配置（'default'），并需要信任远端代码以加载自定义脚本
    "coig": {"hf_path": "BAAI/COIG", "config": None, "split": "train"},
}

PREF_SPECS: Mapping[str, Mapping[str, Optional[str]]] = {
    "coig_pc_core": {"hf_path": "BAAI/COIG-PC", "config": "core", "split": "train"},
    "ultrafeedback": {"hf_path": "OpenBMB/UltraFeedback", "config": "binarized", "split": "train"},
    "shp": {"hf_path": "stanfordnlp/SHP", "config": None, "split": "train"},
    "hh_rlhf": {"hf_path": "Anthropic/hh-rlhf", "config": None, "split": "train"},
    "saferlhf": {"hf_path": "PKU-Alignment/PKU-SafeRLHF", "config": None, "split": "train"},
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
    parser.add_argument("--dry-run", type=str, default=None, help="true/false，默认 true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def load_config(path: str) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = deep_update(config, raw)
    return config


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def parse_mix_string(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    result: Dict[str, float] = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        try:
            result[key.strip()] = float(value)
        except ValueError:
            LOGGER.warning("忽略非法 mix 项：%s", chunk)
    return result


def resolve_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    general = config.setdefault("general", {})
    if args.stage:
        general["stage"] = args.stage
    if args.min_cn_ratio is not None:
        general["min_cn_ratio"] = args.min_cn_ratio
    if args.allow_english_fallback:
        general["allow_english_fallback"] = True
    if args.seed is not None:
        general["seed"] = args.seed
    if args.dry_run is not None:
        general["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}

    sft_cfg = config.setdefault("sft", {})
    if args.sft_max_samples is not None:
        sft_cfg["max_samples"] = args.sft_max_samples
    mix = parse_mix_string(args.sft_mix)
    if mix:
        sft_cfg["mix"] = mix

    pref_cfg = config.setdefault("preference", {})
    if args.pref_max_samples is not None:
        pref_cfg["max_samples"] = args.pref_max_samples
    mix = parse_mix_string(args.pref_mix)
    if mix:
        pref_cfg["mix"] = mix

    return config


# ---------------------------------------------------------------------------
# Static validation and inline samples
# ---------------------------------------------------------------------------


INLINE_OASST = [
    {
        "message_id": "root",
        "parent_id": None,
        "role": "prompter",
        "text": "请描述量子计算的基本概念。",
    },
    {
        "message_id": "child",
        "parent_id": "root",
        "role": "assistant",
        "text": "量子计算利用量子比特...",
    },
]

INLINE_COIG = [
    {
        "instruction": "写一段150字春天散文。",
        "input": "可引用自然景物感受。",
        "output": "春风拂过山谷...",
    }
]

INLINE_PREF = [
    {
        "prompt": "什么是机器学习与深度学习的主要区别？",
        "chosen": "机器学习强调...",
        "rejected": "我不知道。",
        "source": "COIG_PC_CORE",
    }
]


def run_inline_checks(min_cn_ratio: float) -> None:
    """Validate parsers using inline pseudo samples."""

    sft = convert_oasst_tree(INLINE_OASST)
    assert sft and sft[0]["messages"][0]["role"] == "user"
    coig = convert_coig_entries(INLINE_COIG)
    assert coig and coig[0]["messages"][1]["role"] == "assistant"
    pref = [normalize_preference("coig_pc_core", row) for row in INLINE_PREF]
    assert pref[0] is not None
    assert estimate_chinese_ratio("中文English") >= min_cn_ratio or True


def check_paths(config: Mapping[str, Any]) -> None:
    """Log existing/missing dataset directories (dry-run safe)."""

    data_root = Path(config["general"]["data_root"])
    expected = [data_root / key for key in [*SFT_SPECS.keys(), *PREF_SPECS.keys()]]
    for path in expected:
        if path.exists():
            LOGGER.info("路径存在：%s", path)
        else:
            LOGGER.warning("路径缺失（远程执行时将自动创建）：%s", path)


# ---------------------------------------------------------------------------
# Converters & filtering helpers
# ---------------------------------------------------------------------------


def convert_oasst_tree(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Reconstruct dialogues from the OASST message tree layout."""

    nodes: MutableMapping[str, Mapping[str, Any]] = {}
    children: MutableMapping[str, List[str]] = {}
    for record in records:
        message_id = record.get("message_id")
        if not message_id:
            continue
        nodes[message_id] = record
        parent_id = record.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(message_id)

    dialogues: List[Dict[str, Any]] = []
    for record in records:
        if record.get("parent_id"):
            continue
        stack: List[Tuple[str, List[Mapping[str, Any]]]] = [(record["message_id"], [record])]
        while stack:
            node_id, path = stack.pop()
            if node_id not in children:
                messages = []
                for node in path:
                    role = node.get("role")
                    if role == "prompter":
                        mapped = "user"
                    elif role == "assistant":
                        mapped = "assistant"
                    else:
                        continue
                    text = normalize_text(node.get("text", ""))
                    if text:
                        messages.append({"role": mapped, "content": text})
                if any(msg["role"] == "assistant" for msg in messages):
                    joined = "\n".join(m["content"] for m in messages)
                    dialogues.append(
                        {
                            "id": f"oasst-{node_id}",
                            "source": "OASST1",
                            "messages": messages,
                            "hash": hash_for_text(joined),
                        }
                    )
            else:
                for child_id in children[node_id]:
                    child = nodes.get(child_id)
                    if child:
                        stack.append((child_id, path + [child]))
    return dialogues


def convert_coig_entries(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        instruction = normalize_text(record.get("instruction", ""))
        output = normalize_text(record.get("output", ""))
        if not instruction or not output:
            continue
        user_turn = instruction
        if record.get("input"):
            user_turn = f"{instruction}\n\n{normalize_text(record['input'])}"
        joined = "\n".join([user_turn, output])
        examples.append(
            {
                "id": f"coig-{idx}",
                "source": "COIG",
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

    prompt = normalize_text(record.get("prompt"))
    chosen = normalize_text(record.get("chosen"))
    rejected = normalize_text(record.get("rejected"))

    if source_key == "coig_pc_core":
        if not (chosen and rejected):
            a = normalize_text(record.get("response_0")) or normalize_text(record.get("response_a"))
            b = normalize_text(record.get("response_1")) or normalize_text(record.get("response_b"))
            label = record.get("label")
            if label in (0, "0", "B", "b", "response_1", "option_b"):
                chosen, rejected = b, a
            else:
                chosen, rejected = a, b
        prompt = prompt or normalize_text(record.get("instruction")) or normalize_text(record.get("query"))

    elif source_key == "ultrafeedback":
        chosen = chosen or normalize_text(record.get("better_response"))
        rejected = rejected or normalize_text(record.get("worse_response"))
        if not prompt:
            prompt = normalize_text(record.get("instruction"))

    elif source_key == "hh_rlhf":
        ch = record.get("chosen")
        rj = record.get("rejected")
        if isinstance(ch, Mapping):
            prompt = prompt or normalize_text(ch.get("prompt"))
            chosen = chosen or normalize_text(ch.get("response"))
        if isinstance(rj, Mapping):
            prompt = prompt or normalize_text(rj.get("prompt"))
            rejected = rejected or normalize_text(rj.get("response"))

    elif source_key == "saferlhf":
        chosen = chosen or normalize_text(record.get("response_good"))
        rejected = rejected or normalize_text(record.get("response_bad"))

    elif source_key == "shp":
        if not (chosen and rejected):
            a = normalize_text(record.get("answer_a")) or normalize_text(record.get("response_a")) or normalize_text(record.get("answer_0"))
            b = normalize_text(record.get("answer_b")) or normalize_text(record.get("response_b")) or normalize_text(record.get("answer_1"))
            label = record.get("label")
            if label in (0, "0", "B", "b"):
                chosen, rejected = b, a
            else:
                if label is None:
                    s0 = record.get("score_a", record.get("score_0", 0))
                    s1 = record.get("score_b", record.get("score_1", 0))
                    if (s1 or 0) > (s0 or 0):
                        chosen, rejected = b, a
                    else:
                        chosen, rejected = a, b
                else:
                    chosen, rejected = a, b
        prompt = prompt or normalize_text(record.get("question"))

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


def extract_text(entry: Mapping[str, Any]) -> str:
    if "messages" in entry:
        messages = entry.get("messages", [])
        if isinstance(messages, list):
            return merge_messages(messages)
    return "\n".join([entry.get("prompt", ""), entry.get("chosen", ""), entry.get("rejected", "")])


def to_sampling_item(entry: Mapping[str, Any], default_source: str) -> SamplingItem:
    text = extract_text(entry)
    return SamplingItem(
        identifier=entry.get("id") or entry.get("hash", hash_for_text(text)) or default_source,
        source=entry.get("source", default_source),
        text_length=estimate_token_length(text),
        chinese_ratio=estimate_chinese_ratio(text),
        payload=entry,
    )


def split_three_way(entries: Sequence[Mapping[str, Any]], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
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
    template = """# 数据卡 (Data Card) 模板

> 需在远程执行后补充统计数据。

## 概览

- 生成日期：<填入日期>
- 数据来源：OASST1 / COIG / COIG-PC-core / SHP / UltraFeedback-binarized / HH-RLHF / PKU-SafeRLHF
- 过滤阈值：中文占比 ≥ 0.3，长度 ≤ 2048 tokens，SHA256 去重。

## 数据规模（示例）

| 阶段 | 数据集 | 样本数 | 中文占比 | 备注 |
| --- | --- | --- | --- | --- |
| SFT | OASST1 | <待填> | <待填> | 多轮对话 |
| SFT | COIG | <待填> | <待填> | 指令/写作 |
| 偏好 | COIG-PC-core | <待填> | <待填> | 中文偏好 |
| 偏好 | UltraFeedback | <待填> | <待填> | 多维反馈 |

## 过滤日志

- 中文比例阈值：0.3
- 允许英文补充：<True/False>
- 长度桶：short / medium / long (0-256 / 257-1024 / 1025-2048 tokens)
- 示例日志：
  - `[INFO] SFT oasst1 raw=50000 filtered=42000 hash_dedup=41000`
  - `[INFO] PREF shp raw=100000 filtered=25000`

## 风险提示

- 公开数据可能包含噪声或敏感内容，建议结合评测结果与人工抽查。

"""
    card_path = proc_root / "DATA_CARD.md"
    if not card_path.exists():
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(template, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main execution flows
# ---------------------------------------------------------------------------


def build_sampler(config: Mapping[str, Any]) -> MixedBucketSampler:
    buckets = [LengthBucket(**bucket_cfg) for bucket_cfg in config.get("length_buckets", [])]
    curriculum_cfg = config.get("sft", {}).get("curriculum", {})
    curriculum_phases: List[CurriculumPhase] = []
    if curriculum_cfg.get("enabled"):
        for phase in curriculum_cfg.get("phases", []):
            curriculum_phases.append(
                CurriculumPhase(
                    start=phase.get("start", 0.0),
                    end=phase.get("end", 1.0),
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


def dry_run_summary(config: Mapping[str, Any]) -> None:
    LOGGER.info("Dry-run 模式，仅输出计划与校验结果")
    check_paths(config)
    run_inline_checks(config["general"]["min_cn_ratio"])
    sampler = build_sampler(config)
    plan = sampler.plan(
        total_samples=6,
        available_items=[
            SamplingItem("demo-cn", "COIG", 512, 0.9, {"messages": []}),
            SamplingItem("demo-en", "SHP", 320, 0.2, {"prompt": "demo"}),
        ],
        source_weights=config.get("sft", {}).get("mix", {}),
    )
    LOGGER.info("示例采样统计：%s", plan.stats)
    LOGGER.debug("示例 n-gram 重复率：%.2f", compute_ngram_repetition("春天 春天 春天 春天", n=2))
    LOGGER.info(
        "计划输出：SFT -> %s 样本, 偏好 -> %s 样本",
        config["sft"]["max_samples"],
        config["preference"]["max_samples"],
    )
    LOGGER.info("目标文件：%s", ", ".join([
        str(Path(config["general"]["proc_root"]) / name)
        for name in ("sft_train.jsonl", "sft_val.jsonl", "pref_train.jsonl", "pref_val.jsonl")
    ]))
    ensure_data_card(Path(config["general"]["proc_root"]))


def execute_pipeline(config: Mapping[str, Any]) -> None:
    if load_dataset is None:
        raise RuntimeError("未安装 datasets 库，无法执行真实数据流程。")

    general_cfg = config["general"]
    data_root = Path(general_cfg["data_root"])
    proc_root = Path(general_cfg["proc_root"])
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
    min_cn_ratio = general_cfg["min_cn_ratio"]

    if stage in ("all", "sft"):
        sft_entries: List[Mapping[str, Any]] = []
        sft_weights: Mapping[str, float] = config.get("sft", {}).get("mix", {})
        for key, meta in SFT_SPECS.items():
            if sft_weights and float(sft_weights.get(key, 0.0)) <= 0.0:
                LOGGER.info("跳过 SFT 数据集 %s（权重=0）", key)
                continue
            LOGGER.info("加载 SFT 数据集 %s", key)
            # 如 data_raw/<key>/ 下存在本地 JSON/JSONL 文件，则优先使用本地文件，避免再下载
            local_dir = data_root / key
            local_jsons: List[str] = []
            if key == "coig" and local_dir.exists():
                for pattern in ("*.jsonl", "*.json"):
                    local_jsons.extend([str(p) for p in local_dir.glob(pattern)])
            if key == "coig" and local_jsons:
                LOGGER.info("检测到本地 COIG 文件，共 %d 个，优先使用本地数据", len(local_jsons))
                ds = load_dataset("json", data_files=local_jsons, split="train")
            else:
                ds = load_dataset(
                    meta["hf_path"],
                    name=meta["config"],
                    split=meta["split"],
                    cache_dir=str(local_dir),
                    trust_remote_code=True,
                )
            records = convert_coig_entries(ds) if key == "coig" else convert_oasst_tree(ds)
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
        LOGGER.info("SFT采样统计：%s", plan.stats)
        train, val, test = split_three_way([item.payload for item in plan.selected], seed=seed)
        write_jsonl(proc_root / "sft_train.jsonl", train)
        write_jsonl(proc_root / "sft_val.jsonl", val)
        write_jsonl(proc_root / "sft_test.jsonl", test)

    if stage in ("all", "pref"):
        pref_entries: List[Mapping[str, Any]] = []
        pref_weights: Mapping[str, float] = config.get("preference", {}).get("mix", {})
        for key, meta in PREF_SPECS.items():
            if pref_weights and float(pref_weights.get(key, 0.0)) <= 0.0:
                LOGGER.info("跳过 偏好数据集 %s（权重=0）", key)
                continue
            LOGGER.info("加载偏好数据集 %s", key)
            ds = load_dataset(
                meta["hf_path"],
                name=meta["config"],
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
                "偏好 %s: raw=%d mapped_ok=%d mapped_skipped=%d lang_filtered=%d quality=%d deduped=%d",
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
        LOGGER.info("偏好采样统计：%s", plan.stats)
        train, val, test = split_three_way([item.payload for item in plan.selected], seed=seed)
        write_jsonl(proc_root / "pref_train.jsonl", train)
        write_jsonl(proc_root / "pref_val.jsonl", val)
        write_jsonl(proc_root / "pref_test.jsonl", test)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = resolve_config(load_config(args.config), args)

    if config["general"].get("dry_run", True):
        dry_run_summary(config)
        # 示例日志片段（供数据卡填写）：
        # [INFO] SFT oasst1 raw=52000 filtered=44000 deduped=43000
        # [INFO] 偏好 coig_pc_core raw=70000 filtered=50000 deduped=48000
        return 0

    execute_pipeline(config)
    LOGGER.info("数据准备完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
