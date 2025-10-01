#!/usr/bin/env python
"""Offline ORPO training script with dry-run planning."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from scripts.utils_data import LengthBucket, MixedBucketSampler, SamplingItem, estimate_chinese_ratio, estimate_token_length

try:  # pragma: no cover
    import torch
    from datasets import Dataset, load_dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore
    from trl import ORPOTrainer, PairwiseDataCollatorWithPadding  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = Any  # type: ignore
    load_dataset = None  # type: ignore
    LoraConfig = Any  # type: ignore
    AutoModelForCausalLM = Any  # type: ignore
    AutoTokenizer = Any  # type: ignore
    ORPOTrainer = Any  # type: ignore
    PairwiseDataCollatorWithPadding = Any  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class OrpoConfig:
    general: Dict[str, Any]
    model: Dict[str, Any]
    lora: Dict[str, Any]
    training: Dict[str, Any]
    orpo: Dict[str, Any]
    data: Dict[str, Any]
    metrics: Dict[str, Any]


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "output_dir": "outputs/align",
        "log_dir": "outputs/align/logs",
        "log_backend": "none",
        "checkpointing_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
    },
    "model": {
        "base_model": "outputs/sft",
        "trust_remote_code": True,
        "gradient_checkpointing": True,
        "max_seq_length": 2048,
    },
    "lora": {
        "enable": True,
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
    },
    "training": {
        "epochs": 1.0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "per_device_eval_batch_size": 2,
        "learning_rate": 8.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "tf32": True,
    },
    "orpo": {
        "beta": 0.2,
        "length_penalty": 0.02,
        "prefer_chinese_ratio": 0.7,
    },
    "data": {
        "train_file": "data_proc/pref_train.jsonl",
        "eval_file": "data_proc/pref_val.jsonl",
        "streaming": False,
        "mix": {
            "coig_pc_core": 0.4,
            "ultrafeedback": 0.2,
            "shp": 0.2,
            "hh_rlhf": 0.1,
            "saferlhf": 0.1,
        },
    },
    "metrics": {
        "log_tokens_per_second": True,
        "sample_eval_prompts": [
            "设计一个面向初学者的 AI 伦理研讨课程大纲。",
            "当用户请求违法内容时，模型应如何回应？",
        ],
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORPO alignment trainer")
    parser.add_argument("--config", type=str, default="configs/orpo.yaml")
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--log-backend", type=str, choices=["none", "tensorboard", "wandb"], default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def load_config(path: str) -> OrpoConfig:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            update = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, update)
    return OrpoConfig(**cfg)


def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: OrpoConfig, args: argparse.Namespace) -> OrpoConfig:
    general = dict(config.general)
    if args.dry_run is not None:
        general["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    if args.log_backend is not None:
        general["log_backend"] = args.log_backend
    return OrpoConfig(
        general=general,
        model=config.model,
        lora=config.lora,
        training=config.training,
        orpo=config.orpo,
        data=config.data,
        metrics=config.metrics,
    )


def dry_run_report(config: OrpoConfig) -> None:
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    train_path = Path(config.data["train_file"])
    eval_path = Path(config.data["eval_file"])
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    total_batch = config.training["per_device_train_batch_size"] * config.training["gradient_accumulation_steps"]
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    sampler = MixedBucketSampler(
        length_buckets=[LengthBucket(name="generic", min_tokens=0, max_tokens=config.model.get("max_seq_length", 2048))],
        target_cn_ratio=config.orpo.get("prefer_chinese_ratio", 0.7),
        seed=config.general.get("seed", 42),
    )
    plan = sampler.plan(
        total_samples=4,
        available_items=[
            SamplingItem("pref-cn", "DPO_EN_ZH_20K", 512, 0.9, {}),
            SamplingItem("pref-en", "SHP", 320, 0.2, {}),
        ],
        source_weights=config.data.get("mix", {}),
    )
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    LOGGER.info("采样后样本数: %d", len(plan.selected))


def setup_logging(log_dir: str, backend: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if backend == "tensorboard":  # pragma: no cover
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            SummaryWriter(log_dir)
        except ImportError:
            LOGGER.warning("tensorboard 未安装，忽略")
    elif backend == "wandb":  # pragma: no cover
        import wandb  # type: ignore

        if not wandb.run:
            wandb.init(project="orpo-alignment", dir=log_dir)


def build_dataset(path: str, sampler: MixedBucketSampler, weights: Mapping[str, float]) -> Dataset:
    if load_dataset is None:
        raise RuntimeError("需要在远程环境安装必要依赖")
    raw = load_dataset("json", data_files=path, split="train")
    items = [
        SamplingItem(
            identifier=row.get("source", "PREF") + row.get("prompt", "")[:12],
            source=row.get("source", "PREF"),
            text_length=estimate_token_length("\n".join([row.get("prompt", ""), row.get("chosen", ""), row.get("rejected", "")])),
            chinese_ratio=estimate_chinese_ratio("\n".join([row.get("prompt", ""), row.get("chosen", ""), row.get("rejected", "")])),
            payload=row,
        )
        for row in raw
    ]
    plan = sampler.plan(total_samples=len(items), available_items=items, source_weights=weights)
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    return Dataset.from_list([item.payload for item in plan.selected])


def train(config: OrpoConfig) -> None:
    if load_dataset is None or torch is None:
        raise RuntimeError("需要在远程环境安装必要依赖")

    setup_logging(config.general["log_dir"], config.general.get("log_backend", "none"))
    sampler = MixedBucketSampler(
        length_buckets=[LengthBucket(name="generic", min_tokens=0, max_tokens=config.model.get("max_seq_length", 2048))],
        target_cn_ratio=config.orpo.get("prefer_chinese_ratio", 0.7),
        seed=config.general.get("seed", 42),
    )

    datasets = {}
    weights = config.data.get("mix", {})
    for split_name, file_key in (("train", "train_file"), ("eval", "eval_file")):
        path = config.data.get(file_key)
        if path:
            datasets[split_name] = build_dataset(path, sampler, weights)

    tokenizer = AutoTokenizer.from_pretrained(config.model["base_model"], trust_remote_code=config.model.get("trust_remote_code", False))
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model["base_model"],
        trust_remote_code=config.model.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16 if config.training.get("bf16") else None,
    )

    if config.lora.get("enable"):
        lora_cfg = LoraConfig(
            r=config.lora.get("r", 16),
            lora_alpha=config.lora.get("alpha", 16),
            lora_dropout=config.lora.get("dropout", 0.05),
            bias=config.lora.get("bias", "none"),
            target_modules=config.lora.get("target_modules"),
        )
        model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=config.general["output_dir"],
        num_train_epochs=config.training["epochs"],
        per_device_train_batch_size=config.training["per_device_train_batch_size"],
        per_device_eval_batch_size=config.training["per_device_eval_batch_size"],
        gradient_accumulation_steps=config.training["gradient_accumulation_steps"],
        learning_rate=config.training["learning_rate"],
        lr_scheduler_type=config.training["lr_scheduler_type"],
        warmup_ratio=config.training["warmup_ratio"],
        weight_decay=config.training["weight_decay"],
        max_grad_norm=config.training["max_grad_norm"],
        logging_dir=config.general["log_dir"],
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=config.general["eval_steps"],
        save_steps=config.general["checkpointing_steps"],
        save_total_limit=config.general["save_total_limit"],
        bf16=config.training.get("bf16", False),
        tf32=config.training.get("tf32", False),
        gradient_checkpointing=config.model.get("gradient_checkpointing", False),
    )

    collator = PairwiseDataCollatorWithPadding(tokenizer=tokenizer)

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        beta=config.orpo.get("beta", 0.2),
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("eval"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(config.general["output_dir"])
    tokenizer.save_pretrained(config.general["output_dir"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = apply_cli_overrides(load_config(args.config), args)

    Path(config.general["output_dir"]).mkdir(parents=True, exist_ok=True)
    if config.general.get("dry_run", True):
        dry_run_report(config)
        return 0

    train(config)
    LOGGER.info("采样后样本数: %d", len(plan.selected))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


