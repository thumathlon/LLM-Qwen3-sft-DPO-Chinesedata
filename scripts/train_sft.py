#!/usr/bin/env python
"""Supervised fine-tuning (SFT) with LoRA/QLoRA and dry-run planning."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from scripts.utils_data import (
    LengthBucket,
    MixedBucketSampler,
    SamplingItem,
    estimate_chinese_ratio,
    estimate_token_length,
    hash_for_text,
    merge_messages,
)

try:  # pragma: no cover - 需在远程环境安装依�?    import torch
    from datasets import Dataset, load_dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = Any  # type: ignore
    load_dataset = None  # type: ignore
    LoraConfig = Any  # type: ignore
    AutoModelForCausalLM = Any  # type: ignore
    AutoTokenizer = Any  # type: ignore
    TrainerCallback = object  # type: ignore
    TrainingArguments = Any  # type: ignore
    SFTTrainer = Any  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    general: Dict[str, Any]
    model: Dict[str, Any]
    lora: Dict[str, Any]
    qlora: Dict[str, Any]
    training: Dict[str, Any]
    data: Dict[str, Any]
    metrics: Dict[str, Any]


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "output_dir": "outputs/sft",
        "log_dir": "outputs/sft/logs",
        "log_backend": "none",
        "checkpointing_steps": 1000,
        "eval_steps": 1000,
        "save_total_limit": 3,
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": True,
        "gradient_checkpointing": True,
        "max_seq_length": 2048,
        "packing": True,
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
    "qlora": {
        "enable": False,
        "quant_dtype": "nf4",
        "double_quant": True,
        "quant_threshold": 6.0,
    },
    "training": {
        "epochs": 1.5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "per_device_eval_batch_size": 2,
        "learning_rate": 2.0e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "tf32": True,
        "dataloader_num_workers": 2,
    },
    "data": {
        "train_file": "data_proc/sft_train.jsonl",
        "eval_file": "data_proc/sft_val.jsonl",
        "dataset_text_field": "messages",
        "streaming": False,
    },
    "metrics": {
        "log_tokens_per_second": True,
        "log_generation_preview": True,
        "generation_preview_prompts": [
            "请写一�?120 字的量子计算科普�?,
            "总结本周项目进展，格式为项目周报�?,
        ],
    },
}


class MetricsCallback(TrainerCallback):  # pragma: no cover - 仅在远程运行
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        LOGGER.info("训练日志�?s", {k: round(v, 4) for k, v in logs.items() if isinstance(v, (int, float))})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT trainer")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--log-backend", type=str, choices=["none", "tensorboard", "wandb"], default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def load_config(path: str) -> SFTConfig:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            update = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, update)
    return SFTConfig(**cfg)


def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: SFTConfig, args: argparse.Namespace) -> SFTConfig:
    general = dict(config.general)
    if args.dry_run is not None:
        general["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    if args.log_backend is not None:
        general["log_backend"] = args.log_backend
    return SFTConfig(
        general=general,
        model=config.model,
        lora=config.lora,
        qlora=config.qlora,
        training=config.training,
        data=config.data,
        metrics=config.metrics,
    )


def dry_run_report(config: SFTConfig) -> None:
    LOGGER.info("Dry-run 模式：不会加载模型或数据")
    train_path = Path(config.data["train_file"])
    eval_path = Path(config.data["eval_file"])
    LOGGER.info("训练数据: %s (存在=%s)", train_path, train_path.exists())
    LOGGER.info("验证数据: %s (存在=%s)", eval_path, eval_path.exists())
    total_batch = config.training["per_device_train_batch_size"] * config.training["gradient_accumulation_steps"]
    estimated_samples = config.training.get("estimated_train_samples", 60000)
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8") as fp:
                estimated_samples = sum(1 for _ in fp)
        except Exception:
            pass
    est_steps = math.ceil(estimated_samples / max(1, total_batch))
    LOGGER.info("估算训练步数: ~%s (基于 %d 条样�?", est_steps, estimated_samples)
    sampler = MixedBucketSampler(
        length_buckets=[LengthBucket(name="generic", min_tokens=0, max_tokens=config.model.get("max_seq_length", 2048))],
        target_cn_ratio=0.7,
        seed=config.general.get("seed", 42),
    )
    plan = sampler.plan(
        total_samples=3,
        available_items=[
            SamplingItem("dry-cn", "MXODE", 480, 0.9, {"messages": []}),
            SamplingItem("dry-en", "SHP", 320, 0.2, {"prompt": "demo"}),
        ],
    )
    LOGGER.info("采样器示例统�? %s", plan.stats)
    LOGGER.info(
        "LoRA 配置：r=%s alpha=%s dropout=%.2f",
        config.lora.get("r"),
        config.lora.get("alpha"),
        config.lora.get("dropout"),
    )
    LOGGER.info("QLoRA 启用: %s", config.qlora.get("enable"))


def setup_logging(log_dir: str, backend: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if backend == "tensorboard":  # pragma: no cover
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            SummaryWriter(log_dir)
        except ImportError:
            LOGGER.warning("tensorboard 未安装，忽略该后�?)
    elif backend == "wandb":  # pragma: no cover
        import wandb  # type: ignore

        if not wandb.run:
            wandb.init(project="sft-alignment", dir=log_dir)


def build_dataset(path: str) -> Dataset:
    if load_dataset is None:
        raise RuntimeError("需要在远程环境安装 datasets")
    return load_dataset("json", data_files=path, split="train")


def apply_sampler(dataset: Dataset, sampler: MixedBucketSampler, weights: Optional[Mapping[str, float]]) -> Dataset:
    records: List[SamplingItem] = []
    for row in dataset:
        messages = row.get("messages") if isinstance(row, Mapping) else None
        text = merge_messages(messages) if isinstance(messages, list) else "\n".join(
            str(row.get(key, "")) for key in ("prompt", "input", "output")
        )
        records.append(
            SamplingItem(
                identifier=row.get("id", hash_for_text(text)),
                source=row.get("source", "SFT"),
                text_length=estimate_token_length(text),
                chinese_ratio=estimate_chinese_ratio(text),
                payload=row,
            )
        )
    plan = sampler.plan(total_samples=len(records), available_items=records, source_weights=weights)
    LOGGER.info("采样后样本数�?d", len(plan.selected))
    return Dataset.from_list([item.payload for item in plan.selected])


def build_formatting_function(tokenizer) -> Callable[[Mapping[str, Any]], str]:
    def _format(example: Mapping[str, Any]) -> str:
        messages = example.get("messages")
        if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:  # pragma: no cover - 保底退�?                pass
        text = example.get("text")
        if isinstance(text, str) and text.strip():
            return text
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        if prompt or response:
            return f"{prompt}\n{response}".strip()
        return merge_messages(messages) if isinstance(messages, list) else ""

    return _format


def train(config: SFTConfig) -> None:
    if load_dataset is None or torch is None:
        raise RuntimeError("需要在远程环境安装 transformers/trl/peft 等依赖�?)

    setup_logging(config.general["log_dir"], config.general.get("log_backend", "none"))

    datasets = {}
    sampler = MixedBucketSampler(
        length_buckets=[LengthBucket(name="generic", min_tokens=0, max_tokens=config.model.get("max_seq_length", 2048))],
        target_cn_ratio=0.7,
        seed=config.general.get("seed", 42),
    )
    for split_name, file_key in (("train", "train_file"), ("validation", "eval_file")):
        data_path = config.data.get(file_key)
        if not data_path:
            continue
        LOGGER.info("加载数据�?s", data_path)
        dataset = build_dataset(data_path)
        datasets[split_name] = apply_sampler(dataset, sampler, weights=config.data.get("mix"))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model["base_model"],
        trust_remote_code=config.model.get("trust_remote_code", False),
    )
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {}
    if config.qlora.get("enable"):
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": config.qlora.get("quant_dtype", "nf4"),
                "bnb_4bit_use_double_quant": config.qlora.get("double_quant", True),
            }
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model["base_model"],
        trust_remote_code=config.model.get("trust_remote_code", False),
        torch_dtype=torch.bfloat16 if config.training.get("bf16") else None,
        **model_kwargs,
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

    dataset_text_field = config.data.get("dataset_text_field")
    formatting_func = None
    if not dataset_text_field:
        formatting_func = build_formatting_function(tokenizer)

    training_args = TrainingArguments(
        output_dir=config.general["output_dir"],
        num_train_epochs=config.training["epochs"],
        per_device_train_batch_size=config.training["per_device_train_batch_size"],
        gradient_accumulation_steps=config.training["gradient_accumulation_steps"],
        per_device_eval_batch_size=config.training["per_device_eval_batch_size"],
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
        gradient_checkpointing=config.model.get("gradient_checkpointing", False),
        bf16=config.training.get("bf16", False),
        tf32=config.training.get("tf32", False),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        dataset_text_field=dataset_text_field,
        formatting_func=formatting_func,
        packing=config.model.get("packing", False),
    )
    trainer.add_callback(MetricsCallback())

    trainer.train()
    trainer.save_model(config.general["output_dir"])
    trainer.tokenizer.save_pretrained(config.general["output_dir"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = apply_cli_overrides(load_config(args.config), args)

    Path(config.general["output_dir"]).mkdir(parents=True, exist_ok=True)
    if config.general.get("dry_run", True):
        dry_run_report(config)
        return 0

    train(config)
    LOGGER.info("SFT 训练完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

