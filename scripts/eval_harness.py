#!/usr/bin/env python
"""Wrapper for lm-eval-harness with dry-run support."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "seed": 42,
        "output_json": "outputs/eval/results.json",
        "output_csv": "outputs/eval/results.csv",
        "log_dir": "outputs/eval/logs",
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "peft_adapter": "outputs/align",
        "trust_remote_code": True,
    },
    "tasks": ["cmmlu", "ceval-valid-lite", "hellaswag", "winogrande"],
    "metrics": {
        "save_raw": True,
        "summarize": True,
    },
    "lm_eval": {
        "limit": None,
        "batch_size": 4,
        "use_cache": True,
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lm-eval-harness wrapper")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def load_config(path: str) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            update = yaml.safe_load(f) or {}
        config = deep_update(config, update)
    return config


def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.dry_run is not None:
        config.setdefault("general", {})["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    return config


def dry_run_summary(config: Mapping[str, Any]) -> None:
    LOGGER.info("Dry-run 模式：不会实际调用 lm-eval-harness")
    LOGGER.info("计划评测任务：%s", ", ".join(config.get("tasks", [])))
    LOGGER.info("输出 JSON：%s", config["general"].get("output_json"))
    LOGGER.info("输出 CSV：%s", config["general"].get("output_csv"))
    # 预期结果 JSON schema 示例：
    # {
    #   "results": {
    #     "cmmlu": {
    #       "acc": 0.62,
    #       "acc_stderr": 0.02
    #     },
    #     "hellaswag": {"acc": 0.75, "acc_stderr": 0.01}
    #   },
    #   "config": {...},
    #   "versions": {...}
    # }


def run_evaluation(config: Mapping[str, Any]) -> None:
    try:  # pragma: no cover
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要在远程环境安装 lm-eval-harness") from exc

    model_cfg = config["model"]
    model_args: Dict[str, Any] = {
        "pretrained": model_cfg["base_model"],
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "batch_size": config["lm_eval"].get("batch_size", 4),
    }
    peft_adapter = model_cfg.get("peft_adapter")
    if peft_adapter:
        model_args["peft"] = peft_adapter
    lm = HFLM(**model_args)
    eval_tasks = config.get("tasks", [])
    eval_kwargs = {
        "tasks": eval_tasks,
        "model": lm,
        "bootstrap_iters": 100,
        "limit": config["lm_eval"].get("limit"),
        "cache": config["lm_eval"].get("use_cache", True),
        "log_samples": False,
    }
    results = evaluator.simple_evaluate(**eval_kwargs)

    output_json = Path(config["general"]["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if config["metrics"].get("summarize", True):
        flat_rows = []
        for task, metrics in results.get("results", {}).items():
            for metric_name, value in metrics.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_val in value.items():
                        flat_rows.append((task, f"{metric_name}/{sub_key}", sub_val))
                else:
                    flat_rows.append((task, metric_name, value))
        output_csv = Path(config["general"]["output_csv"])
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "metric", "value"])
            writer.writerows(flat_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    if config["general"].get("dry_run", True):
        dry_run_summary(config)
        return 0

    run_evaluation(config)
    LOGGER.info("评测完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
