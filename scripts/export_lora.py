#!/usr/bin/env python
"""Export LoRA adapters or merged weights (dry-run by default)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "dry_run": True,
        "adapter_dir": "outputs/align",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "outputs/adapters",
        "merge_weights": False,
    }
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LoRA adapters")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="可复用 SFT 配置文件")
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def load_config(path: str) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # 允许从 SFT 配置读取路径
        general = raw.get("general", {})
        config["general"].update({
            "adapter_dir": general.get("output_dir", config["general"]["adapter_dir"]),
            "merge_weights": general.get("merge_weights", config["general"].get("merge_weights", False)),
        })
        if "model" in raw:
            config["general"].update({"base_model": raw["model"].get("base_model", config["general"]["base_model"])})
    return config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.dry_run is not None:
        config["general"]["dry_run"] = str(args.dry_run).lower() not in {"false", "0", "no"}
    return config


def dry_run_summary(config: Mapping[str, Any]) -> None:
    LOGGER.info("Dry-run 模式：不会导出权重")
    LOGGER.info("Adapter 目录：%s", config["general"].get("adapter_dir"))
    LOGGER.info("输出目录：%s", config["general"].get("output_dir"))
    # 粗略估算（以 r=16, hidden=4096, layers=32 计算）
    estimated_mb = (2 * 16 * 4096 * 32 * 2) / (1024 ** 2)
    LOGGER.info("预估导出大小：约 %.1f MB（需远程验证）", estimated_mb)


def export_lora(config: Mapping[str, Any]) -> None:
    try:  # pragma: no cover
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要在远程环境安装 transformers/peft") from exc

    adapter_dir = Path(config["general"]["adapter_dir"])
    output_dir = Path(config["general"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(config["general"]["base_model"], trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)

    if config["general"].get("merge_weights", False):
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        peft_model.save_pretrained(output_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    if config["general"].get("dry_run", True):
        dry_run_summary(config)
        return 0

    export_lora(config)
    LOGGER.info("LoRA 导出完成")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
