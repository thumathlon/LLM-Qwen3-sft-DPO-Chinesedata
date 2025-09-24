"""Static import tests for training scripts."""

from __future__ import annotations

import pytest

import scripts.train_sft as train_sft
import scripts.train_orpo as train_orpo
import scripts.train_dpo as train_dpo


@pytest.mark.skip(reason="no local run")
def test_sft_dry_run_report() -> None:
    train_sft.dry_run_report(train_sft.load_config("configs/sft.yaml"))


@pytest.mark.skip(reason="no local run")
def test_orpo_dry_run_report() -> None:
    train_orpo.dry_run_report(train_orpo.load_config("configs/orpo.yaml"))


@pytest.mark.skip(reason="no local run")
def test_dpo_dry_run_report() -> None:
    train_dpo.dry_run_report(train_dpo.load_config("configs/dpo.yaml"))
