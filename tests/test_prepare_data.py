"""Static import tests for scripts.prepare_data."""

from __future__ import annotations

import pytest

import scripts.prepare_data as prepare_data


@pytest.mark.skip(reason="no local run")
def test_inline_samples_pass_validation() -> None:
    prepare_data.run_inline_checks()
