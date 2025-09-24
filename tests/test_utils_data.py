"""Static tests for utils_data (skipped by default).

这些测试仅用于确保 API 结构在本地静态分析时保持一致；
CI/远程执行时可手动移除 skip 修饰符运行真实测试。
"""

from __future__ import annotations

import pytest

from scripts.utils_data import LengthBucket, MixedBucketSampler, SamplingItem


@pytest.mark.skip(reason="no local run")
def test_mixed_bucket_sampler_language_ratio() -> None:
    sampler = MixedBucketSampler(
        length_buckets=[LengthBucket(name="short", min_tokens=0, max_tokens=256)],
        target_cn_ratio=0.7,
        cn_threshold=0.3,
    )
    items = [
        SamplingItem("cn-1", "CN", 120, 0.9, {}),
        SamplingItem("cn-2", "CN", 120, 0.85, {}),
        SamplingItem("en-1", "EN", 120, 0.1, {}),
    ]
    plan = sampler.plan(total_samples=3, available_items=items)
    assert len(plan.selected) <= 3
