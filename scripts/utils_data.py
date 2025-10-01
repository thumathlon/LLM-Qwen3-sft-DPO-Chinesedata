"""Shared data utilities for the LLM fine-tuning pipeline.

This module contains reusable helpers for text normalization, language ratio
estimation, n-gram repetition detection, SHA256 deduplication and curriculum
aware sampling. All functions are pure and safe to import in dry-run mode.

Example
-------
>>> bucket = LengthBucket(name="short", min_tokens=0, max_tokens=256)
>>> sampler = MixedBucketSampler(length_buckets=[bucket])
>>> item = SamplingItem(
...     identifier="demo-1",
...     source="OASST1",
...     text_length=128,
...     chinese_ratio=0.9,
...     payload={"messages": []},
... )
>>> plan = sampler.plan(total_samples=1, available_items=[item])
>>> len(plan.selected)
1

The dry-run unit tests import this module to ensure type integrity, but do not
execute any heavy operations.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CONTROL_CHAR_PATTERN = re.compile(r"[\u0000-\u001f\u007f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip ASCII control characters."""

    if not text:
        return ""
    normalized = CONTROL_CHAR_PATTERN.sub("", text.replace("\u3000", " "))
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def estimate_chinese_ratio(text: str) -> float:
    """Estimate Chinese character coverage in *text*.

    The function considers Chinese Han characters as positive samples and
    English alphabet letters as the denominator. Non-letter characters are
    ignored, resulting in a conservative estimate.
    """

    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha() or HAN_PATTERN.match(ch)]
    if not letters:
        return 0.0
    chinese = sum(1 for ch in letters if HAN_PATTERN.match(ch))
    return chinese / len(letters)


def estimate_token_length(text: str) -> int:
    """Rough token length heuristic (4 characters per token)."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def hash_for_text(text: str) -> str:
    """Return a SHA256 hash for *text* encoded as UTF-8."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def merge_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Concatenate message contents into a single newline-delimited string."""

    return "\n".join(msg.get("content", "") for msg in messages if isinstance(msg, Mapping))


def dedupe_by_hash(records: Iterable[Mapping[str, object]], key: str) -> List[Mapping[str, object]]:
    """Remove duplicate entries using the SHA256 hash stored in *key*."""

    seen: set[str] = set()
    result: List[Mapping[str, object]] = []
    for record in records:
        digest = record.get(key)
        if not isinstance(digest, str):
            continue
        if digest in seen:
            continue
        seen.add(digest)
        result.append(record)
    return result


def compute_ngram_repetition(text: str, n: int = 4) -> float:
    """Compute a simple n-gram repetition ratio (auto-adapts to Chinese)."""

    if not text:
        return 0.0

    contains_han = bool(HAN_PATTERN.search(text))
    if contains_han:
        tokens = [ch for ch in text if not ch.isspace()]
    else:
        tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    unique = set(ngrams)
    return 1.0 - len(unique) / len(ngrams)


@dataclass(frozen=True)
class LengthBucket:
    """Inclusive token boundaries for curriculum-aware sampling."""

    name: str
    min_tokens: int
    max_tokens: int

    def contains(self, token_count: int) -> bool:
        return self.min_tokens <= token_count <= self.max_tokens


@dataclass
class CurriculumPhase:
    """Optional curriculum phase weighting certain buckets or sources."""

    start: float
    end: float
    weights: Mapping[str, float] = field(default_factory=dict)

    def applies(self, progress: float) -> bool:
        return self.start <= progress <= self.end


@dataclass
class SamplingItem:
    """Container for sampler metadata."""

    identifier: str
    source: str
    text_length: int
    chinese_ratio: float
    payload: Mapping[str, object]

    def is_chinese(self, threshold: float) -> bool:
        return self.chinese_ratio >= threshold


@dataclass
class SamplingPlan:
    """Result bundle returned from :class:`MixedBucketSampler`."""

    selected: List[SamplingItem]
    rejected: List[SamplingItem]
    stats: Mapping[str, float]


class MixedBucketSampler:
    """Language-aware bucket sampler with curriculum schedule support."""

    def __init__(
        self,
        length_buckets: Sequence[LengthBucket],
        target_cn_ratio: float = 0.7,
        cn_threshold: float = 0.3,
        allow_english_fallback: bool = False,
        curriculum: Optional[Sequence[CurriculumPhase]] = None,
        seed: int = 42,
    ) -> None:
        self.length_buckets = list(length_buckets)
        self.target_cn_ratio = target_cn_ratio
        self.cn_threshold = cn_threshold
        self.allow_english_fallback = allow_english_fallback
        self.curriculum = list(curriculum or [])
        self._rng = random.Random(seed)

    def plan(
        self,
        total_samples: int,
        available_items: Sequence[SamplingItem],
        *,
        progress: float = 1.0,
        source_weights: Optional[Mapping[str, float]] = None,
    ) -> SamplingPlan:
        if total_samples <= 0 or not available_items:
            return SamplingPlan(selected=[], rejected=list(available_items), stats={"requested": float(total_samples), "selected": 0.0})

        chinese_pool: List[SamplingItem] = []
        english_pool: List[SamplingItem] = []
        for item in available_items:
            (chinese_pool if item.is_chinese(self.cn_threshold) else english_pool).append(item)

        desired_cn = int(round(total_samples * self.target_cn_ratio))
        if not self.allow_english_fallback and len(chinese_pool) < desired_cn:
            LOGGER.warning(
                "可用中文样本不足：当前=%d，目标=%d。采样量将按中文池截断；可考虑启用 allow_english_fallback 或降低 target_cn_ratio",
                len(chinese_pool),
                desired_cn,
            )

        if not self.allow_english_fallback:
            total_samples = min(total_samples, len(chinese_pool))

        target_cn = int(round(total_samples * self.target_cn_ratio))
        if not self.allow_english_fallback:
            target_cn = min(target_cn, len(chinese_pool))

        weights = self._merge_weights(source_weights, progress)
        chinese_selected = self._select_weighted(chinese_pool, target_cn, weights)
        remaining_slots = total_samples - len(chinese_selected)

        if self.allow_english_fallback:
            combined_pool = [item for item in available_items if item not in chinese_selected]
        else:
            combined_pool = [item for item in chinese_pool if item not in chinese_selected]

        rest_selected = self._select_weighted(combined_pool, remaining_slots, weights)

        selected = chinese_selected + rest_selected
        rejected = [item for item in available_items if item not in selected]
        stats = {
            "requested": float(total_samples),
            "selected": float(len(selected)),
            "chinese_selected": float(sum(item.is_chinese(self.cn_threshold) for item in selected)),
            "english_selected": float(sum(not item.is_chinese(self.cn_threshold) for item in selected)),
            "avg_length": float(sum(item.text_length for item in selected) / max(1, len(selected))),
        }
        return SamplingPlan(selected=selected, rejected=rejected, stats=stats)

    def _bucket_label(self, item: SamplingItem) -> str:
        for bucket in self.length_buckets:
            if bucket.contains(item.text_length):
                return bucket.name
        return "overflow"

    def _merge_weights(self, provided: Optional[Mapping[str, float]], progress: float) -> Dict[str, float]:
        weights = dict(provided or {})
        for phase in self.curriculum:
            if phase.applies(progress):
                for key, value in phase.weights.items():
                    weights[key] = weights.get(key, 1.0) * float(value)
                break
        return weights

    def _select_weighted(self, pool: Sequence[SamplingItem], k: int, weights: Mapping[str, float]) -> List[SamplingItem]:
        if k <= 0 or not pool:
            return []
        bucket_groups: MutableMapping[str, MutableMapping[str, List[SamplingItem]]] = {}
        for item in pool:
            bucket_groups.setdefault(self._bucket_label(item), {}).setdefault(item.source, []).append(item)
        for source_map in bucket_groups.values():
            for items in source_map.values():
                self._rng.shuffle(items)

        bucket_order = list(bucket_groups.keys())
        selected: List[SamplingItem] = []
        while bucket_order and len(selected) < k:
            for bucket in list(bucket_order):
                source_map = bucket_groups.get(bucket, {})
                if not source_map:
                    bucket_order.remove(bucket)
                    continue
                for source in list(source_map.keys()):
                    items = source_map.get(source, [])
                    if not items:
                        source_map.pop(source, None)
                        continue
                    weight = max(1e-3, weights.get(source, 1.0))
                    steps = max(1, int(round(weight)))
                    for _ in range(steps):
                        if not items or len(selected) >= k:
                            break
                        selected.append(items.pop())
                    if not items:
                        source_map.pop(source, None)
                    if len(selected) >= k:
                        break
                if not source_map:
                    bucket_order.remove(bucket)
                if len(selected) >= k:
                    break
        return selected


# Alias exported for consumers
__all__ = [
    "LengthBucket",
    "CurriculumPhase",
    "SamplingItem",
    "SamplingPlan",
    "MixedBucketSampler",
    "normalize_text",
    "estimate_chinese_ratio",
    "estimate_token_length",
    "hash_for_text",
    "dedupe_by_hash",
    "compute_ngram_repetition",
    "merge_messages",
]
LOGGER = logging.getLogger(__name__)



