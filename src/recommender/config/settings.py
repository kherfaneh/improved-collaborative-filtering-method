"""Minimal project configuration foundation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RANDOM_SEED = 42
# Implementation choice for reproducibility, not a paper-specified value.

DEFAULT_TRAIN_RATIO = 0.8
# Default 80/20 training ratio used by the reproduction splitter.


@dataclass(frozen=True, slots=True)
class ResearchSettings:
    """Configuration holder for current and future experiment parameters."""

    random_seed: int = DEFAULT_RANDOM_SEED
    train_test_ratio: float = DEFAULT_TRAIN_RATIO
    neighbor_count: int | None = None
    top_n: int | None = None
    rho_threshold: float | None = None
