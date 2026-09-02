"""Minimal project configuration foundation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class ResearchSettings:
    """Configuration holder for current and future experiment parameters."""

    random_seed: int = DEFAULT_RANDOM_SEED
    train_test_ratio: float | None = None
    neighbor_count: int | None = None
    top_n: int | None = None
    rho_threshold: float | None = None
