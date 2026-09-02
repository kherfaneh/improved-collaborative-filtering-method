"""Random train/test splitting for rating observations.

Same seed plus the same incoming rating order yields the same membership. A
different physical input order can change the split, so deterministic upstream
ordering is the responsibility of future adapters when exact replay matters.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from recommender.config.settings import DEFAULT_RANDOM_SEED, DEFAULT_TRAIN_RATIO
from recommender.preprocessing.dataset import RatingDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Immutable pair of train and test datasets."""

    train: RatingDataset
    test: RatingDataset


def _validate_train_ratio(train_ratio: float) -> None:
    if not isinstance(train_ratio, (int, float)) or isinstance(train_ratio, bool):
        raise ValueError("train_ratio must be a finite numeric value")
    if not math.isfinite(float(train_ratio)):
        raise ValueError("train_ratio must be finite")
    if not 0 < float(train_ratio) < 1:
        raise ValueError("train_ratio must satisfy 0 < train_ratio < 1")


def random_train_test_split(
    dataset: RatingDataset,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> DatasetSplit:
    """Randomly split rating observations into train and test datasets.

    This reproduces the paper's plain random rating-level 80/20 methodology.
    It does not enforce user/item presence in training, so cold-start cases may
    appear naturally and should be handled later at the evaluation layer.
    """

    _validate_train_ratio(train_ratio)

    ratings = list(dataset.ratings)
    total_ratings = len(ratings)
    rng = random.Random(seed)
    rng.shuffle(ratings)

    train_size = int(total_ratings * float(train_ratio))
    train_ratings = ratings[:train_size]
    test_ratings = ratings[train_size:]

    logger.info(
        "split ratings total=%s train=%s test=%s train_ratio=%s seed=%s",
        total_ratings,
        len(train_ratings),
        len(test_ratings),
        train_ratio,
        seed,
    )

    return DatasetSplit(
        train=RatingDataset(train_ratings),
        test=RatingDataset(test_ratings),
    )
