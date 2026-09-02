"""Tests for random train/test splitting."""

from __future__ import annotations

import random

import pytest

from recommender.config.settings import DEFAULT_RANDOM_SEED
from recommender.preprocessing import RatingDataset, random_train_test_split


def make_dataset(size: int = 10) -> RatingDataset:
    from recommender.domain import Rating

    ratings = [
        Rating(
            user_id=f"user_{index % 4}",
            item_id=f"movie_{index}",
            rating=float(index + 1),
        )
        for index in range(size)
    ]
    return RatingDataset(ratings)


def membership(dataset: RatingDataset) -> set[tuple[str, str]]:
    return {(rating.user_id, rating.item_id) for rating in dataset.ratings}


def test_split_80_20_counts_for_ten_ratings() -> None:
    dataset = make_dataset(10)

    split = random_train_test_split(dataset, 0.8, seed=42)

    assert split.train.rating_count == 8
    assert split.test.rating_count == 2
    assert split.train.rating_count + split.test.rating_count == dataset.rating_count


def test_split_completeness_and_disjointness() -> None:
    dataset = make_dataset(10)

    split = random_train_test_split(dataset, 0.8, seed=42)

    train_members = membership(split.train)
    test_members = membership(split.test)

    assert train_members.isdisjoint(test_members)
    assert train_members | test_members == membership(dataset)


def test_same_seed_is_reproducible() -> None:
    dataset = make_dataset(10)

    split1 = random_train_test_split(dataset, 0.8, seed=42)
    split2 = random_train_test_split(dataset, 0.8, seed=42)

    assert membership(split1.train) == membership(split2.train)
    assert membership(split1.test) == membership(split2.test)


def test_different_seeds_change_membership() -> None:
    dataset = make_dataset(10)

    split1 = random_train_test_split(dataset, 0.8, seed=42)
    split2 = random_train_test_split(dataset, 0.8, seed=99)

    assert membership(split1.train) != membership(split2.train)


def test_original_dataset_remains_unchanged() -> None:
    dataset = make_dataset(10)
    original = dataset.ratings

    _ = random_train_test_split(dataset, 0.8, seed=42)

    assert dataset.ratings == original


def test_split_preserves_rating_values_exactly() -> None:
    dataset = make_dataset(10)

    split = random_train_test_split(dataset, 0.8, seed=42)

    combined = {
        (rating.user_id, rating.item_id): rating.rating
        for rating in (*split.train.ratings, *split.test.ratings)
    }

    assert combined == {
        (rating.user_id, rating.item_id): rating.rating for rating in dataset.ratings
    }


@pytest.mark.parametrize(
    "train_ratio",
    [0, 1, -0.1, 1.2, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_train_ratios_fail(train_ratio: float) -> None:
    dataset = make_dataset(10)

    with pytest.raises(ValueError):
        random_train_test_split(dataset, train_ratio=train_ratio, seed=42)


def test_empty_dataset_splits_to_two_empty_datasets() -> None:
    dataset = RatingDataset([])

    split = random_train_test_split(dataset, 0.8, seed=DEFAULT_RANDOM_SEED)

    assert split.train.rating_count == 0
    assert split.test.rating_count == 0


def test_single_rating_follows_floor_behavior() -> None:
    dataset = make_dataset(1)

    split = random_train_test_split(dataset, 0.8, seed=42)

    assert split.train.rating_count == 0
    assert split.test.rating_count == 1


def test_mixed_string_identifiers_work() -> None:
    from recommender.domain import Rating

    dataset = RatingDataset(
        [
            Rating(user_id="user_101", item_id="movie_X", rating=4.0),
            Rating(user_id="user_202", item_id="movie_Y", rating=5.0),
        ]
    )

    split = random_train_test_split(dataset, 0.5, seed=42)

    assert split.train.rating_count == 1
    assert split.test.rating_count == 1
    assert membership(split.train) | membership(split.test) == membership(dataset)


def test_split_does_not_modify_global_random_state() -> None:
    dataset = make_dataset(10)
    state_before = random.getstate()

    _ = random_train_test_split(dataset, 0.8, seed=42)

    state_after = random.getstate()
    assert state_before == state_after
