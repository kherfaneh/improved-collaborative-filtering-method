"""Tests for domain models."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from recommender.domain import Rating


def test_create_valid_rating_with_integer_ids() -> None:
    rating = Rating(user_id=1, item_id=10, rating=4.5)

    assert rating.user_id == 1
    assert rating.item_id == 10
    assert rating.rating == 4.5


def test_create_valid_rating_with_string_ids() -> None:
    rating = Rating(user_id="user_102", item_id="movie_50", rating=5)

    assert rating.user_id == "user_102"
    assert rating.item_id == "movie_50"
    assert rating.rating == 5


def test_rating_is_immutable() -> None:
    rating = Rating(user_id=1, item_id=10, rating=4)

    with pytest.raises(FrozenInstanceError):
        rating.rating = 3  # type: ignore[misc]


@pytest.mark.parametrize("invalid_rating", [nan, inf, -inf])
def test_rejects_non_finite_rating_values(invalid_rating: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Rating(user_id=1, item_id=10, rating=invalid_rating)


@pytest.mark.parametrize("invalid_rating", ["5", None, True])
def test_rejects_non_numeric_rating_values(invalid_rating: object) -> None:
    with pytest.raises(TypeError, match="numeric"):
        Rating(user_id=1, item_id=10, rating=invalid_rating)  # type: ignore[arg-type]
