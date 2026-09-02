"""Tests for the sparse indexed rating dataset."""

from types import MappingProxyType

import pytest

from recommender.data import InMemoryRatingRepository
from recommender.domain import Rating
from recommender.preprocessing import DuplicateRatingError, RatingDataset


def tiny_ratings() -> list[Rating]:
    return [
        Rating(user_id=1, item_id=1, rating=5),
        Rating(user_id=1, item_id=2, rating=3),
        Rating(user_id=2, item_id=1, rating=4),
        Rating(user_id=2, item_id=3, rating=2),
        Rating(user_id=3, item_id=2, rating=4),
        Rating(user_id=3, item_id=3, rating=5),
    ]


def test_basic_dataset_counts() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.rating_count == 6
    assert dataset.user_count == 3
    assert dataset.item_count == 3


def test_unique_users_and_items_are_read_only_sets() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.users == frozenset({1, 2, 3})
    assert dataset.items == frozenset({1, 2, 3})


def test_user_rating_index() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dict(dataset.user_ratings(1)) == {1: 5, 2: 3}


def test_item_rating_index() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dict(dataset.item_ratings(1)) == {1: 5, 2: 4}


def test_direct_rating_lookup_distinguishes_missing_values() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.get_rating(1, 1) == 5
    assert dataset.get_rating(1, 3) is None


def test_rated_items_represents_i_u() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.rated_items(1) == frozenset({1, 2})


def test_users_who_rated_item() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.users_who_rated(1) == frozenset({1, 2})


def test_co_rated_items_represents_intersection() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.co_rated_items(1, 2) == frozenset({1})


def test_union_rated_items_represents_union() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dataset.union_rated_items(1, 2) == frozenset({1, 2, 3})


def test_unknown_ids_return_empty_values_or_none() -> None:
    dataset = RatingDataset(tiny_ratings())

    assert dict(dataset.user_ratings("unknown-user")) == {}
    assert dict(dataset.item_ratings("unknown-item")) == {}
    assert dataset.get_rating("unknown-user", "unknown-item") is None
    assert dataset.rated_items("unknown-user") == frozenset()
    assert dataset.users_who_rated("unknown-item") == frozenset()
    assert dataset.co_rated_items("unknown-user", 1) == frozenset()
    assert dataset.union_rated_items("unknown-user", "other-user") == frozenset()


def test_duplicate_user_item_ratings_raise_explicit_error() -> None:
    ratings = [
        Rating(user_id=1, item_id=10, rating=3),
        Rating(user_id=1, item_id=10, rating=5),
    ]

    with pytest.raises(DuplicateRatingError, match="duplicate rating"):
        RatingDataset(ratings)


def test_empty_dataset_behaves_predictably() -> None:
    dataset = RatingDataset([])

    assert dataset.rating_count == 0
    assert dataset.user_count == 0
    assert dataset.item_count == 0
    assert dataset.users == frozenset()
    assert dataset.items == frozenset()
    assert dict(dataset.user_ratings(1)) == {}
    assert dict(dataset.item_ratings(1)) == {}
    assert dataset.get_rating(1, 1) is None
    assert dataset.rated_items(1) == frozenset()
    assert dataset.users_who_rated(1) == frozenset()


def test_returned_rating_mappings_cannot_corrupt_dataset_state() -> None:
    dataset = RatingDataset(tiny_ratings())
    user_ratings = dataset.user_ratings(1)
    item_ratings = dataset.item_ratings(1)

    assert isinstance(user_ratings, MappingProxyType)
    assert isinstance(item_ratings, MappingProxyType)

    with pytest.raises(TypeError):
        user_ratings[99] = 1  # type: ignore[index]

    with pytest.raises(TypeError):
        item_ratings[99] = 1  # type: ignore[index]

    assert dict(dataset.user_ratings(1)) == {1: 5, 2: 3}
    assert dict(dataset.item_ratings(1)) == {1: 5, 2: 4}


def test_original_input_list_mutation_does_not_corrupt_dataset() -> None:
    ratings = tiny_ratings()
    dataset = RatingDataset(ratings)

    ratings.append(Rating(user_id=99, item_id=99, rating=1))

    assert dataset.rating_count == 6
    assert dataset.get_rating(99, 99) is None


def test_mixed_string_identifiers_do_not_require_sequential_integer_ids() -> None:
    dataset = RatingDataset(
        [
            Rating(user_id="user_A", item_id="movie_10", rating=4.25),
            Rating(user_id="user_B", item_id="movie_10", rating=3.75),
        ]
    )

    assert dataset.users == frozenset({"user_A", "user_B"})
    assert dataset.items == frozenset({"movie_10"})
    assert dataset.get_rating("user_A", "movie_10") == 4.25
    assert dataset.users_who_rated("movie_10") == frozenset({"user_A", "user_B"})


def test_dataset_can_be_created_from_repository_abstraction() -> None:
    repository = InMemoryRatingRepository(tiny_ratings())

    dataset = RatingDataset.from_repository(repository)

    assert dataset.rating_count == 6
    assert dataset.get_rating(1, 1) == 5
