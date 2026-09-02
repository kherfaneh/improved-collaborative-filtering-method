"""Tests for the in-memory rating repository."""

from recommender.data import InMemoryRatingRepository
from recommender.domain import Rating


def tiny_ratings() -> list[Rating]:
    return [
        Rating(user_id=1, item_id=1, rating=5),
        Rating(user_id=1, item_id=2, rating=3),
        Rating(user_id=2, item_id=1, rating=4),
        Rating(user_id=2, item_id=3, rating=2),
        Rating(user_id=3, item_id=2, rating=4),
        Rating(user_id=3, item_id=3, rating=5),
    ]


def test_returns_all_ratings() -> None:
    ratings = tiny_ratings()
    repository = InMemoryRatingRepository(ratings)

    assert repository.get_all_ratings() == tuple(ratings)


def test_retrieves_ratings_for_one_user() -> None:
    repository = InMemoryRatingRepository(tiny_ratings())

    assert repository.get_user_ratings(1) == (
        Rating(user_id=1, item_id=1, rating=5),
        Rating(user_id=1, item_id=2, rating=3),
    )


def test_retrieves_ratings_for_one_item() -> None:
    repository = InMemoryRatingRepository(tiny_ratings())

    assert repository.get_item_ratings(3) == (
        Rating(user_id=2, item_id=3, rating=2),
        Rating(user_id=3, item_id=3, rating=5),
    )


def test_unknown_user_returns_empty_tuple() -> None:
    repository = InMemoryRatingRepository(tiny_ratings())

    assert repository.get_user_ratings("unknown-user") == ()


def test_unknown_item_returns_empty_tuple() -> None:
    repository = InMemoryRatingRepository(tiny_ratings())

    assert repository.get_item_ratings("unknown-item") == ()


def test_original_input_list_mutation_does_not_corrupt_repository() -> None:
    ratings = tiny_ratings()
    repository = InMemoryRatingRepository(ratings)

    ratings.append(Rating(user_id=99, item_id=99, rating=1))

    assert len(repository.get_all_ratings()) == 6
    assert repository.get_user_ratings(99) == ()
