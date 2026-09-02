"""Sparse indexed representation of explicit rating data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from recommender.data.repository import RatingRepository
from recommender.domain.models import Rating
from recommender.domain.types import ItemId, RatingValue, UserId


class DuplicateRatingError(ValueError):
    """Raised when multiple ratings exist for one `(user_id, item_id)` pair."""


class RatingDataset:
    """Read-only sparse view of ratings indexed by users and items.

    This class represents the mathematical rating data needed by future UBCF
    stages. It does not represent a physical data source and does not implement
    similarity, prediction, statistics, sparsity, or train/test splitting.
    """

    def __init__(self, ratings: Iterable[Rating]) -> None:
        self._ratings = tuple(ratings)
        user_ratings: dict[UserId, dict[ItemId, RatingValue]] = {}
        item_ratings: dict[ItemId, dict[UserId, RatingValue]] = {}
        rating_count = 0

        for rating in self._ratings:
            by_user = user_ratings.setdefault(rating.user_id, {})
            if rating.item_id in by_user:
                raise DuplicateRatingError(
                    "duplicate rating for "
                    f"user_id={rating.user_id!r}, item_id={rating.item_id!r}"
                )

            by_user[rating.item_id] = rating.rating
            item_ratings.setdefault(rating.item_id, {})[rating.user_id] = rating.rating
            rating_count += 1

        self._user_ratings = {
            user_id: MappingProxyType(dict(item_ratings_for_user))
            for user_id, item_ratings_for_user in user_ratings.items()
        }
        self._item_ratings = {
            item_id: MappingProxyType(dict(user_ratings_for_item))
            for item_id, user_ratings_for_item in item_ratings.items()
        }
        self._users = frozenset(self._user_ratings)
        self._items = frozenset(self._item_ratings)
        self._rating_count = rating_count

    @classmethod
    def from_repository(cls, repository: RatingRepository) -> RatingDataset:
        """Create a dataset from any repository matching `RatingRepository`."""
        return cls(repository.get_all_ratings())

    @property
    def rating_count(self) -> int:
        """Number of explicit ratings."""
        return self._rating_count

    @property
    def ratings(self) -> tuple[Rating, ...]:
        """Return the original ratings as an immutable sequence."""
        return self._ratings

    @property
    def user_count(self) -> int:
        """Number of unique users."""
        return len(self._users)

    @property
    def item_count(self) -> int:
        """Number of unique items."""
        return len(self._items)

    @property
    def users(self) -> frozenset[UserId]:
        """Unique user identifiers as an order-independent set."""
        return self._users

    @property
    def items(self) -> frozenset[ItemId]:
        """Unique item identifiers as an order-independent set."""
        return self._items

    def user_ratings(self, user_id: UserId) -> Mapping[ItemId, RatingValue]:
        """Return the user's ratings as `{item_id: rating_value}`."""
        return self._user_ratings.get(user_id, MappingProxyType({}))

    def item_ratings(self, item_id: ItemId) -> Mapping[UserId, RatingValue]:
        """Return the item's ratings as `{user_id: rating_value}`."""
        return self._item_ratings.get(item_id, MappingProxyType({}))

    def get_rating(self, user_id: UserId, item_id: ItemId) -> RatingValue | None:
        """Return `r_ui` when present, otherwise `None` for an unrated pair."""
        return self._user_ratings.get(user_id, {}).get(item_id)

    def rated_items(self, user_id: UserId) -> frozenset[ItemId]:
        """Return `I_u`, the set of items rated by user `u`."""
        return frozenset(self._user_ratings.get(user_id, ()))

    def users_who_rated(self, item_id: ItemId) -> frozenset[UserId]:
        """Return the set of users who rated an item."""
        return frozenset(self._item_ratings.get(item_id, ()))

    def co_rated_items(self, user_u: UserId, user_v: UserId) -> frozenset[ItemId]:
        """Return `I_u intersection I_v`, the co-rated item set."""
        return self.rated_items(user_u) & self.rated_items(user_v)

    def union_rated_items(self, user_u: UserId, user_v: UserId) -> frozenset[ItemId]:
        """Return `I_u union I_v`, the rated-item union set."""
        return self.rated_items(user_u) | self.rated_items(user_v)
