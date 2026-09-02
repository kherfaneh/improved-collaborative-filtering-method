"""In-memory implementation of the rating repository contract."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from recommender.domain.models import Rating
from recommender.domain.types import ItemId, UserId


class InMemoryRatingRepository:
    """Store explicit ratings in immutable in-memory collections."""

    def __init__(self, ratings: Iterable[Rating]) -> None:
        self._ratings = tuple(ratings)

        user_index: dict[UserId, list[Rating]] = defaultdict(list)
        item_index: dict[ItemId, list[Rating]] = defaultdict(list)

        for rating in self._ratings:
            user_index[rating.user_id].append(rating)
            item_index[rating.item_id].append(rating)

        self._ratings_by_user = {
            user_id: tuple(user_ratings)
            for user_id, user_ratings in user_index.items()
        }
        self._ratings_by_item = {
            item_id: tuple(item_ratings)
            for item_id, item_ratings in item_index.items()
        }

    def get_all_ratings(self) -> tuple[Rating, ...]:
        """Return all ratings in insertion order."""
        return self._ratings

    def get_user_ratings(self, user_id: UserId) -> tuple[Rating, ...]:
        """Return ratings made by one user."""
        return self._ratings_by_user.get(user_id, ())

    def get_item_ratings(self, item_id: ItemId) -> tuple[Rating, ...]:
        """Return ratings received by one item."""
        return self._ratings_by_item.get(item_id, ())
