"""Technology-agnostic rating repository contract."""

from __future__ import annotations

from typing import Protocol

from recommender.domain.models import Rating
from recommender.domain.types import ItemId, UserId


class RatingRepository(Protocol):
    """Structural interface for rating data access.

    A `Protocol` keeps the boundary lightweight: future storage adapters only
    need matching methods and do not need to inherit from a shared base class.
    """

    def get_all_ratings(self) -> tuple[Rating, ...]:
        """Return every rating available from the data source."""

    def get_user_ratings(self, user_id: UserId) -> tuple[Rating, ...]:
        """Return ratings made by `user_id`, or an empty tuple if unknown."""

    def get_item_ratings(self, item_id: ItemId) -> tuple[Rating, ...]:
        """Return ratings for `item_id`, or an empty tuple if unknown."""
