"""Immutable domain models used by the recommender project."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from recommender.domain.types import ItemId, RatingValue, UserId


@dataclass(frozen=True, slots=True)
class Rating:
    """Explicit rating given by one user to one item.

    The model intentionally avoids dataset-specific rating ranges. It only
    validates fundamental domain constraints that apply to explicit ratings.
    """

    user_id: UserId
    item_id: ItemId
    rating: RatingValue

    def __post_init__(self) -> None:
        if isinstance(self.rating, bool) or not isinstance(self.rating, Real):
            raise TypeError("rating must be a numeric value")

        if not isfinite(float(self.rating)):
            raise ValueError("rating must be finite")
