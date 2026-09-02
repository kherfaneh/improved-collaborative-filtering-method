"""Repository abstractions and data-source adapters."""

from recommender.data.in_memory_repository import InMemoryRatingRepository
from recommender.data.repository import RatingRepository

__all__ = ["InMemoryRatingRepository", "RatingRepository"]
