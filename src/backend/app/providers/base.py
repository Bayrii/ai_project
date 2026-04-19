from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas.common import PropertyType


class BaseTabularProvider(ABC):
    @abstractmethod
    def get_features(self, property_type: PropertyType, listing_id: str | int) -> dict[str, Any]:
        """Return engineered tabular features for a given listing id."""
