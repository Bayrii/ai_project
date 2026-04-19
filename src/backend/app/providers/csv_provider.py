from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.providers.base import BaseTabularProvider
from app.schemas.common import PropertyType


class CSVTabularProvider(BaseTabularProvider):
    def __init__(self, apartment_csv: Path, house_csv: Path, id_column: str = "house_id") -> None:
        self.id_column = id_column
        self._frames: dict[PropertyType, pd.DataFrame] = {
            "apartment": self._load_frame(apartment_csv),
            "house": self._load_frame(house_csv),
        }

    def _load_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Provider CSV not found: {path}")

        df = pd.read_csv(path)
        if self.id_column not in df.columns:
            raise ValueError(f"CSV file {path} must contain '{self.id_column}' column")

        df[self.id_column] = pd.to_numeric(df[self.id_column], errors="coerce")
        df = df.dropna(subset=[self.id_column]).copy()
        df[self.id_column] = df[self.id_column].astype("int64")
        return df.set_index(self.id_column, drop=False)

    def get_features(self, property_type: PropertyType, listing_id: str | int) -> dict[str, Any]:
        frame = self._frames[property_type]

        try:
            listing_key = int(str(listing_id).strip())
        except ValueError as exc:
            raise KeyError(f"Invalid listing_id: {listing_id}") from exc

        if listing_key not in frame.index:
            raise KeyError(f"listing_id={listing_key} not found for {property_type}")

        row = frame.loc[listing_key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        result: dict[str, Any] = {}
        for key, value in row.to_dict().items():
            if key == self.id_column:
                continue
            if pd.isna(value):
                continue
            result[str(key)] = value

        return result
