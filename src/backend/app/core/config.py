from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_PROJECT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "AI Project Prediction API"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    max_images_per_request: int = 20

    @property
    def repo_root(self) -> Path:
        # .../src/backend/app/core/config.py -> repo root is parents[4]
        return Path(__file__).resolve().parents[4]

    @property
    def tabular_apartment_model_path(self) -> Path:
        return self.repo_root / "models" / "tabular" / "apartment" / "apartment_tabular_model.joblib"

    @property
    def tabular_house_model_path(self) -> Path:
        return self.repo_root / "models" / "tabular" / "house" / "house_tabular_model.joblib"

    @property
    def tabular_apartment_summary_path(self) -> Path:
        return self.repo_root / "models" / "tabular" / "apartment" / "apartment_tabular_model_summary.json"

    @property
    def tabular_house_summary_path(self) -> Path:
        return self.repo_root / "models" / "tabular" / "house" / "house_tabular_model_summary.json"

    @property
    def multimodal_apartment_model_path(self) -> Path:
        return self.repo_root / "models" / "multimodal" / "apartment" / "multimodal_model.pt"

    @property
    def multimodal_house_model_path(self) -> Path:
        return self.repo_root / "models" / "multimodal" / "house" / "multimodal_model.pt"

    @property
    def provider_apartment_csv_path(self) -> Path:
        return (
            self.repo_root
            / "data"
            / "apartment"
            / "satilir_properties_apartment_feature_engineered_with_house_id.csv"
        )

    @property
    def provider_house_csv_path(self) -> Path:
        return (
            self.repo_root
            / "data"
            / "house"
            / "satilir_properties_house_feature_engineered_with_house_id.csv"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
