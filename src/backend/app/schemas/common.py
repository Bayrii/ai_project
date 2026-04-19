from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PropertyType = Literal["apartment", "house"]
ModeType = Literal["tabular", "multimodal"]
YesNoType = Literal["Yes", "No"]


class TabularPredictRequest(BaseModel):
    listing_id: str | int | None = None
    tabular_features: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_input_source(self) -> "TabularPredictRequest":
        if self.listing_id is None and not self.tabular_features:
            raise ValueError("Provide either listing_id or tabular_features.")
        return self


class RawFeatureRequest(BaseModel):
    """Raw user-facing schema that mirrors the source listing fields.

    Note: `price` is accepted for parity with source schema but ignored at inference time.
    """

    model_config = ConfigDict(extra="forbid")

    price: float = Field(..., description="Listing price (accepted but ignored during prediction).")
    rooms: int = Field(..., ge=0, description="Number of rooms")
    area_m2: float = Field(..., gt=0, description="Property area in m2")
    land_area_sot: float = Field(..., ge=0, description="Land area in sot")
    floor: int = Field(..., ge=0, description="Floor number")
    has_document: YesNoType = Field(..., description="Yes or No")
    address: str = Field(..., min_length=1, description="Address in 'City, Region' format")
    temirli: YesNoType = Field(..., description="Yes or No")
    qaz: YesNoType = Field(..., description="Yes or No")
    su: YesNoType = Field(..., description="Yes or No")
    isiq: YesNoType = Field(..., description="Yes or No")
    avtodayanacaq: YesNoType = Field(..., description="Yes or No")
    telefon: YesNoType = Field(..., description="Yes or No")
    internet: YesNoType = Field(..., description="Yes or No")
    pvc_pencere: YesNoType = Field(..., description="Yes or No")
    balkon: YesNoType = Field(..., description="Yes or No")
    kabel_tv: YesNoType = Field(..., description="Yes or No")
    lift: YesNoType = Field(..., description="Yes or No")
    kombi: YesNoType = Field(..., description="Yes or No")
    metbex_mebeli: YesNoType = Field(..., description="Yes or No")
    merkezi_qizdirici_sistem: YesNoType = Field(..., description="Yes or No")
    kondisioner: YesNoType = Field(..., description="Yes or No")
    esyali: YesNoType = Field(..., description="Yes or No")
    hovuz: YesNoType = Field(..., description="Yes or No")
    duzelme: YesNoType = Field(..., description="Yes or No")


class RawApartmentPredictRequest(RawFeatureRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "price": 125000,
                "rooms": 3,
                "area_m2": 95,
                "land_area_sot": 0,
                "floor": 7,
                "has_document": "Yes",
                "address": "Bakı, Yasamal",
                "temirli": "Yes",
                "qaz": "Yes",
                "su": "Yes",
                "isiq": "Yes",
                "avtodayanacaq": "No",
                "telefon": "No",
                "internet": "Yes",
                "pvc_pencere": "Yes",
                "balkon": "Yes",
                "kabel_tv": "No",
                "lift": "Yes",
                "kombi": "Yes",
                "metbex_mebeli": "No",
                "merkezi_qizdirici_sistem": "No",
                "kondisioner": "No",
                "esyali": "No",
                "hovuz": "No",
                "duzelme": "No",
            }
        },
    )


class RawHousePredictRequest(RawFeatureRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "price": 240000,
                "rooms": 4,
                "area_m2": 160,
                "land_area_sot": 3.5,
                "floor": 2,
                "has_document": "Yes",
                "address": "Bakı, Xəzər",
                "temirli": "Yes",
                "qaz": "Yes",
                "su": "Yes",
                "isiq": "Yes",
                "avtodayanacaq": "Yes",
                "telefon": "Yes",
                "internet": "Yes",
                "pvc_pencere": "Yes",
                "balkon": "Yes",
                "kabel_tv": "Yes",
                "lift": "No",
                "kombi": "Yes",
                "metbex_mebeli": "Yes",
                "merkezi_qizdirici_sistem": "No",
                "kondisioner": "Yes",
                "esyali": "No",
                "hovuz": "No",
                "duzelme": "No",
            }
        },
    )


class PredictionResponse(BaseModel):
    request_id: str
    property_type: PropertyType
    mode: ModeType
    model_name: str
    model_version: str
    tabular_source: Literal["request", "provider"]
    predicted_log_price: float
    predicted_price_azn: float
    warnings: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    key: str
    mode: ModeType
    model_name: str
    model_version: str
    source_path: str
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
