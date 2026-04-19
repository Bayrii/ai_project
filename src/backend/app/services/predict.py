from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image

from app.models.registry import ModelRegistry
from app.providers.base import BaseTabularProvider
from app.schemas.common import (
    PredictionResponse,
    PropertyType,
    RawApartmentPredictRequest,
    RawFeatureRequest,
    RawHousePredictRequest,
    TabularPredictRequest,
)
from app.services.preprocess import map_raw_to_engineered, order_features


def _resolve_tabular_input(
    property_type: PropertyType,
    listing_id: str | int | None,
    tabular_features: dict[str, Any] | None,
    provider: BaseTabularProvider,
) -> tuple[dict[str, Any], str]:
    if tabular_features:
        return tabular_features, "request"

    if listing_id is not None:
        try:
            return provider.get_features(property_type, listing_id), "provider"
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    raise HTTPException(status_code=422, detail="Provide either listing_id or tabular_features")


def predict_tabular(
    *,
    property_type: PropertyType,
    payload: TabularPredictRequest,
    request_id: str,
    registry: ModelRegistry,
    provider: BaseTabularProvider,
) -> PredictionResponse:
    artifact = registry.get_tabular(property_type)

    features, source = _resolve_tabular_input(
        property_type=property_type,
        listing_id=payload.listing_id,
        tabular_features=payload.tabular_features,
        provider=provider,
    )

    model_input, warnings = order_features(features, artifact.feature_names)
    pred_raw = float(artifact.model.predict(model_input)[0])

    if artifact.target_transform == "log1p":
        pred_log = pred_raw
        pred_price = float(np.expm1(np.clip(pred_log, 0.0, 30.0)))
    else:
        pred_price = pred_raw
        pred_log = float(np.log1p(max(pred_price, 0.0)))

    return PredictionResponse(
        request_id=request_id,
        property_type=property_type,
        mode="tabular",
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        tabular_source=source,
        predicted_log_price=pred_log,
        predicted_price_azn=pred_price,
        warnings=warnings,
    )


def predict_tabular_from_raw(
    *,
    property_type: PropertyType,
    payload: RawApartmentPredictRequest | RawHousePredictRequest,
    request_id: str,
    registry: ModelRegistry,
) -> PredictionResponse:
    artifact = registry.get_tabular(property_type)

    raw_payload = payload if isinstance(payload, RawFeatureRequest) else RawFeatureRequest(**payload.model_dump())
    engineered_features, mapping_warnings = map_raw_to_engineered(
        property_type=property_type,
        raw_payload=raw_payload,
        expected_features=artifact.feature_names,
    )

    model_input, ordering_warnings = order_features(engineered_features, artifact.feature_names)
    pred_raw = float(artifact.model.predict(model_input)[0])

    if artifact.target_transform == "log1p":
        pred_log = pred_raw
        pred_price = float(np.expm1(np.clip(pred_log, 0.0, 30.0)))
    else:
        pred_price = pred_raw
        pred_log = float(np.log1p(max(pred_price, 0.0)))

    return PredictionResponse(
        request_id=request_id,
        property_type=property_type,
        mode="tabular",
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        tabular_source="request",
        predicted_log_price=pred_log,
        predicted_price_azn=pred_price,
        warnings=[*mapping_warnings, *ordering_warnings],
    )


def _build_clip_embedding(image_bytes: list[bytes], clip_model: Any, image_transform: Any, max_images: int) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    for payload in image_bytes[:max_images]:
        try:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
            tensors.append(image_transform(img))
        except Exception:
            continue

    if not tensors:
        raise HTTPException(status_code=422, detail="No valid images were provided.")

    batch = torch.stack(tensors)
    with torch.no_grad():
        embeddings = clip_model.encode_image(batch)
        embeddings = F.normalize(embeddings.float(), dim=-1)
    return embeddings.mean(dim=0, keepdim=True)


def predict_multimodal(
    *,
    property_type: PropertyType,
    listing_id: str | int | None,
    tabular_features: dict[str, Any] | None,
    image_bytes: list[bytes],
    max_images: int,
    request_id: str,
    registry: ModelRegistry,
    provider: BaseTabularProvider,
) -> PredictionResponse:
    artifact = registry.get_multimodal(property_type)

    features, source = _resolve_tabular_input(
        property_type=property_type,
        listing_id=listing_id,
        tabular_features=tabular_features,
        provider=provider,
    )

    model_input, warnings = order_features(features, artifact.tab_features)

    if len(image_bytes) > max_images:
        warnings.append(f"Received {len(image_bytes)} images; only the first {max_images} were used.")

    safe_scale = np.where(artifact.scaler_scale == 0.0, 1.0, artifact.scaler_scale)
    tab_norm = (model_input - artifact.scaler_mean.reshape(1, -1)) / safe_scale.reshape(1, -1)

    emb = _build_clip_embedding(
        image_bytes=image_bytes,
        clip_model=artifact.clip_model,
        image_transform=artifact.image_transform,
        max_images=max_images,
    )

    artifact.fusion_model.eval()
    with torch.no_grad():
        pred_norm = artifact.fusion_model(emb, torch.tensor(tab_norm, dtype=torch.float32)).item()

    pred_log = float(pred_norm * artifact.y_std + artifact.y_mean)
    pred_price = float(np.expm1(np.clip(pred_log, 6.0, 16.5)))

    return PredictionResponse(
        request_id=request_id,
        property_type=property_type,
        mode="multimodal",
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        tabular_source=source,
        predicted_log_price=pred_log,
        predicted_price_azn=pred_price,
        warnings=warnings,
    )


def predict_multimodal_from_raw(
    *,
    property_type: PropertyType,
    payload: RawApartmentPredictRequest | RawHousePredictRequest,
    image_bytes: list[bytes],
    max_images: int,
    request_id: str,
    registry: ModelRegistry,
) -> PredictionResponse:
    artifact = registry.get_multimodal(property_type)

    raw_payload = payload if isinstance(payload, RawFeatureRequest) else RawFeatureRequest(**payload.model_dump())
    engineered_features, mapping_warnings = map_raw_to_engineered(
        property_type=property_type,
        raw_payload=raw_payload,
        expected_features=artifact.tab_features,
    )

    model_input, ordering_warnings = order_features(engineered_features, artifact.tab_features)
    warnings = [*mapping_warnings, *ordering_warnings]

    if len(image_bytes) > max_images:
        warnings.append(f"Received {len(image_bytes)} images; only the first {max_images} were used.")

    safe_scale = np.where(artifact.scaler_scale == 0.0, 1.0, artifact.scaler_scale)
    tab_norm = (model_input - artifact.scaler_mean.reshape(1, -1)) / safe_scale.reshape(1, -1)

    emb = _build_clip_embedding(
        image_bytes=image_bytes,
        clip_model=artifact.clip_model,
        image_transform=artifact.image_transform,
        max_images=max_images,
    )

    artifact.fusion_model.eval()
    with torch.no_grad():
        pred_norm = artifact.fusion_model(emb, torch.tensor(tab_norm, dtype=torch.float32)).item()

    pred_log = float(pred_norm * artifact.y_std + artifact.y_mean)
    pred_price = float(np.expm1(np.clip(pred_log, 6.0, 16.5)))

    return PredictionResponse(
        request_id=request_id,
        property_type=property_type,
        mode="multimodal",
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        tabular_source="request",
        predicted_log_price=pred_log,
        predicted_price_azn=pred_price,
        warnings=warnings,
    )
