from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.schemas.common import PropertyType, RawFeatureRequest


def parse_tabular_features_json(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("tabular_features must be a JSON object")
    return parsed


def coerce_feature_value(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, bool):
        return float(int(value))

    return float(value)


def order_features(input_features: dict[str, Any], expected_features: list[str]) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []

    incoming_keys = set(input_features.keys())
    expected_keys = set(expected_features)

    missing = [name for name in expected_features if name not in input_features]
    extras = sorted(incoming_keys - expected_keys)

    if missing:
        warnings.append(f"Filled {len(missing)} missing feature(s) with 0.")
    if extras:
        preview = extras[:5]
        tail = "..." if len(extras) > 5 else ""
        warnings.append(f"Ignored {len(extras)} extra feature(s): {preview}{tail}")

    invalid_count = 0
    ordered_values: list[float] = []

    for feature_name in expected_features:
        raw_value = input_features.get(feature_name, 0.0)
        try:
            ordered_values.append(coerce_feature_value(raw_value))
        except (TypeError, ValueError):
            invalid_count += 1
            ordered_values.append(0.0)

    if invalid_count:
        warnings.append(f"Converted {invalid_count} non-numeric feature value(s) to 0.")

    return np.asarray([ordered_values], dtype=np.float32), warnings


_BINARY_FEATURE_NAMES = [
    "has_document",
    "avtodayanacaq",
    "balkon",
    "duzelme",
    "esyali",
    "hovuz",
    "internet",
    "isiq",
    "kabel_tv",
    "kombi",
    "kondisioner",
    "lift",
    "merkezi_qizdirici_sistem",
    "metbex_mebeli",
    "pvc_pencere",
    "qaz",
    "su",
    "telefon",
    "temirli",
]


def _yes_no_to_float(value: str) -> float:
    return 1.0 if str(value).strip().lower() == "yes" else 0.0


def _normalize_category(value: str) -> str:
    return value.strip().lower()


def _split_address(address: str) -> tuple[str | None, str | None]:
    if not address or not address.strip():
        return None, None

    parts = address.split(",", maxsplit=1)
    city = parts[0].strip() if parts[0].strip() else None

    if len(parts) == 1:
        return city, None

    region = parts[1].strip()
    return city, region if region else None


def _activate_one_hot(
    *,
    features: dict[str, float],
    expected_features: list[str],
    prefix: str,
    value: str | None,
    warnings: list[str],
) -> None:
    candidate_keys = [name for name in expected_features if name.startswith(prefix)]
    if not candidate_keys:
        return

    normalized_to_key = { _normalize_category(name[len(prefix):]): name for name in candidate_keys }

    selected_key: str | None = None

    if value is not None:
        selected_key = normalized_to_key.get(_normalize_category(value))

    if selected_key is None:
        selected_key = next((name for name in candidate_keys if name.endswith("_infrequent_sklearn")), None)

    if selected_key is None and any(name.endswith("_None") for name in candidate_keys):
        selected_key = next(name for name in candidate_keys if name.endswith("_None"))

    if selected_key is None:
        if value is not None:
            warnings.append(f"No one-hot bucket found for '{prefix}' and value '{value}'.")
        return

    features[selected_key] = 1.0

    if value is not None and _normalize_category(value) not in normalized_to_key:
        warnings.append(f"Mapped unknown category '{value}' to fallback bucket for '{prefix}'.")


def _rooms_group_label(rooms: int) -> str:
    if rooms <= 2:
        return "1-2"
    if rooms <= 4:
        return "3-4"
    if rooms <= 6:
        return "5-6"
    return "infrequent_sklearn"


def _rooms_group_code(rooms: int) -> float:
    if rooms <= 2:
        return 0.0
    if rooms <= 4:
        return 1.0
    if rooms <= 6:
        return 2.0
    return 3.0


def map_raw_to_engineered(
    *,
    property_type: PropertyType,
    raw_payload: RawFeatureRequest,
    expected_features: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Map raw listing input fields to engineered model features.

    The mapping is deterministic and defaults unknown/non-applicable engineered
    columns to 0, then fills known transformed values.
    """

    warnings: list[str] = ["Input field 'price' is ignored during inference."]
    features: dict[str, float] = {name: 0.0 for name in expected_features}

    rooms = max(int(raw_payload.rooms), 0)
    area_m2 = max(float(raw_payload.area_m2), 0.0)
    land_area_sot = max(float(raw_payload.land_area_sot), 0.0)
    floor = max(int(raw_payload.floor), 0)

    yes_ratio = float(np.mean([_yes_no_to_float(getattr(raw_payload, name)) for name in _BINARY_FEATURE_NAMES]))

    for field_name in _BINARY_FEATURE_NAMES:
        feature_name = f"yes_no_binary__{field_name}"
        if feature_name in features:
            features[feature_name] = _yes_no_to_float(getattr(raw_payload, field_name))

    if "num__rooms" in features:
        features["num__rooms"] = float(rooms)

    if "num__feature_score" in features:
        features["num__feature_score"] = yes_ratio

    if "num__log_area_m2" in features:
        features["num__log_area_m2"] = float(np.log1p(area_m2))

    if "num__log_floor" in features:
        features["num__log_floor"] = float(np.log1p(floor))

    if "num__log_area_per_room" in features:
        features["num__log_area_per_room"] = float(np.log1p(area_m2 / max(rooms, 1)))

    city, region = _split_address(raw_payload.address)

    if property_type == "apartment":
        if "num__rooms_group_code" in features:
            features["num__rooms_group_code"] = _rooms_group_code(rooms)

        high_floor_flag = 1.0 if floor >= 9 else 0.0
        if "num__high_floor_flag" in features:
            features["num__high_floor_flag"] = high_floor_flag
        if "num__high_floor_with_lift" in features:
            features["num__high_floor_with_lift"] = high_floor_flag * _yes_no_to_float(raw_payload.lift)

        _activate_one_hot(
            features=features,
            expected_features=expected_features,
            prefix="cat_ohe__rooms_group_",
            value=_rooms_group_label(rooms),
            warnings=warnings,
        )
        _activate_one_hot(
            features=features,
            expected_features=expected_features,
            prefix="cat_ohe__address_part_1_",
            value=city,
            warnings=warnings,
        )
        _activate_one_hot(
            features=features,
            expected_features=expected_features,
            prefix="cat_ohe__address_part_2_",
            value=region,
            warnings=warnings,
        )

    else:
        land_area_m2 = land_area_sot * 100.0
        free_land_ratio = 0.0 if land_area_m2 <= 0 else max(land_area_m2 - area_m2, 0.0) / max(land_area_m2, 1.0)

        if "num__free_land_ratio" in features:
            features["num__free_land_ratio"] = float(free_land_ratio)
        if "num__log_land_area_sot" in features:
            features["num__log_land_area_sot"] = float(np.log1p(land_area_sot))
        if "num__log_rooms_per_floor" in features:
            features["num__log_rooms_per_floor"] = float(np.log1p(rooms / max(floor, 1)))

        _activate_one_hot(
            features=features,
            expected_features=expected_features,
            prefix="cat_ohe_mid__address_part_1_",
            value=city,
            warnings=warnings,
        )
        _activate_one_hot(
            features=features,
            expected_features=expected_features,
            prefix="cat_ohe_mid__address_part_2_",
            value=region,
            warnings=warnings,
        )

    return features, warnings
