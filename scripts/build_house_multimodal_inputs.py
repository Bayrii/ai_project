from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _canon_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
        else:
            out[col] = s.fillna("").astype(str).str.strip().str.lower()
    return out


def _link_to_slug(link: object) -> str:
    value = "" if pd.isna(link) else str(link).strip()
    value = re.sub(r"^https?://[^/]+/", "", value)
    return value.strip("/").lower()


def _folder_to_slug(folder_name: str) -> str:
    return re.sub(r"^[0-9]+_", "", folder_name).lower()


def _extract_house_id(folder_name: object) -> int | None:
    if pd.isna(folder_name) or folder_name is None:
        return None
    match = re.match(r"^(\d+)_", str(folder_name))
    if not match:
        return None
    return int(match.group(1))


def _resolve_photo_root(project_root: Path) -> Path:
    candidates = [
        project_root / "satilir_photos" / "satilir_photos",
        project_root / "satilir_photos",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        "Could not find photo root. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _build_occurrence_mapping(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    match_cols: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Occurrence-aware row mapping from left_df to right_df.

    Returns:
      - mapped_right_idx: Int64 Series aligned to left_df index.
      - left_keys: tuple-key Series for left_df.
      - right_keys: tuple-key Series for right_df.
    """
    left_key_frame = _canon_frame(left_df[match_cols])
    right_key_frame = _canon_frame(right_df[match_cols])

    left_keys = pd.Series(
        list(map(tuple, left_key_frame.to_numpy())),
        index=left_df.index,
    )
    right_keys = pd.Series(
        list(map(tuple, right_key_frame.to_numpy())),
        index=right_df.index,
    )

    idx_by_right_key: dict[tuple, list[int]] = {}
    for idx, key in right_keys.items():
        idx_by_right_key.setdefault(key, []).append(int(idx))

    mapped_right_idx = pd.Series(index=left_df.index, data=pd.NA, dtype="Int64")
    for left_idx, key in left_keys.items():
        candidates = idx_by_right_key.get(key, [])
        if candidates:
            mapped_right_idx.at[left_idx] = candidates.pop(0)

    return mapped_right_idx, left_keys, right_keys


def _cost_clean_to_raw_group(
    house_clean_row: pd.Series,
    house_raw_row: pd.Series,
    numeric_cols: list[str],
    binary_cols: list[str],
) -> float:
    rel_terms: list[float] = []
    weights = {
        "price": 1.0,
        "area_m2": 0.7,
        "land_area_sot": 0.5,
    }
    for col in numeric_cols:
        c_val = float(house_clean_row[col])
        r_val = float(house_raw_row[col])
        rel_diff = abs(c_val - r_val) / (abs(r_val) + 1.0)
        rel_terms.append(weights.get(col, 0.5) * rel_diff)

    if binary_cols:
        mismatch = (house_clean_row[binary_cols].to_numpy() != house_raw_row[binary_cols].to_numpy()).mean()
        bin_penalty = 0.4 * float(mismatch)
    else:
        bin_penalty = 0.0

    return float(sum(rel_terms) + bin_penalty)


def _map_house_clean_to_raw(
    house_clean: pd.DataFrame,
    house_raw: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict]:
    """Map each house_clean row to one house_raw row.

    Pass 1: exact occurrence-aware full-row match.
    Pass 2: for unmatched rows, solve optimal assignment inside
            (address, rooms, floor) groups using a numeric+binary cost.
    """
    full_cols = [c for c in house_clean.columns if c in house_raw.columns]
    if not full_cols:
        raise ValueError("No shared columns found between house_clean and house_raw")

    mapped_raw_idx, _, _ = _build_occurrence_mapping(house_clean, house_raw, full_cols)
    method = pd.Series(index=house_clean.index, data="", dtype="object")
    method.loc[mapped_raw_idx.notna()] = "exact_clean_to_raw"

    used_raw_idx = set(mapped_raw_idx.dropna().astype(int).tolist())
    unmatched_clean_idx = mapped_raw_idx[mapped_raw_idx.isna()].index.tolist()

    if unmatched_clean_idx:
        remaining_raw_idx = [idx for idx in house_raw.index if int(idx) not in used_raw_idx]

        key_cols = [c for c in ["address", "rooms", "floor"] if c in full_cols]
        if len(key_cols) < 3:
            raise ValueError(
                "Cannot run group assignment fallback: required keys "
                "['address', 'rooms', 'floor'] were not all found."
            )

        numeric_cols = [c for c in ["price", "area_m2", "land_area_sot"] if c in full_cols]
        binary_cols = [c for c in full_cols if c not in key_cols + numeric_cols]

        house_clean_canon = _canon_frame(house_clean[full_cols])
        house_raw_canon = _canon_frame(house_raw[full_cols])

        clean_groups: dict[tuple, list[int]] = {}
        for clean_idx in unmatched_clean_idx:
            key = tuple(house_clean_canon.loc[clean_idx, key_cols].to_numpy())
            clean_groups.setdefault(key, []).append(int(clean_idx))

        raw_groups: dict[tuple, list[int]] = {}
        for raw_idx in remaining_raw_idx:
            key = tuple(house_raw_canon.loc[raw_idx, key_cols].to_numpy())
            raw_groups.setdefault(key, []).append(int(raw_idx))

        failed_groups: list[tuple] = []
        assignment_costs: list[float] = []

        for key, clean_idx_group in clean_groups.items():
            raw_idx_group = raw_groups.get(key, [])
            if len(raw_idx_group) < len(clean_idx_group):
                failed_groups.append((key, len(clean_idx_group), len(raw_idx_group)))
                continue

            cost_matrix = []
            for clean_idx in clean_idx_group:
                clean_row = house_clean_canon.loc[clean_idx]
                row_costs = []
                for raw_idx in raw_idx_group:
                    raw_row = house_raw_canon.loc[raw_idx]
                    row_costs.append(
                        _cost_clean_to_raw_group(
                            clean_row,
                            raw_row,
                            numeric_cols=numeric_cols,
                            binary_cols=binary_cols,
                        )
                    )
                cost_matrix.append(row_costs)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r_i, c_j in zip(row_ind, col_ind):
                clean_idx = clean_idx_group[int(r_i)]
                raw_idx = raw_idx_group[int(c_j)]
                mapped_raw_idx.at[clean_idx] = raw_idx
                method.at[clean_idx] = "group_assignment_clean_to_raw"
                assignment_costs.append(float(cost_matrix[int(r_i)][int(c_j)]))

        if failed_groups:
            raise ValueError(
                "Could not complete group assignment for some (address, rooms, floor) groups. "
                f"Example: {failed_groups[:3]}"
            )

        if mapped_raw_idx.isna().any():
            raise ValueError(
                "Fallback group assignment ended with unmatched house_clean rows: "
                f"{int(mapped_raw_idx.isna().sum())}"
            )

        stats = {
            "exact_clean_to_raw": int((method == "exact_clean_to_raw").sum()),
            "group_assignment_clean_to_raw": int((method == "group_assignment_clean_to_raw").sum()),
            "group_assignment_cost_mean": float(pd.Series(assignment_costs).mean()) if assignment_costs else 0.0,
            "group_assignment_cost_max": float(pd.Series(assignment_costs).max()) if assignment_costs else 0.0,
        }
    else:
        stats = {
            "exact_clean_to_raw": int((method == "exact_clean_to_raw").sum()),
            "group_assignment_clean_to_raw": 0,
            "group_assignment_cost_mean": 0.0,
            "group_assignment_cost_max": 0.0,
        }

    return mapped_raw_idx.astype("Int64"), method, stats


def build_house_multimodal_inputs(project_root: Path) -> dict:
    data_dir = project_root / "data"
    house_dir = data_dir / "house"
    photo_root = _resolve_photo_root(project_root)

    source_raw_path = data_dir / "satilir_properties.csv"
    house_clean_path = house_dir / "satilir_properties_house_cleaned.csv"
    house_raw_path = house_dir / "satilir_properties_house.csv"
    house_fe_path = house_dir / "satilir_properties_house_feature_engineered.csv"
    house_target_path = house_dir / "satilir_properties_house_target.csv"

    source_raw = pd.read_csv(source_raw_path, low_memory=False)
    house_clean = pd.read_csv(house_clean_path, low_memory=False)
    house_raw = pd.read_csv(house_raw_path, low_memory=False)
    house_fe = pd.read_csv(house_fe_path, low_memory=False)
    house_target = pd.read_csv(house_target_path, low_memory=False)

    source_raw.columns = [c.strip() for c in source_raw.columns]
    house_clean.columns = [c.strip() for c in house_clean.columns]
    house_raw.columns = [c.strip() for c in house_raw.columns]
    house_fe.columns = [c.strip() for c in house_fe.columns]
    house_target.columns = [c.strip() for c in house_target.columns]

    if "link" not in source_raw.columns:
        raise ValueError("Column 'link' is missing in satilir_properties.csv")
    if "price" not in house_target.columns:
        raise ValueError("Column 'price' is missing in satilir_properties_house_target.csv")

    if len(house_clean) > len(house_raw):
        raise ValueError(
            "House row mismatch: cleaned rows cannot exceed house_raw rows: "
            f"cleaned={len(house_clean)}, house_raw={len(house_raw)}"
        )

    if len(house_clean) != len(house_fe) or len(house_clean) != len(house_target):
        raise ValueError(
            "House row mismatch: cleaned="
            f"{len(house_clean)}, feature_engineered={len(house_fe)}, target={len(house_target)}"
        )

    # Step 1: map house_raw -> source_raw occurrence-aware exact on shared raw columns.
    raw_match_cols = [c for c in house_raw.columns if c in source_raw.columns]
    if not raw_match_cols:
        raise ValueError("No shared columns found between house_raw and satilir_properties")

    mapped_source_idx_by_raw, _, _ = _build_occurrence_mapping(
        house_raw,
        source_raw,
        raw_match_cols,
    )
    if mapped_source_idx_by_raw.isna().any():
        raise ValueError(
            "Could not map every house_raw row to satilir_properties. "
            f"Unmapped raw rows: {int(mapped_source_idx_by_raw.isna().sum())}"
        )

    # Step 2: map house_clean -> house_raw with exact pass + assignment fallback.
    mapped_raw_idx_by_clean, method_clean_to_raw, clean_to_raw_stats = _map_house_clean_to_raw(
        house_clean,
        house_raw,
    )

    folder_names = [p.name for p in photo_root.iterdir() if p.is_dir()]
    slug_to_folder: dict[str, str] = {}
    duplicate_slug_keys: set[str] = set()
    for folder in folder_names:
        slug = _folder_to_slug(folder)
        if slug in slug_to_folder:
            duplicate_slug_keys.add(slug)
            continue
        slug_to_folder[slug] = folder

    image_count_cache: dict[str, int] = {}

    def get_image_count(folder_name: str) -> int:
        if folder_name not in image_count_cache:
            folder_path = photo_root / folder_name
            if not folder_path.exists():
                image_count_cache[folder_name] = 0
            else:
                image_count_cache[folder_name] = sum(
                    1
                    for p in folder_path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
        return image_count_cache[folder_name]

    bridge_rows: list[dict] = []
    for house_idx in house_clean.index:
        raw_idx_val = mapped_raw_idx_by_clean.at[house_idx]
        if pd.isna(raw_idx_val):
            continue

        raw_idx = int(raw_idx_val)
        source_idx = int(mapped_source_idx_by_raw.at[raw_idx])

        link = str(source_raw.at[source_idx, "link"]).strip()
        link_slug = _link_to_slug(link)
        image_folder_name = slug_to_folder.get(link_slug)
        if image_folder_name is None:
            continue

        house_id = _extract_house_id(image_folder_name)
        if house_id is None:
            continue

        bridge_rows.append(
            {
                "house_clean_idx": int(house_idx),
                "house_raw_idx": raw_idx,
                "source_raw_idx": source_idx,
                "mapping_method": method_clean_to_raw.at[house_idx],
                "house_id": house_id,
                "price": pd.to_numeric(house_target.at[house_idx, "price"], errors="coerce"),
                "image_folder_name": image_folder_name,
                "n_images": int(get_image_count(image_folder_name)),
                "link": link,
            }
        )

    bridge = pd.DataFrame(bridge_rows).sort_values("house_clean_idx").reset_index(drop=True)
    if bridge.empty:
        raise ValueError(
            "No house rows were linked to image folders. "
            "Cannot build multimodal house inputs safely."
        )

    for int_col in ["house_clean_idx", "house_raw_idx", "source_raw_idx", "house_id", "n_images"]:
        bridge[int_col] = bridge[int_col].astype("Int64")

    matched_house_idx = bridge["house_clean_idx"].astype(int).tolist()

    house_tabular = house_fe.iloc[matched_house_idx].reset_index(drop=True).copy()
    house_prices = pd.to_numeric(
        house_target.iloc[matched_house_idx]["price"],
        errors="coerce",
    ).reset_index(drop=True)

    house_tabular.insert(0, "house_id", bridge["house_id"].astype("Int64").reset_index(drop=True))
    house_tabular.insert(1, "price", house_prices)

    tabular_out = house_dir / "satilir_properties_house_feature_engineered_with_house_id.csv"
    house_tabular.to_csv(tabular_out, index=False, encoding="utf-8")

    id_price = bridge[
        [
            "house_id",
            "price",
            "image_folder_name",
            "n_images",
            "house_clean_idx",
            "house_raw_idx",
            "source_raw_idx",
            "link",
            "mapping_method",
        ]
    ].copy()

    id_price_out = house_dir / "satilir_house_id_price_folder.csv"
    id_price.to_csv(id_price_out, index=False, encoding="utf-8")

    unmatched_idx = mapped_raw_idx_by_clean[mapped_raw_idx_by_clean.isna()].index.tolist()
    unmatched_examples = house_clean.loc[
        unmatched_idx,
        ["price", "rooms", "area_m2", "land_area_sot", "floor", "address"],
    ]
    unmatched_examples = unmatched_examples.head(25).fillna("").to_dict(orient="records")

    report = {
        "house_clean_rows": int(len(house_clean)),
        "house_raw_rows": int(len(house_raw)),
        "house_feature_engineered_rows": int(len(house_fe)),
        "house_target_rows": int(len(house_target)),
        "source_raw_rows": int(len(source_raw)),
        "shared_raw_match_columns": int(len(raw_match_cols)),
        "raw_to_source_mapped_rows": int(mapped_source_idx_by_raw.notna().sum()),
        "clean_to_raw_mapped_rows": int(mapped_raw_idx_by_clean.notna().sum()),
        "clean_to_raw_exact_rows": int(clean_to_raw_stats["exact_clean_to_raw"]),
        "clean_to_raw_group_assignment_rows": int(clean_to_raw_stats["group_assignment_clean_to_raw"]),
        "clean_to_raw_group_cost_mean": float(clean_to_raw_stats["group_assignment_cost_mean"]),
        "clean_to_raw_group_cost_max": float(clean_to_raw_stats["group_assignment_cost_max"]),
        "mapped_with_image_folder": int(len(bridge)),
        "unmatched_house_rows": int(len(house_clean) - mapped_raw_idx_by_clean.notna().sum()),
        "linked_coverage_ratio": float(len(bridge) / len(house_clean)),
        "unique_house_id": int(id_price["house_id"].nunique()),
        "duplicate_folder_slug_keys": int(len(duplicate_slug_keys)),
        "outputs": {
            "house_tabular_with_house_id_csv": str(tabular_out),
            "house_id_price_folder_csv": str(id_price_out),
        },
        "unmatched_examples": unmatched_examples,
    }

    report_out = house_dir / "house_multimodal_alignment_report.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = build_house_multimodal_inputs(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
