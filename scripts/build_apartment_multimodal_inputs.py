from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


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
) -> pd.Series:
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

    return mapped_right_idx


def _map_apartment_clean_to_raw(
    apartment_clean: pd.DataFrame,
    apartment_raw: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    match_cols = [c for c in apartment_raw.columns if c in apartment_clean.columns]
    if not match_cols:
        raise ValueError("No shared columns found between apartment_clean and apartment_raw")

    mapped_raw_idx = _build_occurrence_mapping(apartment_clean, apartment_raw, match_cols)
    method = pd.Series(index=apartment_clean.index, data="", dtype="object")
    method.loc[mapped_raw_idx.notna()] = "exact_clean_to_raw"

    # Lightweight fallback for any unmatched rows.
    used_raw_idx = set(mapped_raw_idx.dropna().astype(int).tolist())

    raw_addr = apartment_raw["address"].fillna("").astype(str).str.strip().str.lower()
    clean_addr = apartment_clean["address"].fillna("").astype(str).str.strip().str.lower()
    raw_price = pd.to_numeric(apartment_raw["price"], errors="coerce")
    clean_price = pd.to_numeric(apartment_clean["price"], errors="coerce")

    key_cols = [c for c in ["rooms", "area_m2", "floor"] if c in apartment_raw.columns and c in apartment_clean.columns]
    raw_key_num = {
        c: pd.to_numeric(apartment_raw[c], errors="coerce") for c in key_cols
    }
    clean_key_num = {
        c: pd.to_numeric(apartment_clean[c], errors="coerce") for c in key_cols
    }

    for clean_idx in mapped_raw_idx.index[mapped_raw_idx.isna()]:
        candidates = apartment_raw.index[(raw_addr == clean_addr.at[clean_idx]) & (raw_price == clean_price.at[clean_idx])]
        candidates = [int(c) for c in candidates if int(c) not in used_raw_idx]

        if len(candidates) > 1 and key_cols:
            narrowed = []
            for cand in candidates:
                ok = True
                for col in key_cols:
                    if raw_key_num[col].at[cand] != clean_key_num[col].at[clean_idx]:
                        ok = False
                        break
                if ok:
                    narrowed.append(cand)
            candidates = narrowed

        if len(candidates) == 1:
            raw_idx = int(candidates[0])
            mapped_raw_idx.at[clean_idx] = raw_idx
            method.at[clean_idx] = "fallback_addr_price"
            used_raw_idx.add(raw_idx)

    return mapped_raw_idx.astype("Int64"), method


def build_apartment_multimodal_inputs(project_root: Path) -> dict:
    data_dir = project_root / "data"
    apartment_dir = data_dir / "apartment"
    photo_root = _resolve_photo_root(project_root)

    source_raw_path = data_dir / "satilir_properties.csv"
    apartment_clean_path = apartment_dir / "satilir_properties_apartment_cleaned.csv"
    apartment_raw_path = apartment_dir / "satilir_properties_apartment.csv"
    apartment_fe_path = apartment_dir / "satilir_properties_apartment_feature_engineered.csv"
    apartment_target_path = apartment_dir / "satilir_properties_apartment_target.csv"

    source_raw = pd.read_csv(source_raw_path, low_memory=False)
    apartment_clean = pd.read_csv(apartment_clean_path, low_memory=False)
    apartment_raw = pd.read_csv(apartment_raw_path, low_memory=False)
    apartment_fe = pd.read_csv(apartment_fe_path, low_memory=False)
    apartment_target = pd.read_csv(apartment_target_path, low_memory=False)

    source_raw.columns = [c.strip() for c in source_raw.columns]
    apartment_clean.columns = [c.strip() for c in apartment_clean.columns]
    apartment_raw.columns = [c.strip() for c in apartment_raw.columns]
    apartment_fe.columns = [c.strip() for c in apartment_fe.columns]
    apartment_target.columns = [c.strip() for c in apartment_target.columns]

    if "link" not in source_raw.columns:
        raise ValueError("Column 'link' is missing in satilir_properties.csv")
    if "price" not in apartment_target.columns:
        raise ValueError("Column 'price' is missing in satilir_properties_apartment_target.csv")

    if len(apartment_clean) != len(apartment_fe) or len(apartment_clean) != len(apartment_target):
        raise ValueError(
            "Apartment row mismatch: cleaned="
            f"{len(apartment_clean)}, feature_engineered={len(apartment_fe)}, target={len(apartment_target)}"
        )

    raw_match_cols = [c for c in apartment_raw.columns if c in source_raw.columns]
    if not raw_match_cols:
        raise ValueError("No shared columns found between apartment_raw and satilir_properties")

    mapped_source_idx_by_raw = _build_occurrence_mapping(
        apartment_raw,
        source_raw,
        raw_match_cols,
    )
    if mapped_source_idx_by_raw.isna().any():
        raise ValueError(
            "Could not map every apartment_raw row to satilir_properties. "
            f"Unmapped raw rows: {int(mapped_source_idx_by_raw.isna().sum())}"
        )

    mapped_raw_idx_by_clean, method_clean_to_raw = _map_apartment_clean_to_raw(
        apartment_clean,
        apartment_raw,
    )

    if mapped_raw_idx_by_clean.isna().any():
        raise ValueError(
            "Could not map every apartment_clean row to apartment_raw. "
            f"Unmapped cleaned rows: {int(mapped_raw_idx_by_clean.isna().sum())}"
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
    for apartment_idx in apartment_clean.index:
        raw_idx = int(mapped_raw_idx_by_clean.at[apartment_idx])
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
                "apartment_clean_idx": int(apartment_idx),
                "apartment_raw_idx": raw_idx,
                "source_raw_idx": source_idx,
                "mapping_method": method_clean_to_raw.at[apartment_idx],
                "house_id": house_id,
                "price": pd.to_numeric(apartment_target.at[apartment_idx, "price"], errors="coerce"),
                "image_folder_name": image_folder_name,
                "n_images": int(get_image_count(image_folder_name)),
                "link": link,
            }
        )

    bridge = pd.DataFrame(bridge_rows).sort_values("apartment_clean_idx").reset_index(drop=True)
    if bridge.empty:
        raise ValueError(
            "No apartment rows were linked to image folders. "
            "Cannot build multimodal apartment inputs safely."
        )

    for int_col in ["apartment_clean_idx", "apartment_raw_idx", "source_raw_idx", "house_id", "n_images"]:
        bridge[int_col] = bridge[int_col].astype("Int64")

    matched_apartment_idx = bridge["apartment_clean_idx"].astype(int).tolist()

    apartment_tabular = apartment_fe.iloc[matched_apartment_idx].reset_index(drop=True).copy()
    apartment_prices = pd.to_numeric(
        apartment_target.iloc[matched_apartment_idx]["price"],
        errors="coerce",
    ).reset_index(drop=True)

    apartment_tabular.insert(0, "house_id", bridge["house_id"].astype("Int64").reset_index(drop=True))
    apartment_tabular.insert(1, "price", apartment_prices)

    tabular_out = apartment_dir / "satilir_properties_apartment_feature_engineered_with_house_id.csv"
    apartment_tabular.to_csv(tabular_out, index=False, encoding="utf-8")

    id_price = bridge[
        [
            "house_id",
            "price",
            "image_folder_name",
            "n_images",
            "apartment_clean_idx",
            "apartment_raw_idx",
            "source_raw_idx",
            "link",
            "mapping_method",
        ]
    ].copy()

    id_price_out = apartment_dir / "satilir_apartment_id_price_folder.csv"
    id_price.to_csv(id_price_out, index=False, encoding="utf-8")

    report = {
        "apartment_clean_rows": int(len(apartment_clean)),
        "apartment_raw_rows": int(len(apartment_raw)),
        "apartment_feature_engineered_rows": int(len(apartment_fe)),
        "apartment_target_rows": int(len(apartment_target)),
        "source_raw_rows": int(len(source_raw)),
        "raw_to_source_mapped_rows": int(mapped_source_idx_by_raw.notna().sum()),
        "clean_to_raw_mapped_rows": int(mapped_raw_idx_by_clean.notna().sum()),
        "clean_to_raw_exact_rows": int((method_clean_to_raw == "exact_clean_to_raw").sum()),
        "clean_to_raw_fallback_rows": int((method_clean_to_raw == "fallback_addr_price").sum()),
        "mapped_with_image_folder": int(len(bridge)),
        "linked_coverage_ratio": float(len(bridge) / len(apartment_clean)),
        "unique_house_id": int(id_price["house_id"].nunique()),
        "duplicate_folder_slug_keys": int(len(duplicate_slug_keys)),
        "outputs": {
            "apartment_tabular_with_house_id_csv": str(tabular_out),
            "apartment_id_price_folder_csv": str(id_price_out),
        },
    }

    report_out = apartment_dir / "apartment_multimodal_alignment_report.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = build_apartment_multimodal_inputs(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
