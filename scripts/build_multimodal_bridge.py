from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _canon_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(0).astype(float)
        else:
            out[col] = s.fillna("").astype(str).str.strip()
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


def build_bridge(project_root: Path) -> dict:
    data_dir = project_root / "data"
    photo_root = project_root / "satilir_photos" / "satilir_photos"

    raw_path = data_dir / "satilir_properties.csv"
    clean_path = data_dir / "satilir_properties_cleaned.csv"
    base_path = data_dir / "satilir_model_base_v3.csv"

    raw = pd.read_csv(raw_path, low_memory=False)
    clean = pd.read_csv(clean_path, low_memory=False)
    base = pd.read_csv(base_path, low_memory=False)

    raw.columns = [c.strip() for c in raw.columns]
    clean.columns = [c.strip() for c in clean.columns]
    base.columns = [c.strip() for c in base.columns]

    if len(clean) != len(base):
        raise ValueError(
            "Row count mismatch: satilir_properties_cleaned.csv "
            f"({len(clean)}) vs satilir_model_base_v3.csv ({len(base)})."
        )

    if "link" not in raw.columns:
        raise ValueError("Column 'link' is missing in satilir_properties.csv")

    match_cols = list(clean.columns)
    missing_in_raw = [c for c in match_cols if c not in raw.columns]
    if missing_in_raw:
        raise ValueError(
            "Missing columns in satilir_properties.csv for key matching: "
            f"{missing_in_raw}"
        )

    raw_key_frame = _canon_frame(raw[match_cols])
    clean_key_frame = _canon_frame(clean[match_cols])

    raw_keys = pd.Series(list(map(tuple, raw_key_frame.to_numpy())), index=raw.index)
    clean_keys = pd.Series(list(map(tuple, clean_key_frame.to_numpy())), index=clean.index)

    first_raw_idx_by_key: dict[tuple, int] = {}
    for idx, key in raw_keys.items():
        first_raw_idx_by_key.setdefault(key, int(idx))

    mapped_raw_idx = clean_keys.map(lambda key: first_raw_idx_by_key.get(key, None))
    map_method = pd.Series(index=clean.index, data="", dtype="object")
    map_method[mapped_raw_idx.notna()] = "key_exact"

    # Fallback for rare rows modified by cleaning/imputation.
    raw_addr = raw["address"].fillna("").astype(str).str.strip()
    clean_addr = clean["address"].fillna("").astype(str).str.strip()
    raw_price = pd.to_numeric(raw["price"], errors="coerce")
    clean_price = pd.to_numeric(clean["price"], errors="coerce")

    for ci in mapped_raw_idx.index[mapped_raw_idx.isna()]:
        candidates = raw.index[(raw_addr == clean_addr.at[ci]) & (raw_price == clean_price.at[ci])]
        if len(candidates) == 1:
            mapped_raw_idx.at[ci] = int(candidates[0])
            map_method.at[ci] = "fallback_addr_price_unique"

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

    bridge_rows = []
    for clean_idx in clean.index:
        raw_idx_val = mapped_raw_idx.at[clean_idx]
        has_raw = pd.notna(raw_idx_val)
        raw_idx = int(raw_idx_val) if has_raw else None

        link = str(raw.at[raw_idx, "link"]).strip() if has_raw else ""
        link_slug = _link_to_slug(link)
        image_folder = slug_to_folder.get(link_slug)
        house_id = _extract_house_id(image_folder)
        has_image = image_folder is not None
        n_images = get_image_count(image_folder) if has_image else 0

        bridge_rows.append(
            {
                "clean_idx": int(clean_idx),
                "model_base_idx": int(clean_idx),
                "raw_idx": raw_idx,
                "mapping_method": map_method.at[clean_idx],
                "price": pd.to_numeric(clean.at[clean_idx, "price"], errors="coerce"),
                "address": clean.at[clean_idx, "address"],
                "link": link,
                "link_slug": link_slug,
                "image_folder_name": image_folder,
                "house_id": house_id,
                "has_image_folder": has_image,
                "n_images": int(n_images),
            }
        )

    bridge = pd.DataFrame(bridge_rows).sort_values("clean_idx").reset_index(drop=True)
    for int_col in ["clean_idx", "model_base_idx", "raw_idx", "house_id", "n_images"]:
        bridge[int_col] = bridge[int_col].astype("Int64")

    bridge_out = data_dir / "satilir_tabular_image_bridge.csv"
    bridge.to_csv(bridge_out, index=False, encoding="utf-8")

    id_price = (
        bridge[bridge["has_image_folder"]]
        .copy()
        .sort_values("clean_idx")
        .loc[
            :,
            [
                "house_id",
                "price",
                "image_folder_name",
                "n_images",
                "clean_idx",
                "raw_idx",
                "link",
            ],
        ]
    )

    id_price_out = data_dir / "satilir_id_price_folder.csv"
    id_price.to_csv(id_price_out, index=False, encoding="utf-8")

    base_with_house_id = base.copy()
    for col in [
        "house_id",
        "image_folder_name",
        "has_image_folder",
        "n_images",
        "clean_idx",
        "raw_idx",
        "mapping_method",
        "link",
    ]:
        base_with_house_id[col] = bridge[col].values

    base_out = data_dir / "satilir_model_base_with_house_id.csv"
    base_with_house_id.to_csv(base_out, index=False, encoding="utf-8")

    clean_price_num = pd.to_numeric(clean["price"], errors="coerce")
    base_price_num = pd.to_numeric(base["num__price"], errors="coerce")
    price_index_match = int((clean_price_num.values == base_price_num.values).sum())

    missing_rows = bridge[~bridge["has_image_folder"]].loc[
        :,
        ["clean_idx", "raw_idx", "mapping_method", "link", "address", "price"],
    ]
    missing_rows = missing_rows.head(25).fillna("").to_dict(orient="records")

    report = {
        "clean_rows": int(len(clean)),
        "raw_rows": int(len(raw)),
        "model_base_rows": int(len(base)),
        "price_index_match_clean_vs_model_base": price_index_match,
        "mapped_to_raw_total": int(mapped_raw_idx.notna().sum()),
        "mapped_exact_key": int((map_method == "key_exact").sum()),
        "mapped_fallback_addr_price_unique": int(
            (map_method == "fallback_addr_price_unique").sum()
        ),
        "unmapped_to_raw": int(mapped_raw_idx.isna().sum()),
        "with_image_folder": int(bridge["has_image_folder"].sum()),
        "without_image_folder": int((~bridge["has_image_folder"]).sum()),
        "unique_house_id_with_image": int(id_price["house_id"].nunique()),
        "duplicate_folder_slug_keys": int(len(duplicate_slug_keys)),
        "outputs": {
            "bridge_csv": str(bridge_out),
            "id_price_csv": str(id_price_out),
            "model_base_with_house_id_csv": str(base_out),
        },
        "missing_image_examples": missing_rows,
    }

    report_out = data_dir / "multimodal_alignment_report.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = build_bridge(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
