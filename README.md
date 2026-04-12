# AI Project Setup Guide

This repository keeps the generated multimodal CSV outputs in git so low-spec machines can run modeling directly without heavy preprocessing.
Use the steps below for the fastest setup.

## 1. Create Environment

Use your preferred Python environment manager.

PowerShell example:

```powershell
conda create -n ai_project python=3.12 -y
conda activate ai_project
pip install -r requariments.txt
```

## 2. Verify Required Source Data

These files are expected before running multimodal notebook:

- `data/satilir_properties.csv`
- `data/satilir_properties_cleaned.csv`
- `data/satilir_model_base_v3.csv`
- `data/satilir_model_base_with_house_id.csv`
- `data/satilir_tabular_image_bridge.csv`
- `data/satilir_id_price_folder.csv`
- `data/satilir_location_encoded_oof_v6_with_house_id.csv`
- `satilir_photos/` directory (photo folders)

Important:
`satilir_photos/` is intentionally gitignored because it is very large. Share this folder separately (external drive, cloud folder, or Git LFS archive) and place it at repo root as `satilir_photos/`.

## 3. Optional: Regenerate Bridge Outputs

You only need this if those CSV files are missing or you want to refresh them after data changes.

Run:

```powershell
python scripts/build_multimodal_bridge.py
```

This regenerates multimodal alignment outputs:

- `data/satilir_tabular_image_bridge.csv`
- `data/satilir_id_price_folder.csv`
- `data/satilir_model_base_with_house_id.csv`
- `data/multimodal_alignment_report.json`

If `data/satilir_model_base_v3.csv` is missing, run `notebooks/arenda-az-Data-Preprocessing.ipynb` first.

## 4. Run Multimodal Notebook

Open and run:

- `notebooks/multimodal_fusion.ipynb`

Run cells in order from top to bottom.

If image folder is missing, the multimodal run will fail after image filtering and merge steps. In that case, run tabular-only notebooks instead.

## 5. Notes Before Pushing

- `.venv/` is ignored and should never be committed.
- Generated CSV outputs used for multimodal training are intentionally tracked to help weaker machines.
- Keep `scripts/build_multimodal_bridge.py` tracked so teammates can regenerate outputs when source data changes.
