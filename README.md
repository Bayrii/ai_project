# AI Project: Real Estate Price Prediction (ML)

This README documents the machine learning side of the project only.

The project predicts listing prices for Azerbaijan real estate using:

- tabular models (structured listing features)
- multimodal models (tabular + image features)

## ML Scope

- Data preparation for apartment and house listings
- Feature engineering and split-specific preprocessing
- Tabular training/evaluation
- Multimodal alignment and fusion training
- Metrics tracking and artifact versioning

## ML Repository Layout

- `data/` - processed datasets, engineered features, and target exports
- `models/` - saved tabular/multimodal model artifacts and metric history
- `notebooks/` - notebook workflow for cleaning, preprocessing, feature engineering, and modeling
- `scripts/` - dataset assembly and ML utility scripts
- `scrape/` - scraping and merge notebooks used for upstream dataset creation
- `satilir_photos/` - listing images for multimodal experiments
- `figures/` - generated figures for analysis/reporting
- `report/` - report sources (LaTeX)

## End-to-End ML Workflow

1. Collect and merge listing data and image references.
2. Clean and normalize raw fields for apartment and house subsets.
3. Build engineered tabular features.
4. Train tabular baselines and tuned models.
5. Align image and tabular records for multimodal subsets.
6. Train/evaluate multimodal fusion models.
7. Save artifacts and update score history.

## Running ML Notebooks

Use the notebooks under `notebooks/`, especially split flows under:

- `notebooks/apartment/`
- `notebooks/house/`

Recommended execution order per split:

1. Data-Cleaning
2. Data-Preprocessing
3. Feature-Engineering
4. Model
5. Multimodal

## ML Artifacts

Typical outputs written during runs:

- tabular model files under `models/tabular/...`
- multimodal model files under `models/multimodal/...`
- latest metrics snapshot in `models/metrics/latest_scores.json`
- historical metric log in `models/metrics/score_history.csv`

## Environment

- Python is required for notebooks, preprocessing scripts, and model training.
- Install dependencies from the repository-level `requirements.txt`.

```bash
pip install -r requirements.txt
```
