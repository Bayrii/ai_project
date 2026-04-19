# AI Project: Real Estate Price Prediction (Arenda.az)

This repository contains a full pipeline for real estate price prediction for Azerbaijan listings, including:

- data collection and preprocessing notebooks
- tabular ML models
- multimodal (tabular + image) models
- a FastAPI backend for inference
- a React frontend for interactive predictions
- report assets and figure generation scripts

## Repository Layout

- `data/` - cleaned, engineered, and target datasets
- `figures/` - generated figures and index
- `models/` - trained model artifacts and score history
- `notebooks/` - end-to-end notebook workflow (cleaning, preprocessing, feature engineering, modeling)
- `report/` - LaTeX report files
- `satilir_photos/` - listing photos used by multimodal flow
- `scrape/` - scraping and merge notebooks
- `scripts/` - utility scripts (feature/multimodal preparation, figure extraction, LaTeX generation)
- `src/backend/` - FastAPI inference service
- `src/frontend/` - React + Vite user interface

## Quick Start

## 1) Backend

From repository root:

```bash
cd src/backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Swagger:

- `http://127.0.0.1:8000/docs`

## 2) Frontend

From repository root:

```bash
cd src/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

UI:

- `http://127.0.0.1:5173`

## Inference Notes

- Tabular and multimodal requests use explicit raw listing fields.
- Multimodal endpoints require file uploads in the `images` field.
- For multimodal requests, submit actual image files (`.jpg`, `.jpeg`, `.png`), not string placeholders.
- `price` is accepted for schema parity and ignored during inference.

## Main API Endpoints

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/predict/tabular/apartment`
- `POST /v1/predict/tabular/house`
- `POST /v1/predict/multimodal/apartment`
- `POST /v1/predict/multimodal/house`

## Development Workflow

Typical modeling flow:

1. scrape and merge raw listing data
2. clean and preprocess tabular data
3. engineer features
4. train and evaluate tabular models
5. build multimodal inputs and train multimodal fusion model
6. serve inference through backend and frontend

## Requirements

- Python environment for backend and notebooks
- Node.js and npm for frontend

If npm is missing on Windows, install Node.js LTS and reopen terminal.
