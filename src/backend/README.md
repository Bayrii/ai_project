# Backend (FastAPI)

FastAPI service for tabular and multimodal real-estate prediction.

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API from `src/backend`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/predict/tabular/apartment`
- `POST /v1/predict/tabular/house`
- `POST /v1/predict/multimodal/apartment`
- `POST /v1/predict/multimodal/house`

## Notes

- The backend loads models once at startup from `models/tabular` and `models/multimodal`.
- Tabular apartment/house requests now use explicit raw fields in Swagger (rooms, area_m2, floor, address, yes/no amenities, etc.) and are mapped to engineered features internally.
- `price` is accepted in raw tabular payloads for schema parity but ignored during inference.
- Multimodal apartment/house requests now also use the same explicit raw fields as multipart form fields plus `images`.
- For multimodal prediction, at least 1 image is required and extra images beyond `AI_PROJECT_MAX_IMAGES_PER_REQUEST` are ignored with a warning.
