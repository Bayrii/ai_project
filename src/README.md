# Source Apps (Backend + Frontend)

This folder contains the application layer used to serve predictions from the trained ML artifacts.

## Structure

- `backend/` - FastAPI service exposing tabular and multimodal prediction endpoints
- `frontend/` - React + Vite client for interactive prediction requests

## Backend (`src/backend`)

Main responsibilities:

- load model artifacts at startup
- validate request payloads (raw feature schema)
- preprocess raw fields into engineered model inputs
- run tabular or multimodal inference
- return prediction values and warnings

Run backend:

```bash
cd src/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Useful URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`
- `http://127.0.0.1:8000/docs`

## Frontend (`src/frontend`)

Main responsibilities:

- provide forms for apartment/house predictions
- support both tabular and multimodal requests
- submit image files via multipart form data for multimodal endpoints
- display predicted prices and model warnings

Run frontend:

```bash
cd src/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend URL:

- `http://127.0.0.1:5173`

## Local Dev Flow

1. Start backend first.
2. Start frontend in a second terminal.
3. Open the frontend and submit predictions.
4. Use backend Swagger (`/docs`) for direct API testing when needed.
