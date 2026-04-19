# Source Apps

This folder contains:

- `backend/` — FastAPI API for tabular and multimodal predictions.
- `frontend/` — React/Vite UI to call the backend prediction endpoints.

## Quick start

### Backend

```bash
cd src/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd src/frontend
npm install
npm run dev
```
