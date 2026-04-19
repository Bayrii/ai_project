# Frontend (React + Vite)

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Start dev server:

```bash
npm run dev
```

## Environment

Optional environment variable:

- `VITE_API_BASE_URL` (default: `http://localhost:8000`)

Create `.env` in this folder if needed:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## UI behavior

- Tabular and multimodal predictions both collect the same explicit raw listing fields.
- Multimodal mode also requires photo upload and shows suggested photo categories (kitchen, bedroom, exterior, etc.).
- The frontend limits uploads to 12 photos per request for a cleaner user flow.
