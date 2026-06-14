# Deployment Plan

## Current Demo Deployment

The initial demo runs as two services:

1. Next.js frontend
2. FastAPI PyTorch model API

```text
Browser
  -> Next.js upload UI
  -> /api/analyze
  -> FastAPI /predict
  -> PyTorch MobileNetV3 checkpoint
  -> JSON response
  -> frontend result display
```

## Local Demo Commands

Run the model API:

```bash
cd model_server
pip install -r requirements-pytorch.txt
uvicorn pytorch_main:app --reload --port 8000
```

Run the frontend:

```bash
npm install
npm run dev
```

Environment:

```text
MODEL_API_URL=http://localhost:8000
```

## Online Deployment Option

Frontend:

- Deploy Next.js to Vercel or Netlify.
- Set `MODEL_API_URL` in deployment environment variables.

Model API:

- Deploy FastAPI to Render, Railway, Hugging Face Spaces, or a cloud VM.
- Install `model_server/requirements-pytorch.txt`.
- Include the model files from `model_server/model_exports`.

## Data And Storage

For the initial demo, images do not need permanent storage.

For the later full capstone version:

- store prediction records in PostgreSQL or SQLite
- store uploaded images only with permission
- keep user feedback for model improvement
- add admin/cooperative officer review tools

## Safety

The API returns `needs_review: true` when confidence is low or the top-two class margin is small. The frontend displays **Needs review** in that case instead of presenting a risky final decision.
