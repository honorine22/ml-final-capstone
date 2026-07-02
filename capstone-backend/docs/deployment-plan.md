# Deployment Plan

MaizeGuard runs as two separate services:

```text
capstone-frontend  Next.js upload interface
capstone-backend   FastAPI PyTorch model API
```

## Runtime Flow

```text
Browser
  -> capstone-frontend /api/analyze
  -> capstone-backend /predict
  -> PyTorch MobileNetV3 checkpoint
  -> JSON response
  -> frontend result display
```

## Local Demo

Run the backend first:

```bash
cd capstone-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

Backend URLs:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

Run the frontend in a second terminal:

```bash
cd capstone-frontend
npm install
cp .env.example .env.local
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

Frontend environment:

```text
MODEL_API_URL=http://127.0.0.1:8000
```

## Backend Deployment

Recommended service: Render, Railway, Hugging Face Spaces, or a university server.

Render settings:

```text
Root directory: capstone-backend
Build command: pip install -r requirements.txt
Start command: ./start.sh
Health check path: /
```

Backend environment variables:

```text
CORS_ORIGINS=https://your-frontend-url.vercel.app
CONFIDENCE_THRESHOLD=0.65
TOP2_MARGIN_THRESHOLD=0.15
```

The backend must include:

```text
model_server/model_exports/maizeguard_public_best_model.pt
model_server/model_exports/maizeguard_model_metadata.json
model_server/model_exports/class_names.json
```

## Frontend Deployment

Recommended service: Vercel.

Vercel settings:

```text
Root directory: capstone-frontend
Build command: npm run build
Framework: Next.js
```

Frontend environment variable:

```text
MODEL_API_URL=https://your-backend-url.onrender.com
```

After deployment, test one upload through the frontend and one direct backend request through `/docs`.

## Safety

The API returns `needs_review: true` when the image is unclear, confidence is low, or the top-two class margin is too small. The frontend displays **Needs review** instead of giving a risky final recommendation.
