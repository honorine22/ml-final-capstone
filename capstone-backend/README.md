# MaizeGuard Backend

FastAPI service for MaizeGuard Rwanda. It loads the PyTorch MobileNetV3 model and returns maize quality predictions for the frontend.

## Local Run

```bash
cd capstone-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API

```text
POST /predict
form-data field: image
```

Response includes `label`, `confidence`, `probabilities`, `needs_review`, `risk`, `action`, and `recommendation`.

## Deployment

Recommended quick deployment: Render web service.

Use:

```text
Root directory: capstone-backend
Build command: pip install -r requirements.txt
Start command: ./start.sh
Health check path: /
```

After deployment, copy the backend URL and set it as `MODEL_API_URL` in the frontend deployment.
