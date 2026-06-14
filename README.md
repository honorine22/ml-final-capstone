# MaizeGuard Rwanda - Initial Software Product Demo

MaizeGuard Rwanda is a capstone MVP for **A Deep Learning-Based Maize Grain Quality Assessment and Post-Harvest Action Recommendation System for Smallholder Farmers in Rwanda**.

The product lets a farmer, cooperative officer, or post-harvest support worker upload a maize image and receive:

- visible quality class
- confidence score
- risk level
- `needs_review` safety status
- recommended post-harvest action

The current demo uses a Next.js frontend, a Next.js API route, and a FastAPI PyTorch model server.

## GitHub Repository

Add the final GitHub link after pushing:

```text
https://github.com/honorine22/ml-final-capstone
```

## Technology Stack

- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS
- API proxy: Next.js route handler at `app/api/analyze/route.ts`
- Model API: FastAPI at `model_server/pytorch_main.py`
- ML framework: PyTorch, torchvision, timm
- Model used for demo: `mobilenetv3_large_100`
- Model export folder: `model_server/model_exports`

## Project Structure

```text
app/                                  Next.js frontend and API route
model_server/                         FastAPI model server
notebooks/                            ML training notebooks
scripts/                              Dataset preparation and training scripts
docs/diagrams/                        Required report/UML/design diagrams
docs/screenshots/                     App screenshots and ML visualizations
model_server/model_exports/           API-ready PyTorch model checkpoint and metadata
reports/models/                       Training metrics, manifests, audit files, and API example
```

## Setup

Install frontend dependencies:

```bash
npm install
```

Create local environment file:

```bash
cp .env.example .env.local
```

Make sure `.env.local` contains:

```text
MODEL_API_URL=http://localhost:8000
```

Install model API dependencies:

```bash
cd model_server
pip install -r requirements-pytorch.txt
```

Run the FastAPI model server:

```bash
uvicorn pytorch_main:app --reload --port 8000
```

Run the frontend in another terminal:

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

Build check:

```bash
npm run build
```

## How The Demo Works

```text
Next.js upload UI
  -> app/api/analyze/route.ts
  -> FastAPI /predict endpoint
  -> PyTorch model checkpoint
  -> JSON result returned to frontend
```

The frontend displays the model result. If the API returns `needs_review: true`, the app shows **Needs review** instead of presenting a risky final decision.

Expected API response shape:

```json
{
  "label": "good",
  "raw_label": "good",
  "confidence": 0.988,
  "confidence_percent": 98.8,
  "needs_review": false,
  "probabilities": {
    "good": 0.988,
    "broken": 0.004,
    "impurity": 0.002,
    "mold_risk": 0.006
  },
  "risk": "Low",
  "action": "Store safely or prepare for sale",
  "recommendation": "The maize appears clean. Store in a dry place and monitor normally."
}
```

## Product Features Demonstrated

- Maize image upload and preview
- Model-backed classification through API
- Risk-aware result display
- Confidence visualization
- `needs_review` safety behavior
- Post-harvest recommendation
- Recent assessment panel
- Dataset and ML readiness section
- API route suitable for Swagger/Postman testing

## ML Notebook

Main notebook:

[notebooks/maizeguard_public_datasets_pytorch_training.ipynb](notebooks/maizeguard_public_datasets_pytorch_training.ipynb)

The notebook includes:

- public dataset discovery/download logic
- strict maize-only data mapping
- dataset audit to avoid non-maize leakage
- data visualizations and class distributions
- image preprocessing and augmentation
- MobileNetV3 model architecture and training setup
- optimizer, learning rate, loss function, and fine-tuning plan
- accuracy, precision, recall, F1-score, confusion matrix, and error analysis
- API response example
- export files for deployment

## Current Model Result

From `reports/models/model_comparison_summary.csv`:

| Model | Best validation macro F1 | Test accuracy | Test precision macro | Test recall macro | Test F1 macro | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Large | 0.94375 | 0.97727 | 0.97500 | 0.97222 | 0.97214 | 16.25 MB |

Model architecture summary:

[model_server/model_exports/maizeguard_model_metadata.json](model_server/model_exports/maizeguard_model_metadata.json)

## Designs, Diagrams, And Screenshots

Required diagrams are in:

[docs/diagrams](docs/diagrams)

Files included:

- `01_research_model.png`
- `02_system_architecture.png`
- `03_use_case_diagram.png`
- `04_class_diagram.png`
- `05_erd.png`
- `06_sequence_diagram.png`
- `07_gantt_chart.png`

ML visualizations and screenshots are in:

[docs/screenshots](docs/screenshots)

These include class distribution, dataset source by class, sample images, training/validation loss, validation accuracy/F1, model comparison, per-class metrics, confusion matrix, and confidence distribution.

## Deployment Plan

Initial demo deployment:

- Frontend: Vercel, Netlify, or local Next.js server
- Model API: FastAPI server running on a local machine or cloud VM
- Model artifact: PyTorch `.pt` checkpoint in `model_server/model_exports`
- API communication: frontend calls `/api/analyze`, which proxies to `MODEL_API_URL`

Later production deployment:

- Host Next.js frontend on Vercel
- Host FastAPI model server on Render, Railway, Hugging Face Spaces, or a university server
- Store prediction records in PostgreSQL or SQLite
- Store uploaded images in controlled cloud storage if required
- Add authentication for cooperative officers/admins

More detail:

[docs/deployment-plan.md](docs/deployment-plan.md)

Video Demo

Link: https://youtu.be/4_IMnTrr39o
