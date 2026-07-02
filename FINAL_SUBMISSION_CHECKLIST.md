# Final Submission Checklist

## Attempt 1

- [ ] GitHub repository link submitted.
- [ ] README has step-by-step install/run instructions.
- [ ] README lists related files.
- [ ] README links the 5-minute demo video.
- [ ] README includes deployed frontend URL or clearly states local demo fallback.
- [ ] README includes backend URL if deployed separately.

## Attempt 2

- [ ] Zip file created from the same repository submitted in Attempt 1.
- [ ] Zip excludes generated folders:
  - `node_modules/`
  - `.next/`
  - `.venv/`
  - `data/raw/`
  - `data/processed/`

## Required Evidence

- [x] Frontend app in `capstone-frontend/`.
- [x] Backend API in `capstone-backend/`.
- [x] Model checkpoint in `capstone-backend/model_server/model_exports/`.
- [x] Final notebook in `capstone-backend/notebooks/`.
- [x] Testing screenshots in `capstone-backend/docs/screenshots/`.
- [x] Diagrams in `capstone-backend/docs/diagrams/`.
- [x] Model metrics in `capstone-backend/reports/models/`.
- [x] External test results in `capstone-backend/reports/external_test/`.
- [x] Deployment plan in `capstone-backend/docs/deployment-plan.md`.

## Commands To Recheck Before Submission

Backend:

```bash
cd capstone-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

Frontend:

```bash
cd capstone-frontend
npm install
cp .env.example .env.local
npm run build
npm run dev
```

Readiness:

```bash
cd capstone-backend
source .venv/bin/activate
python scripts/check_capstone_ready.py
```

## Demo Video Flow

1. Show repository structure.
2. Show README run instructions.
3. Run/show backend `/docs`.
4. Run/show frontend upload page.
5. Upload one good image.
6. Upload one broken/impurity/mold-risk image.
7. Show `Needs review` for uncertain or external image.
8. Show notebook metrics, confusion matrix, and external test summary.
9. Explain deployment plan: backend first, frontend second.

