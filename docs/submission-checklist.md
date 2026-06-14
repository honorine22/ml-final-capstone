# Initial Software Demo Submission Checklist

## ML Track Requirements

| Requirement | Repo Evidence | Status |
| --- | --- | --- |
| Model notebook | `notebooks/maizeguard_public_datasets_pytorch_training.ipynb` | Done |
| Data visualization | `docs/screenshots/01_class_distribution_by_split.png`, `02_dataset_source_by_class.png`, `05_sample_images_by_class.png` | Done |
| Data engineering | strict mapping and audit in notebook; `reports/models/dataset_mapping_audit.json` | Done |
| Model architecture | `model_server/model_exports/maizeguard_model_metadata.json` | Done |
| Performance metrics | `reports/models/model_comparison_summary.csv`, confusion matrix screenshots | Done |
| Deployment option | Next.js UI, Next.js API route, FastAPI Swagger/Postman-testable API | Done |

## GitHub ZIP Requirements

| Requirement | Repo Evidence | Status |
| --- | --- | --- |
| README description | `README.md` | Done |
| GitHub repo link | `README.md` placeholder to replace after push | Needs final link |
| Setup instructions | `README.md`, `.env.example`, `model_server/requirements-pytorch.txt` | Done |
| Designs/diagrams | `docs/diagrams/` | Done |
| App screenshots/visuals | `docs/screenshots/` | Done |
| Deployment plan | `docs/deployment-plan.md` | Done |
| Video demo guidance | `docs/video-demo-script.md` | Done |
| Code files | `app/`, `model_server/`, `scripts/`, `notebooks/` | Done |

## Rubric Alignment

| Rubric Item | Evidence |
| --- | --- |
| Review requirements and tools | README, ML notebook, dataset docs, diagrams, API flow |
| Development environment setup | npm setup, FastAPI setup, `.env.example`, build verified |
| Navigation and layout structures | Next.js app has project, workflow, results, demo/data sections |

## Before Zipping

1. Replace the GitHub placeholder in `README.md`.
2. Confirm `node_modules` and `.next` are not included in the ZIP.
3. Confirm model files in `model_server/model_exports` are included if the supervisor expects offline demo.
4. Record a 5-10 minute demo video focused on functionality.
