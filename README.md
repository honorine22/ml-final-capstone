# MaizeGuard Rwanda - Initial Software Demo

MaizeGuard Rwanda is an initial software product demo for the capstone project **A Deep Learning-Based Maize Grain Quality Assessment and Post-Harvest Action Recommendation System for Smallholder Farmers in Rwanda**.

The current version focuses on the initial product experience: image upload/capture mockup, image preview, demo prediction endpoint, visible quality category display, confidence/risk output, recommendation rules, demo metrics, recent assessment records, and dataset readiness. The ML model is represented through a mock `/api/analyze` route so the supervisor can review the product flow while model training continues.

## GitHub Repository

Add your GitHub repository link here after pushing:

`https://github.com/<your-username>/<your-repo-name>`

## Technology Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Lucide React icons
- Mock prediction API route

## Setup

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the application:

```text
http://localhost:3000
```

Build for production:

```bash
npm run build
```

## Product Features Demonstrated

- Farmer/cooperative-facing maize quality assessment interface
- Image upload/camera capture mockup with local preview
- Mock `/api/analyze` route for the future ML inference endpoint
- Classification scenarios for good, broken, impurity-contaminated, discolored, and mold-risk maize grain
- Confidence score and risk-level display
- Rule-based post-harvest recommendation output
- Recent assessment history panel
- Demo metrics for supervisor review
- Dataset and ML-readiness section

## Dataset and ML Plan

Recommended sources:

- EfficientMaize: good/bad maize classification baseline with 4,846 raw images and augmented data
- GrainSet maize subset: normal, damaged/DU, and impurity kernel examples
- Deep Learning based Corn Kernel Classification dataset: good, defective, and impurity classes
- Local Rwanda phone images: final validation under local lighting/background conditions

See [docs/dataset-sources.md](docs/dataset-sources.md) and [docs/ml-roadmap.md](docs/ml-roadmap.md) for dataset links, class mapping, training steps, and integration plan.

## Model Comparison Notebook

The repo includes a ready-to-run notebook and script:

- [notebooks/model_comparison.ipynb](notebooks/model_comparison.ipynb)
- [scripts/train_compare.py](scripts/train_compare.py)

After placing images into `data/processed/train`, `data/processed/val`, and `data/processed/test`, run:

```bash
python scripts/train_compare.py --data-dir data/processed --epochs 12
```

The script compares Custom CNN, MobileNetV2, EfficientNetB0, and ResNet50, then saves the ranking to `reports/model_comparison.json`.

Current first-run result using CK-CNN individual kernel images:

- Best CNN by macro F1: `ResNet50`
- Test accuracy: `0.773`
- Macro F1-score: `0.740`
- Lightweight frontend model: `public/model/maize_linear_model.json`

The frontend currently uses the exported lightweight model for browser-side predictions and keeps the safer visual review guardrail so unknown samples are not marked as low risk.

## Designs and Screenshots

Design assets to include in the final submission:

- Screenshot of the home assessment screen
- Screenshot of risk and recommendation state
- Figma mockup link, if created
- System architecture and UML diagrams from the report

## Deployment Plan

The frontend can be deployed on Vercel or Netlify. The future ML-backed version can use:

1. Next.js frontend for user interface
2. FastAPI or Next.js backend for image upload and prediction endpoint
3. TensorFlow/Keras or PyTorch model exported for inference
4. PostgreSQL or SQLite for storing assessment records and feedback
5. Cloud storage for uploaded maize images, if needed

## Video Demo Guidance

For the required 5-10 minute video, focus on:

1. Opening the app and explaining the user problem briefly
2. Demonstrating image upload/capture mockup
3. Switching between quality scenarios
4. Explaining confidence, risk level, and recommendation
5. Showing recent records and deployment plan

Avoid a long research introduction; focus on product functionality.
