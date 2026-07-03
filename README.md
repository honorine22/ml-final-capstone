# MaizeGuard Rwanda - Final Product Demo

MaizeGuard Rwanda is a machine-learning web prototype for visible maize grain quality assessment and post-harvest action recommendation. A user uploads a maize image, the system sends it to a PyTorch model API, and the interface displays:

- predicted visible quality class
- confidence score and class probabilities
- risk level
- `needs_review` safety status
- recommended post-harvest action

The current model supports four classes: `good`, `broken`, `impurity`, and `mold_risk`.

## Repository Link

```text
https://github.com/honorine22/ml-final-capstone
```

## Demo Video And Deployment

Video demo:

```text
https://youtu.be/4_IMnTrr39o
```

Deployed app or local demo URL:

```text
Add deployed URL here after deployment.
```

For local demonstration:

```text
Frontend: http://localhost:3000
Model API docs: http://localhost:8000/docs
```

## Technology Stack

- Frontend: Next.js 14, React, TypeScript, Tailwind CSS
- API proxy: Next.js route handler at `capstone-frontend/app/api/analyze/route.ts`
- Model API: FastAPI at `capstone-backend/model_server/pytorch_main.py`
- ML framework: PyTorch, torchvision, timm
- Model: MobileNetV3 Large with ImageNet transfer learning
- Model artifact: `capstone-backend/model_server/model_exports/maizeguard_public_best_model.pt`

## Step-By-Step Setup

1. Create and activate a Python environment for the backend.

```bash
cd capstone-backend
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the model API dependencies.

```bash
pip install -r requirements.txt
```

3. Run the FastAPI model server.

```bash
./start.sh
```

4. In a second terminal, install frontend dependencies.

```bash
cd capstone-frontend
npm install
```

5. Create the local frontend environment file.

```bash
cp .env.example .env.local
```

6. Confirm `capstone-frontend/.env.local` contains:

```text
MODEL_API_URL=http://localhost:8000
```

7. Run the frontend.

```bash
npm run dev
```

8. Open the app.

```text
http://localhost:3000
```

## Related Project Files

```text
capstone-frontend/                    Next.js frontend and API route
capstone-backend/                     FastAPI PyTorch model server
capstone-backend/model_server/        API code and model loading logic
capstone-backend/model_server/model_exports/
                                      API-ready model checkpoint and metadata
capstone-backend/notebooks/           final ML training notebooks
capstone-backend/scripts/             testing and dataset utility scripts
capstone-backend/data/external_test/  test images used for demo validation
capstone-backend/reports/models/      model metrics and evaluation reports
capstone-backend/reports/external_test/
                                      external test predictions and summary
capstone-backend/docs/screenshots/    testing screenshots and ML result images
capstone-backend/docs/diagrams/       project diagrams from the capstone design
```

Generated local folders such as `node_modules/`, `.next/`, `.venv/`, `data/raw/`, and `data/processed/` are not required inside the final zip because they can be recreated. Raw training datasets were removed from the submission copy to keep the repository focused on the final product, model artifact, testing evidence, and deployment files.

## How The Product Works

```text
Browser upload UI
  -> Next.js /api/analyze
  -> FastAPI /predict
  -> PyTorch MobileNetV3 model
  -> JSON result
  -> frontend result display
```

The API response includes `label`, `raw_label`, `confidence`, `probabilities`, `needs_review`, `risk`, `action`, and `recommendation`.

If the model sees an unclear, tiny, or uncertain image, the frontend displays **Needs review** instead of forcing a risky final decision.

## Core Functionalities Demonstrated

- Upload a maize image.
- Preview the full uploaded photo.
- Send the image through the Next.js API route to the FastAPI model server.
- Classify visible maize quality as `good`, `broken`, `impurity`, or `mold_risk`.
- Display confidence and class probabilities.
- Convert model output into a post-harvest action.
- Mark unsafe or uncertain predictions as `Needs review`.
- Test with multiple data values and image sources.

## Testing Results

Run all verification checks:

```bash
cd capstone-frontend
npm run build

cd ../capstone-backend
source .venv/bin/activate
python scripts/evaluate_test_images.py
python scripts/check_capstone_ready.py
```

Latest controlled public holdout result from `capstone-backend/reports/models/model_metrics_summary.csv`:

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Test samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Large | 0.9773 | 0.9750 | 0.9722 | 0.9721 | 44 |

External/domain-shift testing from `capstone-backend/reports/external_test/summary.json`:

| Test group | Samples | Raw accuracy | Needs review | Final decision accuracy |
| --- | ---: | ---: | ---: | ---: |
| CK-CNN good | 18 | 1.0000 | 2 | 1.0000 |
| CK-CNN broken | 9 | 0.8889 | 1 | 0.8750 |
| CK-CNN impurity | 8 | 1.0000 | 0 | 1.0000 |
| CK-CNN mold risk | 9 | 1.0000 | 0 | 1.0000 |
| EfficientMaize good | 12 | 0.0833 | 12 | Not applicable |
| EfficientMaize bad unresolved | 12 | Not applicable | 11 | Not applicable |

The EfficientMaize results are intentionally treated as cross-domain evidence, not final model quality. They show that images from a different dataset style can confuse the model, so the app uses the `Needs review` safety layer.

Screenshots and visual evidence:

- `capstone-backend/docs/screenshots/00_app_interface_home.png`
- `capstone-backend/docs/screenshots/01_app_interface_home.png`
- `capstone-backend/docs/screenshots/01_class_distribution_by_split.png`
- `capstone-backend/docs/screenshots/03_sample_images_by_class.png`
- `capstone-backend/docs/screenshots/06_training_validation_loss.png`
- `capstone-backend/docs/screenshots/08_confusion_matrix_raw_argmax.png`
- `capstone-backend/docs/screenshots/10_per_class_metrics.png`
- `capstone-backend/docs/screenshots/11_mistakes_raw_argmax.png`

## Performance On Different Environments

- Kaggle GPU issue handled: the notebook detects unsupported CUDA devices such as P100 with newer PyTorch builds and falls back to CPU instead of crashing.
- Local Mac test: FastAPI loads the PyTorch model on CPU and returns predictions through `/predict`.
- Web build test: `npm run build` passes and produces the optimized Next.js production build.
- API integration test: posting an image through `/api/analyze` returns the live model result expected by the frontend.

## Analysis

The project achieved the main objective of building a working software prototype that connects image upload, model inference, confidence display, risk mapping, and post-harvest recommendation.

The controlled CK-CNN holdout results are strong, with 97.73% accuracy and 97.21% macro F1. This shows that the exported model, preprocessing, class order, and API integration work correctly for the public training distribution.

The project partially missed the broader real-world objective because external images from different sources still show domain shift. Some EfficientMaize images are predicted as `impurity` or `broken` even when the folder label is broad `good` or `bad`. The system handles this by marking low-quality or uncertain external samples as `Needs review`, which is safer than claiming final food-safety certification.

## Discussion

The important milestone is that the capstone moved from a mockup to an end-to-end executable product. The frontend no longer gives fake default results; it calls a real model API and shows model evidence.

The results are useful because they reveal both success and limitation. The model performs well on controlled public kernel images, but real farmer phone images require more local data. This is an important finding because agricultural AI systems must be tested with the same image style expected in the field.

The `Needs review` behavior is also important. It makes the product more responsible for smallholder use because it avoids telling a farmer to store or sell a batch when the model is uncertain.

## Recommendations And Future Work

- Collect local Rwanda maize photos from markets, cooperatives, and post-harvest handling locations.
- Label local samples with support from agronomy or cooperative officers.
- Retrain or fine-tune the model with real phone images, not only public kernel datasets.
- Keep `mold_risk` as a visible-risk category and avoid claiming laboratory aflatoxin detection.
- Add a small database for prediction history and supervisor/cooperative feedback.
- Deploy the frontend online and host the FastAPI model server on Render, Railway, Hugging Face Spaces, or a university server.
- Add offline-friendly guidance for farmers with low connectivity.

## Submission Notes

Attempt 1: submit the GitHub repository link, deployed/local demo link, and video demo link.

Attempt 2: submit a zip file of this repository. Generated folders such as `node_modules`, `.next`, `.venv`, `data/raw`, and `data/processed` have been excluded from this cleaned submission copy.
