# ML Roadmap And Final Status

## Final Implemented Stack

- Model: MobileNetV3 Large
- Framework: PyTorch with `timm`
- Classes: `good`, `broken`, `impurity`, `mold_risk`
- API: FastAPI `/predict`
- Frontend integration: Next.js `/api/analyze`

## Current Result

From `reports/models/model_metrics_summary.csv`:

```text
accuracy: 0.9773
macro precision: 0.9750
macro recall: 0.9722
macro F1: 0.9721
test samples: 44
```

## Final Testing Strategy

1. Controlled public holdout test using CK-CNN-style images.
2. External/domain-shift test using EfficientMaize images.
3. API smoke test using FastAPI `/predict`.
4. Frontend integration test using Next.js `/api/analyze`.
5. Production build test using `npm run build`.

## Optional Accuracy Improvement

CK-CNNLW can strengthen the implementation by adding synthetic segmented crops to the training set. Its synthesized cluster images should be converted into object-level crops using:

```bash
python scripts/prepare_ckcnnlw_synthetic_crops.py
```

Use the output as training augmentation only. Do not use synthetic crops as the main final test set, because that would overstate real-world performance.

## Known Limitation

The model performs best on images similar to its public training source. External phone-style or different-resolution images can produce unreliable raw predictions. The deployed system therefore uses `needs_review` when an image is unclear, very small, or uncertain.

## Future Work

- Collect local Rwanda maize images.
- Add expert labeling for local images.
- Fine-tune the model on real phone-camera samples.
- Add a database for prediction history and cooperative feedback.
- Deploy the model API to a stable cloud service.
