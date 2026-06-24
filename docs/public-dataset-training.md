# MaizeGuard Public Dataset Training

This is the recommended public-only machine learning setup for the capstone.

## Dataset Classes

Use four safe classes:

- `good`
- `broken`
- `impurity`
- `mold_risk`

`mold_risk` is intentionally named as a visible-risk class. Public maize/corn datasets often do not provide laboratory-confirmed mold labels, so unclear `bad` images should not be forced into mold.

## Public Dataset Mapping

Use CK-CNN as the main dataset:

- CK-CNN good corn -> `good`
- CK-CNN defective, damaged, broken corn -> `broken`
- CK-CNN impurity -> `impurity`
- CK-CNN rotten/fungal when clearly labeled -> `mold_risk`

Use GrainSet maize as the larger support dataset:

- healthy/sound maize -> `good`
- damaged/unsound maize -> `broken`
- impurity labels -> `impurity`
- mold/fungal/rotten labels -> `mold_risk`

Use EfficientMaize as support:

- good -> `good`
- bad -> use only when the folder/label or visual class is clear enough for `broken` or `mold_risk`

Do not treat every broad `bad` image as mold.

## Notebook

The latest notebook has been copied into the repo:

```text
notebooks/maizeguard_public_datasets_pytorch_training.ipynb
```

It contains the full ML process:

- Kaggle setup
- public dataset discovery
- mapping into the four safe classes
- manifest creation
- class balancing
- train/validation/test split
- sample visualization
- augmentation
- model comparison with PyTorch/timm
- test metrics
- confusion matrix
- full-image deployment prediction
- `needs_review` safety rule
- export for API use

## Local Script Version

Prepare the public dataset:

```bash
python scripts/prepare_public_dataset.py \
  --roots data/raw \
  --out-dir data/public_prepared \
  --report-dir reports/public_training
```

If `mold_risk` keeps confusing the model because the public labels are weak, train a safer 3-class model:

```bash
python scripts/prepare_public_dataset.py \
  --roots data/raw \
  --out-dir data/public_prepared \
  --report-dir reports/public_training \
  --exclude-mold-risk
```

The preparation step now writes:

```text
reports/public_training/dataset_mapping_audit.json
```

Use this file to confirm the notebook is not training on `sorg`, `sorghum`, `GrainSet-tiny`, rice, or wheat folders.

Train and compare models:

```bash
python scripts/train_pytorch_public.py \
  --manifest-dir reports/public_training \
  --out-dir reports/public_training \
  --head-epochs 3 \
  --finetune-epochs 8
```

If Kaggle shows `CUDA error: no kernel image is available for execution on the device`, force CPU:

```bash
python scripts/train_pytorch_public.py \
  --manifest-dir reports/public_training \
  --out-dir reports/public_training \
  --force-cpu
```

The training script includes:

- CUDA smoke test with automatic CPU fallback
- capped class weights
- focal loss with label smoothing
- optional balanced sampler only when you pass `--balanced-sampler`
- gradient clipping
- early stopping
- validation macro F1 model selection
- saved `test_predictions_and_errors.csv`

The optional training script can compare:

- `mobilenetv3_large_100`
- `tf_efficientnetv2_b0`
- `convnext_tiny`

The best model is selected by macro F1 and exported as:

```text
reports/public_training/maizeguard_public_best_model.pt
reports/public_training/maizeguard_model_metadata.json
```

## API Export

Copy the two exported files to:

```text
model_server/model_exports/maizeguard_public_best_model.pt
model_server/model_exports/maizeguard_model_metadata.json
```

Run the PyTorch model server:

```bash
cd model_server
pip install -r requirements-pytorch.txt
uvicorn pytorch_main:app --reload --port 8000
```

Next.js calls this through:

```text
MODEL_API_URL=http://localhost:8000
```

The frontend continues to call:

```text
/api/analyze
```

The model stays on the backend.
