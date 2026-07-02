# Data Folder

Only lightweight test data should remain in the final repository.

```text
data/external_test/
```

This folder contains demo validation images used by `scripts/evaluate_test_images.py`.

Generated or downloaded training folders should not be included in the final zip:

```text
data/raw/
data/processed/
```

The final model already lives in:

```text
model_server/model_exports/maizeguard_public_best_model.pt
```

Current model classes:

- `good`
- `broken`
- `impurity`
- `mold_risk`
