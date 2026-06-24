# Model Exports

The FastAPI PyTorch server loads the final Kaggle-trained model from this folder.

Required files:

- `maizeguard_public_best_model.pt`
- `maizeguard_model_metadata.json`
- `class_names.json`

Run with:

```bash
uvicorn pytorch_main:app --reload --port 8000
```

The compatibility command below also works and loads the same PyTorch app:

```bash
uvicorn main:app --reload --port 8000
```
