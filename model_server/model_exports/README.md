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

The older TensorFlow/Keras API path can still use this folder for:

- `maizeguard_model.keras`
- `class_names.json`

Example Kaggle export:

```python
best_model.save("maizeguard_model.keras")

import json
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
```

The API expects `class_names.json` to use the same class order as the model output layer.
