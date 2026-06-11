# ML Integration Roadmap

This document lists the practical pieces needed to turn the MaizeGuard Rwanda interface into a model-backed capstone demo.

## Recommended Dataset Sources

| Source | Use in project | Notes |
| --- | --- | --- |
| EfficientMaize: A Lightweight Dataset for Maize Classification on Resource-Constrained Devices | Start binary good/bad classification | Mendeley Data DOI `10.17632/r6vvm5jkh6.2`; raw dataset has 4,846 images: 2,635 good and 2,211 bad; augmented set has 28,910 images. |
| GrainSet maize subset | Expand to normal, damaged/unsound, and impurity categories | Maize Figshare DOI `10.6084/m9.figshare.22987562.v2`; GrainSet contains about 19K maize kernel images with normal, defective/DU grains, and impurities. |
| Deep Learning based Corn Kernel Classification dataset | Support good, defective, and impurity classes | CIDIS/CK-CNN dataset has 6,600 images for 3-class classification at 224x224. |
| Local Rwanda phone images | Final validation and demo credibility | Keep local images for validation/demo review unless enough labeled samples are collected. |

Full links are listed in [dataset-sources.md](dataset-sources.md).

## Target Class Mapping

| App class | Training source | Recommendation |
| --- | --- | --- |
| Good maize grain | EfficientMaize good, GrainSet normal | Store safely or prepare for sale |
| Broken or damaged grain | GrainSet damaged/DU, corn defective | Sort before storage |
| Impurity-contaminated grain | GrainSet impurity, corn impurity | Clean and re-screen |
| Visible mold-risk grain | Only clearly labeled mold/fungal/rotten public samples or local validation examples | Do not store; refer to cooperative facility |

## Implementation Steps

1. Download datasets and document license/DOI/source.
2. Create `data/raw`, `data/processed/train`, `data/processed/val`, and `data/processed/test`.
3. Normalize images to one input size, such as 224x224.
4. Train and compare PyTorch/timm models from the public-only notebook: MobileNetV3, EfficientNetV2B0, and ConvNeXtTiny.
5. Evaluate accuracy, precision, recall, F1-score, confusion matrix, model size, and inference time.
6. Export selected model to `.pt` and optional ONNX for backend inference.
7. Serve the model through FastAPI and keep Next.js calling `/api/analyze`.

## Current Public-Only Training Plan

The latest workflow is stored in `notebooks/maizeguard_public_datasets_pytorch_training.ipynb`.

The repo script version is:

- `scripts/prepare_public_dataset.py`
- `scripts/train_pytorch_public.py`
- `model_server/pytorch_main.py`

The current safe public classes are `good`, `broken`, `impurity`, and `mold_risk`. The `mold_risk` class is used only when source labels clearly indicate mold, fungus, infection, rotten, or similar visible-risk wording.

## First Local Training Result

Dataset used: CK-CNN individual kernel images prepared into `good`, `broken`, `impurity`, and early `mold`/rotten support.

Short CNN comparison run: 3 epochs, 224x224 images, ImageNet transfer weights where available.

| Model | Test accuracy | Macro F1 | Model size | Inference time / batch |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 0.773 | 0.740 | 90.72 MB | 734.93 ms |
| MobileNetV2 | 0.636 | 0.556 | 9.25 MB | 334.96 ms |
| EfficientNetB0 | 0.614 | 0.469 | 16.33 MB | 552.95 ms |
| Custom CNN | 0.409 | 0.145 | 1.13 MB | 67.44 ms |

Current best research model from this first run: **ResNet50**.

Deployment note: ResNet50 performed best on the first small test set, but MobileNetV2 is still a strong candidate for lightweight mobile/web deployment after more data and longer fine-tuning.

## Demo Boundary

The frontend calls `/api/analyze`, and the Next.js route proxies to `MODEL_API_URL`. For the latest PyTorch notebook export, run `model_server/pytorch_main.py` with `maizeguard_public_best_model.pt` and `maizeguard_model_metadata.json`.
