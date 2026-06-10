# ML Integration Roadmap

This document lists the practical pieces needed to turn the MaizeGuard Rwanda interface into a model-backed capstone demo.

## Recommended Dataset Sources

| Source | Use in project | Notes |
| --- | --- | --- |
| EfficientMaize: A Lightweight Dataset for Maize Classification on Resource-Constrained Devices | Start binary good/bad classification | Mendeley Data DOI `10.17632/r6vvm5jkh6.2`; raw dataset has 4,846 images: 2,635 good and 2,211 bad; augmented set has 28,910 images. |
| GrainSet maize subset | Expand to normal, damaged/unsound, and impurity categories | Maize Figshare DOI `10.6084/m9.figshare.22987562.v2`; GrainSet contains about 19K maize kernel images with normal, defective/DU grains, and impurities. |
| Deep Learning based Corn Kernel Classification dataset | Support good, defective, and impurity classes | CIDIS/CK-CNN dataset has 6,600 images for 3-class classification at 224x224. |
| Local Rwanda phone images | Final validation and demo credibility | Collect 100-300 images across lighting/backgrounds if time allows. |

Full links are listed in [dataset-sources.md](dataset-sources.md).

## Target Class Mapping

| App class | Training source | Recommendation |
| --- | --- | --- |
| Good maize grain | EfficientMaize good, GrainSet normal | Store safely or prepare for sale |
| Broken or damaged grain | GrainSet damaged/DU, corn defective | Sort before storage |
| Impurity-contaminated grain | GrainSet impurity, corn impurity | Clean and re-screen |
| Discolored grain | Local images plus bad/defective examples | Sell quickly or refer for review |
| Visible mold-risk grain | Local labeled samples or curated bad examples | Do not store; refer to cooperative facility |

## Implementation Steps

1. Download datasets and document license/DOI/source.
2. Create `data/raw`, `data/processed/train`, `data/processed/val`, and `data/processed/test`.
3. Normalize images to one input size, such as 224x224.
4. Train baseline custom CNN, then MobileNetV2, EfficientNetB0, and ResNet50.
5. Evaluate accuracy, precision, recall, F1-score, confusion matrix, model size, and inference time.
6. Export selected model to `.keras`, SavedModel, ONNX, or TensorFlow.js.
7. Replace the mock `/api/analyze` endpoint with a FastAPI or Next.js inference adapter.

## First Local Training Result

Dataset used: CK-CNN individual kernel images prepared into `good`, `broken`, `impurity`, and `mold`.

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

The app now includes a lightweight exported baseline model for browser-side predictions at `public/model/maize_linear_model.json`. The `/api/analyze` endpoint remains shaped like the final ML endpoint and can later be replaced by ResNet50/MobileNetV2 inference.
