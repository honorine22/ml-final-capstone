# Dataset Sources

The final MaizeGuard model uses public maize/corn kernel data for training and separate external images for validation.

## Current Classes

- `good`
- `broken`
- `impurity`
- `mold_risk`

`mold_risk` means visible quality risk only. It is not a laboratory aflatoxin result.

## Main Public Sources

## CK-CNN / Deep Learning Based Corn Kernel Classification

- GitHub: https://github.com/vision-cidis/CK-CNN
- Best use: main supervised public source
- Mapping:

```text
good      -> good
defective -> broken
impurity  -> impurity
rotten    -> mold_risk, only where clearly labeled
```

## CK-CNNLW Synthesized Clusters

- GitHub: https://github.com/vision-cidis/CK-CNNLW
- Best use: optional synthetic augmentation for training only
- Repository note: the project describes a segmentation-classification pipeline with Mask R-CNN segmentation and a lightweight CNN classifier. The classifier receives single segmented elements and supports good, defective, and impurity categories.

The `dataset/synthesized` folder contains cluster images and VIA polygon annotations. Do not place a full cluster image directly into one class folder. Instead, crop each annotated polygon into a single-kernel or impurity image.

Optional preparation script:

```bash
git clone https://github.com/vision-cidis/CK-CNNLW.git data/raw/CK-CNNLW
python scripts/prepare_ckcnnlw_synthetic_crops.py \
  --source-root data/raw/CK-CNNLW/dataset/synthesized \
  --out-dir data/synthetic_ckcnnlw_crops
```

Recommended use:

```text
data/synthetic_ckcnnlw_crops/train/good       -> add to training/good
data/synthetic_ckcnnlw_crops/train/broken     -> add to training/broken
data/synthetic_ckcnnlw_crops/train/impurity   -> add to training/impurity
data/synthetic_ckcnnlw_crops/train/mold_risk  -> add only if labels are clearly rotten/fungal/mold
```

Keep CK-CNNLW synthetic crops out of final test metrics. Use them for training only, then test again on CK-CNN holdout and external phone-style images.

## EfficientMaize

- Dataset: https://data.mendeley.com/datasets/r6vvm5jkh6/2
- DOI: `10.17632/r6vvm5jkh6.2`
- License: CC BY 4.0
- Best use: external/domain-shift test data and possible future support

The broad `bad` label should not automatically be mapped to `mold_risk`.

## GrainSet Maize

- Project: https://grainnet.github.io/GrainSet.html
- Paper DOI: https://doi.org/10.1038/s41597-023-02660-8
- Best use: future maize kernel quality expansion when labels clearly match the project classes

## Final Test Data

The final repo keeps lightweight test images in:

```text
data/external_test/
```

Run:

```bash
python scripts/evaluate_test_images.py
```

The output is saved to:

```text
reports/external_test/
```
