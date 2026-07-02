# MaizeGuard Test Images

This directory separates two different tests. Do not add these files to training.

## `ckcnn_sanity/`

The model's existing CK-CNN holdout images. Use these to verify that the exported
checkpoint, API class order, and preprocessing still match the notebook. Strong
performance here does not prove that the model works on real farmer photographs.

## `efficientmaize/`

Cross-domain images from EfficientMaize version 2. These were not used by the
current CK-CNN-only model. The source labels are `good` and broad `bad`.
`bad_unresolved` must not be reported as broken or mold-risk ground truth without
manual expert review.

- Dataset: https://data.mendeley.com/datasets/r6vvm5jkh6/2
- DOI: 10.17632/r6vvm5jkh6.2
- Authors: Emmanuel Asante, Obed Appiah, and Peter Appiahene
- License: CC BY 4.0

Run `node scripts/download_external_test_images.mjs` to recreate the external set.

## Recommended use

1. Upload every CK-CNN sanity image through `/api/analyze` and verify its label.
2. Upload the EfficientMaize images and record all probabilities, not only the top class.
3. Treat incorrect or overconfident EfficientMaize results as domain-shift evidence.
4. Keep a separate local-phone test set under `local_phone/` when real samples are collected.
