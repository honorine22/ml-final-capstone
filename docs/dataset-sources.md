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
