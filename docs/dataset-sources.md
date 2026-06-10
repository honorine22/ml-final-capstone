# Dataset Sources for MaizeGuard Rwanda

These are the dataset sources that match the original capstone description and literature review.

## 1. EfficientMaize

- Dataset page: https://data.mendeley.com/datasets/r6vvm5jkh6/2
- Article page: https://www.sciencedirect.com/science/article/pii/S2352340924002300
- DOI: `10.17632/r6vvm5jkh6.2`
- License: CC BY 4.0
- Best use: first good-vs-bad maize classifier.
- Classes: `good`, `bad`
- Size: 4,846 raw images and 28,910 augmented images.
- Notes: images were captured using a 12MP phone camera and labeled by seed experts.

Recommended mapping:

```text
good -> good
bad  -> discolored / damaged / review-needed
```

## 2. GrainSet Maize

- Project page: https://grainnet.github.io/GrainSet.html
- Code and dataset links: https://github.com/hellodfan/GrainSet
- Maize Figshare DOI: https://doi.org/10.6084/m9.figshare.22987562.v2
- Paper DOI: https://doi.org/10.1038/s41597-023-02660-8
- License note: the project page lists Creative Commons BY-NC-SA 4.0 for non-commercial use.
- Best use: multi-class maize kernel quality classifier.
- Size: maize subset has about 19K single-kernel images.
- Labels include normal kernels, damage/unsound categories, and impurities.

Recommended mapping:

```text
normal       -> good
DU/damaged   -> broken or discolored
impurities   -> impurity
```

## 3. Deep Learning Based Corn Kernel Classification / CK-CNN

- Dataset page: https://www.cidis.espol.edu.ec/es/content/dataset-deep-learning-based-corn-kernel-classification
- GitHub dataset: https://github.com/vision-cidis/CK-CNN/tree/master/dataset
- Paper page: https://openaccess.thecvf.com/content_CVPRW_2020/html/w5/Velesaca_Deep_Learning_Based_Corn_Kernel_Classification_CVPRW_2020_paper.html
- Best use: good, defective, and impurity classifier support.
- Classification dataset: 6,600 images for 3 classes at 224x224.
- Classes: good corn kernel, defective corn kernel, impurity.

Recommended mapping:

```text
good      -> good
defective -> broken / damaged
impurity  -> impurity
```

## Recommended Training Strategy

1. Use EfficientMaize first for binary `good` vs `not_good`.
2. Add CK-CNN to separate `good`, `broken/damaged`, and `impurity`.
3. Add GrainSet maize to strengthen `normal`, `damaged/unsound`, and `impurity` recognition.
4. Collect a small local Rwanda validation set for images like mixed-color maize cobs, phone photos, poor lighting, and market samples.
5. Keep `mold-risk` conservative unless you have clear labeled mold images. If the model is unsure, show `Needs quality review` instead of `Good`.

## Folder Preparation

Place images into:

```text
data/processed/train/good
data/processed/train/broken
data/processed/train/impurity
data/processed/train/discolored
data/processed/train/mold

data/processed/val/<same classes>
data/processed/test/<same classes>
```

Then run:

```bash
python scripts/train_compare.py --data-dir data/processed --epochs 12
```
