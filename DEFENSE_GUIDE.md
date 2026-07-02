# MaizeGuard Defense Guide

## One-Minute Project Summary

MaizeGuard Rwanda is a web-based machine learning prototype for visible maize grain quality assessment. A user uploads a maize image, the frontend sends it to a FastAPI model server, and the system returns a quality class, confidence score, risk level, and post-harvest recommendation.

The system supports four visible classes:

- `good`
- `broken`
- `impurity`
- `mold_risk`

The system is a decision-support prototype. It does not claim to perform laboratory aflatoxin detection or official food safety certification.

## Approved Scope Alignment

Proposal objective:

> Develop and evaluate a software prototype that classifies visible maize grain quality risks from images and recommends practical post-harvest actions.

Implemented:

- Image upload and preview.
- Next.js API proxy.
- FastAPI PyTorch prediction server.
- MobileNetV3 Large model checkpoint.
- Four-class visible quality classification.
- Confidence and probability output.
- Risk mapping and recommendation rules.
- `Needs review` safety logic for uncertain predictions.
- Testing with controlled holdout images and external/domain-shift images.

## Model Details To Know

Model:

- MobileNetV3 Large.
- Transfer learning from ImageNet.
- Final classifier adapted to four maize classes.
- Input image size: `320 x 320`.
- Framework: PyTorch, torchvision, timm.

Why MobileNetV3:

- It is lighter than large CNNs such as ResNet50.
- It is practical for low-resource deployment.
- It gives a good accuracy-size balance for a farmer-facing prototype.

Preprocessing:

- Convert image to RGB.
- Resize to model input size.
- Convert to tensor.
- Normalize using ImageNet mean and standard deviation.

Output:

- Softmax probabilities over the four classes.
- Highest probability is the raw predicted class.
- Additional safety checks decide whether to show final recommendation or `Needs review`.

## Dataset And Mapping

Public datasets were used because local farmer images were limited.

Main mapping:

- CK-CNN good corn -> `good`
- CK-CNN defective corn -> `broken`
- CK-CNN impurity -> `impurity`
- Rotten/mold-like samples where clearly labeled -> `mold_risk`

CK-CNNLW synthesized crops:

- Used as augmentation idea and support for individual kernel crops.
- `good` -> `good`
- `defective` -> `broken`
- `others` -> `impurity`
- Nothing from CK-CNNLW is forced into `mold_risk`.

Important defense point:

> I avoided forcing unclear `bad` images into mold because wrong labels would make the model learn the wrong thing.

## Current Results

Controlled public holdout:

- Accuracy: `97.73%`
- Macro precision: `97.50%`
- Macro recall: `97.22%`
- Macro F1: `97.21%`
- Test samples: `44`
- Mistakes: `1`

External/domain-shift testing:

- CK-CNN good: `18/18` raw correct, with 2 needing review.
- CK-CNN broken: `8/9` raw correct.
- CK-CNN impurity: `8/8` raw correct.
- CK-CNN mold risk: `9/9` raw correct.
- EfficientMaize external images showed domain shift and were mostly routed to `Needs review`.

How to explain this:

> The model performs well on controlled public maize kernel images, but cross-dataset phone or differently styled images can still confuse it. That is why the system includes a `Needs review` layer and why future work focuses on local Rwanda maize photos.

## Testing Strategies

Use these in the demo and defense:

1. Functional testing:
   - Upload image.
   - Preview full photo.
   - API returns prediction.
   - Result displays label, confidence, risk, and recommendation.

2. Class testing:
   - Test `good`.
   - Test `broken`.
   - Test `impurity`.
   - Test `mold_risk`.

3. Edge case testing:
   - Very small image.
   - Blank/overexposed image.
   - Dark image.
   - Uncertain external image.

4. API testing:
   - Test FastAPI `/docs`.
   - Test `POST /predict`.
   - Test Next.js `/api/analyze`.

5. Environment testing:
   - Kaggle training notebook.
   - Local Mac CPU backend.
   - Next.js production build.
   - Planned online deployment with Vercel plus Render/Railway/Hugging Face Spaces.

## Backend Logic

Files:

- `capstone-backend/model_server/pytorch_main.py`
- `capstone-backend/model_server/main.py`

Important functions:

- `recommendation_for`: maps class to risk/action.
- `needs_review`: checks confidence and top-two probability margin.
- `image_quality_review`: rejects tiny, blank, overexposed, or dark images.
- `make_batch_views`: creates full image plus large tiles.
- `predict_views`: averages full-image and tile predictions.

Why tile inference:

> Some batch photos contain many kernels. Full-image resizing may hide small risk evidence, so the API also checks large tiles and combines their probabilities.

## Frontend Logic

Files:

- `capstone-frontend/app/page.tsx`
- `capstone-frontend/app/api/analyze/route.ts`

Flow:

1. User selects an image.
2. Frontend validates file type and size.
3. Frontend previews the full image.
4. Next.js sends image to `/api/analyze`.
5. `/api/analyze` forwards image to FastAPI `/predict`.
6. FastAPI returns JSON.
7. Frontend displays result.

## Important Limitations

- The model is trained mostly on public datasets, not enough local Rwanda farmer photos.
- It only detects visible quality signs.
- It cannot confirm aflatoxin scientifically.
- `mold_risk` means visible risk, not laboratory proof.
- External image styles can cause domain shift.
- More local data is needed before real field deployment.

## Future Work

- Collect local Rwanda maize images.
- Label with cooperative/agronomy support.
- Fine-tune model using real phone photos.
- Add prediction history database.
- Add cooperative officer feedback.
- Improve deployment and offline-friendly guidance.
- Add explainability visuals such as Grad-CAM for supervisor/user trust.

## Short Answers To Common Questions

Why did you use public datasets?

> Local data collection was limited within the capstone timeline, so I used public maize/corn datasets to build a working prototype and kept local-data collection as future work.

Why is `Needs review` important?

> It prevents the system from giving unsafe advice when confidence is low, the image is unclear, or another risk class is close to the top prediction.

Why not detect aflatoxin?

> Aflatoxin cannot be confirmed from ordinary RGB images with laboratory certainty. My system only screens visible quality risk and recommends further checking where needed.

What is macro F1?

> Macro F1 calculates F1-score for each class and averages them equally, so it is useful when classes are imbalanced.

Why not only accuracy?

> Accuracy can hide weak performance on minority classes. For this project, mold-risk and impurity are important, so macro F1, precision, recall, and confusion matrix are more informative.

What is deployment architecture?

> The frontend is a Next.js app. It calls its own `/api/analyze` route, which forwards the image to a FastAPI backend. The backend loads the PyTorch model and returns JSON.

