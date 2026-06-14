# Video Demo Script

Target length: 5-10 minutes.

## 1. Opening

State the project name:

```text
MaizeGuard Rwanda: a maize grain quality assessment and post-harvest recommendation MVP.
```

Briefly say the problem: farmers and cooperative officers need quick visible-quality screening before storage or sale.

## 2. App Demonstration

Open the frontend:

```text
http://localhost:3000
```

Show:

- upload area
- image preview
- assessment result
- confidence
- risk level
- recommended action
- recent assessments

## 3. Model-Backed Flow

Explain:

```text
The frontend does not load the model directly. It sends the image to /api/analyze, then the Next.js API route forwards it to FastAPI /predict.
```

Open:

```text
http://localhost:8000/docs
```

Show the `/predict` endpoint.

## 4. ML Notebook

Open:

```text
notebooks/maizeguard_public_datasets_pytorch_training.ipynb
```

Show:

- dataset mapping
- class distribution
- sample images
- model architecture
- training history
- confusion matrix
- accuracy, precision, recall, F1-score

## 5. Design And Diagrams

Show:

```text
docs/diagrams
docs/screenshots
```

Mention architecture, use case, class diagram, ERD, sequence diagram, and Gantt chart.

## 6. Close

End with:

```text
This is an initial MVP for supervisor feedback. The next stage is user testing with local maize images and improved deployment.
```
