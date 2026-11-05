# Inference and Model Export

## Model export

A trained model must be exported together with the components required to reproduce its inference behavior:

- **Model:** trained model and its configuration.
- **Preprocessing:** fitted preprocessing pipeline.
- **Thresholding:** selected decision threshold and associated configuration.
- **Manifest:** metadata identifying the model, version, and required artifacts.

This ensures inference uses the same transformations and decision threshold selected during experimentation.

## Inference architecture

`InferenceRunner` coordinates the inference process without implementing model-specific logic.

```text
          Input data
              |
              v
       Preprocessing
              |
              v
       Model adaptation
              |
              v
         Model
              |
         +----+----+
         |         |
         v         v
       Score    Threshold
         |         |
         +----+----+
              |
              v
         Prediction
```

## InferenceRunner

The runner receives three components:

- `prep`: transforms incoming data using the fitted preprocessing pipeline.
- `model`: adapts input representation and computes anomaly scores and predictions.
- `thresholding`: provides the selected decision threshold.

It exposes three operations:

- **`score(X)`:** returns anomaly scores without applying a threshold.
- **`predict(X)`:** returns binary predictions using the configured threshold.
- **`predict_with_labels(X, y)`:** returns scores, processed labels, and predictions, preserving label alignment after preprocessing.

The runner separates inference orchestration from model and preprocessing implementations.

## Design rationale

This architecture provides:

- Consistent inference across different model families.
- Reuse of fitted preprocessing transformations.
- Separation between scoring and binary classification.
- A single inference interface for exported models.
- Independence between model implementation and inference orchestration.