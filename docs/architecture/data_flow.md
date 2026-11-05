# Data Flow

## 1. Overview

The data flow describes how raw network traffic observations are transformed, used for model development, and eventually evaluated.

The general workflow is:

```text
Raw Dataset
     |
     v
Dataset Preparation
     |
     v
Train / Validation / Test Split
     |
     v
Preprocessing
     |
     v
Model Training
     |
     v
Anomaly Scoring
     |
     v
Threshold Selection
     |
     v
Final Evaluation
```

## 2. Dataset Preparation

The `data/` module handles dataset preparation and splitting.

The objective is to produce separate training, validation, and test sets.

The splitting strategy must preserve the intended ordering of observations when temporal evaluation is required.

Dataset preparation also generates metadata describing the resulting dataset, supporting traceability and reproducibility.

## 3. Preprocessing

The `preprocessing/` module transforms observations into representations suitable for the selected model.

Preprocessing can involve two stages:

- **Pointwise preprocessing:** Imputation, feature selection, scaling, and dimensionality reduction.
- **Temporal preprocessing:** Window construction and optional temporal feature extraction.

The exact transformations depend on the selected configuration.

Preprocessing transformations are fitted using training data and subsequently applied to validation and test data. This prevents information from evaluation datasets from influencing the fitted transformations.

For models requiring normal-only training data, the training subset is restricted to normal observations according to the corresponding training strategy.


## 4. Input adaptation
BaseModel is responsible for adapting the processed data to the input requirements of its architecture.

So

- Prep: transforms the data
- Model: decides what representation it consumes

```
                    X_processed
                         |
              +----------+----------+
              |          |          |
              v          v          v
          IsoForest      AE     Transformer
              |          |          |
           Flatten     Flatten    Keep 3D
              |          |          |
              v          v          v
          (N, W*F)    (N, W*F)   (N, W, F)

```



Model-independent transformations belong to preprocessing. Architecture-specific input adaptations belong to the model.


## 5. Model Training

The selected model receives the preprocessed training data.

Training behavior depends on the model family:

- Reconstruction-based models learn to represent or reconstruct their training distribution.
- Isolation-based models identify observations that deviate from the learned distribution.
- Hybrid approaches combine different detection mechanisms.

Some models require normal-only training data, while others may support different training strategies.

The trainer is responsible for executing the corresponding optimization procedure.

## 6. Anomaly Scoring

After training, the model generates anomaly scores for observations or sequences.

Scores represent the degree to which an observation deviates from the learned normal behavior.

The score convention is standardized so that higher scores indicate greater anomaly likelihood.

At this stage, the model produces continuous scores rather than necessarily making binary anomaly decisions.

## 7. Threshold Selection

The `thresholding/` module determines how continuous anomaly scores are converted into binary predictions.

When threshold selection is required, validation data is used to determine the decision threshold according to the configured strategy.

This separates model learning from the operational decision of classifying an observation as anomalous.

Some detection approaches may use alternative decision mechanisms and therefore not require a fitted threshold.

## 8. Evaluation

The `evaluation/` module measures detector performance.

Validation data supports model development and threshold selection, while test data is reserved for final evaluation.

Evaluation can include:

- Precision and recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Inference time.

The test set should not influence preprocessing fitting, model training, hyperparameter optimization, or threshold selection.

## 9. Important Data Handling Principles

The data flow follows several principles:

1. Dataset splitting precedes preprocessing fitting.
2. Preprocessing transformations are fitted on training data and reused for validation and test data.
3. Model-specific training requirements determine whether training observations must be filtered.
4. Validation data supports model development and threshold selection.
5. Test data is reserved for final performance assessment.
6. The original observation ordering must be preserved when temporal transformations are applied.
7. Effective preprocessing configurations and dataset metadata should be tracked for reproducibility.