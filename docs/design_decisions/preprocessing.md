# Preprocessing Architecture and Design Decisions

## 1. Overview

Preprocessing is implemented as an independent component of the experiment pipeline.

Its purpose is to transform raw network observations into representations suitable for the selected model.

The preprocessing architecture separates transformations into two stages:

1. **Pointwise preprocessing:** Operations applied independently to individual observations.
2. **Temporal preprocessing:** Operations that introduce or manipulate relationships between observations.

This separation allows preprocessing strategies to be configured independently of model architectures.

## 2. Processing Stages

The general preprocessing sequence is:

```text
Raw observations
        │
        ▼
[ Pointwise Pipeline ]
        │
        ▼
[ Windowing ] ─────────── y
        │                  │
        ▼                  │
[ Temporal Pipeline ]      │
       │                   │
       ▼                   ▼
Model Input             y_windowed


```

### 2.1 Pointwise Preprocessing

Pointwise preprocessing handles feature-level transformations.

Depending on the configuration, it can include:

- Imputation of missing values.
- Feature selection using variance-based criteria.
- Feature scaling.
- Dimensionality reduction using PCA.

The transformation order is intentional:

1. Imputation.
2. Optional feature selection.
3. Scaling.
4. Optional dimensionality reduction.

Imputation precedes feature selection and scaling to ensure that subsequent transformations operate on valid numerical inputs.

Scaling precedes PCA because dimensionality reduction should not be dominated by features with larger numerical ranges.

### 2.2 Windowing

Windowing groups consecutive observations into sequences.

Given a window size \(W\), each sequence contains \(W\) consecutive observations.

Window construction is controlled by parameters such as:

- Window size.
- Stride.
- Labeling strategy.

The original observation ordering is preserved.

A window size of one represents the pointwise configuration: no meaningful temporal context is introduced.

### 2.3 Temporal Preprocessing

Temporal preprocessing handles transformations that depend on the constructed sequences.

It can include temporal features such as differences between consecutive observations.

These transformations allow models to receive information about changes between observations in addition to their original feature values.

## 3. Shared Windowing Across Model Families

Windowing is treated as a shared preprocessing operation rather than an exclusive capability of sequence-based architectures.

This is an intentional methodological decision designed to improve experimental comparability.

Both pointwise and sequence-oriented models can receive windows containing the same observations.

However, their input representations differ.

| Model family | Input representation |
|---|---|
| AE / VAE | Flattened windows: `(N, W × F)` |
| Isolation Forest | Flattened windows: `(N, W × F)` |
| Transformer | Sequence representation: `(N, W, F)` |

Here:

- `N` represents the number of windows.
- `W` represents the window size.
- `F` represents the number of features per observation.

Flattening allows models designed for individual feature vectors to process contextual information without explicitly modeling sequential structure.

Transformers retain the sequence dimensions and can explicitly learn relationships between positions within a window.

This design helps investigate two distinct questions:

1. Does providing neighboring observations improve anomaly detection?
2. Does explicitly modeling sequential relationships provide additional benefits?

A window size of one provides a pointwise baseline without meaningful temporal context.

**Important:** Shared window construction improves comparability but does not guarantee a completely fair comparison. Model capacity, optimization, scoring granularity, and computational requirements must also be considered.

## 4. Fitting and Data Leakage

Preprocessing transformations must be fitted using training data only.

Once fitted, the same transformations are applied to validation and test data.

This prevents information from evaluation datasets from influencing the learned preprocessing parameters.

For models requiring normal-only training data, fitting behavior must also respect the corresponding training methodology.

In particular, transformations that learn statistical parameters from data must not inadvertently incorporate anomalous training observations when the intended training protocol is normal-only.

## 5. Training Data Requirements

Preprocessing is independent of the model's training-data requirements.

Different model families may require different subsets of the preprocessed training data.

For example:

- Reconstruction-based anomaly detectors such as AE and VAE are trained using normal observations.
- Isolation Forest can use different training strategies depending on the experimental setup.
- Hybrid models can combine components with different training requirements.

These restrictions belong to the training methodology rather than the preprocessing architecture itself.

## 6. Window Labeling

Window construction introduces a labeling decision when observation-level labels are available.

The project considers different strategies:

- `last`: The window inherits the label of its final observation.
- `any`: The window is positive if at least one observation is anomalous.
- `majority`: The window label is determined by the majority of observation labels.

The choice affects the resulting class distribution and the interpretation of evaluation metrics.

For the objective of identifying individual suspicious network observations, the `last` strategy is preferred because it associates each window with its final observation.

## 7. Design Rationale

The preprocessing architecture is designed around three principles:

1. **Separation of concerns:** Feature-level and temporal transformations are independently configurable.
2. **Model independence:** Preprocessing does not dictate the architecture or implementation of the selected model.
3. **Experimental comparability:** Shared window construction allows pointwise and sequence-oriented architectures to access equivalent observation context.

This design facilitates controlled experiments comparing different anomaly detection approaches while keeping preprocessing decisions explicit and reproducible.