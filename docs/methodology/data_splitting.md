# Data Splitting

## Dataset partitions

The dataset is divided into three independent partitions:

- **Training:** used to fit preprocessing transformations and train models.
- **Validation:** used for model comparison, hyperparameter optimization, and threshold selection.
- **Test:** reserved for final evaluation.

## Temporal splitting

The project uses temporal splitting to preserve observation order across partitions.

Since explicit timestamps are unavailable, row order is used as an implicit temporal ordering.

This assumes the original dataset ordering is meaningful. It does not establish actual elapsed time or guarantee uniformly sampled observations.

## Leakage prevention

To prevent information from validation and test data influencing training:

- Split data before fitting preprocessing transformations.
- Fit preprocessing transformations using training data only.
- Apply the fitted transformations to validation and test data.
- Do not use test data for model selection, tuning, or threshold selection.

For normal-only training strategies, anomalous observations are excluded from model training after preprocessing.

## Dataset metadata

Dataset preparation produces a manifest containing relevant information about the processed dataset and its partitions.

This supports traceability and helps identify the data used in an experiment.

## Design rationale

Temporal splitting provides a more realistic evaluation setting when observations are ordered in time, avoiding the potential leakage introduced by randomly distributing correlated observations across partitions.