# Evaluation Methodology

## Model comparison

PR-AUC (Area Under the Precision-Recall Curve) is the primary metric for comparing models.

It evaluates ranking performance across different thresholds, making it suitable for comparing anomaly scores without committing to a particular operating threshold.

PR-AUC is particularly relevant when anomaly detection involves class imbalance.

## Threshold selection

After model training, the thresholding strategy selects an operating threshold using validation data.

The selected threshold converts continuous anomaly scores into binary predictions:

- Score >= threshold: anomaly.
- Score < threshold: normal.

Threshold selection is configurable and independent of model implementation.

## Validation metrics

Validation data serves two purposes:

1. Model comparison and hyperparameter optimization using PR-AUC.
2. Threshold selection and evaluation of the resulting operating point.

Threshold-dependent metrics, such as precision and recall, may be optimistic because the threshold is selected using the same validation data.

Similarly, repeated model selection using validation PR-AUC can lead to optimistic estimates of model performance.

These metrics should therefore be treated as development estimates, not unbiased estimates of generalization.

## Final evaluation

The test set is reserved for evaluating the selected model and its operating threshold.

It must not be used for:

- Hyperparameter optimization.
- Model selection.
- Threshold selection.

Final test metrics provide the primary estimate of performance on unseen data.

## Evaluation principles

- Use PR-AUC for threshold-independent model comparison.
- Select thresholds using validation data.
- Report threshold-dependent metrics alongside the selected threshold.
- Keep the test set independent of development decisions.