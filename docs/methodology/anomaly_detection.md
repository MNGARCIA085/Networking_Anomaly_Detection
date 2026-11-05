# Anomaly Detection Methodology

## Detection approach

The framework supports multiple anomaly detection strategies, including reconstruction-based, isolation-based, and hybrid approaches.

All models expose a common scoring interface. Scores are oriented consistently: higher values indicate greater anomaly likelihood.

## Reconstruction-based detection

Autoencoders (AE, VAE, Transformer AE) learn patterns from training data and use reconstruction error as an anomaly score.

The underlying assumption is that observations unlike the learned normal patterns will generally produce higher reconstruction errors.

These models are trained using normal observations only. This reduces the risk of learning anomalous behavior as part of the normal reference.

## Isolation-based detection

Isolation Forest identifies observations that are easier to isolate from the rest of the data.

Unlike reconstruction-based models, it does not require learning to reconstruct normal observations. Its training strategy can therefore differ.

## Hybrid detection

AE + Isolation Forest combines both approaches:

1. Train an AE using normal observations.
2. Extract latent representations.
3. Train Isolation Forest on those representations.
4. Use Isolation Forest anomaly scores for detection.

This allows isolation-based detection to operate on learned representations rather than the original feature space.

## Important assumptions

- Training data should adequately represent normal network behavior for reconstrucion models.
- Anomalies should exhibit characteristics distinguishable from learned normal patterns.
- Reconstruction error is not necessarily equivalent to anomaly probability.
- Anomaly scores require an appropriate threshold to produce binary predictions.

These assumptions are methodological and may not hold equally well across all datasets or attack types.