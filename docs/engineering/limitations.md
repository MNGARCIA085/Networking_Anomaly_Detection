# Limitations and Future Work

## Methodological limitations

### Temporal information

The dataset does not provide explicit timestamps. Row order is used as an implicit temporal ordering.

Consequently, the framework cannot directly model elapsed time, irregular sampling intervals, or time-dependent rates.

### Anomaly detection assumptions

Reconstruction-based approaches assume that anomalies generally differ from patterns learned from normal observations.

This assumption may not hold for every attack type, particularly attacks that resemble legitimate traffic.

### Threshold selection

Thresholds are selected using validation data. Threshold-dependent validation metrics may therefore be optimistic.

Final performance must be assessed using the independent test set.

### Model comparison

Using common preprocessing and window construction improves experimental comparability, but does not guarantee equivalent model capacity, inductive biases, optimization budgets, or computational costs.

## Engineering considerations

- Reproducibility can be affected by hardware, numerical precision, and nondeterministic operations.
- Exported models depend on compatible preprocessing artifacts and configuration.
- Inference performance may vary across model architectures and input representations.

## Future work

Potential extensions include:

- Incorporating explicit timestamps and elapsed-time information.
- Exploring additional temporal anomaly detection approaches.
- Improving experiment reproducibility and artifact versioning.
- Expanding model comparison to include computational and operational constraints.
- Integrating the detector with an external API and incident investigation system.

These are potential improvements rather than commitments or assumptions about current implementation.