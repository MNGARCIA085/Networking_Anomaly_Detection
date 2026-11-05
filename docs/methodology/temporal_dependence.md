# Temporal Assumptions

## 1. Motivation

Network traffic is not necessarily composed of independent observations. Relationships between consecutive observations may contain information relevant to anomaly detection.

For example, changes in traffic volume, destination diversity, or communication patterns may provide contextual information that cannot be captured by analyzing individual observations independently.

The project therefore explores both pointwise and sequence-based anomaly detection approaches.

## 2. Implicit Temporal Ordering

The dataset does not provide explicit timestamps.

Temporal preprocessing therefore assumes that the order in which observations appear in the dataset represents a meaningful temporal ordering.

This is an implicit assumption rather than a verified property of the data.

Consequently:

- No actual elapsed time between observations is available.
- The temporal distance between consecutive observations is unknown.
- Sequence positions represent observation order, not fixed time intervals.
- The resulting sequences may not correspond to uniformly sampled time series.

This distinction is important when interpreting the behavior and performance of temporal models.

## 3. Pointwise vs. Temporal Modeling

The project supports two general approaches.

### Pointwise Detection

Each observation is treated independently.

The model learns patterns associated with individual network observations without explicitly considering their surrounding context.

Examples include Autoencoders and Isolation Forest.

### Sequence-Based Detection

Observations are grouped into sequences, allowing models to learn relationships between observations.

Architectures such as Transformer-based Autoencoders can use contextual information from neighboring observations when reconstructing or evaluating a sequence.

The underlying hypothesis is that temporal context may improve anomaly detection by capturing patterns that are difficult to identify from individual observations.

## 4. Windowing Assumptions

Sequence-based models require observations to be grouped into windows.

Window construction introduces additional assumptions:

- Consecutive observations are considered related.
- A fixed sequence length defines the context available to the model.
- Sequence construction depends on the original observation ordering.
- Window boundaries determine which observations can provide context to one another.

The labeling strategy also influences how sequence-level predictions relate to individual observations.

The project supports different window-labeling policies, such as `last`, `any`, and `majority`.

## 5. Methodological Limitations

The absence of timestamps limits the temporal interpretation of the resulting sequences.

In particular, the project cannot directly model:

- Actual elapsed time between observations.
- Irregular sampling intervals.
- Time-dependent traffic rates.
- Relationships based on explicit temporal intervals.

Furthermore, meaningful temporal dependence cannot be assumed solely because observations are presented in an ordered dataset.

The usefulness of sequence-based modeling must therefore be evaluated empirically.

## 6. Experimental Implication

Pointwise models provide an important baseline against which sequence-based approaches can be compared.

The objective is to determine whether incorporating observation context improves anomaly detection performance enough to justify the additional modeling complexity.

Any observed improvement should be interpreted in light of the implicit ordering assumption and the absence of explicit timestamps.




There are actually two separate questions:

1. Is the observation order meaningful? Are consecutive rows related in the real network process?
2. Does that relationship improve detection? Does using neighboring observations produce better results than pointwise detection?

Your current methodology assumes the first and experimentally investigates the second.
