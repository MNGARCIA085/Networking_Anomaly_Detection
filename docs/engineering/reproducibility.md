# Reproducibility

## Objective

The project aims to make experiments reproducible by tracking the information required to identify and reconstruct an experiment's execution.

## Main mechanisms

### Configuration tracking

- Hydra provides structured experiment configuration.
- Effective configurations are logged to MLflow.
- Model, preprocessing, training, and tuning settings are recorded.

### Data traceability

- Dataset preparation produces manifests describing processed datasets and partitions.
- Train, validation, and test splits are preserved.
- Experiments should reference the specific dataset version used.

### Experiment tracking

MLflow records parameters, metrics, artifacts, and experiment information.

This provides traceability between configurations, results, and exported models.

### Dependency management

`pyproject.toml` defines package metadata and dependencies.

Docker provides a controlled execution environment, reducing differences between host environments.

### Randomness

Where applicable, random seeds should be controlled for:

- Dataset splitting.
- Model initialization.
- Data shuffling.
- Hyperparameter optimization.

Seeds and relevant execution settings should be recorded with the experiment.

## Reproducibility considerations

Reproducibility is not necessarily equivalent to bitwise-identical results.

Differences may arise from hardware, numerical precision, library versions, parallel execution, and nondeterministic operations.

The objective is to preserve enough information to understand, trace, and reliably repeat experiments under equivalent conditions.