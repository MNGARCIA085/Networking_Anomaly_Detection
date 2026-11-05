# Experiment Tracking

## MLflow

The framework uses MLflow to track experiments, training runs, and hyperparameter optimization.

Each run records relevant information to support comparison, reproducibility, and model selection.

## Logged information

Depending on the experiment, tracked information includes:

- **Parameters:** effective model, preprocessing, training, and tuning configurations.
- **Metrics:** training metrics, validation PR-AUC, threshold-dependent metrics, and inference time.
- **Artifacts:** trained models, preprocessing objects, threshold information, and relevant metadata.
- **Tags:** identifiers such as model family and experiment information.

Effective configurations are logged to capture the actual settings used, including relevant defaults, rather than relying only on the original YAML files.

The selected decision threshold is logged separately from the model configuration.

## Tuning integration

Optuna trials are tracked as nested runs under a parent tuning run. This preserves the relationship between the overall optimization process and its individual trials.

## Candidate registry

Candidate selection is maintained separately from MLflow tracking. A SQLite-backed registry stores candidate information and supports selecting models according to defined criteria.

MLflow provides experiment history; the candidate registry supports selection.

## Design rationale

This separation provides:

- Traceability of experimental decisions.
- Comparison across model families and configurations.
- Access to trained artifacts.
- Reproducibility through effective configuration snapshots.
- Separation between experiment tracking and final candidate selection.