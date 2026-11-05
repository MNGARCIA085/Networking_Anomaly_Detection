# Tuning and Candidate Selection

## Hyperparameter optimization

The framework uses Optuna to explore hyperparameter configurations and optimize a selected objective metric.

The optimization process:

1. Samples configurations from the defined search space.
2. Executes an experiment for each trial.
3. Evaluates each trial using the configured metric.
4. Records trial results and identifies the best configuration.

PR-AUC is used as the primary optimization objective, with maximization as the default direction.

Optuna trials are tracked as nested MLflow runs under a parent tuning experiment.


## Sample

Base config is merged with the sample parameters.

```
Base configuration ─────────────┐
                               │
Tuning configuration ──> Sampler
                               │
                               ▼
                       Sampled parameters
                               │
                               ▼
                    Merge with base config
                               │
                               ▼
                    Effective trial config
                               │
                               ▼
                           Experiment
```


## Candidate selection

Hyperparameter optimization and candidate selection are separate processes.

- **Tuning** identifies promising configurations within an experiment.
- **Candidate selection** compares trained model candidates and chooses one according to deployment-oriented criteria.

For example, candidate selection may:

1. Filter models that meet a minimum PR-AUC requirement.
2. Select the candidate with the lowest inference time among those that qualify.

Candidate metadata and results are maintained through a candidate registry backed by SQLite.

## Design rationale

Separating these processes allows model selection to consider criteria beyond predictive performance, such as inference cost.

It also avoids coupling the optimization objective to the final deployment requirements.