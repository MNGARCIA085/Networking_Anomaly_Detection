# Project Structure

```text
├── config/
│   ├── data/
│   ├── models/
│   ├── model_type/
│   ├── prep/
│   ├── thresholding/
│   ├── training/
│   └── tuning/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
├── src/network_anomaly_detection/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
│   ├── infra/
│   ├── models/
│   ├── preprocessing/
│   ├── selection/
│   ├── thresholding/
│   ├── training/
│   ├── tuning/
│   └── utils/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── mock_model_store/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Configuration

`config/` contains Hydra configuration groups:

- `data/`: dataset and splitting settings.
- `model_type/`: model family selection.
- `models/`: model-specific parameters.
- `prep/`: preprocessing configuration.
- `training/`: training and optimization settings.
- `thresholding/`: threshold-selection strategies.
- `tuning/`: hyperparameter optimization settings.

## Source package

`src/network_anomaly_detection/` contains the main implementation.

| Module | Responsibility |
|---|---|
| `data/` | Dataset loading, splitting, and preparation. |
| `preprocessing/` | Pointwise and temporal transformations. |
| `models/` | Model implementations and common interfaces. |
| `training/` | Training strategies and trainers. |
| `thresholding/` | Operating threshold selection. |
| `evaluation/` | Model evaluation and metrics. |
| `tuning/` | Hyperparameter optimization. |
| `selection/` | Candidate comparison and selection. |
| `experiment/` | Experiment orchestration and lifecycle. |
| `inference/` | Model loading and inference execution. |
| `infra/` | Experiment-tracking integrations, including MLflow. |
| `utils/` | Shared utilities. |

## Scripts

`scripts/` contains executable entry points for common workflows, including:

- Dataset preparation.
- Model training and resuming.
- Hyperparameter tuning.
- Candidate benchmarking and selection.
- Final evaluation.
- Model export and inference.

## Data and artifacts

- `data/raw/`: original datasets.
- `data/processed/`: prepared datasets, splits, and manifests.
- `mock_model_store/`: example exported model artifacts and manifests.
- MLflow stores experiment metadata, metrics, and artifacts.

## Testing

`tests/` separates:

- `unit/`: individual component behavior.
- `integration/`: interactions between components and workflows.

## Project configuration

- `pyproject.toml`: package metadata, dependencies, and development configuration.
- `Dockerfile`: containerized execution environment.
- `README.md`: project introduction and setup instructions.