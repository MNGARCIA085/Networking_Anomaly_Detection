# Architecture Overview

## 1. Purpose

The Network Anomaly Detection project provides a modular framework for developing, training, evaluating, and comparing machine learning approaches for network traffic anomaly detection.

Rather than implementing a single detection model, the project supports experimentation with different model families, preprocessing strategies, training procedures, and anomaly detection methodologies.

The architecture aims to facilitate experimentation while maintaining separation of concerns, extensibility, and reproducibility.

The system covers the following activities:

- Dataset preparation and splitting.
- Data preprocessing and transformation.
- Model construction and training.
- Hyperparameter optimization.
- Anomaly scoring and threshold selection.
- Candidate model selection.
- Performance evaluation.
- Experiment tracking and model artifact export.
- Model loading and inference.

## 2. Architectural Principles

### 2.1 Modularity

Models, preprocessing components, trainers, thresholding strategies, and tuning implementations are independent components.

This allows individual components to be modified or replaced without requiring changes throughout the system.

### 2.2 Separation of Concerns

Different responsibilities are handled by dedicated modules.

For example:

- Model implementations are separated from training procedures.
- Dataset preparation is separated from preprocessing transformations.
- Hyperparameter optimization is separated from final candidate selection.
- Anomaly scoring is separated from threshold selection.
- Experiment orchestration is separated from individual component implementations.

### 2.3 Extensibility

The architecture uses registries and common interfaces to facilitate the integration of new implementations.

Adding a new model, preprocessing strategy, trainer, or tuning implementation should primarily involve implementing the corresponding component and registering it, rather than modifying the experiment orchestration logic.

### 2.4 Configuration-Driven Experiments

Hydra is used to compose experiment configurations.

Configuration groups separate concerns such as model family, model parameters, training, preprocessing, hyperparameter tuning, and thresholding.

This allows different experimental combinations to be evaluated without modifying the underlying implementation.

Configuration composition is handled by Hydra, while application-level validation handles structural correctness and compatibility constraints between components.

### 2.5 Reproducibility

Experiments are configured explicitly, and their effective configurations, metrics, and relevant artifacts are tracked.

MLflow is used for experiment tracking, while Docker provides a reproducible execution environment.

Dataset preparation also produces metadata that helps identify the data used in experiments.

## 3. High-Level Architecture

The application is implemented as a Python package under `src/network_anomaly_detection/`.

Its main modules are organized around the following responsibilities.

| Module | Responsibility |
|---|---|
| `data/` | Dataset preparation, splitting, and generation of training, validation, and test sets. |
| `preprocessing/` | Feature-level and temporal transformations, including imputation, scaling, windowing, and temporal features. |
| `models/` | Model implementations and their anomaly-scoring mechanisms. |
| `training/` | Training procedures and optimization logic. |
| `experiment/` | Experiment orchestration and lifecycle management. |
| `tuning/` | Hyperparameter optimization and search-space management. |
| `thresholding/` | Strategies for selecting anomaly decision thresholds. |
| `selection/` | Selection of the best candidate model according to configurable criteria. |
| `evaluation/` | Performance evaluation and metric computation. |
| `inference/` | Model loading and inference functionality. |
| `infra/` | Infrastructure integrations, including MLflow logging. |
| `utils/` | Shared utilities. |

### 3.1 Component Interaction

The following diagram illustrates the conceptual experiment workflow.

```text
                  Configuration
                      |
                      v
             Configuration Validation
                      |
                      v
              Experiment Pipeline
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     Preprocessing  Model      Trainer
          |           |           |
          +-----------+-----------+
                      |
                      v
                Anomaly Scoring
                      |
                      v
              Threshold Selection
                      |
                      v
                  Evaluation
                      |
                      v
               Experiment Tracking
                    (MLflow)
```

This diagram represents the main responsibilities and their conceptual relationships. It does not imply that every component is implemented as a direct dependency of the experiment pipeline.

## 4. Configuration Architecture

The `config/` directory contains the Hydra configuration hierarchy.

Configuration is organized into independent groups, each responsible for a different aspect of an experiment.

| Configuration group | Responsibility | Examples |
|---|---|---|
| `model_type/` | Selects the model family or detection approach. | AE, VAE, Isolation Forest, Transformer, AE + Isolation Forest. |
| `models/` | Defines model-specific architectural and algorithmic parameters. | Encoder dimensions, latent dimension, number of estimators. |
| `training/` | Defines training behavior and optimization parameters. | Optimizer, learning rate, scheduler, epochs. |
| `prep/` | Defines preprocessing transformations. | Imputation, scaling, windowing, temporal features. |
| `tuning/` | Defines hyperparameter optimization settings and search spaces. | Parameter ranges, number of trials, optimization metric. |
| `thresholding/` | Defines anomaly threshold selection behavior. | Constrained F1, no threshold. |
| `data/` | Defines dataset-related preparation settings. | Dataset paths and splitting parameters. |

The distinction between `model_type/` and `models/` is intentional:

- `model_type/` determines which model family or detection approach is used.
- `models/` specifies the parameters of the selected implementation.

This separates the choice of detection approach from its specific parameterization.

The root `config.yaml` composes the selected configuration groups.

Hydra handles configuration composition, while application-level validation handles compatibility constraints between components.

## 5. Data Preparation and Preprocessing

Dataset preparation and preprocessing are separate responsibilities.

The `data/` module handles the preparation of datasets, including splitting the available observations into training, validation, and test sets.

The `preprocessing/` module handles transformations applied to those datasets.

The general distinction is:

- **Data preparation:** Determines which observations belong to each dataset partition.
- **Preprocessing:** Transforms the observations into representations suitable for the selected model.

Preprocessing can involve pointwise transformations, such as imputation and scaling, as well as temporal transformations, such as window construction and temporal feature extraction.

The specific preprocessing sequence and its fitting behavior depend on the selected configuration and model requirements.



### 5.1. Preprocessing injection

The experiment pipeline receives an already fitted preprocessing instance through dependency injection.

This separates preprocessing construction and fitting from experiment orchestration, allowing preprocessing to be configured, fitted, and reused independently.

The pipeline is therefore responsible for coordinating the experiment, not necessarily constructing or fitting every component internally.



## 6. Experiment Lifecycle

An experiment follows a general lifecycle:

1. **Configuration:** Select the model family, model parameters, preprocessing strategy, training procedure, and other experiment parameters.
2. **Validation:** Validate configuration structure and compatibility between selected components.
3. **Data preparation:** Prepare and split the dataset into training, validation, and test sets.
4. **Preprocessing:** Fit and apply the configured transformations according to the selected methodology.
5. **Model construction:** Instantiate the selected model implementation.
6. **Training:** Execute the corresponding training procedure.
7. **Scoring:** Generate anomaly scores for the relevant observations.
8. **Threshold selection:** Determine a decision threshold when required by the selected strategy.
9. **Evaluation:** Evaluate detector performance using the selected metrics.
10. **Tracking:** Record configurations, metrics, and relevant artifacts.

Hyperparameter optimization can execute multiple experiments using different configurations to identify promising candidates.

## 7. Hyperparameter Optimization and Candidate Selection

Hyperparameter optimization and candidate selection are distinct stages.

### Hyperparameter Optimization

The `tuning/` module handles the search for promising configurations.

It evaluates different combinations of hyperparameters according to a configured objective, such as maximizing PR-AUC.

Optuna is used to manage the optimization process.

### Candidate Selection

The `selection/` module handles the selection of a final candidate among previously evaluated experiments.

Selection can involve multiple criteria rather than simply maximizing a single metric.

For example:

1. Consider only candidates whose PR-AUC exceeds a specified minimum.
2. Among the eligible candidates, select the one with the lowest inference time.

This allows model selection to account for operational requirements in addition to predictive performance.

## 8. Experiment Tracking and Infrastructure

The `infra/` module contains infrastructure integrations, including MLflow logging.

MLflow provides experiment tracking for configurations, metrics, and artifacts.

Docker provides a controlled execution environment to improve reproducibility across different machines.

The project also supports exporting selected models into a model store containing the artifacts required for subsequent inference.

## 9. Execution Layer

The `scripts/` directory provides entry points for the main project operations.

These scripts expose workflows such as:

- Dataset preparation.
- Experiment training.
- Hyperparameter optimization.
- Cross-validation.
- Candidate benchmarking.
- Final model evaluation.
- Model export.
- Model inference.
- Experiment inspection and resumption.

The scripts provide an execution interface to the application package. Core functionality should remain within the package rather than being duplicated across scripts.

## 10. Testing Strategy

Tests are organized into two main categories:

- **Unit tests:** Validate individual components and their behavior in isolation.
- **Integration tests:** Validate interactions between components and execution workflows.

This separation helps identify whether failures originate in individual implementations or in interactions between different parts of the system.

## 11. Scope and Boundaries

The primary responsibility of this project is developing and evaluating network anomaly detection models.

Model serving through an external API and automated incident investigation are considered separate concerns.

The inference module provides functionality for loading and executing exported models, while an external API or incident investigation system can consume the resulting predictions independently.