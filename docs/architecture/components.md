# Core Components and Interfaces

## Experiment orchestration

The experiment layer coordinates the execution lifecycle.

- **`ExperimentPipeline`**: coordinates the configured components and experiment workflow.
- **`Experiment`**: represents an experiment execution and its associated lifecycle.

The orchestration layer delegates specialized responsibilities to individual components rather than implementing model-specific behavior.

## Core interfaces

### BaseModel

Defines the common model contract:

- `adapt_input()`: adapts preprocessed data to the model's expected representation.
- `score()`: computes anomaly scores.
- `predict()`: converts scores into binary predictions using a threshold.
- `get_config()`: exposes model configuration.

### BasePrep

Defines the preprocessing workflow, including pointwise transformations, windowing, temporal transformations, and training-data selection.

### Trainers

Training implementations are independent of model implementations. Trainers handle optimization and training-specific behavior through a common construction and execution workflow.

## Registries and factories

The framework uses registries to map component identifiers to implementations:

- `MODEL_REGISTRY`
- `PREP_REGISTRY`
- `TRAINER_REGISTRY`
- `TUNING_REGISTRY`

Factory functions, such as `build_model()` and `build_trainer()`, centralize component construction.

This allows new implementations to be registered without modifying experiment orchestration.

## Supporting components

| Component | Responsibility |
|---|---|
| Thresholding | Determines the operating threshold from validation scores. |
| Evaluation | Calculates performance metrics. |
| Tuning | Searches hyperparameter configurations. |
| Selection | Chooses candidates according to selection criteria. |
| Tracking | Logs experiment information and artifacts through MLflow. |
| InferenceRunner | Coordinates preprocessing, model adaptation, scoring, and prediction. |

## Component interaction

```text
             Hydra Configuration
                     |
                     v
              Config Validation
                     |
                     v
             ExperimentPipeline
                     |
          +----------+----------+
          |          |          |
          v          v          v
         Prep      Model     Trainer
          |          |          |
          +----------+----------+
                     |
                     v
                Evaluation
                     |
          +----------+----------+
          |                     |
          v                     v
      Thresholding            Tuning
          |                     |
          +----------+----------+
                     |
                     v
              Candidate Selection
                     |
                     v
                MLflow / Export
                     |
                     v
                InferenceRunner
```

*Conceptual overview: this diagram illustrates responsibilities and workflow relationships, not necessarily direct dependencies or execution order between every component.*

## Design principles

- Models and trainers evolve independently.
- Preprocessing is independent of model architecture.
- Registries decouple component selection from construction.
- Experiment orchestration coordinates rather than implements specialized logic.
- Training and inference reuse compatible model and preprocessing interfaces.