# Configuration Architecture

## Purpose

The project uses configuration-driven experiments to make model development flexible, reproducible, and easy to extend.

Hydra handles configuration composition, Pydantic validates configuration structure, and application-level validation checks compatibility between components.

## Configuration organization

Configuration is divided into independent groups:

- `model_type/`: selects the model family (AE, VAE, Isolation Forest, etc.).
- `models/`: defines model-specific hyperparameters and architecture.
- `training/`: defines optimizer, scheduler, and training settings.
- `prep/`: defines preprocessing operations and parameters.
- `tuning/`: defines the hyperparameter search space and optimization settings.
- `thresholding/`: defines the threshold-selection strategy.
- `data/`: defines dataset and splitting settings.

This separation avoids coupling unrelated configuration concerns and allows components to be combined independently.

## Validation responsibilities

Configuration validation is divided into three levels:

1. **Hydra:** composes configuration groups and resolves selected configurations.
2. **Pydantic:** validates configuration structure, types, and field constraints.
3. **ConfigValidator:** checks application-level compatibility between components, such as whether a model supports a particular input representation.

Hydra composition does not replace application-level validation.

## Component construction

Registries map configuration identifiers to their corresponding implementations:

- `MODEL_REGISTRY`
- `PREP_REGISTRY`
- `TRAINER_REGISTRY`
- `TUNING_REGISTRY`

Factory functions use these registries to instantiate components without requiring the experiment pipeline to know their concrete implementations.

## Design rationale

This architecture provides:

- Independent evolution of models, preprocessing, training, and tuning.
- Reusable configurations and easier experimentation.
- Explicit compatibility checks.
- Reduced coupling between configuration, orchestration, and implementation.
- A straightforward path to adding new components.

The configuration describes what should be executed; application code determines how components are constructed and coordinated.