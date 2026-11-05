# Testing Strategy

## Objective

Testing verifies individual component behavior, architectural contracts, and integration across the experiment workflow.

## Unit tests

Unit tests focus on individual components in isolation.

Relevant areas include:

- Model input adaptation and anomaly scoring.
- Preprocessing transformations and label alignment.
- Trainer behavior and training-data selection.
- Threshold selection.
- Evaluation metrics.
- Configuration validation.
- Registry and factory construction.

Mock components can be used to test orchestration without requiring actual model training.

## Integration tests

Integration tests verify interactions between components and workflows.

Examples include:

- Configuration composition and validation.
- Preprocessing and model input adaptation.
- Experiment pipeline execution.
- Training, scoring, and threshold selection.
- Model export and subsequent loading for inference.

## Architectural contracts

Tests should verify shared interface expectations, including:

- Higher scores consistently indicate greater anomaly likelihood.
- Preprocessing preserves feature and label alignment.
- Model input adaptation produces compatible representations.
- Exported artifacts can be loaded and used for inference.

## Design rationale

Separating unit and integration tests helps identify whether failures originate within individual components or from their interactions.

Testing shared contracts also makes it safer to introduce new model families, preprocessing strategies, and trainers without breaking existing workflows.