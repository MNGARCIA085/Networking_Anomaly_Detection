# Network Anomaly Detector

## 1. Data Flow

**Raw Data → Preprocessing → Model Construction → Training → Threshold Selection → Evaluation**

- **Preprocessing:** Pointwise transformations → windowing → temporal transformations. The order is intentional.
- **Model construction:** Builds the selected model using its configuration.
- **Training:** Fits the model using its corresponding trainer.
- **Threshold selection:** Determines the decision threshold using validation scores and labels.
- **Evaluation:** Measures detection performance on the complete validation set.

## 2. Source Structure

- `data/`: Data loading and window construction.
- `preprocessing/`: Shared preprocessing pipeline and reusable components.
- `models/`: Model implementations, organized by family (`nn/`, `ml/`) and composite models.
- `training/`: Base trainer, training infrastructure, losses, callbacks, and optimizers.
- `thresholding/`: Threshold selection strategies.
- `evaluation/`: Evaluation metrics and reporting.
- `experiment/`: Experiment orchestration and pipeline execution.
- `tuning/`: Hyperparameter optimization.
- `inference/`: Model loading, scoring, and inference utilities.
- `infra/`: Logging, experiment tracking, and candidate/model selection.

Model-specific folders can contain their own `model.py`, `trainer.py` or `prep.py`, and `tuning.py` or configuration schemas, depending on their needs.


models/ contains model-specific implementations, while training/ provides shared training infrastructure. A model can supply its own trainer when its training procedure requires specialized logic; otherwise, it can reuse the common training machinery.


## 3. Key Design Decisions

- **Temporal assumptions:** The dataset's implicit ordering is assumed to preserve temporal relationships. Windowing constructs sequences, enabling temporal models such as TransformerAE.

- **Modularity:** Base classes define shared interfaces and behavior. Specialized models, trainers, and preprocessing implementations inherit from them and override what they need. This allows each model family to be developed largely independently.

- **Preprocessing injection:** The main experiment pipeline receives an already fitted preprocessing instance. This allows preprocessing to be configured and fitted separately before being passed to the experiment, rather than requiring the pipeline to construct and fit it internally.

- **Trainer injection and resumption:** The experiment can also receive an existing trainer instance. This supports workflows such as resuming training from a previously saved trainer, provided its model and training state have been restored appropriately.(*)

- **Normal-only training:** AE, VAE, and TransformerAE are trained exclusively on normal observations, using available labels to filter the data. Their preprocessing is fitted using the filtered training data as well.

- **Validation:** Normal-only validation data is used for training-related purposes, such as early stopping. The complete validation set is used for threshold selection and final evaluation.

- **Reproducibility:** Experiments should record their effective configurations, preprocessing state, and evaluation results so that model behavior can be reproduced and compared.

(*) Passing an existing trainer supports resuming training only if the relevant state is restored. For neural networks, that generally includes model weights, optimizer state, scheduler state, and the training step or epoch.





-----
Models
 canonical X
                   │
                   ▼
            adapt_input()
                   │
                   ▼
             model-ready X
                   │
          ┌────────┴────────┐
          ▼                 ▼
       score()           predict()
          │                 │
          ▼                 ▼
 anomaly score        native IF threshold




---
Inference runner

 TRAINING
                       │
                       ▼
                  MLflow run
                       │
                       ▼
              ┌─────────────────┐
              │     loader      │
              │   MLflow →      │
              │ InferenceRunner │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ InferenceRunner │
              │                 │
              │ prep            │
              │ windowing       │
              │ adapt_input     │
              │ model           │
              │ threshold       │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
          FastAPI           Benchmark
                              │
                              ▼
                         inference_ms




---
Experimnet

start MLflow run
      │
      ├── status = running
      │
      ├── save config
      ├── fit Prep
      ├── save Prep artifact
      ├── transform train/val
      │
      ├── build model
      ├── build trainer
      │
      ├── train
      │      └── checkpoints → run artifacts/checkpoints/
      │
      ├── save final artifacts / metrics
      │
      └── status = finished
             OR
          status = failed

Note. There is no architectural reason to inject a pre-fitted threshold into Experiment.



---
Prep


BasePrep constructs and orchestrates the preprocessing stages.
Pointwise and temporal stages are implemented as PreprocessingPipeline objects, which contain an ordered sequence of preprocessing components. Windowing is a separate stage because it operates on both X and y.


Prep
  = transforms the data

Model
  = decides what representation it consumes

  ADAPT iput belongs to the model then




Raw X
 ↓
Pointwise prep
 ↓
Windowing
 ↓
Temporal prep
 ↓
X_processed

Then the model decides how to consume it:

X_processed
     │
     ├── IsoForest → flatten → (N, seq_len × features)
     │
     ├── AE        → flatten → (N, seq_len × features)
     │
     └── Transformer → keep 3D sequence
                       (N, seq_len, features)







Overall execution:

X
 │
 ▼
[ Pointwise Pipeline ]
 │
 ▼
[ Windowing ] ─────────── y
 │                         │
 ▼                         │
[ Temporal Pipeline ]      │
 │                         │
 ▼                         ▼
X_processed             y_windowed




------
Training

Prep produces the canonical representation; the model layer adapts its input shape; Trainer owns training; and tuning remains a separate first-class component.

Training
│
├── NN training
│   └── epoch-based + checkpoint callback
│
└── Classical training
    └── fit + persist fitted artifact




network_anomaly_detection/
│
├── training/
│   ├── base_trainer.py
│   ├── callbacks/
│   │   ├── checkpoint.py
│   │   ├── early_stopping.py
│   │   └── ...
│   ├── optimizers/
│   │   └── ...
│   └── ...
│
├── models/
│   ├── ae/
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── prep.py
│   │   └── tuning.py
│   │
│   ├── vae/
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── prep.py
│   │   └── tuning.py
│   │
│   └── isoforest/
│       ├── model.py
│       ├── trainer.py
│       ├── prep.py
│       └── tuning.py


training/ = reusable infrastructure

BaseTrainer
    ├── training loop
    ├── validation
    ├── optimizer creation
    ├── scheduler
    ├── callback handling
    ├── checkpoint integration
    └── logging hooks



models/<model>/trainer.py = model-specific behavior

This is also where I'd put differences such as:

custom loss
multiple model outputs
special training steps
unusual validation logic
model-specific optimizer requirements
special handling of latent variables


----------------
BaseTrainer
    │
    ├── IsoForestTrainer
    │
    └── NNTrainer
          │
          ├── AETrainer
          ├── VAETrainer
          └── TransformerAETrainer


BaseTrainer: minimal common contract, mainly fit(...).
NNTrainer: generic NN machinery — dataloaders, epochs, validation, optimizer, callbacks, etc.
AETrainer: only AE-specific behavior/configuration.
IsoForestTrainer: independent of NNTrainer, but still satisfies BaseTrainer.



data for aes....

| Data | Purpose | Samples |
|---|---|---|
| Training | Learn normal patterns | Normal only |
| Validation for early stopping | Monitor reconstruction loss | Normal only |
| Validation for model evaluation and threshold selection | Evaluate anomaly detection | Normal + anomalous |







----------
Tuning

1. Tuner orchestrates the search. It receives the base configuration, training/validation data, preprocessing/model execution dependencies, evaluator, and tuning settings.

2. The sampler builds each trial configuration. A model-specific sampler (e.g. AESampler, IsoForestSampler) receives the base config, deep-copies it, samples hyperparameters using Optuna's trial, and returns a complete configuration. It changes only tuned parameters, preserving fixed settings.

3. Each Optuna trial runs an experiment. The objective uses the sampled config to construct preprocessing, model, and trainer, then trains and evaluates on validation data. It returns the configured objective metric (e.g. PR-AUC) to Optuna.

4. Optuna manages the search. It proposes trial parameters, receives objective values, and selects subsequent configurations according to the study's optimization strategy.

5. The best trial is recorded. Persist the best parameters, objective value, and configuration; optionally rerun the best configuration as a normal tracked Experiment.

Information the Tuner needs

- Base resolved configuration.

- X_train, y_train, X_val, y_val.

- Sampler for the selected model.

- Objective metric and direction (maximize or minimize).

- Number of trials and study settings.

- The execution dependencies needed to build and run each trial.

Key distinction: the sampler decides what configuration to try; the Experiment/Pipeline executes it; the evaluator measures its performance; Optuna decides what to try next