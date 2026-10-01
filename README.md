# Server Anomaly Detection

This project implements a machine-learning pipeline for detecting anomalies in sequential server measurements.

The system is designed to identify observations that deviate from the normal behavior learned from server monitoring data. It includes data preprocessing, temporal windowing, multiple anomaly-detection model families, experiment tracking, hyperparameter tuning, evaluation, model persistence, and inference benchmarking.

The project is structured as a reusable ML template, with an emphasis on reproducibility, extensibility, and separation between experimentation and production-oriented components.

## Project Scope

The project currently explores several approaches to server anomaly detection, including:

* Statistical and classical anomaly-detection baselines
* Isolation Forest
* Autoencoders
* Variational Autoencoders
* Transformer-based autoencoders
* Configurable preprocessing and temporal transformations
* Multiple thresholding strategies
* Experiment tracking with MLflow
* Hyperparameter optimization with Optuna
* Model evaluation and inference benchmarking

The final model and configuration are selected using the validation data. The test set is reserved for final evaluation after model and threshold selection.

Detailed architectural decisions, preprocessing choices, windowing strategy, scoring, thresholding, and experiment methodology are documented separately.

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install the project

Install the package and its dependencies:

```bash
pip install -e .
```

**Note** CPU Torch
```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```  

For training and experimentation:

```bash
pip install -e ".[train]"
```



The exact optional dependency groups are defined in `pyproject.toml`.

## Usage

The project provides command-line entry points for the main ML workflows.

### Training

Run a training experiment:

```bash
python -m scripts.train
```

Hydra configuration can be overridden from the command line. For example:

```bash
python -m scripts.train model_type=ae
```

Multirun

```bash
python -m scripts.train -m model_type=ae,transformer
```

Parallel execution

```bash
python scripts/train.py -m \
    model=ae,transformer \
    hydra.launcher.n_jobs=2
```


### Hyperparameter Tuning

Run an Optuna-based tuning experiment:

```bash
python -m scripts.tune
python -m scripts.tune model_type=ae
python -m scripts.tune -m model_type=ae,isoforest
```



### Inference

Run inference using a trained model:

```bash
python -m scripts.inference
```

### Evaluation

Evaluate model predictions and anomaly scores:

```bash
python -m scripts.final_evaluation
```

The available configuration options and experiment parameters are defined under `config/`.

## Tests

Run the test suite with:

```bash
pytest
```

The test suite covers core components as well as integration points such as preprocessing, model entries, experiments, inference, thresholding, and evaluation.

## Docker

The project provides separate Docker targets for training and inference.

### Build the images

Build the training image:

```bash
docker build --target train -t anomaly-detection:train-cpu .
```

Build the inference image:

```bash
docker build --target inference -t anomaly-detection:inference-cpu .
```

### Run training

The training container mounts the project data directory as read-only:

```bash
docker build \
    --target train \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
    -t anomaly-detection:train-cpu .

docker run -it --rm \
    -v "$(pwd)/data:/app/data:ro" \
    anomaly-detection:train-cpu
```



### Run inference

Inference can use a persisted model from the local model store:

```bash
docker build \
    --target inference \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
    -t anomaly-detection:inference-cpu .
    
    
docker run --rm -it \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/mock_model_store:/app/mock_model_store:ro" \
    anomaly-detection:inference-cpu

```

The mounted directories allow the container to access the dataset and persisted model artifacts without copying them into the image.

The Docker images keep training dependencies separate from the smaller inference runtime.

## MLflow

The project uses MLflow for experiment tracking.

Start the local MLflow server with:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host localhost \
    --port 5000
```

The default local tracking database is:

```text
mlflow.db
```

Experiment artifacts are stored under:

```text
mlruns/
```

To reset the local MLflow database:

```bash
rm mlflow.db
```

To remove the stored artifacts:

```bash
rm -rf mlruns/
```

## Documentation

The README intentionally provides a high-level overview of the project.

More detailed documentation covers:

* Project architecture
* Data preprocessing
* Sequential windowing and temporal representations
* Model architecture and model interfaces
* Experiment design
* Hyperparameter tuning
* Anomaly scoring
* Threshold selection
* Evaluation methodology
* Model selection
* Inference benchmarking
* Deployment and monitoring

See the project documentation for the detailed rationale behind these decisions.

## Dataset

The initial dataset used is provided as part of the DeepLearning.AI & Stanford Online Machine Learning Specialization, in the anomaly detection course material.

DeepLearning.AI & Stanford Online. Machine Learning Specialization.
https://www.deeplearning.ai/specializations/machine-learning

The dataset is used here for experimentation and methodology development; it is not presented as a real-world production dataset.

