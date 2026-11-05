# Model Architecture

## Supported model families

The framework supports different anomaly detection approaches:

- **Autoencoder (AE):** learns to reconstruct normal observations; reconstruction error is used as the anomaly score.
- **Variational Autoencoder (VAE):** probabilistic reconstruction-based approach.
- **Transformer AE:** uses self-attention to model relationships within input sequences.
- **Isolation Forest:** detects anomalies through isolation-based partitioning.
- **AE + Isolation Forest:** combines learned representations from an AE with Isolation Forest scoring.

## Common interface

All models inherit from `BaseModel`, which defines a common interface:

```python
class BaseModel(ABC):

    @abstractmethod
    def adapt_input(self, X: np.ndarray) -> np.ndarray:
        """Adapt canonical preprocessed data to model input."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score for each input sample."""

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.score(X) >= threshold

    @abstractmethod
    def get_config(self):
        pass
```

The interface establishes three important contracts:

- **Input adaptation:** each model handles the conversion from canonical preprocessed data to its required representation.
- **Anomaly scoring:** every model returns an anomaly score, with higher values consistently indicating greater anomaly likelihood.
- **Prediction:** the base class applies a common thresholding rule to model scores.

`get_config()` exposes model configuration for experiment tracking and reproducibility.

## Input representations

Models can use different representations of the same preprocessed data:

- Feed-forward models can flatten windows from `(N, W, F)` into `(N, W × F)`.
- Sequence models can retain `(N, W, F)`.
- Pointwise inputs can use `(N, F)`.

This keeps preprocessing independent of model-specific input requirements.




## Example: AE + Isolation Forest

## Core idea

The Autoencoder learns a compressed representation of normal network traffic. Isolation Forest then detects anomalies within this latent representation space.

This combines representation learning with isolation-based anomaly detection.

### Training workflow

The hybrid model is trained in two stages:

1. Train the Autoencoder using normal training observations.
2. Encode normal training observations into latent vectors and fit Isolation Forest on those representations.

```text
       Normal training data
                |
                v
          Preprocessing
                |
                v
         Model adaptation
                |
                v
       AEIsolationForestTrainer
                |
          +-----+------+
          |            |
          v            |
       AETrainer       |
          |            |
          v            |
      Trained AE       |
          |            |
          v            |
       Encoder         |
          |            |
          v            |
    Normal latent      |
       vectors         |
          |            |
          v            |
    Isolation Forest <-+
          |
          v
      Trained hybrid model
```

Validation data is transformed and encoded using the trained Autoencoder. Isolation Forest produces anomaly scores, which are subsequently used for threshold selection and evaluation.

### Inference workflow

```text
       Incoming data
             |
             v
        Preprocessing
             |
             v
       Trained AE
         Encoder
             |
             v
       Latent vectors
             |
             v
      Trained Isolation
          Forest
             |
             v
       Anomaly scores
             |
             v
        Thresholding
             |
             v
      Normal / Anomaly
```

### Important considerations

- Both training stages use normal observations.
- Isolation Forest operates on latent representations, not directly on the original feature space.
- The trained Autoencoder encoder and Isolation Forest must be exported together.
- The final anomaly score comes from Isolation Forest, with higher values indicating greater anomaly likelihood.




## Design rationale

The common interface decouples experiment orchestration from model implementation. New model families can be introduced without changing the overall experiment workflow, provided they satisfy the interface and compatibility requirements.


