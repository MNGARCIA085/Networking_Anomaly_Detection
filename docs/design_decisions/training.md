# Training Strategies

## 1. Overview

Training procedures are separated from model architectures.

The `models/` module defines model behavior and anomaly-scoring mechanisms, while `training/` implements the optimization procedures required by each model family.

Different anomaly detection approaches have different training requirements. In particular, reconstruction-based models generally follow a normal-only training strategy, whereas other approaches may use different assumptions.

## 2. Normal-Only Training

Reconstruction-based models, such as Autoencoders and Variational Autoencoders, are trained using normal observations.

The underlying assumption is that a model trained to represent normal behavior will produce higher reconstruction errors for observations that deviate from that behavior.

Consequently, the training dataset is filtered to retain only normal observations.

This strategy also applies to the training of sequence-based reconstruction models, subject to the configured window-labeling policy.

Other model families, such as Isolation Forest, may follow different training strategies.

## 3. Validation Data Usage

Validation data serves two distinct purposes, which require different subsets.

### 3.1 Training-Related Validation

For models using normal-only training, validation data is also filtered to retain normal observations when used by training callbacks.

Examples include:

- Early Stopping.
- Learning-rate scheduling based on validation loss.

This ensures that training-related decisions are based on the model's ability to represent normal observations rather than on a mixture of normal and anomalous samples.

### 3.2 Threshold Selection

The complete validation dataset, including normal and anomalous observations, is used for threshold selection when the configured strategy requires labeled validation data.

This allows threshold-selection strategies to optimize the desired detection objective using both classes.

Threshold selection is performed separately from model optimization.

### 3.3 Final Evaluation

The complete validation dataset can also be used for validation-stage evaluation and model analysis.

The test dataset remains reserved for final evaluation and must not influence model training, preprocessing fitting, hyperparameter optimization, or threshold selection.

## 4. Training and Evaluation Separation

The training process and anomaly decision process have different objectives.

Training attempts to learn the expected behavior of the selected model.

Threshold selection determines how model scores are translated into anomaly decisions.

For reconstruction-based models, these objectives imply different data requirements:

| Operation | Dataset |
|---|---|
| Model training | Normal training observations |
| Early Stopping | Normal validation observations |
| Learning-rate scheduling | Normal validation observations |
| Threshold selection | Complete validation dataset |
| Validation evaluation | Complete validation dataset |
| Final evaluation | Complete test dataset |

The exact training-data requirements depend on the selected model family and training strategy.

## 5. Design Rationale

The separation between training-related validation and threshold selection is intentional.

Using normal-only validation observations for reconstruction-based training callbacks ensures that optimization decisions focus on normal behavior.

Using the complete validation dataset for threshold selection allows the decision threshold to account for the distinction between normal and anomalous observations.

This keeps model optimization separate from the selection of an operational anomaly decision threshold.

## 6. Important Considerations

- Preprocessing transformations must be fitted using training data only.
- Anomalous observations must not enter normal-only training subsets.
- Training callbacks must receive the appropriate validation subset.
- Threshold selection must use validation data rather than test data.
- The test dataset must remain untouched until final evaluation.
- Different model families may require different training strategies; normal-only training should not be treated as a universal requirement for every detector.


## Note

Normal-only validation loss is not a measure of anomaly detection performance.It measures how well the model reconstructs normal observations. PR-AUC, recall, precision and related metrics on the complete validation set measure detection performance. Both are useful, but they answer different questions.