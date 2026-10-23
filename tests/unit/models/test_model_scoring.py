import numpy as np
import pytest
import torch

from network_anomaly_detection.models.nn.ae.model import AEModel
from network_anomaly_detection.models.nn.transformer.model import TransformerAEModel


@pytest.fixture
def ae_model():
    return AEModel(
        cfg={
            "encoder_dims": [8, 4],
            "decoder_dims": [4, 8],
        },
        input_shape=(3, 2),
    )


@pytest.fixture
def transformer_model():
    return TransformerAEModel(
        cfg={
            "d_model": 8,
            "nhead": 2,
            "dim_feedforward": 16,
            "num_encoder_layers": 1,
            "dropout": 0.0,
        },
        input_shape=(3, 2),
    )


def test_ae_score_returns_one_finite_score_per_sample(ae_model):
    X = np.random.default_rng(42).normal(size=(5, 3, 2))
    X_adapted = ae_model.adapt_input(X)

    scores = ae_model.score(X_adapted)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_ae_score_matches_mean_squared_reconstruction_error(ae_model):
    X = np.random.default_rng(42).normal(size=(4, 3, 2))
    X_adapted = ae_model.adapt_input(X)
    X_tensor = torch.as_tensor(X_adapted, dtype=torch.float32)

    scores = ae_model.score(X_adapted)

    with torch.no_grad():
        reconstruction = ae_model.model(X_tensor)
        expected = torch.mean(
            (X_tensor - reconstruction) ** 2,
            dim=1,
        ).cpu().numpy()

    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-7)


def test_transformer_score_returns_one_finite_score_per_window(
    transformer_model,
):
    X = np.random.default_rng(42).normal(size=(5, 3, 2))

    scores = transformer_model.score(X)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_transformer_score_matches_mean_squared_reconstruction_error(
    transformer_model,
):
    X = np.random.default_rng(42).normal(size=(4, 3, 2))
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    scores = transformer_model.score(X)

    with torch.no_grad():
        reconstruction = transformer_model.model(X_tensor)
        expected = torch.mean(
            (X_tensor - reconstruction) ** 2,
            dim=(1, 2),
        ).cpu().numpy()

    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "model_type",
    ["ae", "transformer"],
)
def test_predict_applies_threshold_and_returns_binary_integer_labels(
    model_type,
    ae_model,
    transformer_model,
    monkeypatch,
):
    model = ae_model if model_type == "ae" else transformer_model

    # Includes a score exactly equal to the threshold.
    scores = np.array([0.1, 0.7, 0.4])

    monkeypatch.setattr(
        model,
        "score",
        lambda X: scores,
    )

    X = np.zeros((3, 3, 2))
    if model_type == "ae":
        X = model.adapt_input(X)

    predictions = model.predict(X, threshold=0.4)

    np.testing.assert_array_equal(predictions, [0, 1, 1])
    assert predictions.dtype.kind in "iu"