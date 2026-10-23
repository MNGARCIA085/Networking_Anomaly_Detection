import numpy as np
import pytest

from network_anomaly_detection.models.nn.ae.model import AEModel
from network_anomaly_detection.models.nn.transformer.model import TransformerAEModel


@pytest.fixture
def ae_config():
    return {
        "encoder_dims": [8, 4],
        "decoder_dims": [4, 8],
    }


@pytest.fixture
def transformer_config():
    return {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "num_encoder_layers": 1,
        "dropout": 0.0,
    }


@pytest.mark.parametrize(
    "input_shape",
    [
        (5, 1, 3),   # Single-timestep windows
        (5, 4, 3),   # Multi-timestep windows
    ],
)
def test_ae_adapt_input_flattens_each_sample_and_preserves_batch(
    ae_config, input_shape
):
    _, seq_len, n_features = input_shape

    model = AEModel(
        cfg=ae_config,
        input_shape=(seq_len, n_features),
    )

    X = np.arange(np.prod(input_shape)).reshape(input_shape)

    adapted = model.adapt_input(X)

    assert adapted.shape == (input_shape[0], seq_len * n_features)

    # Each flattened row must correspond to the same original sample.
    np.testing.assert_array_equal(
        adapted,
        X.reshape(input_shape[0], -1),
    )


def test_ae_adapt_input_preserves_batch_dimension_for_single_sample(ae_config):
    model = AEModel(
        cfg=ae_config,
        input_shape=(1, 3),
    )

    X = np.array([[[1.0, 2.0, 3.0]]])

    adapted = model.adapt_input(X)

    assert adapted.shape == (1, 3)
    np.testing.assert_array_equal(adapted, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize(
    "input_shape",
    [
        (5, 1, 3),
        (5, 4, 3),
    ],
)
def test_transformer_adapt_input_preserves_sequence_structure(
    transformer_config, input_shape
):
    _, seq_len, n_features = input_shape

    model = TransformerAEModel(
        cfg=transformer_config,
        input_shape=(seq_len, n_features),
    )

    X = np.arange(np.prod(input_shape)).reshape(input_shape)

    adapted = model.adapt_input(X)

    assert adapted.shape == input_shape
    np.testing.assert_array_equal(adapted, X)


def test_transformer_adapt_input_preserves_batch_dimension_for_single_sample(
    transformer_config,
):
    model = TransformerAEModel(
        cfg=transformer_config,
        input_shape=(4, 3),
    )

    X = np.ones((1, 4, 3))

    adapted = model.adapt_input(X)

    assert adapted.shape == (1, 4, 3)
    np.testing.assert_array_equal(adapted, X)


def test_transformer_rejects_invalid_configured_input_shape(
    transformer_config,
):
    with pytest.raises(ValueError, match="expects input_shape"):
        TransformerAEModel(
            cfg=transformer_config,
            input_shape=(3,),
        )