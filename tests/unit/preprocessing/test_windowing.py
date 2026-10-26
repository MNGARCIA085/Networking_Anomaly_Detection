import numpy as np
import pytest

from network_anomaly_detection.data.windowing import Windowing


def test_transform_creates_correct_windows_and_stride():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400],
        [50, 500],
    ])

    windowing = Windowing(seq_len=3, stride=2)

    result = windowing.transform(X)

    expected = np.array([
        [
            [10, 100],
            [20, 200],
            [30, 300],
        ],
        [
            [30, 300],
            [40, 400],
            [50, 500],
        ],
    ])

    np.testing.assert_array_equal(result, expected)


def test_default_strategy_is_last():
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 0, 1, 0])

    windowing = Windowing(seq_len=2, stride=1)

    X_windows, y_windows = windowing.transform(X, y)

    expected_X = np.array([
        [[0, 1], [2, 3]],
        [[2, 3], [4, 5]],
        [[4, 5], [6, 7]],
    ])

    np.testing.assert_array_equal(X_windows, expected_X)
    np.testing.assert_array_equal(y_windows, [0, 1, 0])


def test_last_strategy_uses_final_observation_label():
    X = np.arange(8).reshape(4, 2)
    y = np.array([1, 0, 0, 1])

    windowing = Windowing(seq_len=3, strategy="last")

    _, y_windows = windowing.transform(X, y)

    np.testing.assert_array_equal(y_windows, [0, 1])


def test_any_strategy_marks_window_positive_if_any_point_is_anomalous():
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 0, 1, 0])

    windowing = Windowing(seq_len=2, strategy="any")

    _, y_windows = windowing.transform(X, y)

    np.testing.assert_array_equal(y_windows, [0, 1, 1])


@pytest.mark.parametrize(
    "y, expected",
    [
        ([0, 0, 0], 0),  # No anomalies
        ([1, 1, 0], 1),  # Majority anomalous
        ([1, 0, 0], 0),  # Minority anomalous
        ([1, 0, 1, 0], 0),  # Tie: negative
    ],
)
def test_majority_strategy(y, expected):
    X = np.arange(len(y)).reshape(-1, 1)

    windowing = Windowing(
        seq_len=len(y),
        strategy="majority",
    )

    _, y_windows = windowing.transform(X, np.array(y))

    np.testing.assert_array_equal(y_windows, [expected])


def test_seq_len_one_preserves_individual_samples():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ])

    windowing = Windowing(seq_len=1, stride=1)

    result = windowing.transform(X)

    assert result.shape == (3, 1, 2)
    np.testing.assert_array_equal(result[:, 0, :], X)


@pytest.mark.parametrize(
    "seq_len,stride",
    [
        (0, 1),
        (-1, 1),
        (2, 0),
        (2, -1),
    ],
)
def test_invalid_window_parameters_raise_value_error(seq_len, stride):
    with pytest.raises(ValueError):
        Windowing(seq_len=seq_len, stride=stride)


def test_invalid_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unknown strategy"):
        Windowing(seq_len=3, strategy="invalid")


def test_transform_rejects_input_with_wrong_dimensions():
    windowing = Windowing(seq_len=2)

    with pytest.raises(ValueError, match="Expected X"):
        windowing.transform(np.array([1, 2, 3]))


def test_transform_rejects_mismatched_feature_and_label_counts():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])

    windowing = Windowing(seq_len=2)

    with pytest.raises(ValueError, match="same number of samples"):
        windowing.transform(X, y)


def test_transform_rejects_insufficient_samples():
    X = np.array([[1], [2]])

    windowing = Windowing(seq_len=3)

    with pytest.raises(ValueError, match="Not enough samples"):
        windowing.transform(X)


def test_transform_rejects_multidimensional_labels():
    X = np.array([[1], [2], [3]])
    y = np.array([[0], [1], [0]])

    windowing = Windowing(seq_len=2)

    with pytest.raises(ValueError, match="Expected y"):
        windowing.transform(X, y)

