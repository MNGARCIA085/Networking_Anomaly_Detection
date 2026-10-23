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


def test_transform_with_labels_marks_window_anomalous_if_any_point_is_anomalous():
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
    np.testing.assert_array_equal(y_windows, [0, 1, 1])


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