import numpy as np

from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.data.windowing import Windowing


class RecordingPrep:
    """Minimal preprocessing stage that records fit inputs and transforms data."""

    def __init__(self, operation):
        self.operation = operation
        self.fit_inputs = []

    def fit(self, X):
        self.fit_inputs.append(np.asarray(X).copy())
        return self

    def transform(self, X):
        return self.operation(np.asarray(X))


class TestableBasePrep(BasePrep):
    """Inject predictable stages without relying on configuration or registries."""

    def __init__(self):
        super().__init__(cfg={})
        self.pointwise = RecordingPrep(lambda X: X + 1)
        self.temporal = RecordingPrep(lambda X: X * 2)

    def build_preprocessing_stages(self):
        self.pointwise_prep = self.pointwise
        self.windowing = Windowing(seq_len=2, stride=1)
        self.temporal_prep = self.temporal

        return self.pointwise_prep, self.windowing, self.temporal_prep


def test_fit_fits_pointwise_on_raw_data_and_temporal_on_windowed_data():
    prep = TestableBasePrep()
    X = np.array([[1], [2], [3]])

    prep.fit(X)

    np.testing.assert_array_equal(prep.pointwise.fit_inputs[0], X)

    expected_temporal_input = np.array([[[2], [3]], [[3], [4]]])
    np.testing.assert_array_equal(
        prep.temporal.fit_inputs[0],
        expected_temporal_input,
    )


def test_transform_applies_stages_in_pointwise_windowing_temporal_order():
    prep = TestableBasePrep()
    X = np.array([[1], [2], [3]])

    prep.fit(X)
    result = prep.transform(X)

    expected = np.array([[[4], [6]], [[6], [8]]])
    np.testing.assert_array_equal(result, expected)


def test_transform_does_not_refit_preprocessing_stages():
    prep = TestableBasePrep()
    X_train = np.array([[1], [2], [3]])
    X_val = np.array([[10], [20], [30]])

    prep.fit(X_train)
    pointwise_fit_count = len(prep.pointwise.fit_inputs)
    temporal_fit_count = len(prep.temporal.fit_inputs)

    prep.transform(X_val)

    assert len(prep.pointwise.fit_inputs) == pointwise_fit_count
    assert len(prep.temporal.fit_inputs) == temporal_fit_count


def test_transform_with_labels_keeps_labels_aligned_with_windows():
    prep = TestableBasePrep()
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 0])

    prep.fit(X, y)
    X_windows, y_windows = prep.transform_with_labels(X, y)

    # Each window inherits the label of its final observation.
    expected_y = np.array([0, 1, 0])

    expected_X = np.array(
        [
            [[4], [6]],
            [[6], [8]],
            [[8], [10]],
        ]
    )

    np.testing.assert_array_equal(X_windows, expected_X)
    np.testing.assert_array_equal(y_windows, expected_y)
    assert len(X_windows) == len(y_windows)
