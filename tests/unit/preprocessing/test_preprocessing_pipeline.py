import numpy as np

from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)


class AddConstant:
    def __init__(self, value):
        self.value = value
        self.fitted = False
        self.fit_calls = 0

    def fit(self, X):
        self.fitted = True
        self.fit_calls += 1
        return self

    def transform(self, X):
        return np.asarray(X) + self.value

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def test_fit_fits_each_step_and_passes_transformed_data_forward():
    first = AddConstant(2)
    second = AddConstant(3)

    pipeline = PreprocessingPipeline([first, second])

    X = np.array([[1], [2]])

    pipeline.fit(X)

    assert first.fitted
    assert second.fitted

    # The final fitted step receives data transformed by the first step.
    np.testing.assert_array_equal(
        pipeline.transform(X),
        np.array([[6], [7]]),
    )


def test_transform_applies_steps_in_order_without_refitting():
    first = AddConstant(2)
    second = AddConstant(3)

    pipeline = PreprocessingPipeline([first, second])

    X = np.array([[1], [2]])

    pipeline.fit(X)
    initial_fit_calls = (first.fit_calls, second.fit_calls)

    result = pipeline.transform(X)

    np.testing.assert_array_equal(result, [[6], [7]])
    assert (first.fit_calls, second.fit_calls) == initial_fit_calls


def test_fit_transform_applies_each_step_sequentially():
    pipeline = PreprocessingPipeline([
        AddConstant(2),
        AddConstant(3),
    ])

    result = pipeline.fit_transform(np.array([[1], [2]]))

    np.testing.assert_array_equal(result, [[6], [7]])