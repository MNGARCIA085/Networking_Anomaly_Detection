import numpy as np
import pandas as pd

from network_anomaly_detection.models.nn.ae.prep import AEPrep


def test_select_training_data_keeps_only_normal_samples():
    prep = AEPrep(cfg={})

    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400],
    ])
    y = np.array([0, 1, 0, 1])

    X_selected, y_selected = prep.select_training_data(X, y)

    np.testing.assert_array_equal(
        X_selected,
        [[10, 100], [30, 300]],
    )
    np.testing.assert_array_equal(y_selected, [0, 0])


def test_select_training_data_preserves_pandas_indices():
    prep = AEPrep(cfg={})

    X = pd.DataFrame(
        {"feature": [10, 20, 30, 40]},
        index=[101, 102, 103, 104],
    )
    y = pd.Series(
        [0, 1, 0, 1],
        index=X.index,
    )

    X_selected, y_selected = prep.select_training_data(X, y)

    assert X_selected.index.tolist() == [101, 103]
    assert y_selected.index.tolist() == [101, 103]

    np.testing.assert_array_equal(X_selected["feature"], [10, 30])
    np.testing.assert_array_equal(y_selected, [0, 0])


def test_select_training_data_handles_column_vector_labels():
    prep = AEPrep(cfg={})

    X = np.array([[10], [20], [30]])
    y = np.array([[0], [1], [0]])

    X_selected, y_selected = prep.select_training_data(X, y)

    np.testing.assert_array_equal(X_selected, [[10], [30]])
    np.testing.assert_array_equal(y_selected, [[0], [0]])


def test_select_training_data_handles_all_normal_samples():
    prep = AEPrep(cfg={})

    X = np.array([[10], [20]])
    y = np.array([0, 0])

    X_selected, y_selected = prep.select_training_data(X, y)

    np.testing.assert_array_equal(X_selected, X)
    np.testing.assert_array_equal(y_selected, y)