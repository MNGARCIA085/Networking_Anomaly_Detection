import joblib
import numpy as np

from network_anomaly_detection.preprocessing.base import BasePrep


def test_default_training_data_selection_preserves_all_samples():
    prep = BasePrep(cfg={})

    X = np.array([[10], [20], [30]])
    y = np.array([0, 1, 0])

    X_selected, y_selected = prep.select_training_data(X, y)

    np.testing.assert_array_equal(X_selected, X)
    np.testing.assert_array_equal(y_selected, y)


def test_save_and_load_preserves_configured_state(tmp_path):
    prep = BasePrep(cfg={"example": "value"})

    path = tmp_path / "prep.pkl"
    prep.save(path)

    loaded = BasePrep.load(path)

    assert loaded.cfg == prep.cfg
    assert loaded.pointwise_prep is None
    assert loaded.windowing is None
    assert loaded.temporal_prep is None


def test_loaded_fitted_prep_preserves_transformation(tmp_path):
    prep = BasePrep(cfg={})
    prep.pointwise_prep = None
    prep.windowing = None
    prep.temporal_prep = None

    path = tmp_path / "prep.pkl"
    joblib.dump(prep, path)

    loaded = BasePrep.load(path)

    X = np.array([[1, 2], [3, 4]])

    np.testing.assert_array_equal(loaded.transform(X), X)