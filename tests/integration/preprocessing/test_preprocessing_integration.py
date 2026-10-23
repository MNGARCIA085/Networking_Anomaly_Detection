import numpy as np
import pytest

from network_anomaly_detection.preprocessing.base import BasePrep


def make_cfg(*, window_size=1, stride=1, delta_enabled=False):
    return {
        "prep": {
            "pointwise": {
                "imputation": {
                    "enabled": True,
                    "name": "simple",
                    "params": {"strategy": "median"},
                },
                "feature_selection": {
                    "enabled": False,
                    "name": "variance_threshold",
                    "params": {"threshold": 0.0},
                },
                "scaler": {
                    "enabled": True,
                    "name": "standard",
                    "params": {
                        "with_mean": True,
                        "with_std": True,
                    },
                },
                "transform": {
                    "enabled": False,
                    "name": "power",
                    "params": {
                        "method": "yeo-johnson",
                        "standardize": False,
                    },
                },
            },
            "windowing": {
                "size": window_size,
                "stride": stride,
            },
            "temporal": {
                "delta": {
                    "enable": delta_enabled,
                    "name": "delta",
                    "params": {"order": 1},
                },
            },
        }
    }


def test_real_imputer_and_scaler_fit_on_training_data_only():
    X_train = np.array([
        [1.0, 10.0],
        [2.0, np.nan],
        [3.0, 30.0],
        [4.0, 40.0],
    ])

    X_val = np.array([
        [5.0, 50.0],
        [100.0, 1000.0],
    ])

    prep = BasePrep(make_cfg())
    prep.fit(X_train)

    # Capture fitted parameters before transforming validation data.
    scaler = prep.pointwise_prep.steps[-1]
    train_mean = scaler.mean_.copy()
    train_scale = scaler.scale_.copy()

    X_val_transformed = prep.transform(X_val)

    # Validation transformation must not refit the scaler.
    np.testing.assert_array_equal(scaler.mean_, train_mean)
    np.testing.assert_array_equal(scaler.scale_, train_scale)

    assert X_val_transformed.shape == (len(X_val), 1, X_val.shape[1])
    assert np.isfinite(X_val_transformed).all()


def test_pointwise_preprocessing_and_windowing_preserve_expected_shape():
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])

    prep = BasePrep(make_cfg(window_size=3, stride=1))
    prep.fit(X_train)

    X_transformed = prep.transform(X_train)

    # Five samples, window length three, stride one -> three windows.
    assert X_transformed.shape == (3, 3, 2)
    assert np.isfinite(X_transformed).all()


def test_transform_with_labels_keeps_window_labels_aligned():
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])
    y_train = np.array([0, 0, 1, 0, 0])

    prep = BasePrep(make_cfg(window_size=3, stride=1))
    prep.fit(X_train, y_train)

    X_windows, y_windows = prep.transform_with_labels(X_train, y_train)

    assert X_windows.shape == (3, 3, 2)
    np.testing.assert_array_equal(y_windows, [1, 1, 1])
    assert len(X_windows) == len(y_windows)


def test_save_and_reload_preserves_fitted_transformations(tmp_path):
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    X_val = np.array([
        [5.0, 50.0],
        [6.0, 60.0],
        [7.0, 70.0],
    ])

    prep = BasePrep(make_cfg(window_size=2))
    prep.fit(X_train)
    expected = prep.transform(X_val)

    path = tmp_path / "prep.pkl"
    prep.save(path)
    loaded = BasePrep.load(path)

    actual = loaded.transform(X_val)

    np.testing.assert_allclose(actual, expected)





"""
| Test | Purpose |
|---|---|
| `test_real_imputer_and_scaler_fit_on_training_data_only` | Verifies that actual sklearn preprocessing components work together and validation data does not change fitted scaler parameters. |
| `test_pointwise_preprocessing_and_windowing_preserve_expected_shape` | Verifies the output dimensions when real pointwise preprocessing feeds into windowing. |
| `test_transform_with_labels_keeps_window_labels_aligned` | Verifies that the integration preserves the window-label aggregation semantics. |
| `test_save_and_reload_preserves_fitted_transformations` | Verifies that a serialized, fitted preprocessing object can be loaded and reused without changing its results. |
"""