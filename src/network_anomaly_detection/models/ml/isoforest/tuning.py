from copy import deepcopy

from network_anomaly_detection.tuning.samplers import (
    sample_window_size,
    sample_scaler,
)


# ============================================================
# Feature selection
# ============================================================

def sample_feature_selection(trial, cfg):

    if not cfg["enabled"]:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.pointwise.feature_selection.name",
        cfg["names"],
    )

    threshold = trial.suggest_float(
        "prep.pointwise.feature_selection.threshold",
        cfg["threshold"]["low"],
        cfg["threshold"]["high"],
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "threshold": threshold,
        },
    }


# ============================================================
# Dimensionality reduction
# ============================================================

def sample_dimensionality(trial, cfg):

    if not cfg["enabled"]:
        return {
            "enabled": False,
        }

    name = trial.suggest_categorical(
        "prep.pointwise.dimensionality.name",
        cfg["names"],
    )

    n_components = trial.suggest_int(
        "prep.pointwise.dimensionality.n_components",
        cfg["n_components"]["low"],
        cfg["n_components"]["high"],
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "n_components": n_components,
        },
    }


# ============================================================
# IsoForest Sampler
# ============================================================

class IsoForestSampler:

    def __init__(self, base_cfg):

        self.base_cfg = deepcopy(base_cfg)

        self.tuning_cfg = (
            base_cfg["model_type"]["tuning"]
        )

    def sample(self, trial):

        cfg = deepcopy(self.base_cfg)
        tuning = self.tuning_cfg

        # ====================================================
        # DATA
        # ====================================================

        cfg["model_type"]["data"]["windowing"]["size"] = (
            sample_window_size(
                trial,
                tuning["data"]["windowing"]["size"],
            )
        )

        # ====================================================
        # PREPROCESSING
        # ====================================================

        cfg["model_type"]["prep"]["pointwise"]["scaler"] = (
            sample_scaler(
                trial,
                tuning["prep"]["pointwise"]["scaler"],
            )
        )

        cfg["model_type"]["prep"]["pointwise"]["feature_selection"] = (
            sample_feature_selection(
                trial,
                tuning["prep"]["pointwise"]["feature_selection"],
            )
        )

        """
        cfg["model_type"]["prep"]["pointwise"]["dimensionality"] = (
            sample_dimensionality(
                trial,
                tuning["prep"]["pointwise"]["dimensionality"],
            )
        )
        """

        # ====================================================
        # MODEL
        # ====================================================

        cfg["model_type"]["models"]["n_estimators"] = (
            trial.suggest_int(
                "model.n_estimators",
                tuning["model_space"]["n_estimators"]["low"],
                tuning["model_space"]["n_estimators"]["high"],
            )
        )

        cfg["model_type"]["models"]["contamination"] = (
            trial.suggest_float(
                "model.contamination",
                tuning["model_space"]["contamination"]["low"],
                tuning["model_space"]["contamination"]["high"],
            )
        )

        return cfg