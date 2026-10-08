from copy import deepcopy




from network_anomaly_detection.tuning.samplers import (
        sample_window_size,
        sample_imputation,
        sample_transform,
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
        "prep.feature_selection.name",
        cfg["names"],
    )

    threshold = trial.suggest_float(
        "prep.feature_selection.threshold",
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
        "prep.dimensionality.name",
        cfg["names"],
    )

    n_components = trial.suggest_int(
        "prep.dimensionality.n_components",
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
# Model
# ============================================================

def sample_model(trial, cfg, base_model_cfg):

    model_cfg = deepcopy(base_model_cfg)

    model_cfg["n_estimators"] = trial.suggest_int(
        "model.n_estimators",
        cfg["n_estimators"]["low"],
        cfg["n_estimators"]["high"],
    )

    model_cfg["contamination"] = trial.suggest_float(
        "model.contamination",
        cfg["contamination"]["low"],
        cfg["contamination"]["high"],
    )

    return model_cfg

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

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        cfg["model_type"]["data"]["windowing"]["size"] = (
            sample_window_size(
                trial,
                tuning["data"]["windowing"]["size"],
            )
        )

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        cfg["model_type"]["prep"]["pointwise"]["scaler"] = (
            sample_scaler(
                trial,
                tuning["prep"]["scaler"],
            )
        )

        cfg["model_type"]["prep"]["pointwise"]["feature_selection"] = (
            sample_feature_selection(
                trial,
                tuning["prep"]["feature_selection"],
            )
        )


        """
        cfg["model_type"]["prep"]["pointwise"]["dimensionality"] = (
            sample_dimensionality(
                trial,
                tuning["prep"]["dimensionality"],
            )
        )
        """

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        cfg["model_type"]["models"] = sample_model(
            trial,
            tuning["model_space"],
            cfg["model_type"]["models"],
        )

        return cfg


