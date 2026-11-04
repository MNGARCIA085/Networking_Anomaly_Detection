from copy import deepcopy

from network_anomaly_detection.tuning.samplers import (
    sample_window_size,
    sample_imputation,
    sample_transform,
    sample_scaler,
    sample_callbacks,
    sample_optimizer,
)

from network_anomaly_detection.models.nn.ae.tuning import sample_dimensions


"""
. Don't tune Isolation Forest's contamination unless you use its built-in prediction 
threshold
"""






# ============================================================
# AE + Isolation Forest Sampler
# ============================================================

class AEIsolationForestSampler:

    def __init__(self, base_cfg, tuning_cfg):
        self.base_cfg = deepcopy(base_cfg)
        self.tuning_cfg = tuning_cfg

    def sample(self, trial):

        cfg = deepcopy(self.base_cfg)
        tuning = self.tuning_cfg

        # ====================================================
        # PREPROCESSING
        # ====================================================

        cfg["prep"]["pointwise"]["imputation"] = (
            sample_imputation(
                trial,
                tuning["prep"]["pointwise"]["imputation"],
            )
        )

        cfg["prep"]["pointwise"]["scaler"] = (
            sample_scaler(
                trial,
                tuning["prep"]["pointwise"]["scaler"],
            )
        )

        cfg["prep"]["windowing"]["size"] = (
            sample_window_size(
                trial,
                tuning["prep"]["windowing"]["size"],
            )
        )

        # ====================================================
        # AUTOENCODER
        # ====================================================

        ae_dimensions = sample_dimensions(
            trial,
            tuning["ae_space"],
        )

        cfg["models"]["ae"]["encoder_dims"] = (
            ae_dimensions["encoder_dims"]
        )

        cfg["models"]["ae"]["decoder_dims"] = (
            ae_dimensions["decoder_dims"]
        )

        # ====================================================
        # ISOLATION FOREST
        # ====================================================

        cfg["models"]["isoforest"]["n_estimators"] = (
            trial.suggest_int(
                "models.isoforest.n_estimators",
                tuning["isoforest_space"]["n_estimators"]["low"],
                tuning["isoforest_space"]["n_estimators"]["high"],
            )
        )

        # ====================================================
        # AE TRAINING
        # ====================================================

        cfg["training"]["optimizer"] = (
            sample_optimizer(
                trial,
                tuning["training_space"]["optimizer"],
            )
        )

        cfg["training"]["callbacks"] = (
            sample_callbacks(
                trial,
                tuning["training_space"]["callbacks"],
            )
        )

        cfg["training"]["epochs"] = (
            trial.suggest_int(
                "training.epochs",
                tuning["training_space"]["epochs"]["low"],
                tuning["training_space"]["epochs"]["high"],
            )
        )

        cfg["training"]["batch_size"] = (
            trial.suggest_categorical(
                "training.batch_size",
                tuning["training_space"]["batch_size"]["choices"],
            )
        )

        # ====================================================
        # THRESHOLDING
        # ====================================================

        cfg["thresholding"] = deepcopy(
            tuning["thresholding"]
        )

        return cfg