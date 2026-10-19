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



class AEIsolationForestSampler:

    def __init__(self, base_cfg):
        self.base_cfg = deepcopy(base_cfg)
        self.tuning_cfg = base_cfg["model_type"]["tuning"]

    def sample(self, trial):
        cfg = deepcopy(self.base_cfg)
        tuning = self.tuning_cfg
        model_cfg = cfg["model_type"]

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------
        model_cfg["data"]["windowing"]["size"] = sample_window_size(
            trial,
            tuning["data"]["windowing"]["size"],
        )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------
        prep_cfg = model_cfg["prep"]["pointwise"]
        prep_tuning = tuning["prep"]

        prep_cfg["imputation"] = sample_imputation(
            trial,
            prep_tuning["imputation"],
        )

        #prep_cfg["transform"] = sample_transform(
        #    trial,
        #    prep_tuning["transform"],
        #)

        prep_cfg["scaler"] = sample_scaler(
            trial,
            prep_tuning["scaler"],
        )

        # ----------------------------------------------------
        # AE ARCHITECTURE
        # ----------------------------------------------------
        dimensions = sample_dimensions(
            trial,
            tuning["ae_space"],
        )

        model_cfg["models"]["ae"]["encoder_dims"] = (
            dimensions["encoder_dims"]
        )
        model_cfg["models"]["ae"]["decoder_dims"] = (
            dimensions["decoder_dims"]
        )

        # ----------------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------------
        iso_cfg = model_cfg["models"]["isoforest"]
        iso_tuning = tuning["isoforest_space"]

        iso_cfg["n_estimators"] = trial.suggest_int(
            "isoforest.n_estimators",
            iso_tuning["n_estimators"]["low"],
            iso_tuning["n_estimators"]["high"],
        )

        #iso_cfg["contamination"] = trial.suggest_float(
        #    "isoforest.contamination",
        #    iso_tuning["contamination"]["low"],
        #    iso_tuning["contamination"]["high"],
        #)

        # ----------------------------------------------------
        # AE TRAINING
        # ----------------------------------------------------
        training_cfg = model_cfg["training"]
        training_tuning = tuning["training_space"]

        training_cfg["optimizer"] = sample_optimizer(
            trial,
            training_tuning["optimizer"],
        )

        training_cfg["callbacks"] = sample_callbacks(
            trial,
            training_tuning["callbacks"],
        )

        training_cfg["epochs"] = trial.suggest_int(
            "training.epochs",
            training_tuning["epochs"]["low"],
            training_tuning["epochs"]["high"],
        )

        training_cfg["batch_size"] = trial.suggest_categorical(
            "training.batch_size",
            training_tuning["batch_size"]["choices"],
        )

        # ----------------------------------------------------
        # THRESHOLDING
        # ----------------------------------------------------
        model_cfg["thresholding"] = deepcopy(tuning["thresholding"])

        return cfg