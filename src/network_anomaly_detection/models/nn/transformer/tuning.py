from copy import deepcopy

from network_anomaly_detection.tuning.samplers import (
    sample_window_size,
    sample_imputation,
    sample_transform,
    sample_scaler,
    sample_callbacks,
    sample_optimizer,
)


def sample_transformer_dimensions(trial, cfg):
    return {
        "d_model": trial.suggest_categorical(
            "d_model",
            cfg["d_model"]["choices"],
        ),
        "nhead": trial.suggest_categorical(
            "nhead",
            cfg["nhead"]["choices"],
        ),
        "num_encoder_layers": trial.suggest_int(
            "num_encoder_layers",
            cfg["num_encoder_layers"]["low"],
            cfg["num_encoder_layers"]["high"],
        ),
        "dim_feedforward": trial.suggest_categorical(
            "dim_feedforward",
            cfg["dim_feedforward"]["choices"],
        ),
        "dropout": trial.suggest_float(
            "dropout",
            cfg["dropout"]["low"],
            cfg["dropout"]["high"],
        ),
    }






class TransformerAESampler:

    def __init__(self, base_cfg):
        self.base_cfg = deepcopy(base_cfg)
        self.tuning_cfg = base_cfg["model_type"]["tuning"]

    def sample(self, trial):
        # Start from complete experiment configuration
        cfg = deepcopy(self.base_cfg)
        tuning = self.tuning_cfg


        print(cfg)

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
        # PREPROCESSING: POINTWISE
        # ====================================================

        cfg["model_type"]["prep"]["pointwise"]["imputation"] = (
            sample_imputation(
                trial,
                tuning["prep"]["pointwise"]["imputation"],
            )
        )

        #cfg["model_type"]["prep"]["pointwise"]["transform"] = (
        #    sample_transform(
        #        trial,
        #        tuning["prep"]["pointwise"]["transform"],
        #    )
        #)

        cfg["model_type"]["prep"]["pointwise"]["scaler"] = (
            sample_scaler(
                trial,
                tuning["prep"]["pointwise"]["scaler"],
            )
        )

        # ====================================================
        # PREPROCESSING: TEMPORAL
        # ====================================================

        # Delta configuration is copied from the tuning config.
        # Remove this block if delta is not being tuned.
        #if "temporal" in tuning["prep"]:
        #    cfg["model_type"]["prep"]["temporal"]["delta"] = (
        #        deepcopy(tuning["prep"]["temporal"]["delta"])
        #    )


        # ====================================================
        # PREPROCESSING: TEMPORAL
        # ====================================================

        use_delta = trial.suggest_categorical(
            "prep.temporal.delta.enabled",
            tuning["prep"]["temporal"]["delta"]["choices"],
        )

        cfg["model_type"]["prep"]["temporal"]["delta"] = {
            "enable": use_delta,
            "name": "delta",
            "params": {"order": 1},
        }



        # ====================================================
        # MODEL
        # ====================================================

        dimensions = sample_transformer_dimensions(
            trial,
            tuning["model_space"],
        )

        for name, value in dimensions.items():
            cfg["model_type"]["models"][name] = value

        # input_dim and seq_len are derived from the prepared
        # input_shape by TransformerAEModel, not sampled here.

        # ====================================================
        # TRAINING
        # ====================================================

        cfg["model_type"]["training"]["optimizer"] = (
            sample_optimizer(
                trial,
                tuning["training_space"]["optimizer"],
            )
        )

        cfg["model_type"]["training"]["callbacks"] = (
            sample_callbacks(
                trial,
                tuning["training_space"]["callbacks"],
            )
        )

        cfg["model_type"]["training"]["epochs"] = (
            trial.suggest_int(
                "training.epochs",
                tuning["training_space"]["epochs"]["low"],
                tuning["training_space"]["epochs"]["high"],
            )
        )

        cfg["model_type"]["training"]["batch_size"] = (
            trial.suggest_categorical(
                "training.batch_size",
                tuning["training_space"]["batch_size"]["choices"],
            )
        )

        # ====================================================
        # THRESHOLDING
        # ====================================================

        if "thresholding" in tuning:
            cfg["model_type"]["thresholding"] = deepcopy(
                tuning["thresholding"]
            )

        return cfg