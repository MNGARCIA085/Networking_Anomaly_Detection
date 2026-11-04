from copy import deepcopy

from network_anomaly_detection.tuning.samplers import (
        sample_window_size,
        sample_imputation,
        sample_transform,
        sample_scaler,
        sample_callbacks,
        sample_optimizer,
    )


# ============================================================
# Model dimensions (model specific)
# ============================================================
def sample_dimensions(trial, cfg):
    encoder_dims = trial.suggest_categorical(
        "encoder_dims",
        [tuple(dims) for dims in cfg["encoder_dims"]["choices"]],
    )

    decoder_dims = trial.suggest_categorical(
        "decoder_dims",
        [tuple(dims) for dims in cfg["decoder_dims"]["choices"]],
    )

    return {
        "encoder_dims": list(encoder_dims),
        "decoder_dims": list(decoder_dims),
    }




# ============================================================
# AE Sampler
# ============================================================

class AESampler:

    def __init__(self, base_cfg, tuning_cfg):
        """ Replace base_cfg with tuning_cfg appropiately """
        self.base_cfg = deepcopy(base_cfg)
        self.tuning_cfg = tuning_cfg

    def sample(self, trial):

        # Start from complete experiment configuration
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

        cfg["prep"]["pointwise"]["transform"] = (
            sample_transform(
                trial,
                tuning["prep"]["pointwise"]["transform"],
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
        # MODEL
        # ====================================================

        dimensions = sample_dimensions(
            trial,
            tuning["model_space"],
        )

        cfg["models"]["encoder_dims"] = (
            dimensions["encoder_dims"]
        )

        cfg["models"]["decoder_dims"] = (
            dimensions["decoder_dims"]
        )

        # ====================================================
        # TRAINING
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



# note: overrids only what is tunned


# correct;