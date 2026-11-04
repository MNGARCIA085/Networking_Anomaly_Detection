from copy import deepcopy

from network_anomaly_detection.tuning.samplers import (
        sample_window_size,
        sample_imputation,
        sample_transform,
        sample_scaler,
        sample_callbacks,
        sample_optimizer,
    )


def sample_vae_dimensions(trial, cfg):
    encoder_dims = trial.suggest_categorical(
        "encoder_dims",
        [tuple(dims) for dims in cfg["encoder_dims"]["choices"]],
    )

    decoder_dims = trial.suggest_categorical(
        "decoder_dims",
        [tuple(dims) for dims in cfg["decoder_dims"]["choices"]],
    )

    latent_dim = trial.suggest_categorical(
        "latent_dim",
        cfg["latent_dim"]["choices"],
    )

    beta = trial.suggest_float(
        "beta",
        cfg["beta"]["low"],
        cfg["beta"]["high"],
        log=cfg["beta"]["log"],
    )

    return {
        "encoder_dims": list(encoder_dims),
        "decoder_dims": list(decoder_dims),
        "latent_dim": latent_dim,
        "beta": beta,
    }




class VAESampler:

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

        dimensions = sample_vae_dimensions(
            trial,
            tuning["model_space"],
        )

        cfg["models"]["encoder_dims"] = (
            dimensions["encoder_dims"]
        )

        cfg["models"]["decoder_dims"] = (
            dimensions["decoder_dims"]
        )

        cfg["models"]["latent_dim"] = (
            dimensions["latent_dim"]
        )

        cfg["models"]["beta"] = (
            dimensions["beta"]
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