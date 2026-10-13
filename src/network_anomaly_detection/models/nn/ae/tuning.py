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


def sample_dimensionsv0(trial, cfg):
    """ only enforces powers of 2; not architecture"""
    def sample_layer(param):
        if param["type"] == "categorical":
            return trial.suggest_categorical(
                param["name"],
                param["choices"],
            )

        if param["type"] == "int":
            return trial.suggest_int(
                param["name"],
                param["low"],
                param["high"],
                step=param.get("step", 1),
            )

        raise ValueError(
            f"Unsupported parameter type: {param['type']}"
        )

    return {
        "encoder_dims": [
            sample_layer(p) for p in cfg["encoder_dims"]
        ],
        "decoder_dims": [
            sample_layer(p) for p in cfg["decoder_dims"]
        ],
    }



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

    def __init__(self, base_cfg):

        self.base_cfg = deepcopy(base_cfg)

        self.tuning_cfg = (
            base_cfg["model_type"]["tuning"]
        )

    def sample(self, trial):

        # Start from complete experiment configuration
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

        # replace the or. conf with the one from tuning!!!
        cfg["model_type"]["prep"]["pointwise"]["imputation"] = (
            sample_imputation(
                trial,
                tuning["prep"]["imputation"], # later adapt to ["prep"]["poinbtwise"]...
            )
        )

        cfg["model_type"]["prep"]["pointwise"]["transform"] = (
            sample_transform(
                trial,
                tuning["prep"]["transform"],
            )
        )

        cfg["model_type"]["prep"]["pointwise"]["scaler"] = (
            sample_scaler(
                trial,
                tuning["prep"]["scaler"],
            )
        )

        # ====================================================
        # MODEL
        # ====================================================

        dimensions = sample_dimensions(
            trial,
            tuning["model_space"],
        )

        cfg["model_type"]["models"]["encoder_dims"] = (
            dimensions["encoder_dims"]
        )

        cfg["model_type"]["models"]["decoder_dims"] = (
            dimensions["decoder_dims"]
        )

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

        cfg["model_type"]["thresholding"] = deepcopy(
            tuning["thresholding"]
        )

        return cfg



# note: overrids only what is tunned