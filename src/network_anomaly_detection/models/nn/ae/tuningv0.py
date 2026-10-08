from copy import deepcopy







from copy import deepcopy


# ============================================================
# Windowing
# ============================================================

def sample_window_size(trial, cfg):

    return trial.suggest_categorical(
        "data.windowing.size",
        cfg.choices,
    )


# ============================================================
# Imputation
# ============================================================

def sample_imputation(trial, cfg):

    name = cfg.names[0]

    strategy = trial.suggest_categorical(
        "prep.imputation.strategy",
        cfg.strategy.choices,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "strategy": strategy,
        },
    }


# ============================================================
# Transform
# ============================================================

def sample_transform(trial, cfg):

    name = cfg.names[0]

    method = trial.suggest_categorical(
        "prep.transform.method",
        cfg.params.method.choices,
    )

    standardize = trial.suggest_categorical(
        "prep.transform.standardize",
        cfg.params.standardize.choices,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {
            "method": method,
            "standardize": standardize,
        },
    }


# ============================================================
# Scaler
# ============================================================

def sample_scaler(trial, cfg):

    name = trial.suggest_categorical(
        "prep.scaler.name",
        cfg.names,
    )

    return {
        "enabled": True,
        "name": name,
        "params": {},
    }


# ============================================================
# Optimizer
# ============================================================

def sample_optimizer(trial, cfg):

    optimizer_name = trial.suggest_categorical(
        "training.optimizer.name",
        cfg.names,
    )

    optimizer_cfg = getattr(
        cfg,
        optimizer_name,
    )

    params = {
        "lr": trial.suggest_float(
            f"training.optimizer.{optimizer_name}.lr",
            optimizer_cfg.lr.low,
            optimizer_cfg.lr.high,
            log=optimizer_cfg.lr.log,
        )
    }

    if optimizer_name == "adam":

        params["betas"] = trial.suggest_categorical(
            "training.optimizer.adam.betas",
            [
                tuple(beta)
                for beta in optimizer_cfg.betas.choices
            ],
        )

    elif optimizer_name == "sgd":

        params["momentum"] = trial.suggest_float(
            "training.optimizer.sgd.momentum",
            optimizer_cfg.momentum.low,
            optimizer_cfg.momentum.high,
        )

        params["weight_decay"] = trial.suggest_float(
            "training.optimizer.sgd.weight_decay",
            optimizer_cfg.weight_decay.low,
            optimizer_cfg.weight_decay.high,
            log=optimizer_cfg.weight_decay.log,
        )

    return {
        "name": optimizer_name,
        "params": params,
    }


# ============================================================
# Callbacks
# ============================================================

def sample_callbacks(trial, cfg):

    callbacks = []

    # --------------------------------------------------------
    # Print loss
    # --------------------------------------------------------

    if cfg.print_loss:

        enabled = trial.suggest_categorical(
            "training.callbacks.print_loss",
            [True, True],
        )

        if enabled:

            callbacks.append({
                "name": "print_loss",
                "params": {},
            })

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if cfg.early_stopping.enabled:

        enabled = trial.suggest_categorical(
            "training.callbacks.early_stopping.enabled",
            [True, True],
        )

        if enabled:

            patience = trial.suggest_int(
                "training.callbacks.early_stopping.patience",
                cfg.early_stopping.patience.low,
                cfg.early_stopping.patience.high,
            )

            callbacks.append({
                "name": "early_stopping",
                "params": {
                    "patience": patience,
                },
            })

    return callbacks


# ============================================================
# Model dimensions
# ============================================================

def sample_dimensions(trial, cfg):

    encoder_dims = [
        trial.suggest_int(
            param.name,
            param.low,
            param.high,
        )
        for param in cfg.encoder_dims
    ]

    decoder_dims = [
        trial.suggest_int(
            param.name,
            param.low,
            param.high,
        )
        for param in cfg.decoder_dims
    ]

    return {
        "encoder_dims": encoder_dims,
        "decoder_dims": decoder_dims,
    }


# ============================================================
# AE Sampler
# ============================================================

class AESampler:

    def __init__(self, base_cfg):

        self.base_cfg = deepcopy(base_cfg)
        self.tuning_cfg = base_cfg.model_type.tuning

    def sample(self, trial):

        cfg = deepcopy(self.base_cfg)
        tuning = self.tuning_cfg

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        cfg.model_type.data.windowing.size = (
            sample_window_size(
                trial,
                tuning.data.windowing.size,
            )
        )

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        cfg.model_type.prep.pointwise.imputation = (
            sample_imputation(
                trial,
                tuning.prep.imputation,
            )
        )

        cfg.model_type.prep.pointwise.transform = (
            sample_transform(
                trial,
                tuning.prep.transform,
            )
        )

        cfg.model_type.prep.pointwise.scaler = (
            sample_scaler(
                trial,
                tuning.prep.scaler,
            )
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        dimensions = sample_dimensions(
            trial,
            tuning.model_space,
        )

        cfg.model_type.models.encoder_dims = (
            dimensions["encoder_dims"]
        )

        cfg.model_type.models.decoder_dims = (
            dimensions["decoder_dims"]
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        cfg.model_type.training.optimizer = (
            sample_optimizer(
                trial,
                tuning.training_space.optimizer,
            )
        )

        cfg.model_type.training.callbacks = (
            sample_callbacks(
                trial,
                tuning.training_space.callbacks,
            )
        )

        cfg.model_type.training.epochs = (
            trial.suggest_int(
                "training.epochs",
                tuning.training_space.epochs.low,
                tuning.training_space.epochs.high,
            )
        )

        cfg.model_type.training.batch_size = (
            trial.suggest_categorical(
                "training.batch_size",
                tuning.training_space.batch_size.choices,
            )
        )

        # ----------------------------------------------------
        # Thresholding
        # ----------------------------------------------------

        cfg.model_type.thresholding = deepcopy(
            tuning.thresholding
        )

        return cfg













#--------------------------------#
#--------------------------------#
#--------------------------------#
#--------------------------------#





class AESamplerv0:

    def __init__(self, base_cfg):

        self.cfg = deepcopy(base_cfg)
        self.tuning = base_cfg.model_type.tuning

    def sample(self, trial):

        cfg = deepcopy(self.cfg)
        tuning = self.tuning

        # ==================================================
        # DATA
        # ==================================================

        cfg.model_type.data.windowing.size = (
            trial.suggest_categorical(
                "window_size",
                tuning.data.windowing.size.choices,
            )
        )

        # ==================================================
        # PREPROCESSING
        # ==================================================

        # Imputation
        cfg.model_type.prep.pointwise.imputation.params.strategy = (
            trial.suggest_categorical(
                "imputation_strategy",
                tuning.prep.imputation.strategy.choices,
            )
        )

        # Transform method
        cfg.model_type.prep.pointwise.transform.params.method = (
            trial.suggest_categorical(
                "transform_method",
                tuning.prep.transform.params.method.choices,
            )
        )

        # Transform standardization
        cfg.model_type.prep.pointwise.transform.params.standardize = (
            trial.suggest_categorical(
                "transform_standardize",
                tuning.prep.transform.params.standardize.choices,
            )
        )

        # Scaler
        cfg.model_type.prep.pointwise.scaler.name = (
            trial.suggest_categorical(
                "scaler",
                tuning.prep.scaler.names,
            )
        )

        # ==================================================
        # MODEL
        # ==================================================

        cfg.model_type.models.encoder_dims = [
            trial.suggest_int(
                param.name,
                param.low,
                param.high,
            )
            for param in tuning.model_space.encoder_dims
        ]

        cfg.model_type.models.decoder_dims = [
            trial.suggest_int(
                param.name,
                param.low,
                param.high,
            )
            for param in tuning.model_space.decoder_dims
        ]

        # ==================================================
        # TRAINING
        # ==================================================

        training = tuning.training_space

        optimizer_name = trial.suggest_categorical(
            "optimizer",
            training.optimizer.names,
        )

        optimizer_space = getattr(
            training.optimizer,
            optimizer_name,
        )

        # Optimizer LR
        optimizer_params = {
            "lr": trial.suggest_float(
                f"{optimizer_name}_lr",
                optimizer_space.lr.low,
                optimizer_space.lr.high,
                log=optimizer_space.lr.log,
            )
        }

        # Adam
        if optimizer_name == "adam":

            optimizer_params["betas"] = (
                trial.suggest_categorical(
                    "adam_betas",
                    [
                        tuple(beta)
                        for beta in optimizer_space.betas.choices
                    ],
                )
            )

        # SGD
        elif optimizer_name == "sgd":

            optimizer_params["momentum"] = (
                trial.suggest_float(
                    "sgd_momentum",
                    optimizer_space.momentum.low,
                    optimizer_space.momentum.high,
                )
            )

            optimizer_params["weight_decay"] = (
                trial.suggest_float(
                    "sgd_weight_decay",
                    optimizer_space.weight_decay.low,
                    optimizer_space.weight_decay.high,
                    log=optimizer_space.weight_decay.log,
                )
            )

        cfg.model_type.training.optimizer.name = optimizer_name

        cfg.model_type.training.optimizer.params = optimizer_params

        # Epochs
        cfg.model_type.training.epochs = (
            trial.suggest_int(
                "epochs",
                training.epochs.low,
                training.epochs.high,
            )
        )

        # Batch size
        cfg.model_type.training.batch_size = (
            trial.suggest_categorical(
                "batch_size",
                training.batch_size.choices,
            )
        )

        # Early stopping patience
        patience = trial.suggest_int(
            "early_stopping_patience",
            training.callbacks.early_stopping.patience.low,
            training.callbacks.early_stopping.patience.high,
        )

        for callback in cfg.model_type.training.callbacks:

            if callback.name == "early_stopping":

                callback.params.patience = patience

        # ==================================================
        # THRESHOLDING
        # ==================================================

        cfg.model_type.thresholding.name = tuning.thresholding.name

        cfg.model_type.thresholding.params.min_recall = (
            tuning.thresholding.params.min_recall
        )

        return cfg