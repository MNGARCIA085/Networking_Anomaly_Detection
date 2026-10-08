# ============================================================
# Windowing
# ============================================================

def sample_window_size(trial, cfg):

    return trial.suggest_categorical(
        "data.windowing.size",
        cfg["choices"],
    )


# ============================================================
# Imputation
# ============================================================

def sample_imputation(trial, cfg):

    return {
        "enabled": True,
        "name": cfg["names"][0],
        "params": {
            "strategy": trial.suggest_categorical(
                "prep.imputation.strategy",
                cfg["strategy"]["choices"],
            ),
        },
    }


# ============================================================
# Transform
# ============================================================

def sample_transform(trial, cfg):

    return {
        "enabled": True,
        "name": cfg["names"][0],
        "params": {
            "method": trial.suggest_categorical(
                "prep.transform.method",
                cfg["params"]["method"]["choices"],
            ),
            "standardize": trial.suggest_categorical(
                "prep.transform.standardize",
                cfg["params"]["standardize"]["choices"],
            ),
        },
    }


# ============================================================
# Scaler
# ============================================================

def sample_scaler(trial, cfg):

    return {
        "enabled": True,
        "name": trial.suggest_categorical(
            "prep.scaler.name",
            cfg["names"],
        ),
        "params": {},
    }


# ============================================================
# Optimizer
# ============================================================

def sample_optimizer(trial, cfg):

    optimizer_name = trial.suggest_categorical(
        "training.optimizer.name",
        cfg["names"],
    )

    optimizer_cfg = cfg[optimizer_name]

    params = {
        "lr": trial.suggest_float(
            f"training.optimizer.{optimizer_name}.lr",
            optimizer_cfg["lr"]["low"],
            optimizer_cfg["lr"]["high"],
            log=optimizer_cfg["lr"].get("log", False),
        )
    }

    # Adam
    if optimizer_name == "adam":

        params["betas"] = trial.suggest_categorical(
            "training.optimizer.adam.betas",
            [
                tuple(beta)
                for beta in optimizer_cfg["betas"]["choices"]
            ],
        )

    # SGD
    elif optimizer_name == "sgd":

        params["momentum"] = trial.suggest_float(
            "training.optimizer.sgd.momentum",
            optimizer_cfg["momentum"]["low"],
            optimizer_cfg["momentum"]["high"],
        )

        params["weight_decay"] = trial.suggest_float(
            "training.optimizer.sgd.weight_decay",
            optimizer_cfg["weight_decay"]["low"],
            optimizer_cfg["weight_decay"]["high"],
            log=optimizer_cfg["weight_decay"].get(
                "log",
                False,
            ),
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

    if cfg["print_loss"]:

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

    if cfg["early_stopping"]["enabled"]:

        enabled = trial.suggest_categorical(
            "training.callbacks.early_stopping.enabled",
            [True, True],
        )

        if enabled:

            patience = trial.suggest_int(
                "training.callbacks.early_stopping.patience",
                cfg["early_stopping"]["patience"]["low"],
                cfg["early_stopping"]["patience"]["high"],
            )

            callbacks.append({
                "name": "early_stopping",
                "params": {
                    "patience": patience,
                },
            })

    return callbacks