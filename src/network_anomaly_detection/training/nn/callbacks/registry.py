from .implementations import PrintLossCallback, EarlyStopping, CheckpointCallback



CALLBACK_REGISTRY = {
    "print_loss": PrintLossCallback,
    "early_stopping": EarlyStopping,
    "checkpoint": CheckpointCallback,
}




# create callbacks dynamically to use with config

def create_callback(name, **params):
    try:
        callback_cls = CALLBACK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown callback: {name}. "
            f"Available: {list(CALLBACK_REGISTRY)}"
        )

    return callback_cls(**params)






def create_callbacks(cfg, checkpoint_dir=None):
    callbacks = []

    for callback_cfg in cfg:

        params = dict(
            callback_cfg.get("params", {})
        )

        if callback_cfg["name"] == "checkpoint":

            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir is required for the checkpoint callback."
                )

            params["directory"] = checkpoint_dir

        callbacks.append(
            create_callback(
                callback_cfg["name"],
                **params,
            )
        )

    return callbacks




# factory; remove later
def create_callbacksvo(cfg):
    callbacks = [] # I can put defaults here if i want
            #EarlyStopping(patience=3),
            #PrintLossCallback(),

    for callback_cfg in cfg:
        callbacks.append(
            create_callback(
                callback_cfg["name"],
                **callback_cfg.get("params", {}),
            )
        )

    return callbacks



def create_callbacksv1(cfg, checkpoint_dir=None):
    callbacks = []

    for callback_cfg in cfg:

        name = callback_cfg["name"]
        params = callback_cfg.get("params", {}).copy()

        if name == "checkpoint":
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir is required when using "
                    "the checkpoint callback."
                )

            params["directory"] = checkpoint_dir

        callbacks.append(
            create_callback(
                name,
                **params,
            )
        )

    return callbacks
