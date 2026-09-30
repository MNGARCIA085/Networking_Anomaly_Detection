from .implementations import PrintLossCallback, EarlyStopping



CALLBACK_REGISTRY = {
    "print_loss": PrintLossCallback,
    "early_stopping": EarlyStopping,
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


def create_callbacks(cfg):
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
