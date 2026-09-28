from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def create_scaler(name, **params):
    try:
        scaler_cls = SCALER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown scaler: {name}")

    return scaler_cls(**params)


# create_scaler("standard") produces StandardScaler()
