from sklearn.impute import SimpleImputer


IMPUTATION_REGISTRY = {
    "simple": SimpleImputer,
}


def create_imputer(name, **params):
    try:
        imputer_cls = IMPUTATION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown imputation method: {name}. "
            f"Available: {list(IMPUTATION_REGISTRY)}"
        )

    return imputer_cls(**params)

