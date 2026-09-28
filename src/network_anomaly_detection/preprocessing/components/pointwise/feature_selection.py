from sklearn.feature_selection import VarianceThreshold


FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
}


def create_feature_selector(name, **params):
    try:
        selector_cls = FEATURE_SELECTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature selection method: {name}. "
            f"Available: {list(FEATURE_SELECTION_REGISTRY)}"
        )

    return selector_cls(**params)
    