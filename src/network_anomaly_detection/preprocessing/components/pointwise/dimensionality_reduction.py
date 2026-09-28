from sklearn.decomposition import PCA, TruncatedSVD


DIMENSIONALITY_REDUCTION_REGISTRY = {
    "pca": PCA,
    "truncated_svd": TruncatedSVD,
}


def create_dimensionality_reducer(name, **params):
    try:
        reducer_cls = DIMENSIONALITY_REDUCTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dimensionality reduction method: {name}. "
            f"Available: {list(DIMENSIONALITY_REDUCTION_REGISTRY)}"
        )

    return reducer_cls(**params)
