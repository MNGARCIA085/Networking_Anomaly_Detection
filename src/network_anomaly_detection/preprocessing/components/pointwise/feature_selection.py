from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold



import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_df = X
        else:
            X = np.asarray(X)
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(X.shape[1])], dtype=object
            )
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        corr = X_df.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        self.selected_features_ = [
            col
            for col in X_df.columns
            if not (upper[col] > self.threshold).any()
        ]

        self.selected_indices_ = np.array(
            [X_df.columns.get_loc(col) for col in self.selected_features_]
        )

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.selected_features_]

        return np.asarray(X)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)






FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
    "correlation": CorrelationSelector,
    "pca": PCA,
}


def create_feature_selector(name, **params):
    try:
        selector_cls = FEATURE_SELECTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature selection method: {name}. "
            f"Available: {list(FEATURE_SELECTION_REGISTRY)}"
        ) from None

    return selector_cls(**params)


"""


prep:
  feature_selection:
    name: correlation
    params:
      threshold: 0.95


prep:
  feature_selection:
    name: pca
    params:
      n_components: 0.95


prep:
  feature_selection:
    name: variance_threshold
    params:
      threshold: 0.0



selector = create_feature_selector(
    name=cfg["name"],
    **cfg.get("params", {}),
)


"""