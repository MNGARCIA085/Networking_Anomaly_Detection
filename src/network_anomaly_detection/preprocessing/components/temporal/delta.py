


class DeltaTransform:
    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order

    # no fit need, but for standardize interface
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_delta = X

        for _ in range(self.order):
            X_delta = X_delta[:, 1:, :] - X_delta[:, :-1, :]

        return X_delta

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



def create_delta(name, **params):
    return DeltaTransform(**params)



"""
    order=1
    [x1, x2, x3, x4]
    →
    [x2-x1, x3-x2, x4-x3]


    order=2
    [x1, x2, x3, x4]
    →
    [(x3-x2)-(x2-x1),
     (x4-x3)-(x3-x2)]
"""