import joblib
import numpy as np


class Windowing:

    def __init__(self, seq_len: int, stride: int = 1):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.seq_len = seq_len
        self.stride = stride

    def _get_starts(self, n_samples):
        if n_samples < self.seq_len:
            raise ValueError(
                f"Not enough samples ({n_samples}) "
                f"for seq_len={self.seq_len}"
            )

        return range(
            0,
            n_samples - self.seq_len + 1,
            self.stride,
        )



    def transform(self, X, y=None): # y is opional for inference
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected X with shape "
                f"(n_samples, n_features), got {X.shape}"
            )

        if y is not None:
            y = np.asarray(y)

            if y.ndim != 1:
                raise ValueError(
                    f"Expected y with shape (n_samples,), got {y.shape}"
                )

            if len(X) != len(y):
                raise ValueError(
                    "X and y must have the same number of samples"
                )

        starts = self._get_starts(len(X))

        X_windows = []
        y_windows = []

        for i in starts:
            X_windows.append(X[i:i + self.seq_len])

            if y is not None:
                # Window is anomalous if any point is anomalous
                y_windows.append(
                    int(np.any(y[i:i + self.seq_len] == 1))
                )

        X_windows = np.stack(X_windows)

        if y is None:
            return X_windows

        return X_windows, np.asarray(y_windows)


    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)