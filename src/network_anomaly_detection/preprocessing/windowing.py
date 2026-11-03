import joblib
import numpy as np


class Windowing:

    def __init__(
        self,
        seq_len: int,
        stride: int = 1,
        strategy: str = "last",
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        if stride < 1:
            raise ValueError("stride must be >= 1")

        if strategy not in {"last", "any", "majority"}:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from: 'last', 'any', 'majority'"
            )

        self.seq_len = seq_len
        self.stride = stride
        self.strategy = strategy

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

    def _label_window(self, y_window):
        if self.strategy == "last":
            return y_window[-1]

        if self.strategy == "any":
            return int(np.any(y_window == 1))

        # Majority: positive only if more than half are anomalous.
        return int(np.mean(y_window == 1) > 0.5)

    def transform(self, X, y=None):
        """Create temporal windows and optionally their labels."""
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

        X_windows = np.stack([
            X[i:i + self.seq_len]
            for i in starts
        ])

        if y is None:
            return X_windows

        y_windows = np.asarray([
            self._label_window(y[i:i + self.seq_len])
            for i in starts
        ])

        return X_windows, y_windows

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


"""
windowing = Windowing(seq_len=20, stride=1, strategy="last")
windowing = Windowing(seq_len=20, strategy="any")
windowing = Windowing(seq_len=20, strategy="majority")
"""