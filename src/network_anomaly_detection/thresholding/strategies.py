import numpy as np
from sklearn.metrics import f1_score, recall_score

from .base import ThresholdStrategy


class ConstrainedF1Threshold(ThresholdStrategy):

    def __init__(
        self,
        min_recall=0.50,
    ):
        self.min_recall = min_recall
        self.threshold = None

    def fit(
        self,
        val_scores=None, # scores
        y_val=None, # y_true; then ill fit it with y_val
    ):

        if val_scores is None:
            raise ValueError(
                "ConstrainedF1Threshold requires "
                "scores and labels."
            )

        if len(val_scores) != len(y_val):
            raise ValueError(
                "val_scores and y_val must have "
                "the same length."
            )

        thresholds = np.unique(val_scores)

        best_threshold = None
        best_f1 = -np.inf

        for threshold in thresholds:

            predictions = (
                val_scores >= threshold
            )

            recall = recall_score(
                y_val,
                predictions,
                zero_division=0,
            )

            if recall < self.min_recall:
                continue

            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        if best_threshold is None:
            raise ValueError(
                "No threshold satisfies the "
                f"minimum recall constraint: "
                f"{self.min_recall}"
            )

        self.threshold = best_threshold

        return self

    def get_threshold(self):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been fitted."
            )

        return self.threshold