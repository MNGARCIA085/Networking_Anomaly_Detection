import joblib
from .registry import create_threshold_strategy

class Thresholding:

    def __init__(self, config):

        self.config = config
        self.strategy = None

        if config:

            self.strategy = (
                create_threshold_strategy(
                    config["name"],
                    **config.get("params", {}),
                )
            )

    def fit(self, scores, y_val): # y_true more than y_val

        if self.strategy is not None:
            self.strategy.fit(scores, y_val)

        return self

    def get_threshold(self):

        if self.strategy is None:
            return None

        return self.strategy.get_threshold()



    # --------save and load--------#
    def save(self, path):

        joblib.dump(
            self,
            path,
        )

    @classmethod
    def load(cls, path):

        return joblib.load(path)