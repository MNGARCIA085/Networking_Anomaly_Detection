from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def fit(
        self,
        X,
        y=None,
        X_val=None,
        y_val=None,
    ):
        raise NotImplementedError