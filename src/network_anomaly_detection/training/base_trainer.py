from abc import ABC, abstractmethod


class BaseTrainer(ABC):



    def __init__(self, model, cfg=None, checkpoint_dir=None):
        self.model = model
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.history = None

    @abstractmethod
    def fit(
        self,
        X,
        y=None,
        X_val=None,
        y_val=None,
    ):
        raise NotImplementedError