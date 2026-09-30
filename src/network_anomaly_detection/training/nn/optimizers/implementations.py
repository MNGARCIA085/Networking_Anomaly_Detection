from abc import ABC, abstractmethod
import torch


class BaseOptimizer(ABC):

    @staticmethod
    @abstractmethod
    def sample(trial, cfg):
        """Return optimizer config dictionary."""
        pass

    @staticmethod
    @abstractmethod
    def create(parameters, cfg):
        """Return instantiated torch optimizer."""
        pass



class AdamOptimizer(BaseOptimizer):

    @staticmethod
    def sample(trial, cfg):

        return {
            "name": "adam",
            "params": {
                "lr": trial.suggest_float(
                    "optimizer.lr",
                    cfg.lr.low,
                    cfg.lr.high,
                    log=cfg.lr.log,
                ),
                "betas": trial.suggest_categorical(
                    "optimizer.betas",
                    cfg.betas.choices,
                ),
            },
        }

    @staticmethod
    def create(parameters, cfg):

        return torch.optim.Adam(
            parameters,
            **cfg["params"],
        )






class SGDOptimizer(BaseOptimizer):

    @staticmethod
    def sample(trial, cfg):

        return {
            "name": "sgd",
            "params": {
                "lr": trial.suggest_float(
                    "optimizer.lr",
                    cfg.lr.low,
                    cfg.lr.high,
                    log=cfg.lr.log,
                ),
                "momentum": trial.suggest_float(
                    "optimizer.momentum",
                    cfg.momentum.low,
                    cfg.momentum.high,
                ),
                "weight_decay": trial.suggest_float(
                    "optimizer.weight_decay",
                    cfg.weight_decay.low,
                    cfg.weight_decay.high,
                    log=cfg.weight_decay.log,
                ),
            },
        }

    @staticmethod
    def create(parameters, cfg):

        return torch.optim.SGD(
            parameters,
            **cfg["params"],
        )