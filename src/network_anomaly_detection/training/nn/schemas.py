from dataclasses import dataclass, field
from typing import List,Any,Dict
import torch.nn as nn
import torch


"""
@dataclass
class TrainState:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = None
    model: Any = None
    stop_training: bool = False
"""


@dataclass
class TrainState:

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    epoch: int = 0
    train_loss: float | None = None
    val_loss: float | None = None

    stop_training: bool = False




@dataclass
class OptimizerConfig:
    name: str
    params: dict = field(default_factory=dict)



@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig
    loss: nn.Module | None

    batch_size: int
    epochs: int

    device: str = "cpu"
    shuffle: bool = True
    num_workers: int = 0

    callbacks: List[Any] = field(default_factory=list)





@dataclass
class TrainingHistory:
    """
    Stores any metric recorded during training.

    Example:
        {
            "train_loss": [0.95, 0.81, 0.72],
            "val_loss":   [1.02, 0.88, 0.79],
            "lr":          [0.001, 0.001, 0.0005]
        }
    """

    metrics: dict[str, list[float]] = field(default_factory=dict)

    def append(self, name: str, value: float):

        self.metrics.setdefault(
            name,
            []
        ).append(float(value))


    def get(self, name: str):

        return self.metrics.get(name, [])


    def as_dict(self):

        return self.metrics