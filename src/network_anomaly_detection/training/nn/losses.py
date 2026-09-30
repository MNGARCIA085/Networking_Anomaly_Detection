import torch.nn as nn


LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
}



def create_loss(cfg):
    loss_cls = LOSS_REGISTRY[cfg["name"]]

    return loss_cls(
        **cfg.get("params", {})
    )




"""

def create_loss(cfg):

    loss_cls = LOSS_REGISTRY[cfg.name]

    return loss_cls()
"""