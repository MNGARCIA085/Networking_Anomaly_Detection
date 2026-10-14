import torch


SCHEDULER_REGISTRY = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}


def create_scheduler(cfg, optimizer):
    if not cfg:
        return None

    name = cfg["name"]

    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {name}")

    scheduler_cls = SCHEDULER_REGISTRY[name]
    scheduler = scheduler_cls(
        optimizer,
        **cfg.get("params", {}),
    )

    return {
        "scheduler": scheduler,
        "monitor": cfg.get("monitor", "val_loss"),
        "interval": cfg.get("interval", "epoch"),
        "frequency": cfg.get("frequency", 1),
        "name": name,
    }
