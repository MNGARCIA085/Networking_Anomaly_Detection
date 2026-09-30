from .implementations import AdamOptimizer, SGDOptimizer


OPTIMIZER_REGISTRY = {
    "adam": AdamOptimizer,
    "sgd": SGDOptimizer,
}


def create_optimizer(cfg, parameters):

    optimizer_cls = OPTIMIZER_REGISTRY[cfg["name"]]

    return optimizer_cls.create(
        parameters,
        cfg,
    )
