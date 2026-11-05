# utils/seeding.py
import os, random
import numpy as np

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        import lightning.pytorch as pl
        torch.manual_seed(seed)
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass