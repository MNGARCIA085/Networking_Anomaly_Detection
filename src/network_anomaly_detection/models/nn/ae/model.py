from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from network_anomaly_detection.models.base_model import BaseModel
from network_anomaly_detection.models.persistence.torch import (
    load_torch_model,
    save_torch_model,
)

from .schemas import AEConfig


class AE(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.config = cfg

        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim

        for dim in cfg.encoder_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, dim),
                nn.ReLU(),
            ])
            in_dim = dim

        self.encoder = nn.Sequential(*encoder_layers[:-1])

        # Decoder
        decoder_layers = []
        in_dim = cfg.encoder_dims[-1]

        for dim in cfg.decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, dim),
                nn.ReLU(),
            ])
            in_dim = dim

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X):
        return self.decoder(self.encoder(X))





class AEModel(BaseModel):
    def __init__(self, cfg: dict, input_shape):
        self.config = cfg
        self.input_shape = tuple(input_shape)

        input_dim = int(np.prod(self.input_shape))

        self.model = AE(
            AEConfig(
                input_dim=input_dim,
                encoder_dims=cfg["encoder_dims"],
                decoder_dims=cfg["decoder_dims"],
            )
        )

    def adapt_input(self, X):
        """Flatten each sample while preserving the batch dimension."""
        return np.asarray(X).reshape(len(X), -1)

    def encode(self, X):
        """Return latent representations for already-adapted inputs."""
        self.model.eval()

        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32)
            return self.model.encoder(X).cpu().numpy()

    def score(self, X):
        """Return reconstruction-error anomaly scores."""
        self.model.eval()

        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32)
            reconstruction = self.model(X)

            return (
                torch.mean((X - reconstruction) ** 2, dim=1)
                .cpu()
                .numpy()
            )

    def predict(self, X, threshold):
        return (self.score(X) >= threshold).astype(int)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_torch_model(self.model, path)

        joblib.dump(
            {
                "config": self.config,
                "input_shape": self.input_shape,
            },
            path / "wrapper_config.pkl",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        wrapper_cfg = joblib.load(path / "wrapper_config.pkl")

        instance = cls.__new__(cls)
        instance.config = wrapper_cfg["config"]
        instance.input_shape = tuple(wrapper_cfg["input_shape"])
        instance.model = load_torch_model(AE, path)

        return instance


    #---------config------------#
    def get_config(self): # later put as abstract in the base class!!!!!
        return {
            "model_type": "ae",
            "input_shape": list(self.input_shape),
            "input_dim": self.model.config.input_dim,
            "encoder_dims": list(self.model.config.encoder_dims),
            "decoder_dims": list(self.model.config.decoder_dims),
        }