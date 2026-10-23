from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from network_anomaly_detection.models.base_model import BaseModel
from network_anomaly_detection.models.nn.vae.schemas import VAEConfig
from network_anomaly_detection.models.persistence.torch import (
    load_torch_model,
    save_torch_model,
)

class VAE(nn.Module):
    def __init__(self, cfg):
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

        # Latent distribution parameters
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim

        for dim in cfg.decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, dim),
                nn.ReLU(),
            ])
            in_dim = dim

        decoder_layers.append(
            nn.Linear(in_dim, cfg.input_dim)
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, X):
        h = self.encoder(X)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        mu, logvar = self.encode(X)
        Z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(Z)

        return reconstruction, mu, logvar






class VAEModel(BaseModel):
    def __init__(self, cfg: dict, input_shape):
        self.config = cfg
        self.input_shape = tuple(input_shape)

        input_dim = int(np.prod(self.input_shape))

        self.model = VAE(
            VAEConfig(
                input_dim=input_dim,
                encoder_dims=cfg["encoder_dims"],
                decoder_dims=cfg["decoder_dims"],
                latent_dim=cfg["latent_dim"],
            )
        )

    def adapt_input(self, X):
        """Flatten each sample while preserving the batch dimension."""
        return np.asarray(X).reshape(len(X), -1)

    def encode(self, X):
        """Return latent means for already-adapted inputs."""
        self.model.eval()

        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32)
            mu, _ = self.model.encode(X)

            return mu.cpu().numpy()

    def score(self, X):
        """Return reconstruction-error anomaly scores."""
        self.model.eval()

        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32)

            # Use the latent mean for deterministic scoring.
            mu, _ = self.model.encode(X)
            reconstruction = self.model.decode(mu)

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
        instance.model = load_torch_model(VAE, path)

        return instance