import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from network_anomaly_detection.models.base_model import BaseModel
from network_anomaly_detection.models.persistence.torch import (
    load_torch_model,
    save_torch_model,
)


# ============================================================
# Configuration
# ============================================================

from dataclasses import dataclass

@dataclass
class TransformerAEConfig:
    input_dim: int
    seq_len: int
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 128
    num_encoder_layers: int = 2
    dropout: float = 0.1


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()

        position = torch.arange(
            seq_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, X):
        return X + self.pe[:, :X.size(1)]


# ============================================================
# Transformer Autoencoder
# ============================================================

class TransformerAE(nn.Module):
    def __init__(self, cfg: TransformerAEConfig):
        super().__init__()

        self.config = cfg

        self.input_projection = nn.Linear(
            cfg.input_dim, cfg.d_model
        )

        self.positional_encoding = PositionalEncoding(
            cfg.d_model, cfg.seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
        )

        self.output_projection = nn.Linear(
            cfg.d_model, cfg.input_dim
        )

    def forward(self, X):
        X = self.input_projection(X)
        X = self.positional_encoding(X)
        X = self.encoder(X)
        return self.output_projection(X)


# ============================================================
# Model Interface
# ============================================================

class TransformerAEModel(BaseModel):
    def __init__(self, cfg: dict, input_shape):
        self.config = cfg
        self.input_shape = tuple(input_shape)

        if len(self.input_shape) != 2:
            raise ValueError(
                "TransformerAEModel expects input_shape "
                "(seq_len, num_feats); "
                f"got {self.input_shape}"
            )

        seq_len, num_feats = self.input_shape

        self.model = TransformerAE(
            TransformerAEConfig(
                input_dim=num_feats,
                seq_len=seq_len,
                d_model=cfg.get("d_model", 64),
                nhead=cfg.get("nhead", 4),
                dim_feedforward=cfg.get("dim_feedforward", 128),
                num_encoder_layers=cfg.get(
                    "num_encoder_layers", 2
                ),
                dropout=cfg.get("dropout", 0.1),
            )
        )

    def adapt_input(self, X):
        """Preserve (batch_size, seq_len, num_feats)."""
        return np.asarray(X)

    def score(self, X):
        """Return one reconstruction-error score per window."""
        self.model.eval()

        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32)
            reconstruction = self.model(X)

            return (
                torch.mean(
                    (X - reconstruction) ** 2,
                    dim=(1, 2),
                )
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
        instance.model = load_torch_model(TransformerAE, path)

        return instance