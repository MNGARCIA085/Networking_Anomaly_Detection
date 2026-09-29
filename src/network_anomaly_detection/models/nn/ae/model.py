import numpy as np
import torch
from torch import nn

from network_anomaly_detection.models.base_model import BaseModel
from .schemas import AEConfig


class AEModel(BaseModel):

    def __init__(self, cfg: AEConfig):
        super().__init__()

        self.config = cfg

        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim

        for dim in cfg.encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim

        encoder_layers = encoder_layers[:-1]

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.encoder_dims[-1]

        for dim in cfg.decoder_dims:
            decoder_layers.append(nn.Linear(in_dim, dim))
            decoder_layers.append(nn.ReLU())
            in_dim = dim

        decoder_layers.append(
            nn.Linear(in_dim, cfg.input_dim)
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)

    def forward(self, X):
        return self.decoder(
            self.encoder(X)
        )

    def score(self, X):

        self.eval()

        with torch.no_grad():

            reconstruction = self(X)

            return torch.mean(
                (X - reconstruction) ** 2,
                dim=1,
            ).cpu().numpy()




"""
X_train = prep.transform(X_train, y_train)

X_train = model.adapt_input(X_train)

trainer.fit(model, X_train)

X_val = prep.transform(X_val, y_val)

X_val = model.adapt_input(X_val)

scores = model.score(X_val)
"""