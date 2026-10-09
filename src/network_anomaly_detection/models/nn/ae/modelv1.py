import numpy as np
import torch
from torch import nn

from network_anomaly_detection.models.base_model import BaseModel
from .schemas import AEConfig


from network_anomaly_detection.models.persistence.torch import save_torch_model,load_torch_model


from pathlib import Path
import joblib



class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()

        self.config = cfg

        # Encoder
        # Encoder with non-linear bottleneck
        encoder_layers = []
        in_dim = cfg.input_dim

        for dim in cfg.encoder_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            in_dim = dim

        # Final projection to bottleneck layer
        encoder_layers.append(nn.Linear(in_dim, cfg.encoder_dims[-1]))
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

    def forward(self, X):
        return self.decoder(
            self.encoder(X)
        )





class AEModel(BaseModel):

    def __init__(self, cfg: dict, input_shape):
        self.config = cfg

        # new
        self.input_shape = tuple(input_shape)

        input_dim = int(np.prod(input_shape))

        self.model = AE(
            AEConfig(
                input_dim=input_dim,
                encoder_dims=cfg["encoder_dims"],
                decoder_dims=cfg["decoder_dims"],
            )
        )


    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)



    """
    def score(self, X):

        self.model.eval()

        with torch.no_grad():

            X = torch.tensor(
                X,
                dtype=torch.float32,
            )

            reconstruction = self.model(X)

            return torch.mean(
                (X - reconstruction) ** 2,
                dim=1,
            ).cpu().numpy()
    """

    # top-k feature MSE
    def score(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            reconstruction = self.model(X_tensor)
            
            # Absolute difference per feature: (N, 79)
            feature_errors = torch.abs(X_tensor - reconstruction)
            
            # Take the mean of the top 10 highest feature errors
            top_k_errors, _ = torch.topk(feature_errors, k=10, dim=1)
            return top_k_errors.mean(dim=1).cpu().numpy()


    # predict
    def predict(self, X, threshold):
        scores = self.score(X)
        return (scores >= threshold).astype(int)



    # save & load
    def save(self, path):
        path = Path(path)
        save_torch_model(self.model, path)
        joblib.dump(
            {"config": self.config, "input_shape": self.input_shape},
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


