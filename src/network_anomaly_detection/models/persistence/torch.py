from pathlib import Path
import joblib
import torch



def save_torch_model(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(
        model.state_dict(),
        path / "weights.pt"
    )

    # Save model config
    cfg = model.config


    joblib.dump(
        cfg,
        path / "config.pkl"
    )


def load_torch_model(model_cls, path):
    path = Path(path)

    # Load config
    cfg = joblib.load(
        path / "config.pkl"
    )


    # Reconstruct model
    model = model_cls(cfg)

    # Load weights
    model.load_state_dict(
        torch.load(
            path / "weights.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model

