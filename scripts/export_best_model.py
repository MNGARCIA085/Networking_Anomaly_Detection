import json
import shutil
from pathlib import Path

import hydra
import mlflow
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # Resolve paths relative to the project root.
    root_dir = Path(to_absolute_path(cfg.paths.root_dir))
    tracking_db = root_dir / "mlflow.db"

    candidate_db_url = f"sqlite:///{tracking_db}"

    # Query the candidate registry.
    registry = CandidateRegistry(candidate_db_url)

    candidates = registry.get_candidates(experiment_id=1)

    if not candidates:
        raise RuntimeError("No candidates found in the registry.")

    best = candidates[0]

    # Adapt this line if your registry returns a dictionary
    # rather than an object with a run_id attribute.
    run_id = (
        best["run_id"]
        if isinstance(best, dict)
        else best.run_id
    )

    print(f"Best run: {run_id}")

    # Use the same MLflow tracking database as the training project.
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    # Export destination.
    model_dir = root_dir / "mock_model_store" / "model_001"

    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True)

    # Download all artifacts logged by the selected run.
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        dst_path=str(model_dir),
    )

    print(f"Artifacts downloaded to: {downloaded_path}")

    # Keep only the artifacts required for inference.
    required_dirs = {"model", "preprocessing", "thresholding"}

    for path in model_dir.iterdir():
        if path.is_dir() and path.name in required_dirs:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    # Read the run's tags to identify the model type.
    run = mlflow.get_run(run_id)
    model_type = run.data.tags.get("model_type", "unknown")

    # Write the manifest.
    manifest = {
        "model": {
            "type": model_type,
            "version": "v1",
            "run_id": run_id,
        }
    }

    manifest_path = model_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to: {manifest_path}")
    print(f"Model bundle exported to: {model_dir}")



if __name__ == "__main__":
    main()