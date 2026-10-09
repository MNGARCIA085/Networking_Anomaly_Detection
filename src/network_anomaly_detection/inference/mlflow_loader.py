from pathlib import Path

from mlflow.tracking import MlflowClient

from .loader import _build_runner


def load_from_mlflow(run_id):

    client = MlflowClient()

    run = client.get_run(run_id)

    local_dir = Path(
        client.download_artifacts(run_id, "")
    )

    return _build_runner(
        model_dir=local_dir,
        model_type=run.data.tags["model_type"],
    )