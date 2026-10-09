import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from pathlib import Path




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):

    root_dir = Path(to_absolute_path(cfg.paths.root_dir))
    tracking_db = root_dir / "mlflow.db"
    candidate_db_url = f"sqlite:///{tracking_db}"

   
    registry = CandidateRegistry(candidate_db_url)

    registry.print_candidates(
        experiment_id=1,
        include_evicted=True,
    )



if __name__ == "__main__":
    main()