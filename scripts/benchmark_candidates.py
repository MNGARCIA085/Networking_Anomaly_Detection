import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from network_anomaly_detection.infra.selection.model_selector import ModelSelector
from pathlib import Path
from network_anomaly_detection.inference.benchmarking import benchmark_candidates





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):


    # load data
    """
    data = DataModule(
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.val_path),
        to_absolute_path(cfg.data.y_val_path),
    )
    X_train, X_val, y_val = data.load()
    """

    data = DataModule(
        "data/data.csv",
        "data/data.csv",
    )
    
    _, _, X_val, y_val = data.load()


    root_dir = Path(to_absolute_path(cfg.paths.root_dir))
    tracking_db = root_dir / "mlflow.db"
    candidate_db_url = f"sqlite:///{tracking_db}"

   
    registry = CandidateRegistry(candidate_db_url)

    registry.print_candidates(
        experiment_id=1,
        include_evicted=True,
    )


    print("selected")

    selector = ModelSelector(
        registry
    )



    #------------------
    benchmark_candidates(
        registry=registry,
        experiment_id=1,
        X_benchmark=X_val[:100], # later the 100 came from config
    )


    selected = selector.select(
        experiment_id=1,
        pr_auc_tolerance=0.005, # later from config!!!
    )


    print(selected)
    print(selected.run_id)
    print(selected.model_family)
    print(selected.val_pr_auc)


    #--------
    aux = registry.candidate_records(experiment_id=1)
    print(aux)






if __name__ == "__main__":
    main()