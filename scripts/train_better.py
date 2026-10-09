import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger

from network_anomaly_detection.experiment.experiment import Experiment



@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    # --------------------------------------------------
    # Data
    # --------------------------------------------------


    """
    data = DataModule(
        "data/arriba.csv",
        "data/arriba.csv",
    )
    """
    

    
    data = DataModule(
        "data/data.csv",
        "data/data.csv",
    )
    


    X_train, y_train, X_val, y_val = data.load()

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    prep_cls = PREP_REGISTRY[
        cfg.model_type.name
    ]

    prep = prep_cls(
        cfg.model_type,
    )

    # Fit preprocessing ONLY on training data.
    prep.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Experiment
    # --------------------------------------------------

    
    #logger = MLFlowLogger()
    logger = MLFlowLogger(
        tracking_db=to_absolute_path(cfg.paths.mlflow_db),
        artifact_dir=to_absolute_path(cfg.paths.mlflow_artifacts),
    )



    #print(type(cfg))
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(cfg, resolve=True)


    experiment = Experiment(
        cfg=cfg,
        logger=logger,
    )


    from network_anomaly_detection.evaluation.evaluator import Evaluator

    evaluator = Evaluator()

    result = experiment.run(
        prep=prep, # already fit
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        run_name='train',
        #run_name=f"{cfg.model_type.name}_baseline",
    )

    return result


if __name__ == "__main__":
    main()




    