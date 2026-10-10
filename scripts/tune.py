import hydra
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
    

    """
    data = DataModule(
        "data/data.csv",
        "data/data.csv",
    )
    """
    data = DataModule(
        "data/X_train.csv",
        "data/X_test.csv",
    )


    X_train, y_train, X_val, y_val = data.load()

    # only AEs...
    mask = y_train == 0
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]

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

    from network_anomaly_detection.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    logger = MLFlowLogger()


    from network_anomaly_detection.tuning.tuner import Tuner

    tuner = Tuner(
            cfg,
            evaluator,
            logger,
        )


    study = tuner.run(
            prep=prep,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=2,
            n_jobs=2,
        )



    print(study)


    #--------------------------------------#
    print(
        study.best_value
    )

    print(
        study.best_params
    )



    # ========= Retrain best model ========= #
    best_cfg = tuner.get_best_config(study)


    print("\n\n\n\nBest config:")
    print(best_cfg)


    print('training with best confgi')




    experiment = Experiment(
        cfg=best_cfg,
        logger=logger,
    )


    evaluator = Evaluator()

    result = experiment.run(
        prep=prep,
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        run_name=f"{cfg.model_type.name}_baseline",
    )


    """
    exp = Experiment(
        model_type=model_type,
        evaluator=Evaluator(),
        logger=MLFlowLogger(
            tracking_db=to_absolute_path(cfg.paths.mlflow_db),
            artifact_dir=to_absolute_path(cfg.paths.mlflow_artifacts),
        ),
    )
    """




    return






if __name__ == "__main__":
    main()




    