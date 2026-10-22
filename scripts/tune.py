import hydra
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger

from network_anomaly_detection.experiment.experiment import Experiment
from network_anomaly_detection.tuning.tuner import Tuner
from network_anomaly_detection.evaluation.evaluator import Evaluator
    

@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):


    # omegaconf -> dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    data_module = DataModule(
                train_path = "data/processed/v1/train.csv",
                val_path = "data/processed/v1/val.csv",
            )
    X_train, y_train, X_val, y_val = data_module.load_train_val()


    # --------------------------------------------------
    # Tuning
    # --------------------------------------------------
    evaluator = Evaluator()
    logger = MLFlowLogger()

    tuner = Tuner(
            cfg,
            evaluator,
            logger,
        )


    study = tuner.run(
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


    print('Training with best config')


    experiment = Experiment(
        cfg=best_cfg,
        logger=logger,
    )


    evaluator = Evaluator()

    # i have to create and fit my best prep

    result = experiment.run(
        prep=prep,
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        #run_name=f"{cfg.model_type.name}_tunned",
    )



    return






if __name__ == "__main__":
    main()




    





