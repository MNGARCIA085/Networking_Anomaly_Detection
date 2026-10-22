import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY
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
    cfg = OmegaConf.to_container(cfg, resolve=True)

    data_module = DataModule(
        train_path="data/processed/v1/train.csv",
        val_path="data/processed/v1/val.csv",
    )
    X_train, y_train, X_val, y_val = data_module.load_train_val()

    evaluator = Evaluator()
    logger = MLFlowLogger()

    with logger.start_run(
        run_name=f"{cfg['model_type']['name']}_tuning"
    ):
        logger.log_tags({
            "run_type": "tuning",
            "status": "running",
        })

        try:


            # pass only what it needs
            cfg.pop('paths', None)


            tuner = Tuner(cfg, evaluator, logger)

            study = tuner.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=1,
                n_jobs=1,
            )

            best_cfg = tuner.get_best_config(study)
            best_trial = study.best_trial
 

            mlflow.log_dict(
                best_cfg,
                artifact_file="best_config.json",
            )

            mlflow.log_metric("best_pr_auc", study.best_value)
            mlflow.log_param("best_trial_number", best_trial.number)


            logger.log_tags({"status": "finished"})


        except Exception as error:
            logger.log_tags({
                "status": "failed",
                "error": str(error),
            })
            raise


    # ========= Retrain best model ========= #
    print('\n Retrain with best model\n')
    prep_cls = PREP_REGISTRY[best_cfg["model_type"]["name"]]
    prep = prep_cls(best_cfg["model_type"]) # later pass more specific conf

    X_train_selected, y_train_selected = (
        prep.select_training_data(X_train, y_train)
    )
    prep.fit(X_train_selected, y_train_selected)


    experiment = Experiment(cfg=best_cfg, logger=logger)
    result = experiment.run(
        prep=prep,
        evaluator=evaluator,
        X_train=X_train_selected,
        y_train=y_train_selected,
        X_val=X_val,
        y_val=y_val,
        run_name="best_config_retrain",
    )

    

    print("Best value:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Best config:", best_cfg)


    return



if __name__ == "__main__":
    main()





