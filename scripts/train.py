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

    data_module = DataModule(
                train_path = "data/processed/v1/train.csv",
                val_path = "data/processed/v1/val.csv",
            )


    X_train, y_train, X_val, y_val = data_module.load_train_val()


    print(X_train.shape)
    print(X_val.shape)

    import pandas as pd

    for name, y in [("Train", y_train), ("Validation", y_val)]:
        proportions = y.value_counts(normalize=True).sort_index()
        counts = y.value_counts().sort_index()

        print(f"\n{name} — {len(y):,} samples")
        print(pd.DataFrame({
            "count": counts,
            "proportion": proportions,
        }).fillna(0).to_string())



    # Final evaluation, when appropriate
    #X_test, y_test = data_module.load_split("test")




    #print("Train labels:", y_train.value_counts().to_dict())
    #print("Validation labels:", y_val.value_counts().to_dict())



    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    prep_cls = PREP_REGISTRY[
        cfg.model_type.name
    ]

    prep = prep_cls(
        cfg.model_type,
    )


    # Prep handles model-specific sample selection
    # For ex, AEs are trained only with normal data
    X_train, y_train = prep.select_training_data(X_train, y_train)

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



    print(len(X_train))


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






# hydra multirun: python -m scripts.train -m model_type=ae,vae


"""
# Sequential
python scripts/train.py -m model=ae,transformer_ae

# Parallel
python scripts/train.py -m model=ae,transformer_ae hydra.launcher.n_jobs=2
"""