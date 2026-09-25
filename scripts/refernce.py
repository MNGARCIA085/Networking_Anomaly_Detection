import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from anomaly_detection.data.data import DataModule
from anomaly_detection.experiments.experiments import Experiment
from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.infra.logging.mlflow_logger import  MLFlowLogger



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):

    # ---------Data -----------#
    # separate from exp. bc of loading in tuning    
    data = DataModule(
        to_absolute_path(cfg.data.train_path),
        to_absolute_path(cfg.data.val_path),
        to_absolute_path(cfg.data.y_val_path),
    )
    X_train, X_val, y_val = data.load()
    

    model_type = cfg.model_type.name
    print(model_type)


    #------Experiment--------------#
    exp = Experiment(
        model_type=model_type,
        evaluator=Evaluator(),
        logger=MLFlowLogger(
            tracking_db=to_absolute_path(cfg.paths.mlflow_db),
            artifact_dir=to_absolute_path(cfg.paths.mlflow_artifacts),
        ),
    )


    cfg_dict = OmegaConf.to_container(cfg.model_type, resolve=True)


    metrics = exp.run(
        #cfg.model_type,
        cfg_dict,
        X_train,
        X_val,
        y_val
    )



    print(metrics)    


    return



if __name__ == "__main__":
    main()





# hydra multirun: python -m scripts.train -m model_type=ae,vae


"""
# Sequential
python scripts/train.py -m model=ae,transformer_ae

# Parallel
python scripts/train.py -m model=ae,transformer_ae hydra.launcher.n_jobs=2
"""