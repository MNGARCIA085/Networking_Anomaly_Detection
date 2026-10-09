from pathlib import Path
import numpy as np

from network_anomaly_detection.data.data_module import DataModule    
from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from network_anomaly_detection.infra.selection.model_selector import ModelSelector
from network_anomaly_detection.inference.benchmarking import benchmark_candidates

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from network_anomaly_detection.inference.mlflow_loader import load_from_mlflow

from network_anomaly_detection.evaluation.evaluator import Evaluator



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):

    # load test data
    #X_test = np.load(to_absolute_path(cfg.data.test_path))
    #y_test = np.load(to_absolute_path(cfg.data.y_test_path))

    data = DataModule(
        "data/data.csv",
        "data/data.csv",
    )
    
    _, _, X_test, y_test = data.load()


    # select best model (first in my tbale, i should add a rank column)
    root_dir = Path(to_absolute_path(cfg.paths.root_dir))
    tracking_db = root_dir / "mlflow.db"


    print(tracking_db)

    candidate_db_url = f"sqlite:///{tracking_db}"

   
    registry = CandidateRegistry(candidate_db_url)

    best = registry.get_candidates(
        experiment_id=1,
    )[0]
    
    print(best.run_id)


    # load best model
    runner = load_from_mlflow(best.run_id)

    evaluator = Evaluator()




    # evaluate
    scores, y_test_w, predictions = runner.predict_with_labels(
        X_test,
        y_test,
    )

    metrics = evaluator.evaluate(
        scores=scores,
        y_true=y_test_w,
        predictions=predictions,
    )
    print(metrics)


    #-----------------see later---------#    
    print(scores.shape)
    print(X_test.shape)
    print(y_test.shape)


    print(runner.prep.windowing.seq_len)


    #print(runner.thresholding....)

    

    """
    window_size = runner.windowing.seq_len
    y_test_aligned = y_test[window_size - 1:]
    assert len(scores) == len(y_test_aligned)
    """







if __name__ == "__main__":
    main()


#python scripts/final_evaluation.py


#python -c "import anomaly_detection; print(anomaly_detection.__file__)"