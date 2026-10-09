import time
import numpy as np

from .mlflow_loader import load_from_mlflow



#--------Benchmark candidates-------------#
def benchmark_candidates(
    registry,
    experiment_id,
    X_benchmark, # e.g. X_val[:100]; same fixed raw val subset
    repetitions=20,
    
):
    candidates = registry.get_candidates( # retained ones
        experiment_id
    )


    # ---------------------------------------------------------
    # Benchmark each candidate
    # ---------------------------------------------------------

    for candidate in candidates:

        run_id = candidate.run_id

        runner = load_from_mlflow(
            run_id
        )


        inf_benchmark = InferenceBenchmark()

        result = inf_benchmark.measure(
            runner,
            X_benchmark,
            repetitions=repetitions,
        )

        inference_ms = result["avg_ms"]

        registry.update_selection_metrics(
            run_id=run_id,
            inference_ms=inference_ms,
        )

        print(
            f"{candidate.model_family:<15}"
            f"{candidate.val_pr_auc:.4f}    "
            f"{inference_ms:.3f} ms"
        )




#-----------Benchmark--------------#
class InferenceBenchmark:

    def measure(
        self,
        runner,
        X,
        repetitions=20,
        warmup=5,
    ):
        """ Measures inference time"""

        # Warm-up
        for _ in range(warmup):
            runner.predict(X)

        start = time.perf_counter()

        for _ in range(repetitions):
            runner.predict(X)

        elapsed = time.perf_counter() - start

        return {
            "total_seconds": elapsed,
            "avg_ms": (
                elapsed / repetitions * 1000
            ),
        }







"""
For the real benchmark, replace the generated X_benchmark with a fixed subset of your actual 
raw validation data:

X_benchmark = X_val[:100]

That is preferable to random data because every candidate sees the same realistic input distribution.

And raw_input_dim should come from the raw dataset, not from any candidate's preprocessor.
"""

