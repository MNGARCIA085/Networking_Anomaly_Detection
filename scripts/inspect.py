import joblib
import mlflow
import numpy as np


def inspect_pipeline(run_id: str) -> None:
    """Inspect a fitted preprocessing pipeline saved as an MLflow artifact."""

    # Download the fitted preprocessing pipeline.
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing/prep.pkl",
    )
    prep = joblib.load(local_path)

    # Inspect pointwise preprocessing.
    print("\n=== POINTWISE PREPROCESSING ===")

    pointwise = prep.pointwise_prep
    steps = pointwise.steps

    for i, step in enumerate(steps):
        print(f"\nStep {i}: {step.__class__.__name__}")
        print(f"Parameters: {step.get_params()}")

        if hasattr(step, "n_features_in_"):
            print(f"Input features: {step.n_features_in_}")

        if hasattr(step, "feature_names_in_"):
            print(f"Input feature names: {step.feature_names_in_.tolist()}")

        # Show learned attributes useful for inspecting fitted transformers.
        for attr in ("statistics_", "mean_", "scale_"):
            if hasattr(step, attr):
                values = getattr(step, attr)
                print(f"{attr}: {values}")

        # Inspect feature selection results.
        if hasattr(step, "get_support"):
            support = step.get_support()

            print(f"Features before selection: {len(support)}")
            print(f"Features retained: {int(support.sum())}")
            print(f"Features removed: {int((~support).sum())}")

            # Recover original column names from the imputer.
            # The scaler may output NumPy arrays, so the selector may not
            # have feature_names_in_ of its own.
            if steps and hasattr(steps[0], "feature_names_in_"):
                feature_names = steps[0].feature_names_in_

                if len(feature_names) == len(support):
                    print("Selected features:")
                    print(feature_names[support].tolist())

                    print("Removed features:")
                    print(feature_names[~support].tolist())
                else:
                    print(
                        "Feature names unavailable: name count does not "
                        "match the selector's input dimensions."
                    )

    # Inspect temporal preprocessing.
    print("\n=== TEMPORAL PREPROCESSING ===")

    temporal = getattr(prep, "temporal_prep", None)

    if temporal is None:
        print("Disabled")
    else:
        print(temporal)

    # Inspect windowing configuration.
    print("\n=== WINDOWING ===")

    windowing = getattr(prep, "windowing", None)

    if windowing is None:
        print("Disabled")
    else:
        print(windowing)

        if hasattr(windowing, "__dict__"):
            print("Configuration:", vars(windowing))


if __name__ == "__main__":
    run_id = "b3099800d92d4764ab606ae079f16f08"
    inspect_pipeline(run_id)