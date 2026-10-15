
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    label_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
):
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.")

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All split sizes must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    rows_before = len(df)

    # Normalize column names.
    df.columns = df.columns.str.strip()

    if not df.columns.is_unique:
        raise ValueError("Duplicate columns after stripping whitespace.")

    if label_column not in df.columns:
        raise ValueError(f"Missing target column: {label_column}")

    # Validate and encode labels.
    labels = df[label_column].astype("string").str.strip()

    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Missing or empty target labels.")

    labels = labels.str.upper()

    if "BENIGN" not in labels.unique():
        raise ValueError("No BENIGN labels found.")

    df[label_column] = labels.ne("BENIGN").astype("int8")

    # Validate numeric features.
    feature_columns = [c for c in df.columns if c != label_column]

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Standardize infinities; imputation is handled by prep.
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Chronological split: preserve the existing row order.
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }

    split_metadata = {}

    for name, split_df in splits.items():
        output_path = output_dir / f"{name}.csv"
        split_df.to_csv(output_path, index=False)

        split_metadata[name] = {
            "file": output_path.name,
            "sha256": sha256(output_path),
            "row_start": int(
                0 if name == "train"
                else train_end if name == "val"
                else val_end
            ),
            "row_end_exclusive": int(
                train_end if name == "train"
                else val_end if name == "val"
                else n
            ),
            "rows": len(split_df),
            "class_counts": {
                str(int(label)): int(count)
                for label, count
                in split_df[label_column].value_counts().items()
            },
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "source_sha256": sha256(input_path),
        "git_commit": git_commit(),
        "label_column": label_column,
        "label_mapping": {
            "BENIGN": 0,
            "all_other_labels": 1,
        },
        "infinity_policy": "replace_with_nan",
        "missing_value_policy": "preserve_for_prep",
        "rows_before": rows_before,
        "rows_after": len(df),
        "feature_columns": feature_columns,
        "missing_values_per_feature": {
            col: int(count)
            for col, count in df[feature_columns].isna().sum().items()
            if count > 0
        },
        "split": {
            "strategy": "temporal",
            "ordering": "original CSV row order; no shuffling",
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "boundaries": {
                "train_end": train_end,
                "val_end": val_end,
                "total_rows": n,
            },
            "splits": split_metadata,
        },
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset saved to: {output_dir}")
    for name, split_df in splits.items():
        print(f"{name}: {len(split_df):,} rows")




@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    prepare_dataset(
        input_path=Path(to_absolute_path(cfg.data.raw_path)),
        output_dir=Path(to_absolute_path(cfg.data.processed_dir)),
        label_column=cfg.data.label_column,
        train_size=float(cfg.data.split.train_size),
        val_size=float(cfg.data.split.val_size),
        test_size=float(cfg.data.split.test_size),
    )


if __name__ == "__main__":
    main()
