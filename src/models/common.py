"""Shared helpers for Phase 5 (training & evaluation) so the two scripts agree on
features, the model pipeline, and the metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "total_pickups"
CATEGORICAL = ["region"]                         # one-hot encoded
NUMERIC = ["lag_1", "lag_2", "lag_3", "lag_4",   # recent demand
           "avg_pickups",                        # EWMA recent average
           "hour_sin", "hour_cos",               # cyclical time-of-day (upgrade #1)
           "dow_sin", "dow_cos"]                 # cyclical day-of-week
MODEL_FEATURES = CATEGORICAL + NUMERIC


def find_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "params.yaml").exists():
            return p
    raise FileNotFoundError(f"params.yaml not found above {start}")


def load_train_test(root: Path):
    tr = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    te = pd.read_parquet(root / "data" / "processed" / "test.parquet")
    return tr, te


def build_pipeline(estimator) -> Pipeline:
    """One-hot the region, pass the numeric features through, then the estimator.
    Bundling preprocessing + model in ONE object prevents train/serve skew."""
    pre = ColumnTransformer(
        [("region_ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, and MAPE-on-nonzero. Predictions are clipped at 0 (demand can't be
    negative). MAPE is computed only where y_true > 0 because it divides by y_true."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    nz = y_true > 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_nonzero_pct": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100),
    }


def baseline_predictions(df: pd.DataFrame) -> dict:
    """The two naive baselines as prediction arrays (no training needed)."""
    return {
        "baseline_lag1": df["lag_1"].to_numpy(dtype=float),       # predict = last interval
        "baseline_ewma": df["avg_pickups"].to_numpy(dtype=float),  # predict = recent EWMA
    }
