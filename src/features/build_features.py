"""Phase 4 - Feature engineering + time-based train/test split.

Reads the per-zone 15-min time series and turns it into a model-ready table:
  - lag features : demand 1..N intervals ago (the model's main predictive inputs)
  - calendar     : hour, day-of-week, month
  - cyclical     : sin/cos of hour and day-of-week, so 23:00 and 00:00 are neighbours
                   (1 hour apart) instead of 23 apart -- this is upgrade #1.
Then splits BY MONTH (Jan+Feb -> train, March -> test) with NO shuffling.

Note on leakage: lags are computed on the FULL series BEFORE the split, so the first
March rows correctly use late-February demand as their history. That is genuinely in
the past at prediction time, so it is NOT leakage. (Shuffling a time series WOULD leak.)

Run from the project root:
    python -m src.features.build_features
Outputs: data/processed/train.parquet, data/processed/test.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def find_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "params.yaml").exists():
            return p
    raise FileNotFoundError(f"params.yaml not found above {start}")


def add_features(ts: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Add per-zone lag features + calendar + cyclical encodings."""
    ts = ts.sort_values(["region", "tbin"], ignore_index=True)

    # Lag features (within each zone): demand k intervals (k*15 min) ago.
    # This is how a regression "forecasts": next demand ~ f(recent demand).
    for k in range(1, n_lags + 1):
        ts[f"lag_{k}"] = ts.groupby("region")["total_pickups"].shift(k)

    # Calendar features straight from the timestamp.
    ts["hour"] = ts["tbin"].dt.hour                 # 0..23
    ts["day_of_week"] = ts["tbin"].dt.dayofweek     # 0=Mon .. 6=Sun
    ts["month"] = ts["tbin"].dt.month               # 1..3

    # Cyclical encodings: map a circular quantity onto a circle so the ends wrap
    # around (e.g. hour 23 sits right next to hour 0). A linear model can't learn
    # that from the raw integer 23 vs 0.
    ts["hour_sin"] = np.sin(2 * np.pi * ts["hour"] / 24)
    ts["hour_cos"] = np.cos(2 * np.pi * ts["hour"] / 24)
    ts["dow_sin"] = np.sin(2 * np.pi * ts["day_of_week"] / 7)
    ts["dow_cos"] = np.cos(2 * np.pi * ts["day_of_week"] / 7)
    return ts


def main() -> None:
    root = find_root()
    params = yaml.safe_load(open(root / "params.yaml"))
    n_lags = params["features"]["n_lags"]
    train_m = params["split"]["train_months"]
    test_m = params["split"]["test_months"]

    src = root / "data" / "interim" / "region_timeseries.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - run src/features/make_regions.py first")
    ts = pd.read_csv(src, parse_dates=["tbin"])
    n_before = len(ts)

    ts = add_features(ts, n_lags)

    lag_cols = [f"lag_{k}" for k in range(1, n_lags + 1)]
    # Drop rows whose features are undefined: the first n_lags slots per zone (NaN lags)
    # and the first slot per zone (NaN EWMA). ~n_lags rows per zone, all in early Jan.
    na_cols = lag_cols + ["avg_pickups"]
    ts = ts.dropna(subset=na_cols).reset_index(drop=True)
    ts[lag_cols] = ts[lag_cols].astype("int32")

    # Time-based split (no shuffle): Jan+Feb train, March test.
    train = ts[ts["month"].isin(train_m)].reset_index(drop=True)
    test = ts[ts["month"].isin(test_m)].reset_index(drop=True)

    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    # ---- report / sanity checks ----
    assert train[na_cols].isna().sum().sum() == 0, "unexpected NaNs in train features"
    model_features = ["region"] + lag_cols + ["avg_pickups", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    print(f"rows: {n_before:,} -> {len(ts):,}  (dropped {n_before - len(ts)} NaN-lag rows, "
          f"{n_lags} per zone)")
    print(f"train (months {train_m}): {len(train):>7,} rows   "
          f"{train['tbin'].min()} -> {train['tbin'].max()}")
    print(f"test  (months {test_m}): {len(test):>7,} rows   "
          f"{test['tbin'].min()} -> {test['tbin'].max()}")
    print(f"clean time split (train ends before test starts): "
          f"{train['tbin'].max() < test['tbin'].min()}")
    print(f"lag_1 vs total_pickups correlation (train): "
          f"{train['lag_1'].corr(train['total_pickups']):.3f}  (high = lags are predictive)")
    print(f"\nall columns      : {list(ts.columns)}")
    print(f"model features   : {model_features}")
    print(f"target           : total_pickups")


if __name__ == "__main__":
    main()
