"""Phase 5 - Train the demand model, compare it to the naive baselines, and log
every approach to a local MLflow experiment.

Approaches compared on the March test set:
  - baseline_lag1  : predict = previous interval's demand
  - baseline_ewma  : predict = EWMA recent average
  - linear_regression / ridge / random_forest

LinearRegression is the DEPLOYED model (saved to models/model.joblib) because it is
accurate, fast, and fully explainable - we do not want a black box for a demand model.

Run from the project root:
    python -m src.models.train
    mlflow ui --backend-store-uri ./mlruns      # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import os
# MLflow 3.x put the simple file-store ('./mlruns') in maintenance mode and errors
# unless we opt in. We only need a local experiment logbook (no registry), so opt in.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.models.common import (MODEL_FEATURES, TARGET, baseline_predictions,
                               build_pipeline, find_root, load_train_test,
                               regression_metrics)


def main() -> None:
    root = find_root()
    tr, te = load_train_test(root)
    Xtr, ytr = tr[MODEL_FEATURES], tr[TARGET]
    Xte, yte = te[MODEL_FEATURES], te[TARGET]
    print(f"train {Xtr.shape} | test {Xte.shape}")

    results: dict[str, dict] = {}

    # 1) baselines (no training)
    for name, pred in baseline_predictions(te).items():
        results[name] = regression_metrics(yte, pred)

    # 2) candidate models (transparent comparison - not a 50-trial black box)
    candidates = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42),
    }
    fitted = {}
    for name, est in candidates.items():
        print(f"  training {name} ...")
        pipe = build_pipeline(est).fit(Xtr, ytr)
        results[name] = regression_metrics(yte, pipe.predict(Xte))
        fitted[name] = pipe

    # 3) MLflow: one run per approach so they are directly comparable in the UI
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    mlflow.set_experiment("nyc-taxi-demand")
    for name, m in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_metrics(m)
            if name in candidates:
                mlflow.log_param("model_type", name)
                mlflow.log_param("n_features", len(MODEL_FEATURES))
                if name == "linear_regression":
                    mlflow.sklearn.log_model(fitted[name], name="model")

    # 4) deploy the explainable model
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(fitted["linear_regression"], root / "models" / "model.joblib")

    # 5) comparison table + headline improvement over the best baseline
    tbl = pd.DataFrame(results).T[["mae", "rmse", "mape_nonzero_pct"]].sort_values("mae")
    print("\n=== test-set comparison (sorted by MAE, lower is better) ===")
    print(tbl.round(3).to_string())
    best_base = min(results["baseline_lag1"]["mae"], results["baseline_ewma"]["mae"])
    lr_mae = results["linear_regression"]["mae"]
    print(f"\nLinearRegression MAE {lr_mae:.3f}  vs  best baseline MAE {best_base:.3f}"
          f"  ->  {100 * (best_base - lr_mae) / best_base:+.1f}% (negative = better)")
    print("saved models/model.joblib")


if __name__ == "__main__":
    main()
