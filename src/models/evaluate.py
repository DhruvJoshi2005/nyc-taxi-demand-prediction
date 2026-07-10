"""Phase 5 - Evaluate the DEPLOYED model (models/model.joblib) on the March test set
against the naive baselines, write reports/metrics.json, and save error-analysis
figures. Model-agnostic: whatever is saved as model.joblib gets evaluated.

Run from the project root:
    python -m src.models.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.common import (MODEL_FEATURES, TARGET, baseline_predictions,
                               find_root, load_train_test, regression_metrics)


def main() -> None:
    root = find_root()
    _, te = load_train_test(root)
    model = joblib.load(root / "models" / "model.joblib")

    y = te[TARGET].to_numpy(dtype=float)
    pred = np.clip(model.predict(te[MODEL_FEATURES]), 0, None)

    # ---- metrics.json: baselines vs deployed model ----
    baselines = {n: regression_metrics(y, p) for n, p in baseline_predictions(te).items()}
    model_m = regression_metrics(y, pred)
    best_base_mae = min(m["mae"] for m in baselines.values())
    metrics = {
        "deployed_model": type(model.named_steps["model"]).__name__,
        "test_period": f"{te['tbin'].min()} .. {te['tbin'].max()}",
        "n_test_rows": int(len(te)),
        "model": model_m,
        "baselines": baselines,
        "model_vs_best_baseline_mae_pct": round(100 * (best_base_mae - model_m["mae"]) / best_base_mae, 2),
    }
    (root / "reports").mkdir(exist_ok=True)
    with open(root / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- error analysis ----
    te = te.assign(pred=pred, abs_err=np.abs(y - pred))
    figdir = root / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    demand = te.groupby("region")[TARGET].sum()
    busy, quiet = int(demand.idxmax()), int(demand.idxmin())

    # Fig 1: actual vs predicted over the week of Mon Mar 7, busiest + quietest zone
    wk = te[(te["tbin"] >= "2016-03-07") & (te["tbin"] < "2016-03-14")]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, r, label in [(axes[0], busy, "busiest"), (axes[1], quiet, "quietest")]:
        s = wk[wk["region"] == r].sort_values("tbin")
        ax.plot(s["tbin"], s[TARGET], label="actual", lw=1.6)
        ax.plot(s["tbin"], s["pred"], label="predicted", lw=1.4, alpha=0.85)
        ax.set(title=f"Zone {r} ({label}) - actual vs predicted (week of Mar 7)",
               ylabel="pickups / 15 min")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "pred_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: where is error worst? by hour-of-day and by zone
    by_hour = te.groupby("hour")["abs_err"].mean()
    by_region = te.groupby("region")["abs_err"].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(by_hour.index, by_hour.values, color="#4c72b0")
    axes[0].set(title="MAE by hour of day (error peaks in rush hours)",
                xlabel="hour of day", ylabel="mean abs error")
    axes[1].bar(range(len(by_region)), by_region.values, color="#dd8452")
    axes[1].set(title="MAE by zone (sorted) - a few hard zones dominate",
                xlabel="zones (sorted low -> high)", ylabel="mean abs error")
    fig.tight_layout()
    fig.savefig(figdir / "error_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    print(json.dumps(metrics, indent=2))
    print(f"\nbusiest zone = {busy}, quietest zone = {quiet}")
    print(f"hardest zone (highest MAE) = {int(by_region.idxmax())} (MAE {by_region.max():.1f})")
    print(f"worst hour (highest MAE) = {int(by_hour.idxmax())}:00 (MAE {by_hour.max():.1f})")
    print("wrote reports/metrics.json, reports/figures/pred_vs_actual.png, error_analysis.png")


if __name__ == "__main__":
    main()
