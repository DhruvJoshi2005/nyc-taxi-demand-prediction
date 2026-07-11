"""Serving logic for the Streamlit app, kept separate from the UI so it can be tested
without a running Streamlit server.

The app does NOT re-cluster anything: it reads the pre-built March test table (which
already has each zone's features for every 15-min slot) and just runs the chosen model.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.common import DEPLOYABLE, MODEL_FEATURES, TARGET, model_path


def load_models(root: Path) -> dict:
    return {name: joblib.load(model_path(root, name)) for name in DEPLOYABLE}


def load_data(root: Path):
    test = pd.read_parquet(root / "data" / "processed" / "test.parquet")
    plot = pd.read_csv(root / "data" / "external" / "plot_data.csv")
    return test, plot


def predict_zone_demand(test: pd.DataFrame, model, tbin: pd.Timestamp) -> pd.DataFrame:
    """For one 15-min slot, return each zone's predicted vs actual pickups."""
    rows = test[test["tbin"] == tbin].sort_values("region")
    if rows.empty:
        return pd.DataFrame(columns=["region", TARGET, "predicted"])
    preds = np.clip(model.predict(rows[MODEL_FEATURES]), 0, None)
    return rows[["region", TARGET]].assign(predicted=preds).reset_index(drop=True)


def make_demand_map(plot: pd.DataFrame, zone_pred: pd.DataFrame, cmap: str = "plasma"):
    """Colour each sampled pickup point by its zone's PREDICTED demand -> a live
    heat-map of the city. Returns a matplotlib Figure."""
    demand = dict(zip(zone_pred["region"], zone_pred["predicted"]))
    colours = plot["region"].map(demand).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 7.6))
    sc = ax.scatter(plot["pickup_longitude"], plot["pickup_latitude"], c=colours,
                    s=4, cmap=cmap, alpha=0.55, linewidths=0)
    ax.set(title="Predicted pickups per zone (next 15 min)",
           xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1.3)
    fig.colorbar(sc, ax=ax, label="predicted pickups")
    fig.tight_layout()
    return fig
