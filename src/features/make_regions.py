"""Phase 3 - Regions (clustering) + per-region 15-minute time series.

Pipeline:
  1. Fit StandardScaler + MiniBatchKMeans(30) on a representative sample of pickup
     coordinates (30 stable zone-centers don't need all 34M points).
  2. Validate the choice of K=30 via a neighbour-distance rule (haversine miles).
  3. Assign a `region` to EVERY trip out-of-core with Dask, count pickups per zone
     per 15-min slot (keeping true zeros), and add a shift-by-1 EWMA feature.
  4. Save the fitted models, the time series, a colored point sample for the app map,
     and a verification figure of the zones.

Run from the project root:
    python -m src.features.make_regions            # full run (reads clean.parquet)
    python -m src.features.make_regions --sample   # smoke test (reads clean_sample.parquet)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed - just save figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import dask.dataframe as dd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances

COORD_COLS = ["pickup_longitude", "pickup_latitude"]
EARTH_RADIUS_MI = 3958.8


def find_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "params.yaml").exists():
            return p
    raise FileNotFoundError(f"params.yaml not found above {start}")


def fit_clusterer(ddf, params: dict, fit_rows: int):
    """Fit StandardScaler + MiniBatchKMeans on a random sample of coordinates.

    Fitting on a sample (not all 34M rows) gives the same stable 30 centres far
    faster, with n_init=10 for a good, reproducible result. MiniBatchKMeans is the
    scalable, mini-batch variant of KMeans (it also supports .partial_fit() to stream
    on-disk chunks if the data ever exceeds RAM).
    """
    k = params["regions"]["n_clusters"]
    seed = params["regions"]["random_state"]
    n = int(ddf.shape[0].compute())
    frac = min(1.0, fit_rows / n)
    sample = ddf[COORD_COLS].sample(frac=frac, random_state=seed).compute()
    print(f"  fitting on {len(sample):,} sampled coordinates (of {n:,}) ...")

    scaler = StandardScaler().fit(sample.to_numpy())
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=params["regions"]["n_init"],
                             random_state=seed)
    kmeans.fit(scaler.transform(sample.to_numpy()))
    # Inference (.predict) needs only the 30 cluster centres; drop the ~2M training
    # labels so the saved model is a few KB instead of megabytes.
    kmeans.labels_ = None
    return scaler, kmeans


def validate_k(scaler, kmeans) -> float:
    """Neighbour-distance justification for K: mean haversine miles from each zone
    centre to its 8 nearest zone centres. Should be roughly city-district sized."""
    centers = scaler.inverse_transform(kmeans.cluster_centers_)      # (K, 2) = lon,lat
    latlon_rad = np.radians(centers[:, [1, 0]])                       # -> (lat, lon)
    d_mi = haversine_distances(latlon_rad) * EARTH_RADIUS_MI          # pairwise miles
    np.fill_diagonal(d_mi, np.inf)
    nearest8 = np.sort(d_mi, axis=1)[:, :8]
    return float(nearest8.mean())


def _assign_region_and_bin(part: pd.DataFrame, scaler, kmeans) -> pd.DataFrame:
    """Runs on one Dask partition: label each trip with its region + 15-min slot."""
    region = kmeans.predict(scaler.transform(part[COORD_COLS].to_numpy())).astype("int16")
    tbin = part["tpep_pickup_datetime"].dt.floor("15min")
    return pd.DataFrame({"region": region, "tbin": tbin.to_numpy()})


def build_timeseries(ddf, scaler, kmeans, params: dict) -> pd.DataFrame:
    """Assign regions out-of-core, count pickups per zone per 15-min slot (keeping
    true zeros), then add a shift-by-1 EWMA 'recent average' feature."""
    k = params["regions"]["n_clusters"]
    freq = params["timeseries"]["resample_freq"]
    alpha = params["timeseries"]["ewma_alpha"]

    meta = pd.DataFrame({"region": pd.Series(dtype="int16"),
                         "tbin": pd.Series(dtype="datetime64[ns]")})
    labelled = ddf.map_partitions(_assign_region_and_bin, scaler, kmeans, meta=meta)
    counts = labelled.groupby(["region", "tbin"]).size().rename("total_pickups").compute()

    # Build the COMPLETE grid: every region x every 15-min slot in the period.
    # Missing (region, slot) pairs are real zero-demand slots -> fill 0 (no faking).
    tbins = counts.index.get_level_values("tbin")
    full_time = pd.date_range(tbins.min(), tbins.max(), freq=freq)
    grid = pd.MultiIndex.from_product([range(k), full_time], names=["region", "tbin"])
    ts = counts.reindex(grid, fill_value=0).reset_index()

    ts = ts.sort_values(["region", "tbin"], ignore_index=True)
    # EWMA of recent demand, SHIFTED by 1 so row t only sees slots before t (no leakage).
    ts["avg_pickups"] = (ts.groupby("region")["total_pickups"]
                           .transform(lambda s: s.ewm(alpha=alpha).mean().shift(1)))
    return ts


def make_plot_data(ddf, scaler, kmeans, params: dict, n_points: int) -> pd.DataFrame:
    """A small colored sample of points (lon, lat, region, color) for the app's map."""
    k = params["regions"]["n_clusters"]
    n = int(ddf.shape[0].compute())
    frac = min(1.0, n_points / n)
    pts = ddf[COORD_COLS].sample(frac=frac, random_state=1).compute()
    pts["region"] = kmeans.predict(scaler.transform(pts.to_numpy())).astype("int16")
    cmap = matplotlib.colormaps["hsv"]
    colors = {r: matplotlib.colors.to_hex(cmap(r / k)) for r in range(k)}
    pts["color"] = pts["region"].map(colors)
    return pts


def save_region_map(plot_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.scatter(plot_df["pickup_longitude"], plot_df["pickup_latitude"],
               c=plot_df["color"], s=3, alpha=0.35, linewidths=0)
    ax.set(title=f"{plot_df['region'].nunique()} demand zones (KMeans on pickup coords)",
           xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster pickups into zones + build time series.")
    ap.add_argument("--sample", action="store_true",
                    help="smoke test on clean_sample.parquet with throwaway outputs")
    args = ap.parse_args()

    root = find_root()
    params = yaml.safe_load(open(root / "params.yaml"))
    sfx = "_sample" if args.sample else ""
    in_path = root / "data" / "interim" / (f"clean{sfx}.parquet")
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found - run src/data/data_ingestion.py first")

    (root / "models").mkdir(exist_ok=True)
    (root / "data" / "external").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Reading {in_path.relative_to(root)} ...")
    ddf = dd.read_parquet(in_path)

    # 1. cluster (fit on a sample; smaller sample in --sample mode)
    print("Fitting scaler + MiniBatchKMeans ...")
    scaler, kmeans = fit_clusterer(ddf, params, fit_rows=200_000 if args.sample else 2_000_000)

    # 2. justify K
    mean_mi = validate_k(scaler, kmeans)
    print(f"  K={params['regions']['n_clusters']}: mean haversine distance to 8 "
          f"nearest zones = {mean_mi:.2f} miles")

    # save the fitted models
    joblib.dump(scaler, root / "models" / f"scaler{sfx}.joblib")
    joblib.dump(kmeans, root / "models" / f"kmeans{sfx}.joblib")

    # 3. per-region 15-min time series (with true zeros + anti-leakage EWMA)
    print("Assigning regions + building 15-min time series ...")
    ts = build_timeseries(ddf, scaler, kmeans, params)
    ts_path = root / "data" / "interim" / f"region_timeseries{sfx}.csv"
    ts.to_csv(ts_path, index=False)

    # 4. colored point sample for the app map + a verification figure
    plot_df = make_plot_data(ddf, scaler, kmeans, params,
                             n_points=20_000 if args.sample else 40_000)
    plot_dir = root / "data" / ("interim" if args.sample else "external")
    plot_df.to_csv(plot_dir / f"plot_data{sfx}.csv", index=False)
    save_region_map(plot_df, root / "reports" / "figures" / f"regions_map{sfx}.png")

    # report
    nz = (ts["total_pickups"] == 0).mean()
    print("\n  --- time series summary ---")
    print(f"  rows (regions x slots)  : {len(ts):,}")
    print(f"  slots per region        : {len(ts) // params['regions']['n_clusters']:,}")
    print(f"  total_pickups mean/median/max: "
          f"{ts['total_pickups'].mean():.1f} / {ts['total_pickups'].median():.0f} / {ts['total_pickups'].max():,}")
    print(f"  share of zero-demand slots  : {100*nz:.1f}%  (kept as real zeros)")
    print(f"\n  wrote models/, {ts_path.relative_to(root)}, plot_data{sfx}.csv, "
          f"regions_map{sfx}.png  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
