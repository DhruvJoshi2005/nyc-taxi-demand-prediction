"""Phase 2 - Data ingestion & cleaning.

Reads the three raw monthly Yellow-Taxi CSVs with Dask (out-of-core), applies the
cleaning rules justified in the Phase-1 EDA, and writes a cleaned Parquet dataset
that later phases build on. Only the columns the demand model needs are kept: the
pickup timestamp and the pickup location (fare & distance are read only so we can
filter on them, then dropped).

Run from the project root:
    python -m src.data.data_ingestion            # full run: all 3 months
    python -m src.data.data_ingestion --sample   # fast smoke test on ~1 partition

Output:
    data/interim/clean.parquet         (full run)
    data/interim/clean_sample.parquet  (--sample)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import dask
import dask.dataframe as dd
import yaml

# Columns read from the raw CSV. datetime + coords are the OUTPUT we keep;
# fare & distance are needed only to filter, then discarded.
READ_COLS = ["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude",
             "trip_distance", "fare_amount"]
NUMERIC = ["pickup_longitude", "pickup_latitude", "trip_distance", "fare_amount"]
KEEP_COLS = ["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude"]


def find_root(start: Path | None = None) -> Path:
    """Walk up from this file until the folder holding params.yaml is found."""
    start = start or Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "params.yaml").exists():
            return p
    raise FileNotFoundError(f"params.yaml not found above {start}")


def read_raw(files: list[str], sample: bool):
    """Lazily read the raw CSV(s) with Dask. In sample mode, read just the first
    partition (~64 MB block) of the first month for a quick code test."""
    dtypes = {c: "float64" for c in NUMERIC}
    if sample:
        return dd.read_csv(files[:1], usecols=READ_COLS, dtype=dtypes).partitions[0]
    return dd.read_csv(files, usecols=READ_COLS, dtype=dtypes)


def build_masks(ddf, clean_cfg: dict):
    """Return the three boolean filters (still lazy - no data read yet)."""
    bb, fa, td = clean_cfg["nyc_bbox"], clean_cfg["fare_amount"], clean_cfg["trip_distance"]
    in_box = (ddf["pickup_longitude"].between(bb["min_longitude"], bb["max_longitude"]) &
              ddf["pickup_latitude"].between(bb["min_latitude"], bb["max_latitude"]))
    ok_fare = (ddf["fare_amount"] > fa["min"]) & (ddf["fare_amount"] <= fa["max"])
    ok_dist = (ddf["trip_distance"] > td["min"]) & (ddf["trip_distance"] <= td["max"])
    return in_box, ok_fare, ok_dist


def main() -> None:
    ap = argparse.ArgumentParser(description="Read + clean the raw taxi CSVs.")
    ap.add_argument("--sample", action="store_true",
                    help="fast smoke test on ~1 partition of the first month")
    args = ap.parse_args()

    root = find_root()
    params = yaml.safe_load(open(root / "params.yaml"))
    raw_dir = Path(params["paths"]["raw_data_dir"])
    files = [str(raw_dir / f) for f in params["data_files"]]
    fmax = params["clean"]["fare_amount"]["max"]
    dmax = params["clean"]["trip_distance"]["max"]

    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    out_path = interim / ("clean_sample.parquet" if args.sample else "clean.parquet")

    t0 = time.time()
    print(f"Reading {'1 sample partition of ' if args.sample else ''}"
          f"{len(files)} file(s) with Dask from {raw_dir} ...")
    ddf = read_raw(files, args.sample)

    in_box, ok_fare, ok_dist = build_masks(ddf, params["clean"])
    keep = in_box & ok_fare & ok_dist

    # --- Pass 1: count rows each rule keeps (a single fused Dask computation) ---
    n_total, n_box, n_fare, n_dist, n_keep = dask.compute(
        ddf.shape[0], in_box.sum(), ok_fare.sum(), ok_dist.sum(), keep.sum())

    def pct(x: int) -> str:
        return f"{100 * x / n_total:5.2f}%"

    print("\n  --- cleaning report (rows passing each rule) ---")
    print(f"  raw rows                : {n_total:>12,}")
    print(f"  inside NYC bounding box : {n_box:>12,}  ({pct(n_box)})")
    print(f"  fare in (0, {fmax:g}]        : {n_fare:>12,}  ({pct(n_fare)})")
    print(f"  distance in (0, {dmax:g}]     : {n_dist:>12,}  ({pct(n_dist)})")
    print(f"  KEPT (all rules AND-ed) : {n_keep:>12,}  ({pct(n_keep)})"
          f"   -> dropped {n_total - n_keep:,} ({pct(n_total - n_keep)})")

    # --- Pass 2: filter, keep only needed columns, parse datetime, write ---
    out = ddf[keep][KEEP_COLS]
    out = out.assign(tpep_pickup_datetime=dd.to_datetime(out["tpep_pickup_datetime"]))
    if out_path.exists():
        shutil.rmtree(out_path)
    out.to_parquet(out_path, engine="pyarrow", write_index=False)

    print(f"\n  wrote {out_path.relative_to(root)}  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
