# NYC Taxi Demand Prediction

Predict the number of taxi pickups in each of **30 NYC zones** for the **next 15 minutes**,
using NYC Yellow Taxi trip data (2016, Jan–Mar). A supervised **regression** problem built on
a **time-series + spatial-clustering** pipeline.

**Pipeline:** `raw trips → clean → cluster into 30 zones → count per zone per 15 min → add features → train → predict → show on a map`

## Status
🚧 In progress — **Phase 0: environment & project setup**.

## Setup (Linux, Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
The raw data (~7 GB) is **not** in this repo; its location is set in [`params.yaml`](params.yaml).

## Layout
| Path | What |
|---|---|
| `data/` | `raw/`, `interim/` (gitignored) + `processed/`, `external/` (small committed artifacts) |
| `src/{data,features,models}/` | pipeline scripts |
| `notebooks/` | EDA & modeling |
| `models/` | saved `joblib` models |
| `reports/` | metrics & figures |
| `app.py` | Streamlit demo (Phase 6) |
| `params.yaml` | all paths & tunable knobs |
