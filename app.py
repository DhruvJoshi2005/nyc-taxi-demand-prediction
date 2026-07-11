"""NYC Taxi Demand Prediction - Streamlit demo (Phase 6).

Pick a March 2016 date & time; see predicted taxi pickups per zone for the next 15
minutes, on a map, with a toggle between the interpretable Linear model and the more
accurate Random Forest. Loads only local joblib models (no cloud / MLflow registry).

Run locally:  streamlit run app.py
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `src` imports work on Streamlit Cloud too

from src.models.common import DEFAULT_MODEL, DEPLOYABLE, MODEL_LABELS, TARGET
from src.serving import load_data, load_models, make_demand_map, predict_zone_demand

st.set_page_config(page_title="NYC Taxi Demand Prediction", page_icon="🚕", layout="wide")


@st.cache_resource
def _models():
    return load_models(ROOT)


@st.cache_data
def _data():
    return load_data(ROOT)


models = _models()
test, plot = _data()

st.title("🚕 NYC Taxi Demand Prediction")
st.caption("Predicted taxi pickups per NYC zone for the next 15 minutes "
           "(demonstrated on the March 2016 hold-out test set).")

with st.sidebar:
    st.header("Controls")
    label = st.selectbox("Model", [MODEL_LABELS[n] for n in DEPLOYABLE],
                         index=DEPLOYABLE.index(DEFAULT_MODEL))
    model_name = {MODEL_LABELS[n]: n for n in DEPLOYABLE}[label]
    date = st.date_input("Date", dt.date(2016, 3, 15),
                         min_value=dt.date(2016, 3, 1), max_value=dt.date(2016, 3, 31))
    hour = st.slider("Hour", 0, 23, 18)
    minute = st.select_slider("Minute", options=[0, 15, 30, 45], value=0)

tbin = pd.Timestamp(dt.datetime.combine(date, dt.time(int(hour), int(minute))))
zp = predict_zone_demand(test, models[model_name], tbin)

if zp.empty:
    st.warning("No data for that time slot.")
else:
    total_pred, total_act = zp["predicted"].sum(), zp[TARGET].sum()
    busiest = int(zp.sort_values("predicted", ascending=False).iloc[0]["region"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted pickups (city-wide)", f"{total_pred:,.0f}")
    c2.metric("Actual pickups", f"{total_act:,.0f}", f"{total_pred - total_act:+,.0f} vs actual")
    c3.metric("Busiest zone", f"#{busiest}")

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(make_demand_map(plot, zp))
    with right:
        st.subheader("Top 10 zones - predicted vs actual")
        top = (zp.sort_values("predicted", ascending=False).head(10)
                 .set_index("region")[["predicted", TARGET]]
                 .rename(columns={TARGET: "actual"}))
        st.bar_chart(top, stack=False, color=["#4c72b0", "#dd8452"])
    st.caption(f"{label}  ·  slot {tbin:%a %Y-%m-%d %H:%M}  ·  brighter = more demand")

# ---- About / info section (appears at the bottom of the page) ----
st.divider()
st.subheader("ℹ️ About this project")
about_left, about_right = st.columns(2)
with about_left:
    st.markdown(
        "**What it does** — predicts NYC taxi pickup demand for the **next 15 minutes** "
        "across **30 city zones**, from 2016 Yellow-Taxi trip data.\n\n"
        "**How it works** — 34.5M trips are cleaned out-of-core with **Dask**, "
        "**KMeans** clusters pickups into 30 demand zones, and each zone becomes a "
        "**15-minute time series**. A regression model uses recent-demand **lag features** "
        "plus **cyclical time-of-day / day-of-week** encodings. Trained on Jan–Feb 2016, "
        "tested on March.\n\n"
        "**Results (March hold-out)** — both models beat the naive *predict-the-last-value* "
        "baseline (MAE 15.6): **Random Forest 13.9**, **Linear Regression 14.9**."
    )
with about_right:
    st.markdown(
        "**Using this demo** — pick a date, time and model on the left:\n"
        "- **Map** — each zone coloured by predicted pickups (brighter = busier).\n"
        "- **Bar chart** — predicted vs actual for the busiest zones.\n"
        "- **Metrics** — city-wide predicted vs actual for that 15-minute slot.\n\n"
        "**Why two models?** *Linear Regression* is interpretable; *Random Forest* is a "
        "bit more accurate — toggle to see the accuracy-vs-explainability trade-off.\n\n"
        "**Code & full write-up** → "
        "[github.com/DhruvJoshi2005/nyc-taxi-demand-prediction]"
        "(https://github.com/DhruvJoshi2005/nyc-taxi-demand-prediction)"
    )
st.caption("Built by Dhruv Joshi · data: NYC TLC Trip Records · experiments tracked with "
           "MLflow, pipeline reproducible with DVC.")
