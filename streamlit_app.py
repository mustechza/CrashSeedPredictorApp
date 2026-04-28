import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from training.backtest import run_backtest

from models.random_forest import RFModel
from analytics.dashboard import plot_equity
from analytics.tracker import calculate_metrics

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v2 - Adaptive Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# MODEL CACHE (IMPORTANT)
# -------------------------------
@st.cache_resource
def get_model(X_train, y_train):
    model = RFModel()
    model.train(X_train, y_train)
    return model

# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data loaded!")

# -------------------------------
# LIVE INPUT
# -------------------------------
new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01)

if st.sidebar.button("Add Round"):
    if st.session_state.df is not None:
        now = pd.Timestamp.now()

        new_row = pd.DataFrame([{
            "rate": str(new_rate),
            "crash": float(new_rate),
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": "live",
            "salt": "live",
            "fetchedAt": now
        }])

        st.session_state.df = pd.concat(
            [st.session_state.df, new_row],
            ignore_index=True
        )

        st.success("Round added")
    else:
        st.warning("Upload data first")

# -------------------------------
# CHECK
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data to begin")
    st.stop()

# -------------------------------
# CLEAN DATA
# -------------------------------
df = clean_data(st.session_state.df)

df_ml = df.sort_values("fetchedAt", ascending=True).reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False).reset_index(drop=True)

# -------------------------------
# UI
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Multiplier History")
st.line_chart(df_ml["crash"])

# -------------------------------
# REQUIRE DATA
# -------------------------------
if len(df_ml) < 30:
    st.warning("Need at least 30 rounds")
    st.stop()

# -------------------------------
# TRAIN MODEL (CACHED)
# -------------------------------
X_train, X_test, y_train, y_test = prepare_data(df_ml)
model = get_model(X_train, y_train)

# -------------------------------
# FEATURE ENGINEERING (LIVE)
# -------------------------------
def calculate_features(df):
    last_10 = df.tail(10)["crash"]

    volatility = last_10.std()
    avg = last_10.mean()

    low_streak = sum(last_10 < 2)
    high_streak = sum(last_10 > 3)

    return volatility, avg, low_streak, high_streak

volatility, avg, low_streak, high_streak = calculate_features(df_ml)

# -------------------------------
# ML PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

prediction = model.predict(X_live)[0]
proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# HYBRID STRATEGY ENGINE
# -------------------------------
confidence = 0

# ML weight
confidence += proba * 50

# Volatility boost
if volatility > 1.5:
    confidence += 15

# Low streak boost (bounce logic)
if low_streak >= 6:
    confidence += 20

# High streak penalty
if high_streak >= 5:
    confidence -= 15

confidence = max(0, min(100, confidence))

# -------------------------------
# SIGNAL LOGIC
# -------------------------------
if confidence > 75:
    signal = "🔥 STRONG BET"
elif confidence > 55:
    signal = "✅ BET"
else:
    signal = "❌ SKIP"

# -------------------------------
# DISPLAY SIGNAL
# -------------------------------
st.subheader("🤖 AI Decision Engine")

col1, col2, col3 = st.columns(3)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Probability", f"{proba:.2%}")

# -------------------------------
# DEBUG INSIGHTS
# -------------------------------
with st.expander("🧠 AI Insights"):
    st.write(f"Volatility: {volatility:.2f}")
    st.write(f"Average: {avg:.2f}")
    st.write(f"Low streak (<2x): {low_streak}")
    st.write(f"High streak (>3x): {high_streak}")

# -------------------------------
# BACKTEST
# -------------------------------
history = run_backtest(df_ml, model)
metrics = calculate_metrics(history)

st.subheader("📊 Performance")

col1, col2 = st.columns(2)
col1.metric("Profit", round(metrics["profit"], 2))
col2.metric("Drawdown", round(metrics["max_drawdown"], 2))

st.subheader("📈 Equity Curve")
plot_equity(history)
/*

import streamlit as st
import pandas as pd

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from training.backtest import run_backtest

from models.random_forest import RFModel
from analytics.dashboard import plot_equity
from analytics.tracker import calculate_metrics

st.set_page_config(layout="wide")
st.title("🚀 Crash AI Live Dashboard")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data loaded!")

# -------------------------------
# LIVE INPUT
# -------------------------------
new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01)

if st.sidebar.button("Add Round"):
    if st.session_state.df is not None:
        now = pd.Timestamp.now()

        new_row = pd.DataFrame([{
            "rate": str(new_rate),
            "crash": float(new_rate),
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": "live",
            "salt": "live",
            "fetchedAt": now
        }])

        st.session_state.df = pd.concat(
            [st.session_state.df, new_row],
            ignore_index=True
        )

        st.success("Round added")
    else:
        st.warning("Upload data first")

# -------------------------------
# MAIN CHECK
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data to begin")
    st.stop()

# -------------------------------
# CLEAN DATA
# -------------------------------
df = clean_data(st.session_state.df)

# 🔥 SPLIT MODES (IMPORTANT FIX)
df_ml = df.sort_values("fetchedAt", ascending=True).reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False).reset_index(drop=True)

# -------------------------------
# UI TABLE (LATEST FIRST)
# -------------------------------
st.subheader("📊 Live Data (Latest Rounds)")

st.dataframe(df_ui.head(20), use_container_width=True)

# -------------------------------
# FULL MULTIPLIER HISTORY
# -------------------------------
st.subheader("📈 Full Multiplier History (Old → New)")

st.line_chart(df_ml["crash"])

with st.expander("View Raw Timeline"):
    st.dataframe(df_ml[["fetchedAt", "crash"]], use_container_width=True)

# -------------------------------
# TRAIN MODEL (ML ORDER)
# -------------------------------
if len(df_ml) < 30:
    st.warning("Need at least 30 rounds")
    st.stop()

X_train, X_test, y_train, y_test = prepare_data(df_ml)

model = RFModel()
model.train(X_train, y_train)

# -------------------------------
# LIVE PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]  # ML-safe order

X_live = last_row[FEATURES]

prediction = model.predict(X_live)[0]
proba = model.predict_proba(X_live)[0][1]

st.subheader("🤖 Prediction")

if prediction == 1:
    st.success(f"BET → {proba:.2%}")
else:
    st.error(f"SKIP → {proba:.2%}")

# -------------------------------
# BACKTEST (ML ORDER ONLY)
# -------------------------------
history = run_backtest(df_ml, model)
metrics = calculate_metrics(history)

st.subheader("📊 Performance")

col1, col2 = st.columns(2)
col1.metric("Profit", round(metrics["profit"], 2))
col2.metric("Drawdown", round(metrics["max_drawdown"], 2))
*/
st.subheader("📈 Equity Curve")
plot_equity(history)
