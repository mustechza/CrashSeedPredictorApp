import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v3 - Adaptive System")

# -------------------------------
# SESSION
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# MODEL CACHE
# -------------------------------
@st.cache_resource
def get_model(X, y):
    model = RFModel()
    model.train(X, y)
    return model

# -------------------------------
# UPLOAD
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file:
    st.session_state.df = load_data(file)
    st.success("Data loaded")

# -------------------------------
# LIVE INPUT
# -------------------------------
new_rate = st.sidebar.number_input("Crash Multiplier", 1.0, step=0.01)

if st.sidebar.button("Add Round"):
    if st.session_state.df is not None:
        now = pd.Timestamp.now()

        row = pd.DataFrame([{
            "rate": str(new_rate),
            "crash": float(new_rate),
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": "live",
            "salt": "live",
            "fetchedAt": now
        }])

        st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)
        st.success("Round added")

# -------------------------------
# CHECK
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data")
    st.stop()

# -------------------------------
# CLEAN
# -------------------------------
df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 50:
    st.warning("Need at least 50 rounds")
    st.stop()

# -------------------------------
# TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = prepare_data(df_ml)
model = get_model(X_train, y_train)

# -------------------------------
# FEATURE ENGINE
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    return {
        "volatility": last_10.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3)
    }

ctx = get_context(df_ml)

# -------------------------------
# PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# CONFIDENCE ENGINE
# -------------------------------
confidence = proba * 50

if ctx["volatility"] > 1.5:
    confidence += 15

if ctx["low_streak"] >= 6:
    confidence += 20

if ctx["high_streak"] >= 5:
    confidence -= 15

confidence = max(0, min(100, confidence))

# -------------------------------
# 🔥 ADAPTIVE MULTIPLIER ENGINE
# -------------------------------
def evaluate_multiplier(df, target, window=80):
    balance = 0
    stake = 1

    start = max(30, len(df) - window)

    for i in range(start, len(df) - 1):
        crash = df.iloc[i+1]["crash"]

        if crash >= target:
            balance += stake * (target - 1)
        else:
            balance -= stake

    return balance

def get_adaptive_multipliers(df):
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

    results = []
    for m in multipliers:
        profit = evaluate_multiplier(df, m)
        results.append((m, profit))

    df_res = pd.DataFrame(results, columns=["m", "profit"])

    # Split tiers
    low = df_res[df_res["m"] <= 1.6]
    mid = df_res[(df_res["m"] > 1.6) & (df_res["m"] <= 2.3)]
    high = df_res[df_res["m"] > 2.3]

    return {
        "low": low.sort_values("profit", ascending=False).iloc[0]["m"],
        "mid": mid.sort_values("profit", ascending=False).iloc[0]["m"],
        "high": high.sort_values("profit", ascending=False).iloc[0]["m"],
        "table": df_res.sort_values("profit", ascending=False)
    }

adaptive = get_adaptive_multipliers(df_ml)

# -------------------------------
# SIGNAL + TARGET
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]
elif confidence > 60:
    signal = "✅ BET"
    target = adaptive["mid"]
elif confidence > 50:
    signal = "⚠️ SMALL BET"
    target = adaptive["low"]
else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# UI - TOP
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("🎯 Target", f"{target}x" if target else "No Trade")

# -------------------------------
# ADAPTIVE TABLE
# -------------------------------
st.subheader("🎯 Adaptive Multiplier Performance")

st.dataframe(adaptive["table"], use_container_width=True)

# -------------------------------
# DATA
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Multiplier History")
st.line_chart(df_ml["crash"])
