import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v3 - Regime Adaptive Engine")

# -------------------------------
# SESSION STATE
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
# UPLOAD DATA
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file:
    st.session_state.df = load_data(file)
    st.success("Data loaded!")

# -------------------------------
# LIVE INPUT
# -------------------------------
new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01)

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
# CHECK DATA
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data to begin")
    st.stop()

# -------------------------------
# CLEAN DATA
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
# CONTEXT FEATURES
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
# REGIME DETECTION
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]

    avg = last_20.mean()
    std = last_20.std()
    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()

    if std > 2.5:
        regime = "⚡ VOLATILE"
    elif low_ratio > 0.6:
        regime = "🔴 CHOPPY"
    elif high_ratio > 0.4:
        regime = "🟢 HOT"
    else:
        regime = "🟡 NORMAL"

    return {
        "regime": regime,
        "avg": avg,
        "std": std,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio
    }

regime_data = detect_regime(df_ml)

# -------------------------------
# ML PREDICTION
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

# Regime adjustment
if regime_data["regime"] == "⚡ VOLATILE":
    confidence += 10
elif regime_data["regime"] == "🔴 CHOPPY":
    confidence -= 20
elif regime_data["regime"] == "🟢 HOT":
    confidence += 15

confidence = max(0, min(100, confidence))

# -------------------------------
# ADAPTIVE MULTIPLIER ENGINE
# -------------------------------
def evaluate_multiplier(df, target, window=80):
    balance = 0
    stake = 1

    start = max(30, len(df) - window)

    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]

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

    res = pd.DataFrame(results, columns=["m", "profit"])

    low = res[res["m"] <= 1.6]
    mid = res[(res["m"] > 1.6) & (res["m"] <= 2.3)]
    high = res[res["m"] > 2.3]

    return {
        "low": low.sort_values("profit", ascending=False).iloc[0]["m"],
        "mid": mid.sort_values("profit", ascending=False).iloc[0]["m"],
        "high": high.sort_values("profit", ascending=False).iloc[0]["m"],
        "table": res.sort_values("profit", ascending=False)
    }


adaptive = get_adaptive_multipliers(df_ml)

# -------------------------------
# SIGNAL ENGINE (REGIME AWARE)
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]

elif confidence > 60:
    signal = "✅ BET"

    if regime_data["regime"] == "🟢 HOT":
        target = adaptive["high"]
    else:
        target = adaptive["mid"]

elif confidence > 50:
    signal = "⚠️ SMALL BET"

    if regime_data["regime"] == "🔴 CHOPPY":
        target = adaptive["low"]
    else:
        target = adaptive["mid"]

else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# UI - TOP DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("🎯 Target", f"{target}x" if target else "No Trade")
col5.metric("🧠 Regime", regime_data["regime"])

# -------------------------------
# LAST 10 MULTIPLIERS (NEW)
# -------------------------------
st.markdown("### 🧾 Last 10 Multipliers")

cols = st.columns(10)
last_10_vals = df_ml.tail(10)["crash"].round(2).tolist()

for i, val in enumerate(last_10_vals):
    if val < 2:
        cols[i].metric(f"{i+1}", f"{val}x", delta="Low", delta_color="inverse")
    elif val > 3:
        cols[i].metric(f"{i+1}", f"{val}x", delta="High", delta_color="normal")
    else:
        cols[i].metric(f"{i+1}", f"{val}x")

# -------------------------------
# INSIGHTS
# -------------------------------
with st.expander("🧠 AI + Regime Insights"):
    st.write(f"Volatility: {ctx['volatility']:.2f}")
    st.write(f"Low streak: {ctx['low_streak']}")
    st.write(f"High streak: {ctx['high_streak']}")
    st.write("---")
    st.write(f"Regime avg: {regime_data['avg']:.2f}")
    st.write(f"Regime std: {regime_data['std']:.2f}")
    st.write(f"Low ratio: {regime_data['low_ratio']:.2%}")
    st.write(f"High ratio: {regime_data['high_ratio']:.2%}")

# -------------------------------
# MULTIPLIER PERFORMANCE TABLE
# -------------------------------
st.subheader("🎯 Adaptive Multiplier Performance")
st.dataframe(adaptive["table"], use_container_width=True)

# -------------------------------
# DATA VIEW
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])
