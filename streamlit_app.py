import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v4 - Adaptive Intelligence Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "version" not in st.session_state:
    st.session_state.version = 0  # used to force retrain

# -------------------------------
# MODEL CACHE (AUTO REFRESH)
# -------------------------------
@st.cache_resource
def get_model(X, y, version):
    model = RFModel()
    model.train(X, y)
    return model

# -------------------------------
# UPLOAD DATA
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file:
    st.session_state.df = load_data(file)
    st.session_state.version += 1
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
        st.session_state.version += 1
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
model = get_model(X_train, y_train, st.session_state.version)

# -------------------------------
# TRUE STREAK CALCULATION
# -------------------------------
def get_true_streak(series, condition):
    streak = 0
    for val in reversed(series):
        if condition(val):
            streak += 1
        else:
            break
    return streak

# -------------------------------
# CONTEXT FEATURES
# -------------------------------
def get_context(df):
    last_20 = df.tail(20)["crash"]

    return {
        "volatility": last_20.std(),
        "low_streak": get_true_streak(last_20, lambda x: x < 2),
        "high_streak": get_true_streak(last_20, lambda x: x > 3),
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME DETECTION
# -------------------------------
def detect_regime(df):
    last_30 = df.tail(30)["crash"]

    avg = last_30.mean()
    std = last_30.std()
    low_ratio = (last_30 < 2).mean()
    high_ratio = (last_30 > 3).mean()

    if std > 2.8:
        regime = "⚡ VOLATILE"
    elif low_ratio > 0.65:
        regime = "🔴 CHOPPY"
    elif high_ratio > 0.45:
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
# SMOOTH CONFIDENCE ENGINE
# -------------------------------
confidence = proba * 60  # stronger ML weight

# volatility boost
confidence += min(ctx["volatility"] * 10, 20)

# streak logic (REAL streak)
if ctx["low_streak"] >= 5:
    confidence += 25

if ctx["high_streak"] >= 4:
    confidence -= 20

# regime adjustments
if regime_data["regime"] == "⚡ VOLATILE":
    confidence += 10
elif regime_data["regime"] == "🔴 CHOPPY":
    confidence -= 25
elif regime_data["regime"] == "🟢 HOT":
    confidence += 20

confidence = max(0, min(100, confidence))

# -------------------------------
# SMART MULTIPLIER ENGINE
# -------------------------------
def evaluate_multiplier(df, target, window=100):
    balance = 0
    stake = 1

    start = max(30, len(df) - window)

    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]

        weight = 1 + (i / len(df))  # recent rounds weighted more

        if crash >= target:
            balance += weight * (target - 1)
        else:
            balance -= weight

    return balance

def get_adaptive_multipliers(df):
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

    results = []
    for m in multipliers:
        profit = evaluate_multiplier(df, m)
        results.append((m, profit))

    res = pd.DataFrame(results, columns=["m", "profit"])

    return {
        "low": res.nsmallest(3, "m").sort_values("profit", ascending=False).iloc[0]["m"],
        "mid": res[(res["m"] > 1.6) & (res["m"] <= 2.3)].sort_values("profit", ascending=False).iloc[0]["m"],
        "high": res.nlargest(3, "m").sort_values("profit", ascending=False).iloc[0]["m"],
        "table": res.sort_values("profit", ascending=False)
    }

adaptive = get_adaptive_multipliers(df_ml)

# -------------------------------
# SIGNAL ENGINE
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
# UI
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("🎯 Target", f"{target}x" if target else "No Trade")
col5.metric("🧠 Regime", regime_data["regime"])

# -------------------------------
# LAST 10 MULTIPLIERS (IMPROVED)
# -------------------------------
st.markdown("### 📉 Last 10 Multipliers")

last_10 = df_ml["crash"].tail(10).to_numpy()[::-1]
cols = st.columns(10)

for i, val in enumerate(last_10):
    if val >= 3:
        color = "#00c853"
    elif val >= 2:
        color = "#64dd17"
    else:
        color = "#d50000"

    cols[i].markdown(
        f"""
        <div style="
            background-color:{color};
            padding:10px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-weight:bold;">
            {val:.2f}x
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# INSIGHTS
# -------------------------------
with st.expander("🧠 AI + Regime Insights"):
    st.write(ctx)
    st.write(regime_data)

# -------------------------------
# MULTIPLIERS TABLE
# -------------------------------
st.subheader("🎯 Adaptive Multiplier Performance")
st.dataframe(adaptive["table"], use_container_width=True)

# -------------------------------
# DATA
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])
