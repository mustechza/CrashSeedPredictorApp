import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI - Institutional Engine")

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
# UPLOAD
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
# CONTEXT
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
# OVERDUE EDGE
# -------------------------------
def overdue_factor(df):
    last = df.tail(15)["crash"]
    streak = 0

    for v in reversed(last):
        if v < 2:
            streak += 1
        else:
            break

    return streak

overdue = overdue_factor(df_ml)

# -------------------------------
# REGIME DETECTION
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]

    avg = last_20.mean()
    std = last_20.std()

    global_std = df["crash"].rolling(100).std().mean()

    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()

    if std > global_std * 1.3:
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

proba = model.predict_proba(X_live)[0].max()
confidence = proba * 100

# -------------------------------
# CONFIDENCE ADJUSTMENTS
# -------------------------------
if ctx["volatility"] > 1.5:
    confidence += 5

if ctx["low_streak"] >= 6:
    confidence += 10

if ctx["high_streak"] >= 5:
    confidence -= 10

if overdue >= 5:
    confidence += 12

if regime_data["regime"] == "⚡ VOLATILE":
    confidence += 5
elif regime_data["regime"] == "🔴 CHOPPY":
    confidence -= 15
elif regime_data["regime"] == "🟢 HOT":
    confidence += 10

confidence = max(0, min(100, confidence))

# -------------------------------
# BACKTEST ENGINE
# -------------------------------
def evaluate_multiplier(df, target, window=80):
    balance = 100
    risk = 0.02

    start = max(30, len(df) - window)

    for i in range(start, len(df) - 1):
        stake = balance * risk
        crash = df.iloc[i + 1]["crash"]

        if crash >= target:
            balance += stake * (target - 1)
        else:
            balance -= stake

    return balance - 100

def get_best_multiplier(df):
    multipliers = np.arange(1.2, 3.1, 0.1)

    results = []
    for m in multipliers:
        profit = evaluate_multiplier(df, m)
        results.append((m, profit))

    res = pd.DataFrame(results, columns=["m", "profit"])
    best = res.sort_values("profit", ascending=False).iloc[0]

    return best["m"], res.sort_values("profit", ascending=False)

target, perf_table = get_best_multiplier(df_ml)

# -------------------------------
# REGIME ADJUST TARGET
# -------------------------------
if regime_data["regime"] == "🔴 CHOPPY":
    target = min(target, 1.6)
elif regime_data["regime"] == "⚡ VOLATILE":
    target = max(target, 2.0)

# -------------------------------
# WIN RATE
# -------------------------------
def win_rate(df, target, window=80):
    wins, total = 0, 0

    start = max(30, len(df) - window)

    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]

        if crash >= target:
            wins += 1
        total += 1

    return wins / total if total else 0

wr = win_rate(df_ml, target)

# -------------------------------
# SIGNAL ENGINE
# -------------------------------
if confidence > 75 and wr > 0.55:
    signal = "🔥 STRONG BET"
elif confidence > 60 and wr > 0.50:
    signal = "✅ BET"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
    target = min(target, 1.6)
else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("🎯 Target", f"{target:.2f}x" if target else "No Trade")
col5.metric("🧠 Regime", regime_data["regime"])
col6.metric("Win Rate", f"{wr:.2%}")

# -------------------------------
# LAST 10 MULTIPLIERS
# -------------------------------
st.markdown("### 📉 Last 10 Multipliers")

last_10 = df_ml["crash"].tail(10).values[::-1]
cols = st.columns(10)

for i, val in enumerate(last_10):
    color = "green" if val >= 2 else "red"

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
with st.expander("🧠 AI Insights"):
    st.write(f"Volatility: {ctx['volatility']:.2f}")
    st.write(f"Low streak: {ctx['low_streak']}")
    st.write(f"High streak: {ctx['high_streak']}")
    st.write(f"Overdue: {overdue}")
    st.write("---")
    st.write(f"Regime avg: {regime_data['avg']:.2f}")
    st.write(f"Regime std: {regime_data['std']:.2f}")

# -------------------------------
# PERFORMANCE TABLE
# -------------------------------
st.subheader("🎯 Multiplier Performance")
st.dataframe(perf_table, use_container_width=True)

# -------------------------------
# DATA VIEW
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])
