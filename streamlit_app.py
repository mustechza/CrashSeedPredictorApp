
import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v4 - Pro Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "last_train_size" not in st.session_state:
    st.session_state.last_train_size = 0

if "proba_history" not in st.session_state:
    st.session_state.proba_history = []

# -------------------------------
# LOAD DATA
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

if len(df_ml) < 100:
    st.warning("Need at least 100 rounds")
    st.stop()

# -------------------------------
# SMART RETRAIN
# -------------------------------
def get_model_incremental(df_ml):
    X, _, y, _ = prepare_data(df_ml)

    current_size = len(df_ml)

    if st.session_state.model is None:
        model = RFModel()
        weights = np.linspace(0.5, 2.0, len(X))
        model.model.fit(X, y, sample_weight=weights)

        st.session_state.model = model
        st.session_state.last_train_size = current_size
        return model

    if current_size - st.session_state.last_train_size >= 10:
        recent = df_ml.tail(300)

        X_r, _, y_r, _ = prepare_data(recent)

        model = RFModel()
        weights = np.linspace(0.5, 2.0, len(X_r))
        model.model.fit(X_r, y_r, sample_weight=weights)

        st.session_state.model = model
        st.session_state.last_train_size = current_size

        st.warning("🔁 Model retrained")

    return st.session_state.model

model = get_model_incremental(df_ml)

# -------------------------------
# ML PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

proba = model.model.predict_proba(X_live)[0][1]

# Smooth predictions
st.session_state.proba_history.append(proba)
st.session_state.proba_history = st.session_state.proba_history[-5:]
proba = np.mean(st.session_state.proba_history)

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
        return "⚡ VOLATILE"
    elif low_ratio > 0.6:
        return "🔴 CHOPPY"
    elif high_ratio > 0.4:
        return "🟢 HOT"
    else:
        return "🟡 NORMAL"

regime = detect_regime(df_ml)

# -------------------------------
# PRO ENGINE
# -------------------------------
def evaluate_trade(crash, target, stake):
    return stake * (target - 1) if crash >= target else -stake

def simulate(df, target):
    balance = 0
    curve = []
    max_bal = 0
    dd_list = []

    for i in range(len(df) - 1):
        crash = df.iloc[i + 1]["crash"]
        pnl = evaluate_trade(crash, target, 1)

        balance += pnl
        max_bal = max(max_bal, balance)
        dd_list.append(max_bal - balance)
        curve.append(balance)

    return balance, max(dd_list) if dd_list else 0, curve

def walk_forward(df, multipliers):
    results = []

    start = 0
    train = 200
    test = 20

    while True:
        train_end = start + train
        test_end = train_end + test

        if test_end >= len(df):
            break

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        scores = []

        for m in multipliers:
            bal, _, _ = simulate(train_df, m)
            scores.append((m, bal))

        best_m = max(scores, key=lambda x: x[1])[0]

        test_bal, dd, _ = simulate(test_df, best_m)

        results.append({
            "multiplier": best_m,
            "profit": test_bal,
            "drawdown": dd
        })

        start += test

    return pd.DataFrame(results)

multipliers = [1.3,1.5,1.8,2.0,2.2,2.5,3.0]
wf_df = walk_forward(df_ml, multipliers)

if not wf_df.empty:
    best_m = wf_df.groupby("multiplier")["profit"].mean().idxmax()

    wf_profit = wf_df["profit"].sum()
    wf_win = (wf_df["profit"] > 0).mean()
    wf_dd = wf_df["drawdown"].mean()
    wf_score = wf_profit * wf_win / (1 + wf_dd)
else:
    best_m = None
    wf_score = -999

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

if regime == "⚡ VOLATILE":
    confidence += 10
elif regime == "🔴 CHOPPY":
    confidence -= 20
elif regime == "🟢 HOT":
    confidence += 15

confidence = max(0, min(100, confidence))

# -------------------------------
# FINAL DECISION
# -------------------------------
if confidence > 75 and wf_score > 0:
    signal = "🔥 STRONG BET"
    target = best_m
elif confidence > 60:
    signal = "✅ BET"
    target = best_m
elif confidence > 50:
    signal = "⚠️ SMALL BET"
    target = best_m
else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Signal", signal)
c2.metric("Confidence", f"{confidence:.1f}%")
c3.metric("ML Prob", f"{proba:.2%}")
c4.metric("🎯 Target", f"{target}x" if target else "No Trade")
c5.metric("🧠 Regime", regime)

# -------------------------------
# WFV METRICS
# -------------------------------
st.subheader("🧪 Walk-Forward Performance")

if not wf_df.empty:
    col1, col2, col3 = st.columns(3)

    col1.metric("WF Profit", f"{wf_profit:.2f}")
    col2.metric("Win Rate", f"{wf_win:.2%}")
    col3.metric("Drawdown", f"{wf_dd:.2f}")

    st.dataframe(wf_df, use_container_width=True)
    st.line_chart(wf_df["profit"])

# -------------------------------
# EQUITY CURVE
# -------------------------------
if target:
    st.subheader("💰 Equity Curve")

    bal, dd, curve = simulate(df_ml.tail(300), target)

    st.line_chart(curve)
    st.metric("Balance", f"{bal:.2f}")
    st.metric("Max DD", f"{dd:.2f}")

# -------------------------------
# DATA
# -------------------------------
st.subheader("📊 Recent Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash Chart")
st.line_chart(df_ml["crash"])
```
