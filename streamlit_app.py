import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v3 - Regime Adaptive Engine (PRO FIXED)")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "rounds_since_training" not in st.session_state:
    st.session_state.rounds_since_training = 0
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = []

# -------------------------------
# WALK-FORWARD VALIDATION
# -------------------------------
def walk_forward_validation(df, n_splits=5):
    split_size = len(df) // (n_splits + 1)
    scores = []

    for i in range(1, n_splits + 1):
        train_end = i * split_size
        test_end = (i + 1) * split_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(test_df) < 10:
            continue

        try:
            X_train, X_test, y_train, y_test = prepare_data(pd.concat([train_df, test_df]))

            model = RFModel()
            model.train(X_train.iloc[:len(train_df)], y_train.iloc[:len(train_df)])

            preds = model.model.predict(X_test.iloc[-len(test_df):])
            score = (preds == y_test.iloc[-len(test_df):]).mean()
            scores.append(score)

        except:
            continue

    return np.mean(scores) if scores else 0.5

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_model(df):
    if len(df) < 50:
        return None

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = RFModel()
    model.train(X_train, y_train)

    accuracy = walk_forward_validation(df)

    st.sidebar.success(f"✅ Model trained | WF Accuracy: {accuracy:.2%}")

    return model

# -------------------------------
# LOAD DATA
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file and st.session_state.df is None:
    st.session_state.df = load_data(file)
    st.session_state.model = train_model(st.session_state.df)
    st.rerun()

if st.session_state.df is None:
    st.stop()

# -------------------------------
# CLEAN DATA
# -------------------------------
df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)

if len(df_ml) < 50:
    st.warning("Need at least 50 rounds")
    st.stop()

if st.session_state.model is None:
    st.session_state.model = train_model(df_ml)

model = st.session_state.model

# -------------------------------
# CONTEXT
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    return {
        "volatility": last_10.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3),
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]

    if last_20.std() > 2.5:
        return "⚡ VOLATILE", 10
    elif (last_20 < 2).mean() > 0.6:
        return "🔴 CHOPPY", -20
    elif (last_20 > 3).mean() > 0.4:
        return "🟢 HOT", 15
    else:
        return "🟡 NORMAL", 0

regime, regime_boost = detect_regime(df_ml)

# -------------------------------
# PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# CONFIDENCE ENGINE (FIXED)
# -------------------------------
confidence = proba * 50

if ctx["volatility"] > 1.5:
    confidence += 15
if ctx["low_streak"] >= 5:
    confidence += 10
if ctx["high_streak"] >= 5:
    confidence -= 15

confidence += regime_boost

# ✅ FIX: SIGMOID CALIBRATION
confidence = 100 * (1 / (1 + np.exp(-0.06 * (confidence - 50))))
confidence = max(0, min(100, confidence))

# -------------------------------
# ANTI-OVERTRADING (FIXED)
# -------------------------------
recent_losses = sum(
    1 for p in st.session_state.predictions_log[-5:]
    if p.get("was_correct") is False
)

cooldown = recent_losses >= 3

if cooldown:
    confidence *= 0.7

# -------------------------------
# SIGNAL ENGINE (FIXED)
# -------------------------------
if regime.startswith("🔴") and confidence < 70:
    signal = "❌ SKIP"
    target = None

elif cooldown and confidence < 75:
    signal = "❌ SKIP"
    target = None

elif confidence > 80:
    signal = "🔥 STRONG BET"
    target = 3.0

elif confidence > 65:
    signal = "✅ BET"
    target = 2.0

elif confidence > 50:
    signal = "⚠️ SMALL BET"
    target = 1.5

else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# LOG PREDICTION
# -------------------------------
if len(st.session_state.predictions_log) == 0 or st.session_state.predictions_log[-1].get("actual") is not None:
    st.session_state.predictions_log.append({
        "signal": signal,
        "confidence": confidence,
        "actual": None,
        "was_correct": None
    })

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 AI DECISION")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Signal", signal)

with col2:
    st.metric("Confidence", f"{confidence:.1f}%")

with col3:
    st.metric("Regime", regime)

# -------------------------------
# LAST 10
# -------------------------------
st.subheader("Last 10 Rounds")
st.write(df_ml["crash"].tail(10).values[::-1])

# -------------------------------
# STATS
# -------------------------------
trade_count = sum(1 for p in st.session_state.predictions_log if p["signal"] != "❌ SKIP")
st.metric("📊 Trades Taken", trade_count)

# -------------------------------
# AUTO REFRESH
# -------------------------------
if st.sidebar.checkbox("Auto Refresh"):
    time.sleep(5)
    st.rerun()
