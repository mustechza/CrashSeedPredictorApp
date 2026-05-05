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
st.title("🚀 Crash AI v3 - Regime Adaptive Engine (PRO)")

# -------------------------------
# SESSION STATE
# -------------------------------
for key, default in {
    "df": None,
    "model": None,
    "rounds_since_training": 0,
    "training_history": [],
    "predictions_log": [],
    "auto_learn": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------
# WALK-FORWARD VALIDATION (FIX 1)
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
# TRAIN / RETRAIN
# -------------------------------
def train_or_retrain_model(df, force=False):
    if len(df) < 50:
        return None

    if st.session_state.model is None or force or st.session_state.rounds_since_training >= 10:
        with st.spinner("🔄 Training AI..."):
            X_train, X_test, y_train, y_test = prepare_data(df)

            model = RFModel()
            model.train(X_train, y_train)

            accuracy = walk_forward_validation(df)

            st.session_state.training_history.append({
                "timestamp": datetime.now(),
                "rounds": len(df),
                "accuracy": accuracy
            })

            st.session_state.model = model
            st.session_state.rounds_since_training = 0

    return st.session_state.model

# -------------------------------
# LIVE INPUT
# -------------------------------
st.sidebar.markdown("## 🎮 Add Round")
new_rate = st.sidebar.number_input("Multiplier", min_value=1.0, step=0.01)

if st.sidebar.button("➕ Add Round") and st.session_state.df is not None:
    now = pd.Timestamp.now()
    row = pd.DataFrame([{
        "rate": str(new_rate),
        "crash": new_rate,
        "prepareTime": now,
        "beginTime": now,
        "endTime": now,
        "hash": f"live_{now.timestamp()}",
        "salt": "live",
        "fetchedAt": now
    }])

    st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)
    st.session_state.rounds_since_training += 1

    # Update last prediction
    if st.session_state.predictions_log:
        last = st.session_state.predictions_log[-1]
        if last["actual"] is None:
            last["actual"] = new_rate
            last["was_correct"] = (new_rate >= (last["target"] or 0))

    if st.session_state.auto_learn and st.session_state.rounds_since_training >= 10:
        train_or_retrain_model(st.session_state.df, True)

    st.rerun()

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])
if file and st.session_state.df is None:
    st.session_state.df = load_data(file)
    train_or_retrain_model(st.session_state.df, True)
    st.rerun()

if st.session_state.df is None:
    st.stop()

# -------------------------------
# CLEAN
# -------------------------------
df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)

model = train_or_retrain_model(df_ml)

# -------------------------------
# CONTEXT
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    last_50 = df.tail(50)["crash"]

    return {
        "volatility": last_10.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3),
        "avg_10": last_10.mean(),
        "avg_50": last_50.mean()
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]

    avg = last_20.mean()
    std = last_20.std()

    if std > 2.5:
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
X_live = df_ml.iloc[[-1]][FEATURES]
proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# CONFIDENCE (FIX 2)
# -------------------------------
confidence = proba * 50

if ctx["volatility"] > 1.5:
    confidence += 15
if ctx["low_streak"] >= 5:
    confidence += 10
if ctx["high_streak"] >= 5:
    confidence -= 15

confidence += regime_boost

# Sigmoid calibration
confidence = 100 * (1 / (1 + np.exp(-0.06 * (confidence - 50))))
confidence = max(0, min(100, confidence))

# -------------------------------
# ANTI-OVERTRADING (FIX 3)
# -------------------------------
recent_losses = sum(
    1 for p in st.session_state.predictions_log[-5:]
    if p.get("was_correct") is False
)

cooldown = recent_losses >= 3

if cooldown:
    confidence *= 0.7

# -------------------------------
# MULTIPLIER ENGINE
# -------------------------------
def evaluate_multiplier(df, target):
    wins = sum(df["crash"] >= target)
    total = len(df)
    win_rate = wins / total if total else 0
    return win_rate

multipliers = [1.5, 2.0, 3.0, 4.0]
scores = {m: evaluate_multiplier(df_ml.tail(100), m) for m in multipliers}
best_m = max(scores, key=scores.get)

# -------------------------------
# SIGNAL ENGINE
# -------------------------------
if regime.startswith("🔴") and confidence < 70:
    signal = "❌ SKIP"
    target = None

elif cooldown and confidence < 75:
    signal = "❌ SKIP"
    target = None

elif confidence > 80:
    signal = "🔥 STRONG BET"
    target = best_m

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
# LOG
# -------------------------------
if len(st.session_state.predictions_log) == 0 or st.session_state.predictions_log[-1]["actual"] is not None:
    st.session_state.predictions_log.append({
        "timestamp": datetime.now(),
        "signal": signal,
        "target": target,
        "confidence": confidence,
        "actual": None,
        "was_correct": None
    })

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("Regime", regime)

# Accuracy
valid = [p for p in st.session_state.predictions_log if p["was_correct"] is not None]
acc = sum(p["was_correct"] for p in valid) / len(valid) if valid else 0
st.metric("📊 Accuracy", f"{acc:.2%}")

# Last 10
st.subheader("Last 10")
st.write(df_ml["crash"].tail(10).values[::-1])

# Training history
if st.session_state.training_history:
    hist = pd.DataFrame(st.session_state.training_history)
    st.line_chart(hist.set_index("timestamp")["accuracy"])

# Data
st.dataframe(df_ml.tail(20))

# Auto refresh
if st.sidebar.checkbox("Auto Refresh"):
    time.sleep(5)
    st.rerun()
