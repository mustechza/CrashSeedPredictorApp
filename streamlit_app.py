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
# SIDEBAR SETTINGS (NEW)
# -------------------------------
st.sidebar.header("💰 Bankroll Settings")

bankroll = st.sidebar.number_input("Bankroll", value=100.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.5, 5.0, 2.0) / 100
max_mg_risk_pct = st.sidebar.slider("Max Martingale Risk %", 5.0, 30.0, 15.0) / 100

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

missing_cols = [col for col in FEATURES if col not in df_ml.columns]
if missing_cols:
    st.error(f"Missing features: {missing_cols}")
    st.stop()

model = get_model(X_train, y_train)

# -------------------------------
# OVERDUE FIXED
# -------------------------------
def overdue_factor(df):
    last = df.tail(15)["crash"].to_numpy()
    streak = 0
    for v in last[::-1]:
        if v < 2:
            streak += 1
        else:
            break
    return streak

overdue = overdue_factor(df_ml)

# -------------------------------
# REGIME
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]
    std = last_20.std()
    global_std = df["crash"].rolling(100).std().mean()

    if std > global_std * 1.3:
        return "⚡ VOLATILE"
    elif (last_20 < 2).mean() > 0.6:
        return "🔴 CHOPPY"
    elif (last_20 > 3).mean() > 0.4:
        return "🟢 HOT"
    else:
        return "🟡 NORMAL"

regime = detect_regime(df_ml)

# -------------------------------
# ML PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES].fillna(0)

proba = model.predict_proba(X_live)[0].max()
confidence = proba * 100

if overdue >= 5:
    confidence += 10

confidence = max(0, min(100, confidence))

# -------------------------------
# MULTIPLIER OPTIMIZATION
# -------------------------------
def evaluate_multiplier(df, target):
    balance = 100
    for i in range(len(df)-1):
        crash = df.iloc[i+1]["crash"]
        if crash >= target:
            balance += (target - 1)
        else:
            balance -= 1
    return balance

def get_best_multiplier(df):
    ms = np.arange(1.2, 3.0, 0.1)
    results = [(m, evaluate_multiplier(df, m)) for m in ms]
    res = pd.DataFrame(results, columns=["m", "profit"])
    best = res.sort_values("profit", ascending=False).iloc[0]["m"]
    return float(best)

target = get_best_multiplier(df_ml)

# Regime adjust
if regime == "🔴 CHOPPY":
    target = min(target, 1.6)
elif regime == "⚡ VOLATILE":
    target = max(target, 2.0)

# -------------------------------
# WIN RATE
# -------------------------------
def win_rate(df, target):
    wins = ((df["crash"].shift(-1) >= target).sum())
    total = len(df) - 1
    return wins / total if total > 0 else 0

wr = win_rate(df_ml, target)

# -------------------------------
# SIGNAL
# -------------------------------
if confidence > 70 and wr > 0.55:
    signal = "🔥 STRONG BET"
elif confidence > 60 and wr > 0.5:
    signal = "✅ BET"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# SMART BET SIZE (KELLY LIGHT)
# -------------------------------
def get_base_bet(bankroll, risk_pct, confidence):
    edge_factor = confidence / 100
    adj_risk = risk_pct * edge_factor
    return bankroll * adj_risk

base_bet = get_base_bet(bankroll, risk_pct, confidence)

# -------------------------------
# SMART MARTINGALE (2 STEP)
# -------------------------------
def smart_martingale(target, base_bet, bankroll, max_risk_pct):
    if target is None:
        return None

    step1 = base_bet

    step2 = (step1 + base_bet) / (target - 1)

    total_risk = step1 + step2
    max_allowed = bankroll * max_risk_pct

    # cap risk
    if total_risk > max_allowed:
        scale = max_allowed / total_risk
        step1 *= scale
        step2 *= scale

    profit1 = step1 * (target - 1)
    profit2 = step2 * (target - 1) - step1

    return {
        "step1": round(step1, 2),
        "step2": round(step2, 2),
        "profit1": round(profit1, 2),
        "profit2": round(profit2, 2),
        "total_risk": round(step1 + step2, 2)
    }

mg = smart_martingale(target, base_bet, bankroll, max_mg_risk_pct)

# -------------------------------
# DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Signal", signal)
c2.metric("Confidence", f"{confidence:.1f}%")
c3.metric("Win Rate", f"{wr:.2%}")
c4.metric("Target", f"{target:.2f}x" if target else "No Trade")
c5.metric("Bankroll", f"{bankroll:.2f}")
c6.metric("Base Bet", f"{base_bet:.2f}")

# -------------------------------
# MARTINGALE UI
# -------------------------------
st.markdown("### 💰 Smart Martingale Plan")

if mg and target:
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Step 1", mg["step1"])
    m2.metric("Step 2", mg["step2"])
    m3.metric("Profit", f"{mg['profit1']} / {mg['profit2']}")
    m4.metric("Total Risk", mg["total_risk"])
else:
    st.info("No trade")

# -------------------------------
# LAST 10 MULTIPLIERS
# -------------------------------
st.markdown("### 📉 Last 10 Multipliers")

last_10 = df_ml["crash"].tail(10).to_numpy()[::-1]
cols = st.columns(10)

for i, val in enumerate(last_10):
    color = "green" if val >= 2 else "red"

    cols[i].markdown(
        f"<div style='background:{color};padding:10px;border-radius:8px;text-align:center;color:white'>{val:.2f}x</div>",
        unsafe_allow_html=True
    )

# -------------------------------
# CHART
# -------------------------------
st.line_chart(df_ml["crash"])
