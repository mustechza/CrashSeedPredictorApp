import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data
from training.trainer import prepare_data
from models.random_forest import RFModel
from models.lstm_model import LSTMModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI V5 - Hybrid Intelligence Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "balance" not in st.session_state:
    st.session_state.balance = 1000.0

if "peak" not in st.session_state:
    st.session_state.peak = 1000.0

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# LOAD DATA
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file:
    st.session_state.df = load_data(file)
    st.success("Data loaded")

# -------------------------------
# LIVE INPUT
# -------------------------------
new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01)

if st.sidebar.button("Add Round") and st.session_state.df is not None:
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

# -------------------------------
# VALIDATION
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data to begin")
    st.stop()

df = clean_data(st.session_state.df)
df = df.sort_values("fetchedAt").reset_index(drop=True)

if len(df) < 120:
    st.warning("Need at least 120 rounds")
    st.stop()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["return"] = df["crash"].pct_change()
df["log_return"] = np.log(df["crash"] / df["crash"].shift(1))
df["mean_10"] = df["crash"].rolling(10).mean()
df["std_10"] = df["crash"].rolling(10).std()
df["momentum"] = df["crash"].diff()
df["below_2"] = (df["crash"] < 2).astype(int)

df = df.dropna()

FEATURES = [
    "crash", "return", "log_return",
    "mean_10", "std_10",
    "momentum", "below_2"
]

# Binary target
df["target"] = (df["crash"] >= 2).astype(int)

# -------------------------------
# REGIME DETECTION
# -------------------------------
def detect_regime(df):
    last = df.tail(20)["crash"]

    if last.std() > 2.5:
        return "VOLATILE"
    elif (last < 2).mean() > 0.6:
        return "CHOPPY"
    elif (last > 3).mean() > 0.4:
        return "HOT"
    else:
        return "NORMAL"

regime = detect_regime(df)

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------
@st.cache_resource(ttl=300)
def get_rf(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = RFModel()
    model.train(X_train, y_train)
    return model

rf_model = get_rf(df)

last_row = df.iloc[[-1]]
rf_proba = rf_model.predict_proba(last_row[FEATURES])[0][1]

# -------------------------------
# LSTM MODEL
# -------------------------------
@st.cache_resource(ttl=300)
def get_lstm(df):
    model = LSTMModel(seq_len=10)
    model.train(df, FEATURES, df["target"])
    return model

lstm_model = get_lstm(df)
lstm_proba = lstm_model.predict_proba(df, FEATURES)

# -------------------------------
# HYBRID PROBABILITY
# -------------------------------
proba = (rf_proba * 0.4) + (lstm_proba * 0.6)

confidence = proba * 100

# Regime adjustment
if regime == "VOLATILE":
    confidence *= 1.1
elif regime == "CHOPPY":
    confidence *= 0.85
elif regime == "HOT":
    confidence *= 1.15

confidence = min(100, max(0, confidence))

# -------------------------------
# WALK-FORWARD MULTIPLIER ENGINE
# -------------------------------
def walk_forward(df, target):
    balance = 0

    for i in range(60, len(df) - 1):
        nxt = df.iloc[i + 1]["crash"]

        if nxt >= target:
            balance += (target - 1)
        else:
            balance -= 1

    return balance

def get_best_multiplier(df):
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    scores = [(m, walk_forward(df, m)) for m in multipliers]
    return sorted(scores, key=lambda x: x[1], reverse=True)

ranked = get_best_multiplier(df)
target = ranked[0][0]

# -------------------------------
# KELLY BET SIZING
# -------------------------------
def kelly(p, b):
    return max(0, (p * (b + 1) - 1) / b)

b = target - 1
fraction = kelly(proba, b)

stake = st.session_state.balance * min(fraction, 0.05)

# -------------------------------
# RISK MANAGEMENT
# -------------------------------
drawdown = (st.session_state.peak - st.session_state.balance) / st.session_state.peak

if drawdown > 0.3:
    signal = "🛑 STOP"
    stake = 0
elif confidence > 70:
    signal = "🔥 STRONG BET"
elif confidence > 55:
    signal = "✅ BET"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
else:
    signal = "❌ SKIP"
    stake = 0

# -------------------------------
# EXECUTE TRADE (SIMULATION)
# -------------------------------
if st.button("Execute Trade") and stake > 0:

    result = df.iloc[-1]["crash"]

    if result >= target:
        profit = stake * (target - 1)
    else:
        profit = -stake

    st.session_state.balance += profit
    st.session_state.peak = max(st.session_state.peak, st.session_state.balance)
    st.session_state.history.append(profit)

# -------------------------------
# METRICS
# -------------------------------
wins = sum(1 for x in st.session_state.history if x > 0)
losses = sum(1 for x in st.session_state.history if x < 0)
total = wins + losses

winrate = wins / total if total > 0 else 0

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 AI DECISION")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Signal", signal)
c2.metric("Confidence", f"{confidence:.1f}%")
c3.metric("Regime", regime)
c4.metric("Target", f"{target}x")
c5.metric("Stake", f"{stake:.2f}")

st.markdown("## 💰 Performance")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Balance", f"{st.session_state.balance:.2f}")
p2.metric("Peak", f"{st.session_state.peak:.2f}")
p3.metric("Win Rate", f"{winrate:.2%}")
p4.metric("Drawdown", f"{drawdown:.2%}")

# -------------------------------
# MULTIPLIER TABLE
# -------------------------------
st.subheader("🎯 Multiplier Ranking")
st.dataframe(pd.DataFrame(ranked, columns=["Multiplier", "Score"]))

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📈 Crash History")
st.line_chart(df["crash"])

st.subheader("📊 Last Rounds")
st.dataframe(df.tail(20))
