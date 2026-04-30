import streamlit as st
import pandas as pd
import numpy as np

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v4 - Institutional Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "history" not in st.session_state:
    st.session_state.history = []

if "balance" not in st.session_state:
    st.session_state.balance = 1000

if "peak_balance" not in st.session_state:
    st.session_state.peak_balance = 1000

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

if st.sidebar.button("Add Round"):
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
# VALIDATE DATA
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data")
    st.stop()

df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 100:
    st.warning("Need at least 100 rounds")
    st.stop()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df_ml["mean_5"] = df_ml["crash"].rolling(5).mean()
df_ml["std_5"] = df_ml["crash"].rolling(5).std()
df_ml["momentum"] = df_ml["crash"].diff()
df_ml["below_2_ratio"] = (df_ml["crash"] < 2).rolling(10).mean()

df_ml = df_ml.dropna()

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

regime = detect_regime(df_ml)

# -------------------------------
# TRAIN REGIME MODELS
# -------------------------------
@st.cache_resource(ttl=120)
def train_models(df):

    models = {}

    for r in ["VOLATILE", "CHOPPY", "HOT", "NORMAL"]:
        sub = df.copy()

        if r == "VOLATILE":
            sub = sub[sub["std_5"] > 2]
        elif r == "CHOPPY":
            sub = sub[sub["below_2_ratio"] > 0.5]
        elif r == "HOT":
            sub = sub[sub["crash"] > 3]

        if len(sub) < 50:
            continue

        X_train, X_test, y_train, y_test = prepare_data(sub)

        model = RFModel()
        model.train(X_train, y_train)

        models[r] = model

    return models

models = train_models(df_ml)

model = models.get(regime, list(models.values())[0])

# -------------------------------
# PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

proba = model.predict_proba(X_live)[0][1]

confidence = proba * 100

# -------------------------------
# MULTIPLIER ENGINE (WALK FORWARD)
# -------------------------------
def walk_forward(df, target):
    balance = 0

    for i in range(50, len(df) - 1):
        train = df.iloc[:i]
        test = df.iloc[i + 1]["crash"]

        if test >= target:
            balance += (target - 1)
        else:
            balance -= 1

    return balance

def get_best_multiplier(df):
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

    scores = [(m, walk_forward(df, m)) for m in multipliers]

    return sorted(scores, key=lambda x: x[1], reverse=True)

ranked = get_best_multiplier(df_ml)
best_target = ranked[0][0]

# -------------------------------
# KELLY BET SIZING
# -------------------------------
def kelly(p, b):
    return max(0, (p * (b + 1) - 1) / b)

b = best_target - 1
fraction = kelly(proba, b)

stake = st.session_state.balance * min(fraction, 0.05)

# -------------------------------
# RISK MANAGEMENT
# -------------------------------
drawdown = (st.session_state.peak_balance - st.session_state.balance) / st.session_state.peak_balance

if drawdown > 0.3:
    signal = "🛑 STOP (DRAWDOWN)"
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

    next_crash = df_ml.iloc[-1]["crash"]

    if next_crash >= best_target:
        profit = stake * (best_target - 1)
    else:
        profit = -stake

    st.session_state.balance += profit
    st.session_state.peak_balance = max(st.session_state.peak_balance, st.session_state.balance)

    st.session_state.history.append(profit)

# -------------------------------
# METRICS
# -------------------------------
wins = sum(1 for x in st.session_state.history if x > 0)
losses = sum(1 for x in st.session_state.history if x < 0)

winrate = wins / max(1, (wins + losses))

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 AI DECISION ENGINE")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Signal", signal)
c2.metric("Confidence", f"{confidence:.1f}%")
c3.metric("Regime", regime)
c4.metric("Target", f"{best_target}x")
c5.metric("Stake", f"{stake:.2f}")

st.markdown("## 💰 Performance")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Balance", f"{st.session_state.balance:.2f}")
p2.metric("Peak", f"{st.session_state.peak_balance:.2f}")
p3.metric("Win Rate", f"{winrate:.2%}")
p4.metric("Drawdown", f"{drawdown:.2%}")

# -------------------------------
# MULTIPLIER TABLE
# -------------------------------
st.subheader("🎯 Multiplier Ranking")
st.dataframe(pd.DataFrame(ranked, columns=["Multiplier", "Score"]))

# -------------------------------
# CHART
# -------------------------------
st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])

st.subheader("📊 Last Rounds")
st.dataframe(df_ui.head(20))
