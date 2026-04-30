import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide")
st.title("🚀 Crash AI v4 Elite - Adaptive Engine")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# DATA LOADER
# -------------------------------
def load_data(file):
    df = pd.read_json(file)
    return df

# -------------------------------
# CLEAN + FEATURE ENGINEERING
# -------------------------------
def clean_data(df):
    df["crash"] = df["crash"].astype(float)
    df = df.sort_values("fetchedAt").reset_index(drop=True)

    # Sequence features
    for i in range(1, 6):
        df[f"lag_{i}"] = df["crash"].shift(i)

    # Rolling stats
    df["mean_5"] = df["crash"].rolling(5).mean()
    df["std_5"] = df["crash"].rolling(5).std()

    df.dropna(inplace=True)
    return df

FEATURES = [f"lag_{i}" for i in range(1, 6)] + ["mean_5", "std_5"]

# Target: hit 2x
def prepare_data(df):
    df["target"] = (df["crash"].shift(-1) >= 2).astype(int)
    df.dropna(inplace=True)

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    return train, test

# -------------------------------
# ENSEMBLE MODEL
# -------------------------------
@st.cache_resource
def train_models(train_df):
    X = train_df[FEATURES]
    y = train_df["target"]

    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression()

    rf.fit(X, y)
    lr.fit(X, y)

    return rf, lr

def predict_ensemble(models, X):
    rf, lr = models
    p1 = rf.predict_proba(X)[:, 1]
    p2 = lr.predict_proba(X)[:, 1]

    return (p1 + p2) / 2

# -------------------------------
# REGIME DETECTION
# -------------------------------
def detect_regime(df):
    last = df.tail(20)["crash"]

    std = last.std()
    low_ratio = (last < 2).mean()

    # trend
    y = last.values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]

    if std > 2.5:
        return "⚡ VOLATILE"
    elif low_ratio > 0.6:
        return "🔴 CHOPPY"
    elif slope > 0.1:
        return "🟢 HOT"
    elif slope < -0.1:
        return "🔻 COOLING"
    else:
        return "🟡 NORMAL"

# -------------------------------
# WALK-FORWARD TRAINING
# -------------------------------
def walk_forward_predict(df):
    preds = []

    for i in range(60, len(df) - 1):
        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        models = train_models(train)
        p = predict_ensemble(models, test[FEATURES])[0]

        preds.append(p)

    return preds

# -------------------------------
# MULTIPLIER ENGINE (RISK BASED)
# -------------------------------
def evaluate_multiplier(df, target, risk=0.02):
    balance = 100

    for i in range(30, len(df) - 1):
        stake = balance * risk
        crash = df.iloc[i + 1]["crash"]

        if crash >= target:
            balance += stake * (target - 1)
        else:
            balance -= stake

    return balance - 100

def get_best_multiplier(df):
    options = [1.5, 2.0, 2.5]

    results = [(m, evaluate_multiplier(df, m)) for m in options]
    best = sorted(results, key=lambda x: x[1], reverse=True)[0]

    return best[0], pd.DataFrame(results, columns=["Multiplier", "Profit"])

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
            "crash": new_rate,
            "fetchedAt": now
        }])

        st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)

# -------------------------------
# CHECK
# -------------------------------
if st.session_state.df is None:
    st.stop()

df = clean_data(st.session_state.df)

if len(df) < 80:
    st.warning("Need more data")
    st.stop()

# -------------------------------
# TRAIN
# -------------------------------
train_df, test_df = prepare_data(df)
models = train_models(train_df)

# -------------------------------
# LIVE PREDICTION
# -------------------------------
last_row = df.iloc[[-2]]
proba = predict_ensemble(models, last_row[FEATURES])[0]

# -------------------------------
# REGIME
# -------------------------------
regime = detect_regime(df)

# -------------------------------
# CONFIDENCE + EDGE
# -------------------------------
edge = proba - 0.5

confidence = proba * 70

if regime == "⚡ VOLATILE":
    confidence *= 1.1
elif regime == "🔴 CHOPPY":
    confidence *= 0.8
elif regime == "🟢 HOT":
    confidence *= 1.15

confidence = np.clip(confidence, 0, 100)

# -------------------------------
# MULTIPLIER
# -------------------------------
target, perf_table = get_best_multiplier(df)

# -------------------------------
# SIGNAL
# -------------------------------
if confidence > 75 and edge > 0.1:
    signal = "🔥 STRONG BET"
elif confidence > 60 and edge > 0.05:
    signal = "✅ BET"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
else:
    signal = "❌ SKIP"
    target = None

# -------------------------------
# TRACK PERFORMANCE
# -------------------------------
if target:
    result = df.iloc[-1]["crash"] >= target
    st.session_state.history.append(result)

winrate = np.mean(st.session_state.history) * 100 if st.session_state.history else 0

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("Target", f"{target}x" if target else "No Trade")
col5.metric("Regime", regime)

st.metric("📈 Win Rate", f"{winrate:.1f}%")

# -------------------------------
# PERFORMANCE TABLE
# -------------------------------
st.subheader("🎯 Multiplier Backtest")
st.dataframe(perf_table, use_container_width=True)

# -------------------------------
# CHART
# -------------------------------
st.subheader("📈 Crash History")
st.line_chart(df["crash"])
