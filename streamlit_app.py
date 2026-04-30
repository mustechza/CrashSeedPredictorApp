import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os

from sklearn.metrics import mean_absolute_error

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI - Institutional Engine")

# -------------------------------
# STORAGE
# -------------------------------
DATA_PATH = "data/live_data.csv"

def load_persistent_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=True)
    return None

def save_persistent_data(df):
    df.to_csv(DATA_PATH, index=False)

# -------------------------------
# SESSION
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_persistent_data()

# -------------------------------
# MODEL VERSION (HASH)
# -------------------------------
def get_data_version(df):
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

# -------------------------------
# MODEL CACHE (FIX #5)
# -------------------------------
@st.cache_resource
def get_model(X, y, version):
    model = RFModel()
    model.train(X, y)
    return model

# -------------------------------
# UPLOAD
# -------------------------------
file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if file:
    st.session_state.df = load_data(file)
    save_persistent_data(st.session_state.df)
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
        save_persistent_data(st.session_state.df)
        st.success("Round added")

# -------------------------------
# CHECK DATA
# -------------------------------
if st.session_state.df is None:
    st.info("Upload data to begin")
    st.stop()

# -------------------------------
# CLEAN
# -------------------------------
df = clean_data(st.session_state.df)

df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 50:
    st.warning("Need at least 50 rounds")
    st.stop()

# -------------------------------
# LEAK-FREE DATA PREP
# -------------------------------
def prepare_data_no_leak(df):
    df = df.copy()

    # Predict NEXT round
    df["target"] = df["crash"].shift(-1)

    # Remove last row (no future)
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURES]
    y = df["target"]

    # Time-based split
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test, df

X_train, X_test, y_train, y_test, df_shifted = prepare_data_no_leak(df_ml)

# -------------------------------
# VALIDATE FEATURES
# -------------------------------
missing_cols = [col for col in FEATURES if col not in df_ml.columns]
if missing_cols:
    st.error(f"Missing features: {missing_cols}")
    st.stop()

# -------------------------------
# TRAIN MODEL (FIX #5)
# -------------------------------
version = get_data_version(df_ml)
model = get_model(X_train, y_train, version)

# -------------------------------
# EVALUATE MODEL
# -------------------------------
y_pred = model.model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# -------------------------------
# CONTEXT
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    return {
        "volatility": last_10.std(),
        "low_streak": int((last_10 < 2).sum()),
        "high_streak": int((last_10 > 3).sum())
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME
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
        "std": std
    }

regime_data = detect_regime(df_ml)

# -------------------------------
# LIVE PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES].fillna(0)

prediction = model.model.predict(X_live)[0]

# -------------------------------
# BACKTEST
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

    return float(best["m"]), res.sort_values("profit", ascending=False)

target, perf_table = get_best_multiplier(df_ml)

# -------------------------------
# DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("🎯 Predicted Next", f"{prediction:.2f}x")
col2.metric("📉 MAE", f"{mae:.3f}")
col3.metric("🎯 Best Target", f"{target:.2f}x")
col4.metric("🧠 Regime", regime_data["regime"])
col5.metric("Volatility", f"{ctx['volatility']:.2f}")

# -------------------------------
# CHART
# -------------------------------
st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])

# -------------------------------
# TABLES
# -------------------------------
st.subheader("🎯 Multiplier Performance")
st.dataframe(perf_table, use_container_width=True)

st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)
