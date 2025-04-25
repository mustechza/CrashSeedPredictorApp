import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ========== CONFIG ==========
st.set_page_config(page_title="Crash Predictor", layout="wide")
MAX_MULTIPLIER = 10.5
SEED_SERVER = st.secrets.get("server_seed", "")
SEED_CLIENT = st.secrets.get("client_seed", "")
SEED_NONCE = st.secrets.get("nonce", 0)

# ========== LOAD DATA ==========
st.sidebar.header("📁 Load Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Multipliers' column", type="csv")

def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    if 'multipliers' in df.columns:
        df['multipliers'] = df['multipliers'].str.replace('x', '').astype(float)
        return df
    else:
        st.error("CSV must contain a 'Multipliers' column.")
        return pd.DataFrame()

if uploaded_file:
    data = load_data(uploaded_file)
else:
    data = pd.DataFrame()

# ========== FEATURE EXTRACTION ==========
def extract_features(series):
    if len(series) < 10:
        return None
    last10 = series[-10:]
    return pd.DataFrame([{
        'mean': np.mean(last10),
        'std': np.std(last10),
        'last': last10[-1],
        'max': max(last10),
        'min': min(last10),
        'last_diff': last10[-1] - last10[-2] if len(last10) > 1 else 0
    }])

# ========== MODEL TRAINING ==========
def train_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# ========== UI: INPUT ==========
st.header("🎯 Crash Multiplier Predictor")

with st.form("prediction_form"):
    raw_input = st.text_input("Enter recent multipliers (comma-separated)", "1.52,1.11,1.32,2.58,2.35,3.99,1.19,1.05")
    feedback = st.text_input("Enter actual next multiplier (optional)", "")
    submit = st.form_submit_button("Submit")

def parse_input(raw):
    try:
        return [min(float(x.strip().lower().replace('x', '')), MAX_MULTIPLIER)
                for x in raw.split(',') if x.strip()]
    except:
        return []

crash_series = parse_input(raw_input)
features = extract_features(crash_series) if len(crash_series) >= 10 else None

# Persistent training data
if 'X_train' not in st.session_state:
    st.session_state.X_train = pd.DataFrame()
if 'y_train' not in st.session_state:
    st.session_state.y_train = pd.Series(dtype=float)

# Add feedback if provided
if submit and feedback and features is not None:
    try:
        fb_val = min(float(feedback), MAX_MULTIPLIER)
        st.session_state.X_train = pd.concat([st.session_state.X_train, features], ignore_index=True)
        st.session_state.y_train = pd.concat([st.session_state.y_train, pd.Series([fb_val])], ignore_index=True)
        st.success("✅ Model trained with new feedback.")
    except:
        st.error("Invalid feedback value.")

# Train if enough data
model = None
if len(st.session_state.X_train) >= 10:
    model = train_model(st.session_state.X_train, st.session_state.y_train)

# Predict
if features is not None and model:
    pred = model.predict(features)[0]
    safe = round(pred * 0.97, 2)
    st.subheader(f"📈 Predicted Next: **{pred:.2f}x**")
    st.info(f"🛡️ Safe Target (3% edge): **{safe:.2f}x**")

    # Threshold alerts
    if pred < 1.2:
        st.error("⚠️ Danger Zone Prediction!")
    elif pred > 3.0:
        st.success("🔥 High Multiplier Expected!")

# ========== MONEY MANAGEMENT ==========
st.header("💰 Strategy Simulator")

col1, col2, col3 = st.columns(3)
with col1:
    strategy = st.selectbox("Strategy", ["Flat", "Martingale", "Anti-Martingale"])
with col2:
    bankroll = st.number_input("Starting Bankroll", 100, 100000, 1000)
with col3:
    base_bet = st.number_input("Base Bet", 1, 100, 10)

if len(st.session_state.X_train) >= 10:
    preds = model.predict(st.session_state.X_train.tail(30))
    actuals = st.session_state.y_train.tail(30).values
    outcome = []
    bal = bankroll
    bet = base_bet

    for p, a in zip(preds, actuals):
        win = a >= p * 0.97
        bal += bet * (p if win else -1)
        outcome.append((p, a, win, bal))
        if strategy == "Martingale":
            bet = bet * 2 if not win else base_bet
        elif strategy == "Anti-Martingale":
            bet = bet * 2 if win else base_bet
        else:
            bet = base_bet

    sim_df = pd.DataFrame(outcome, columns=["Pred", "Actual", "Win", "Bankroll"])
    st.dataframe(sim_df.style.applymap(lambda x: 'background-color: #d4edda' if isinstance(x, bool) and x else 'background-color: #f8d7da', subset=["Win"]))

    st.line_chart(sim_df["Bankroll"])
    winrate = (sim_df["Win"].sum() / len(sim_df)) * 100
    st.metric("📊 Win Rate", f"{winrate:.2f}%")

# ========== ACCURACY ==========
st.header("📋 Recent Prediction Accuracy")
if len(st.session_state.X_train) >= 10:
    preds = model.predict(st.session_state.X_train.tail(20))
    actuals = st.session_state.y_train.tail(20).values
    errors = np.abs(preds - actuals)
    result_df = pd.DataFrame({
        "Predicted": preds,
        "Actual": actuals,
        "Error": errors,
        "Win": actuals >= preds * 0.97
    })
    st.dataframe(result_df.style.applymap(lambda x: 'background-color: #d4edda' if isinstance(x, bool) and x else 'background-color: #f8d7da', subset=["Win"]))
    st.metric("MAE (Last 20)", f"{np.mean(errors):.2f}")

# ========== SEED (OPTIONAL) ==========
with st.expander("🔐 Provably Fair (Seed Debug)"):
    st.text(f"Server Seed: {SEED_SERVER}")
    st.text(f"Client Seed: {SEED_CLIENT}")
    st.text(f"Nonce: {SEED_NONCE}")
    st.warning("Seed-based validation not active yet. Coming soon.")

# ========== FOOTER ==========
st.caption("🔁 Built with ❤️ for Crash Prediction & Simulation")
