import streamlit as st
import pandas as pd
import numpy as np
import hmac
import hashlib
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

HISTORY_FILE = "prediction_history.csv"
SERVER_SEED = st.secrets.get("SERVER_SEED", "01a24e141597617f167daef1901514260952f2e64a49adcd829e6813c80305ac")

# --- Provably Fair Multiplier ---
def get_multiplier_from_seed(server_seed, client_seed, nonce):
    message = f"{client_seed}:{nonce}".encode()
    h = hmac.new(server_seed.encode(), message, hashlib.sha256).hexdigest()

    if h.startswith("0000"):
        return 100.0

    r = int(h[:13], 16)
    X = r / 2**52
    if X == 1: return 1.0
    crash_point = 99 / (1 - X)
    return round(max(1.0, crash_point) / 100, 2)

# --- ML Model Training ---
def train_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# --- Load History ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "client_seed", "nonce", "actual", "predicted", "result"])

# --- Save Prediction ---
def save_prediction(client_seed, nonce, actual, predicted):
    df = load_history()
    result = "Win" if round(predicted, 2) == round(actual, 2) else "Loss"
    df.loc[len(df)] = [datetime.now(), client_seed, nonce, actual, predicted, result]
    df.to_csv(HISTORY_FILE, index=False)

# --- Reset with Backup ---
def reset_with_backup():
    if os.path.exists(HISTORY_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"prediction_history_backup_{timestamp}.csv"
        os.rename(HISTORY_FILE, backup_file)
        st.sidebar.success(f"Backup created: {backup_file}")
    else:
        st.sidebar.warning("No history to reset.")

# --- Streamlit UI ---
st.title("Crash Predictor: ML + Seed Logic + Feedback")
st.sidebar.title("Controls")

if st.sidebar.button("Reset & Backup History"):
    reset_with_backup()

st.header("1. Submit Recent Crash Data")
recent_data_input = st.text_input("Enter last 8 crash multipliers (comma-separated):", "1.52,1.11,1.32,2.58,2.35,3.99,1.19,1.05")
recent_data = [float(x.strip()) for x in recent_data_input.split(",") if x.strip()]
if len(recent_data) < 8:
    st.warning("Need 8 values.")
    st.stop()

# --- ML Prediction ---
X = pd.DataFrame([recent_data[i:i+4] for i in range(len(recent_data)-4)])
y = recent_data[4:]
model = train_model(X, y)

st.header("2. Predict Next Crash")
last_input = pd.DataFrame([recent_data[-4:]])
ml_prediction = round(float(model.predict(last_input)[0]), 2)
st.success(f"ML Prediction: **{ml_prediction}x**")

st.header("3. Submit Live Result")
with st.form("live_result_form"):
    client_seed = st.text_input("Client Seed", "97439433b0745d23902d5c53fd1de03d")
    nonce = st.number_input("Nonce", value=15141, step=1)
    actual_multiplier = st.number_input("Actual Crash Multiplier", value=1.00, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Submit")

if submitted:
    # Predict from seed
    seed_prediction = get_multiplier_from_seed(SERVER_SEED, client_seed, int(nonce))
    st.write(f"Seed-based Prediction: **{seed_prediction}x**")

    # Save prediction result
    save_prediction(client_seed, nonce, actual_multiplier, ml_prediction)
    st.success("Result saved with feedback.")

# --- History Table ---
st.header("📊 Prediction History (Last 20)")
history = load_history().tail(20)
if not history.empty:
    def highlight_result(row):
        color = 'green' if row['result'] == 'Win' else 'red'
        return [f'color: {color}' if col == 'result' else '' for col in row.index]

    st.dataframe(history.style.apply(highlight_result, axis=1))

    total = len(history)
    wins = (history['result'] == 'Win').sum()
    losses = total - wins
    win_rate = (wins / total) * 100 if total else 0
    st.metric("Wins", wins)
    st.metric("Losses", losses)
    st.metric("Accuracy (%)", round(win_rate, 2))
else:
    st.info("No predictions yet.")
