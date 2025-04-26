import streamlit as st
import pandas as pd
import hmac
import hashlib
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- Config ---
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

# --- Load and Save History ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "client_seed", "nonce", "actual", "predicted", "result"])

def save_prediction(client_seed, nonce, actual, predicted):
    df = load_history()
    result = "Win" if round(predicted, 2) == round(actual, 2) else "Loss"
    df.loc[len(df)] = [datetime.now(), client_seed, nonce, actual, predicted, result]
    df.to_csv(HISTORY_FILE, index=False)

def reset_with_backup():
    if os.path.exists(HISTORY_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"prediction_history_backup_{timestamp}.csv"
        os.rename(HISTORY_FILE, backup_file)
        st.sidebar.success(f"Backup created: {backup_file}")
    else:
        st.sidebar.warning("No history to reset.")

# --- UI Layout ---
st.set_page_config(page_title="Crash Predictor", layout="centered")
st.title("🎯 Crash Predictor (Seed-Based Only)")

# --- Sidebar ---
st.sidebar.title("Controls")
if st.sidebar.button("Reset & Backup History"):
    reset_with_backup()

st.sidebar.info("Predictions based on server seed + client seed + nonce.")

# --- Prediction Form ---
st.header("1. Live Crash Result Submission")

# Session state for nonce
if "last_nonce" not in st.session_state:
    st.session_state.last_nonce = 0

with st.form("prediction_form"):
    client_seed = st.text_input("Client Seed", "97439433b0745d23902d5c53fd1de03d")
    nonce = st.number_input("Nonce", value=st.session_state.last_nonce, step=1)
    actual_multiplier = st.number_input("Actual Crash Multiplier", value=1.00, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Submit Result")

if submitted:
    predicted = get_multiplier_from_seed(SERVER_SEED, client_seed, int(nonce))
    st.success(f"Predicted Multiplier: **{predicted}x**")
    
    save_prediction(client_seed, nonce, actual_multiplier, predicted)
    st.success("✅ Result saved!")

    # Auto-increment nonce
    st.session_state.last_nonce = nonce + 1

# --- History Table ---
st.header("2. 📊 Prediction History (Latest 20)")
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Wins", wins)
    col2.metric("Losses", losses)
    col3.metric("Accuracy (%)", round(win_rate, 2))

    # Accuracy Trend Chart
    st.subheader("3. 📈 Accuracy Trend")
    history['rolling_accuracy'] = (history['result'] == 'Win').rolling(window=10).mean() * 100

    fig, ax = plt.subplots()
    ax.plot(history['rolling_accuracy'], marker='o')
    ax.set_title('Rolling Accuracy (Last 10)')
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('Prediction #')
    ax.grid(True)
    st.pyplot(fig)

else:
    st.info("No predictions yet. Submit a live result first.")
