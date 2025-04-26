import streamlit as st
import pandas as pd
import numpy as np
import hmac
import hashlib
import os
from datetime import datetime
import plotly.graph_objects as go

# --- Constants ---
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

# --- History Management ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "client_seed", "nonce", "actual", "predicted", "result"])

def save_prediction(client_seed, nonce, actual, predicted):
    df = load_history()
    result = "Win" if actual > predicted else "Loss"
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

# --- Streamlit UI ---
st.set_page_config(page_title="Crash Predictor", layout="wide")
st.title("🎯 Crash Predictor (Seed-Based)")
st.sidebar.title("Controls")

if st.sidebar.button("🔄 Reset & Backup History"):
    reset_with_backup()

# --- Step 1: Live Prediction ---
st.header("1. Predict Next Multiplier")

col1, col2 = st.columns(2)
with col1:
    client_seed = st.text_input("Client Seed", "97439433b0745d23902d5c53fd1de03d")
with col2:
    last_nonce = int(load_history()['nonce'].max()) + 1 if not load_history().empty else 0
    nonce = st.number_input("Nonce (auto-filled)", value=last_nonce, step=1)

predicted_multiplier = get_multiplier_from_seed(SERVER_SEED, client_seed, int(nonce))

# Logic to adjust prediction if too high
if predicted_multiplier > 4.0:
    predicted_multiplier = 2.0

st.success(f"📈 Predicted Multiplier: **{predicted_multiplier}x**")

# --- Step 2: Enter Actual Result ---
st.header("2. Submit Live Result")
with st.form("live_result_form"):
    actual_multiplier = st.number_input("Actual Crash Multiplier", value=1.00, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Submit Result")

if submitted:
    save_prediction(client_seed, nonce, actual_multiplier, predicted_multiplier)
    st.success("✅ Result saved!")

# --- Step 3: Prediction History ---
st.header("📊 Prediction History")

history = load_history()

if not history.empty:
    col3, col4, col5 = st.columns(3)
    total = len(history)
    wins = (history['result'] == 'Win').sum()
    losses = total - wins
    win_rate = (wins / total) * 100 if total else 0

    col3.metric("Total Predictions", total)
    col4.metric("Wins", wins)
    col5.metric("Accuracy (%)", round(win_rate, 2))

    # Animated Win/Loss Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=(history['result'] == 'Win').astype(int).cumsum(),
        mode='lines+markers',
        line=dict(color='limegreen', width=3),
        name="Cumulative Wins",
    ))
    fig.update_layout(
        title="Win Trend Over Time",
        xaxis_title="Prediction #",
        yaxis_title="Cumulative Wins",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show styled table
    def highlight_result(row):
        color = 'green' if row['result'] == 'Win' else 'red'
        return [f'color: {color}' if col == 'result' else '' for col in row.index]

    st.dataframe(history.tail(20).style.apply(highlight_result, axis=1))

else:
    st.info("No predictions yet.")

# --- Step 4: Loss Recovery Suggestion ---
st.header("🛡️ Loss Recovery Calculator")

if not history.empty:
    last_result = history.iloc[-1]['result']
    if last_result == "Loss":
        st.warning("⚠️ Last round was a Loss. Suggesting Recovery Bet...")
        recovery_target = round(predicted_multiplier + 0.10, 2)
        st.info(f"🎯 Suggested Recovery Target: **{recovery_target}x** (Next Predicted must beat {recovery_target}x)")
    else:
        st.success("✅ No recovery needed. Last round was a Win!")
else:
    st.info("Submit some results to activate recovery suggestions.")
