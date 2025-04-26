import streamlit as st
import pandas as pd
import numpy as np
import hmac
import hashlib
import os
import time
from datetime import datetime

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

# --- Load and Save History ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "client_seed", "nonce", "actual", "predicted", "result"])

def save_prediction(client_seed, nonce, actual, predicted):
    df = load_history()
    result = "Win" if actual >= predicted else "Loss"
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

# --- Streamlit Layout ---
st.set_page_config(page_title="Crash Predictor", layout="wide")
st.title("🎯 Crash Predictor: Seed Logic + Live Feedback")
st.sidebar.title("Controls")

if st.sidebar.button("Reset & Backup History"):
    reset_with_backup()

st.header("1. Submit Live Crash Result")

# Autofill client_seed and nonce
history = load_history()
if not history.empty:
    last_nonce = history['nonce'].max() + 1
    default_client_seed = history['client_seed'].iloc[-1]
else:
    last_nonce = 1
    default_client_seed = "your_default_client_seed_here"

with st.form("live_result_form"):
    client_seed = st.text_input("Client Seed", default_client_seed)
    nonce = st.number_input("Nonce", value=int(last_nonce), step=1)
    actual_multiplier = st.number_input("Actual Crash Multiplier", value=1.00, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Submit")

if submitted:
    seed_prediction = get_multiplier_from_seed(SERVER_SEED, client_seed, int(nonce))
    st.success(f"Seed-based Prediction: **{seed_prediction}x**")

    save_prediction(client_seed, nonce, actual_multiplier, seed_prediction)
    st.success("Result saved.")

    history = load_history()

st.markdown("---")

# --- Prediction History ---
st.header("📊 Prediction History (Last 20)")
if not history.empty:
    def highlight_result(row):
        color = 'green' if row['result'] == 'Win' else 'red'
        return [f'color: {color}' if col == 'result' else '' for col in row.index]

    st.dataframe(history.tail(20).style.apply(highlight_result, axis=1))
else:
    st.info("🕒 No predictions yet.")

st.markdown("---")

# --- Performance Summary ---
st.markdown("## 📋 Performance Summary")
if not history.empty and "result" in history.columns:
    total = len(history)
    wins = (history['result'] == 'Win').sum()
    losses = total - wins
    win_rate = (wins / total) * 100 if total else 0

    streak = 0
    last_result = None
    for result in reversed(history['result']):
        if last_result is None or result == last_result:
            streak += 1
            last_result = result
        else:
            break
    current_streak = f"{streak} {last_result}s" if last_result else "N/A"
else:
    total = wins = losses = win_rate = 0
    current_streak = "N/A"

win_color = "green" if win_rate >= 55 else ("orange" if 45 <= win_rate < 55 else "red")
streak_color = "green" if "Win" in current_streak else "red"

col1, col2, col3, col4 = st.columns(4)

with col1:
    placeholder = st.empty()
    for i in range(0, wins+1):
        placeholder.markdown(f"<h3 style='text-align: center; color: green;'>{i}</h3>", unsafe_allow_html=True)
        time.sleep(0.005)
    st.markdown("<p style='text-align: center;'>Wins</p>", unsafe_allow_html=True)

with col2:
    placeholder = st.empty()
    for i in range(0, losses+1):
        placeholder.markdown(f"<h3 style='text-align: center; color: red;'>{i}</h3>", unsafe_allow_html=True)
        time.sleep(0.005)
    st.markdown("<p style='text-align: center;'>Losses</p>", unsafe_allow_html=True)

with col3:
    placeholder = st.empty()
    for i in range(0, int(win_rate)+1):
        placeholder.markdown(f"<h3 style='text-align: center; color: {win_color};'>{i}%</h3>", unsafe_allow_html=True)
        time.sleep(0.005)
    st.markdown("<p style='text-align: center;'>Win Rate</p>", unsafe_allow_html=True)

with col4:
    st.markdown(f"<h3 style='text-align: center; color: {streak_color};'>{current_streak}</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Current Streak</p>", unsafe_allow_html=True)

st.markdown("---")
