import streamlit as st
import pandas as pd
import numpy as np
import hmac
import hashlib
import os
from datetime import datetime
import altair as alt

# Constants
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

# --- Load History ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "client_seed", "nonce", "actual", "predicted", "result", "recovery_bet"])

# --- Save Prediction ---
def save_prediction(client_seed, nonce, actual, predicted):
    df = load_history()
    result = "Win" if actual >= predicted else "Loss"

    recovery_bet = 1
    if result == "Loss":
        recovery_bet = round(2 * (1 / predicted), 2)

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "client_seed": client_seed,
        "nonce": nonce,
        "actual": actual,
        "predicted": predicted,
        "result": result,
        "recovery_bet": recovery_bet
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
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

# --- App UI ---
st.set_page_config(page_title="Crash Predictor", layout="wide")

st.title("🚀 Crash Predictor (Seed Based)")

col1, col2 = st.columns([2, 2])

with col1:
    st.header("Submit Live Data")
    with st.form("live_data_form"):
        client_seed = st.text_input("Client Seed", "97439433b0745d23902d5c53fd1de03d")
        last_nonce = load_history()["nonce"].max() if not load_history().empty else 0
        nonce = st.number_input("Nonce (autofilled)", value=int(last_nonce) + 1, step=1)
        submitted = st.form_submit_button("Predict")

with col2:
    if submitted:
        # Predict
        predicted_multiplier = get_multiplier_from_seed(SERVER_SEED, client_seed, int(nonce))

        # Adjust prediction if needed
        if predicted_multiplier > 4:
            suggested_multiplier = 2.0
        else:
            suggested_multiplier = predicted_multiplier

        st.metric("🎯 Predicted Multiplier", f"{predicted_multiplier}x")
        st.metric("✅ Suggested Multiplier", f"{suggested_multiplier}x")

        st.success("Now enter the *actual* result below to record and track accuracy.")

st.divider()

st.header("Submit Actual Result")
col3, col4 = st.columns(2)

with col3:
    actual_multiplier = st.number_input("Actual Crash Multiplier", value=1.00, step=0.01, format="%.2f")
    confirm_actual = st.button("Submit Result")

if confirm_actual and submitted:
    save_prediction(client_seed, nonce, actual_multiplier, suggested_multiplier)
    st.success("Result saved with feedback!")

# --- History Section ---
st.divider()
st.subheader("📊 Prediction History (Last 15)")
history = load_history().tail(15)

if not history.empty:
    def color_result(val):
        color = 'green' if val == 'Win' else 'red'
        return f'color: {color}'

    st.dataframe(history.style.applymap(color_result, subset=["result"]), use_container_width=True)

    win_rate = (history['result'] == 'Win').mean() * 100
    st.metric("Accuracy (%)", f"{win_rate:.2f}%")

    # Animated chart
    chart_data = history.copy()
    chart_data["index"] = range(len(chart_data))

    line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X('index', title='Prediction Number'),
        y=alt.Y('actual', title='Actual Multiplier'),
        color=alt.Color('result', scale=alt.Scale(domain=['Win', 'Loss'], range=['green', 'red']))
    ).properties(
        width=800,
        height=300
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)
else:
    st.info("No predictions yet.")

# --- Sidebar Reset Button ---
with st.sidebar:
    st.title("Settings")
    if st.button("Reset & Backup History"):
        reset_with_backup()
