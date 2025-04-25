import streamlit as st
import hashlib
import hmac
import struct
import binascii

st.title("Crash Predictor - Seed-Based Calculator")

# ========= Provably Fair Algorithm ==========
def get_hmac_sha256(server_seed, client_seed_nonce):
    return hmac.new(
        server_seed.encode(),
        client_seed_nonce.encode(),
        hashlib.sha256
    ).hexdigest()

def calculate_crash_point(hmac_digest):
    hs = hmac_digest
    if int(hs[:8], 16) == 0:
        return 1.0

    h = int(hs[:13], 16)
    e = 2**52
    crash = (100 * e - h) / (e - h)
    return max(1.0, round(crash, 2))

def compute_next_multiplier(server_seed, client_seed, nonce):
    client_seed_nonce = f"{client_seed}:{nonce}"
    hmac_digest = get_hmac_sha256(server_seed, client_seed_nonce)
    return calculate_crash_point(hmac_digest)

# ========== UI Inputs ==========
st.header("Seed Inputs")

server_seed = st.text_input("Server Seed", value="01a24e141597617f167daef1901514260952f2e64a49adcd829e6813c80305ac")
client_seed = st.text_input("Client Seed", value="97439433b0745d23902d5c53fd1de03d")
nonce = st.number_input("Nonce", value=15141, step=1)

st.header("Optional: Recent Crash Data")
recent_crashes = st.text_area("Recent Crash Multipliers (comma-separated)", value="1.52,1.11,1.32,2.58,2.35,3.99,1.19,1.05")

# ========== Calculation ==========
if st.button("Predict Next Multiplier"):
    predicted = compute_next_multiplier(server_seed, client_seed, nonce)
    st.subheader(f"🎯 Predicted Crash Multiplier: `{predicted}`")

    # Optional display of recent values
    if recent_crashes:
        try:
            values = [float(x.strip()) for x in recent_crashes.split(",") if x.strip()]
            st.line_chart(values + [predicted])
        except:
            st.error("Invalid crash data format.")

# ========== Info ==========
st.caption("Prediction is based on provably fair HMAC-based hash using seeds and nonce.")
