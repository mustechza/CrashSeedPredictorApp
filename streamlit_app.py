import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from playwright.sync_api import sync_playwright

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="wide")
st.title("🚀 Crash AI v3 - Playwright Engine")

# -------------------------------
# STORAGE
# -------------------------------
CSV_FILE = "crash_live.csv"

def append_csv(row):
    file_exists = os.path.exists(CSV_FILE)
    row.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)

# -------------------------------
# PLAYWRIGHT SCRAPER
# -------------------------------
class CrashScraper:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

        self.page.goto("https://bc.game/crash")
        self.page.wait_for_timeout(8000)

        self.last_value = None

    def get_latest(self):
        try:
            elements = self.page.query_selector_all("td.text-secondary")

            if not elements:
                return None

            latest_text = elements[-1].inner_text().replace("x", "").strip()
            latest = float(latest_text)

            if latest == self.last_value:
                return None

            self.last_value = latest
            return latest

        except Exception as e:
            print("Scraper error:", e)
            return None

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    if os.path.exists(CSV_FILE):
        st.session_state.df = pd.read_csv(CSV_FILE)
    else:
        st.session_state.df = None

if "scraper" not in st.session_state:
    st.session_state.scraper = CrashScraper()

# -------------------------------
# MODEL
# -------------------------------
@st.cache_resource
def get_model(X, y):
    model = RFModel()
    model.train(X, y)
    return model

# -------------------------------
# LIVE MODE
# -------------------------------
run_live = st.sidebar.toggle("▶️ Live Playwright Feed")

if run_live:
    new_rate = st.session_state.scraper.get_latest()

    if new_rate is not None:
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

        if st.session_state.df is None:
            st.session_state.df = row
        else:
            st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)

        append_csv(row)

        st.session_state.df = st.session_state.df.tail(500)

    time.sleep(2)
    st.rerun()

# -------------------------------
# CHECK DATA
# -------------------------------
if st.session_state.df is None:
    st.info("Waiting for live Playwright data...")
    st.stop()

# -------------------------------
# CLEAN
# -------------------------------
df = clean_data(st.session_state.df)

df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 50:
    st.warning("Collecting data... need 50+ rounds")
    st.stop()

# -------------------------------
# TRAIN
# -------------------------------
X_train, X_test, y_train, y_test = prepare_data(df_ml)
model = get_model(X_train, y_train)

# -------------------------------
# CONTEXT
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    return {
        "volatility": last_10.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3)
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]

    std = last_20.std()
    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()

    if std > 2.5:
        regime = "⚡ VOLATILE"
    elif low_ratio > 0.6:
        regime = "🔴 CHOPPY"
    elif high_ratio > 0.4:
        regime = "🟢 HOT"
    else:
        regime = "🟡 NORMAL"

    return {"regime": regime, "std": std}

regime_data = detect_regime(df_ml)

# -------------------------------
# ML PREDICTION
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# CONFIDENCE
# -------------------------------
confidence = proba * 50

if ctx["volatility"] > 1.5:
    confidence += 15
if ctx["low_streak"] >= 6:
    confidence += 20
if ctx["high_streak"] >= 5:
    confidence -= 15

confidence = max(0, min(100, confidence))

# -------------------------------
# SIGNAL
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
elif confidence > 60:
    signal = "✅ BET"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
else:
    signal = "❌ SKIP"

# -------------------------------
# UI
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Signal", signal)
c2.metric("Confidence", f"{confidence:.1f}%")
c3.metric("ML Prob", f"{proba:.2%}")
c4.metric("Regime", regime_data["regime"])

st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash Chart")
st.line_chart(df_ml["crash"])
