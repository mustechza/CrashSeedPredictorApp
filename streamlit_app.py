import streamlit as st
import pandas as pd

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data

from models.random_forest import RFModel
from training.backtest import run_backtest
from analytics.dashboard import plot_equity
from analytics.tracker import calculate_metrics

st.set_page_config(layout="wide")
st.title("🚀 Crash AI Live Dashboard")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# UPLOAD
# -------------------------------
st.sidebar.header("📂 Upload JSON")

uploaded_file = st.sidebar.file_uploader("Upload file", type=["json"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data uploaded!")

# -------------------------------
# LIVE INPUT
# -------------------------------
st.sidebar.header("➕ Add New Round")

new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01)

if st.sidebar.button("Add Round"):
    if st.session_state.df is not None:
        now = pd.Timestamp.now()

        new_row = pd.DataFrame([{
            "rate": str(new_rate),
            "crash": float(new_rate),
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": "live",
            "salt": "live",
            "fetchedAt": now
        }])

        st.session_state.df = pd.concat(
            [st.session_state.df, new_row],
            ignore_index=True
        )

        st.success(f"Added: {new_rate}")
    else:
        st.warning("Upload data first")

# -------------------------------
# MAIN
# -------------------------------
if st.session_state.df is None:
    st.info("Upload a dataset to begin")
    st.stop()

# Clean data
df = clean_data(st.session_state.df)

st.subheader("📊 Live Data (Latest Rounds)")

# Ensure datetime is valid
df["fetchedAt"] = pd.to_datetime(df["fetchedAt"], errors="coerce")

# Sort properly
df_sorted = df.sort_values("fetchedAt", ascending=False).reset_index(drop=True)

# Show latest 20
st.dataframe(df_sorted.head(20), use_container_width=True)
# -------------------------------
# TRAIN MODEL
# -------------------------------
if len(df) < 30:
    st.warning("Need at least 30 clean rows")
    st.stop()

X_train, X_test, y_train, y_test = prepare_data(df)

model = RFModel()
model.train(X_train, y_train)

# -------------------------------
# SAFE LIVE PREDICTION (FIXED)
# -------------------------------
last_row = df.iloc[[-1]]  # KEEP AS DATAFRAME

# Ensure feature consistency
if not all(f in last_row.columns for f in FEATURES):
    st.error("Feature mismatch — check data pipeline")
    st.stop()

X_live = last_row[FEATURES]

# DEBUG (optional)
# st.write("Train shape:", X_train.shape)
# st.write("Live shape:", X_live.shape)

prediction = model.predict(X_live)[0]
proba = model.predict_proba(X_live)[0][1]

# -------------------------------
# DISPLAY PREDICTION
# -------------------------------
st.subheader("🤖 Next Prediction")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("✅ BET (≥ 1.5x)")
    else:
        st.error("⛔ SKIP")

with col2:
    st.metric("Confidence", f"{proba:.2%}")

# -------------------------------
# BACKTEST
# -------------------------------
history = run_backtest(df, model)
metrics = calculate_metrics(history)

st.subheader("📊 Performance")

col1, col2 = st.columns(2)
col1.metric("Profit", round(metrics["profit"], 2))
col2.metric("Max Drawdown", round(metrics["max_drawdown"], 2))

# -------------------------------
# EQUITY CURVE
# -------------------------------
st.subheader("📈 Equity Curve")
plot_equity(history)

# -------------------------------
# QUICK STATS
# -------------------------------
st.subheader("📌 Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Rounds", len(df))
col2.metric("Last Crash", round(df["crash"].iloc[-1], 2))
col3.metric("Hit Rate ≥1.5x", f"{(df['crash'] >= 1.5).mean():.2%}")

# -------------------------------
# FULL MULTIPLIER HISTORY
# -------------------------------
st.subheader("📈 Full Multiplier History (First → Last)")

# Ensure correct time order (oldest to newest)
df_history = df.sort_values(by="fetchedAt", ascending=True).reset_index(drop=True)

st.line_chart(df_history["crash"])

# Optional table view
with st.expander("📊 View Raw Multiplier List"):
    st.dataframe(
        df_history[["fetchedAt", "crash"]],
        use_container_width=True
    )
