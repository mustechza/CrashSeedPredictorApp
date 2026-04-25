import streamlit as st
import pandas as pd

from data.loader import load_data
from data.cleaner import clean_data
from training.trainer import prepare_data
from training.backtest import run_backtest

from models.random_forest import RFModel
from analytics.dashboard import plot_equity
from analytics.tracker import calculate_metrics

st.set_page_config(layout="wide")
st.title("🚀 Crash AI Live Dashboard")

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# SIDEBAR - UPLOAD
# -------------------------------
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success("Data loaded successfully!")

# -------------------------------
# SIDEBAR - LIVE INPUT
# -------------------------------
st.sidebar.header("➕ Add New Round")

new_rate = st.sidebar.number_input("Crash Multiplier (rate)", min_value=1.0, step=0.01)

if st.sidebar.button("Add Round"):
    if st.session_state.df is not None:
        new_row = pd.DataFrame([{
            "rate": str(new_rate),
            "prepareTime": pd.Timestamp.now(),
            "beginTime": pd.Timestamp.now(),
            "endTime": pd.Timestamp.now(),
            "hash": "live_input",
            "salt": "live_input",
            "fetchedAt": pd.Timestamp.now()
        }])

        st.session_state.df = pd.concat(
            [st.session_state.df, new_row],
            ignore_index=True
        )

        st.success(f"Added new round: {new_rate}")
    else:
        st.warning("Upload data first!")

# -------------------------------
# MAIN APP
# -------------------------------
if st.session_state.df is None:
    st.info("Upload a JSON file to begin")
    st.stop()

# Clean data
df = clean_data(st.session_state.df)

# -------------------------------
# DISPLAY DATA
# -------------------------------
st.subheader("📊 Live Data")
st.dataframe(df.tail(20), use_container_width=True)

# -------------------------------
# MODEL TRAINING
# -------------------------------
if len(df) < 30:
    st.warning("Need at least 30 rounds for stable predictions")
    st.stop()

X_train, X_test, y_train, y_test = prepare_data(df)

model = RFModel()
model.train(X_train, y_train)

# -------------------------------
# LIVE PREDICTION
# -------------------------------
features = [
    "rolling_mean",
    "rolling_std",
    "round_duration",
    "prep_gap",
    "delta"
]

last_row = df.iloc[-1]
X_live = [last_row[features].values]

prediction = model.predict(X_live)[0]
proba = model.predict_proba(X_live)[0][1]

st.subheader("🤖 Next Round Prediction")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("✅ BET (Target ≥ 1.5x)")
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
# EXTRA INSIGHTS
# -------------------------------
st.subheader("📌 Quick Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rounds", len(df))
col2.metric("Last Crash", round(df["crash"].iloc[-1], 2))
col3.metric(
    "Hit Rate ≥1.5x",
    f"{(df['crash'] >= 1.5).mean():.2%}"
)
