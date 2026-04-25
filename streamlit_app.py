import streamlit as st

from data.loader import load_data
from data.cleaner import clean_data
from training.trainer import prepare_data
from training.backtest import run_backtest

from models.random_forest import RFModel
from analytics.dashboard import plot_equity
from analytics.tracker import calculate_metrics

st.set_page_config(layout="wide")
st.title("🚀 Crash AI Trading Dashboard")

# Load data
df = load_data()
df = clean_data(df)

st.subheader("📊 Raw Data")
st.dataframe(df.tail())

# Train model
X_train, X_test, y_train, y_test = prepare_data(df)

model = RFModel()
model.train(X_train, y_train)

# Backtest
history = run_backtest(df, model)

# Metrics
metrics = calculate_metrics(history)

col1, col2 = st.columns(2)
col1.metric("Profit", round(metrics["profit"], 2))
col2.metric("Max Drawdown", round(metrics["max_drawdown"], 2))

# Plot
st.subheader("📈 Equity Curve")
plot_equity(history)
