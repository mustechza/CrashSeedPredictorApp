import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crash Predictor", layout="centered")
st.title("🚀 Crash Seed Predictor App")

CSV_FILE = "training_data.csv"

# =====================
# 📂 Load or create training CSV
# =====================
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["mean", "std", "last", "max", "min", "change", "target"])
    df.to_csv(CSV_FILE, index=False)

def load_training_data():
    df = pd.read_csv(CSV_FILE)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

# =====================
# 🤖 Train model
# =====================
def train_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# =====================
# 🧠 Feature Extraction
# =====================
def extract_features(vals):
    vals = vals[-10:]
    return pd.DataFrame([{
        "mean": np.mean(vals),
        "std": np.std(vals),
        "last": vals[-1],
        "max": max(vals),
        "min": min(vals),
        "change": vals[-1] - vals[-2] if len(vals) > 1 else 0
    }])

# =====================
# 🧮 Parse Input
# =====================
def parse_input(text):
    try:
        return [min(float(x.strip().lower().replace("x", "")), 10.5) for x in text.split(",") if x.strip()]
    except:
        return []

# =====================
# 📥 Input Section
# =====================
with st.form("prediction_form"):
    input_data = st.text_input("Enter last 10 crash multipliers (comma-separated)", value="1.2, 1.1, 2.3, 3.1, 1.9, 1.5, 1.0, 2.2, 1.6, 1.4")
    actual_value = st.text_input("Next actual multiplier (optional, for feedback)")
    submitted = st.form_submit_button("🔁 Submit & Retrain")

# =====================
# 🔮 Prediction
# =====================
recent = parse_input(input_data)
features = extract_features(recent) if len(recent) >= 10 else None

if features is not None:
    X, y = load_training_data()
    model = train_model(X, y)

    prediction = model.predict(features)[0]
    safe_target = round(prediction * 0.97, 2)

    st.subheader(f"🎯 Predicted: {prediction:.2f}")
    st.success(f"🛡️ Safe target (3% edge): {safe_target:.2f}")

    # Display Indicators
    st.write("📊 Stats on input:")
    st.write(f"Mean: {features['mean'].values[0]:.2f}")
    st.write(f"Std Dev: {features['std'].values[0]:.2f}")
    st.write(f"Last Change: {features['change'].values[0]:.2f}")

    # Feedback handling
    if submitted and actual_value:
        try:
            actual = float(actual_value)
            actual = min(actual, 10.5)
            new_row = features.copy()
            new_row["target"] = actual
            new_row.to_csv(CSV_FILE, mode="a", header=False, index=False)
            st.success("✅ Model updated with new feedback!")
        except:
            st.error("⚠️ Invalid actual value. Enter a valid number.")

# =====================
# 📈 Accuracy Trend
# =====================
df = pd.read_csv(CSV_FILE)
if len(df) >= 30:
    last_30 = df.tail(30)
    X_30 = last_30.drop(columns=["target"])
    y_30 = last_30["target"]
    preds = model.predict(X_30)

    results = pd.DataFrame({
        "Predicted": preds,
        "Actual": y_30,
        "Error": np.abs(preds - y_30),
        "Win/Loss": np.where(preds * 0.97 <= y_30, "✅ Win", "❌ Loss")
    })

    # Color-coded
    def highlight_winloss(val):
        color = "#d4edda" if val == "✅ Win" else "#f8d7da"
        return f"background-color: {color}"

    st.subheader("📊 Prediction Results (Last 30)")
    st.dataframe(results.style.applymap(highlight_winloss, subset=["Win/Loss"]))

    # Summary
    total = len(results)
    wins = (results["Win/Loss"] == "✅ Win").sum()
    losses = total - wins
    st.markdown(f"**✅ Wins:** {wins} | ❌ Losses: {losses} | 🧠 Accuracy: `{wins/total*100:.1f}%`")

    # Chart
    st.subheader("📈 Error Over Time")
    fig, ax = plt.subplots()
    ax.plot(results["Error"], marker="o", label="Absolute Error")
    ax.set_title("Prediction Error")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Not enough data for accuracy trend (need 30+ rows).")
