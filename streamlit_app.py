import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

st.set_page_config(layout="wide")
st.title("🚀 Crash AI v3 - Regime Adaptive Engine with Continuous Learning")

# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "last_training_time" not in st.session_state:
    st.session_state.last_training_time = None
if "rounds_since_training" not in st.session_state:
    st.session_state.rounds_since_training = 0
if "training_history" not in st.session_state:
    st.session_state.training_history = []
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = []

# -------------------------------
# MODEL MANAGEMENT WITH CONTINUOUS LEARNING
# -------------------------------
def train_or_retrain_model(df, force=False):
    """Train or retrain model with current data"""
    min_rounds = 50
    retrain_threshold = 10  # Retrain every 10 new rounds
    
    if len(df) < min_rounds:
        st.warning(f"Need at least {min_rounds} rounds. Currently: {len(df)}")
        return None
    
    # Check if retraining is needed
    needs_retraining = (
        st.session_state.model is None or
        force or
        st.session_state.rounds_since_training >= retrain_threshold
    )
    
    if needs_retraining:
        with st.spinner("🔄 Training AI with latest data..."):
            X_train, X_test, y_train, y_test = prepare_data(df)
            model = RFModel()
            model.train(X_train, y_train)
            
            # Calculate accuracy on test set
            accuracy = model.model.score(X_test, y_test)
            
            # Log training event
            st.session_state.training_history.append({
                "timestamp": datetime.now(),
                "rounds": len(df),
                "accuracy": accuracy,
                "new_rounds_since_last": st.session_state.rounds_since_training
            })
            
            st.session_state.model = model
            st.session_state.last_training_time = datetime.now()
            st.session_state.rounds_since_training = 0
            
            return model
    
    return st.session_state.model

def incremental_learn(df, new_round):
    """Add new round and retrain if needed"""
    st.session_state.df = pd.concat([df, new_round], ignore_index=True)
    st.session_state.rounds_since_training += 1
    
    # Retrain model
    model = train_or_retrain_model(st.session_state.df)
    
    return model

# -------------------------------
# LIVE INPUT WITH AUTO-LEARNING
# -------------------------------
st.sidebar.markdown("## 🎮 Add New Round")
st.sidebar.markdown("Each new round helps the AI learn and improve!")

new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01, key="live_multiplier")

col1, col2 = st.sidebar.columns(2)
with col1:
    add_button = st.button("➕ Add Round", use_container_width=True)
with col2:
    auto_learn = st.checkbox("🤖 Auto-learn", value=True, help="Automatically retrain AI after every 10 rounds")

if add_button and st.session_state.df is not None:
    now = pd.Timestamp.now()
    
    # Calculate features for the new round
    last_round = st.session_state.df.iloc[-1] if len(st.session_state.df) > 0 else None
    
    row = pd.DataFrame([{
        "rate": str(new_rate),
        "crash": float(new_rate),
        "prepareTime": now,
        "beginTime": now,
        "endTime": now,
        "hash": f"live_{now.timestamp()}",
        "salt": "live",
        "fetchedAt": now
    }])
    
    # Add to dataset
    st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)
    st.session_state.rounds_since_training += 1
    
    # Show immediate feedback
    st.sidebar.success(f"✅ Round added! New round: {new_rate}x")
    
    # Auto-learn if enabled
    if auto_learn and st.session_state.rounds_since_training >= 10:
        st.sidebar.info("🔄 Auto-learning triggered...")
        train_or_retrain_model(st.session_state.df, force=True)
    
    # Log prediction accuracy if we had a prediction
    if st.session_state.predictions_log:
        last_pred = st.session_state.predictions_log[-1]
        if last_pred["actual"] is None:
            last_pred["actual"] = new_rate
            last_pred["was_correct"] = (
                (last_pred["prediction"] == "🔥 STRONG BET" and new_rate >= last_pred["target"]) or
                (last_pred["prediction"] == "✅ BET" and new_rate >= last_pred["target"]) or
                (last_pred["prediction"] == "⚠️ SMALL BET" and new_rate >= last_pred["target"]) or
                (last_pred["prediction"] == "❌ SKIP")
            )
            st.sideend insight
    
    st.rerun()

# -------------------------------
# BATCH UPLOAD FOR HISTORICAL DATA
# -------------------------------
st.sidebar.markdown("## 📁 Historical Data")
file = st.sidebar.file_uploader("Upload JSON (once)", type=["json"])

if file and st.session_state.df is None:
    with st.spinner("Loading historical data..."):
        st.session_state.df = load_data(file)
        st.success(f"✅ Loaded {len(st.session_state.df)} historical rounds!")
        
        # Initial training
        train_or_retrain_model(st.session_state.df, force=True)
        st.rerun()

# -------------------------------
# BULK ADD MULTIPLE ROUNDS
# -------------------------------
with st.sidebar.expander("📊 Batch Add Multiple Rounds"):
    st.markdown("Add multiple rounds at once (comma-separated)")
    batch_multipliers = st.text_input("Multipliers (e.g., 1.5, 2.3, 1.2, 4.5)")
    
    if st.button("Add Batch"):
        if batch_multipliers and st.session_state.df is not None:
            multipliers = [float(x.strip()) for x in batch_multipliers.split(",")]
            now = pd.Timestamp.now()
            
            new_rows = []
            for i, m in enumerate(multipliers):
                new_rows.append({
                    "rate": str(m),
                    "crash": m,
                    "prepareTime": now + timedelta(seconds=i),
                    "beginTime": now + timedelta(seconds=i),
                    "endTime": now + timedelta(seconds=i+5),
                    "hash": f"batch_{now.timestamp()}_{i}",
                    "salt": "batch",
                    "fetchedAt": now + timedelta(seconds=i)
                })
            
            batch_df = pd.DataFrame(new_rows)
            st.session_state.df = pd.concat([st.session_state.df, batch_df], ignore_index=True)
            st.session_state.rounds_since_training += len(multipliers)
            
            st.sidebar.success(f"✅ Added {len(multipliers)} rounds!")
            
            if auto_learn and st.session_state.rounds_since_training >= 10:
                train_or_retrain_model(st.session_state.df, force=True)
            
            st.rerun()

# -------------------------------
# MODEL PERFORMANCE DASHBOARD
# -------------------------------
if st.session_state.df is not None and st.session_state.model is not None:
    st.sidebar.markdown("## 📈 Model Status")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Rounds", len(st.session_state.df))
    with col2:
        st.metric("Rounds since training", st.session_state.rounds_since_training)
    
    if st.session_state.last_training_time:
        st.sidebar.caption(f"Last trained: {st.session_state.last_training_time.strftime('%H:%M:%S')}")
    
    if st.sidebar.button("🔄 Force Retrain Now"):
        train_or_retrain_model(st.session_state.df, force=True)
        st.sidebar.success("Model retrained!")
        st.rerun()
    
    # Training history
    if st.session_state.training_history:
        with st.sidebar.expander("📊 Training History"):
            history_df = pd.DataFrame(st.session_state.training_history)
            st.line_chart(history_df.set_index("timestamp")["accuracy"])

# -------------------------------
# CHECK DATA
# -------------------------------
if st.session_state.df is None:
    st.info("📤 Upload your JSON file to begin. The AI will learn from every round you add!")
    st.stop()

# -------------------------------
# CLEAN DATA
# -------------------------------
df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 50:
    st.warning(f"Need at least 50 rounds for AI training. Currently: {len(df_ml)}")
    st.info("📝 Add more rounds using the sidebar to help the AI learn!")
    st.stop()

# -------------------------------
# TRAIN/GET MODEL
# -------------------------------
model = train_or_retrain_model(df_ml)

if model is None:
    st.error("Model training failed. Please add more data.")
    st.stop()

# -------------------------------
# CONTEXT FEATURES
# -------------------------------
def get_context(df):
    last_10 = df.tail(10)["crash"]
    last_50 = df.tail(50)["crash"]
    
    return {
        "volatility": last_10.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3),
        "trend": "up" if last_50.mean() < last_10.mean() else "down",
        "avg_10": last_10.mean(),
        "avg_50": last_50.mean()
    }

ctx = get_context(df_ml)

# -------------------------------
# REGIME DETECTION (ENHANCED)
# -------------------------------
def detect_regime(df):
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    
    avg = last_20.mean()
    std = last_20.std()
    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()
    
    # Detect regime shifts
    trend_shift = abs(last_50.mean() - avg) / max(last_50.std(), 0.1)
    
    if std > 2.5:
        regime = "⚡ VOLATILE"
        confidence_boost = 10
    elif low_ratio > 0.6:
        regime = "🔴 CHOPPY"
        confidence_boost = -20
    elif high_ratio > 0.4:
        regime = "🟢 HOT"
        confidence_boost = 15
    else:
        regime = "🟡 NORMAL"
        confidence_boost = 0
    
    # Adjust for trend shifts
    if trend_shift > 1.5:
        regime += " (Trend Shift!)"
        confidence_boost += 5
    
    return {
        "regime": regime,
        "avg": avg,
        "std": std,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "confidence_boost": confidence_boost,
        "trend_shift": trend_shift
    }

regime_data = detect_regime(df_ml)

# -------------------------------
# ML PREDICTION WITH CONFIDENCE INTERVAL
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

# Get prediction probability
proba = model.predict_proba(X_live)[0][1]

# Get feature importance for explanation
feature_importance = dict(zip(FEATURES, model.model.feature_importances_))
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]

# -------------------------------
# CONFIDENCE ENGINE (ENHANCED)
# -------------------------------
confidence = proba * 50

if ctx["volatility"] > 1.5:
    confidence += 15
elif ctx["volatility"] > 2.5:
    confidence += 25

if ctx["low_streak"] >= 6:
    confidence += 20
elif ctx["low_streak"] >= 4:
    confidence += 10

if ctx["high_streak"] >= 5:
    confidence -= 15

# Regime adjustment
confidence += regime_data["confidence_boost"]

# Learning progress boost - model gets more confident with more data
learning_progress = min(len(df_ml) / 500, 0.2)  # Up to 20% boost at 500 rounds
confidence *= (1 + learning_progress)

confidence = max(0, min(100, confidence))

# -------------------------------
# ADAPTIVE MULTIPLIER ENGINE (ENHANCED)
# -------------------------------
def evaluate_multiplier(df, target, window=100):
    """Enhanced evaluation with risk-adjusted returns"""
    balance = 0
    stake = 1
    wins = 0
    total_trades = 0
    
    start = max(30, len(df) - window)
    
    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]
        
        if crash >= target:
            profit = stake * (target - 1)
            balance += profit
            wins += 1
        else:
            balance -= stake
        
        total_trades += 1
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    sharpe = balance / np.std([balance]) if total_trades > 1 else 0
    
    return balance, win_rate, sharpe

def get_adaptive_multipliers(df):
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    results = []
    for m in multipliers:
        profit, win_rate, sharpe = evaluate_multiplier(df, m)
        results.append({
            "m": m, 
            "profit": profit, 
            "win_rate": win_rate,
            "sharpe": sharpe,
            "score": profit * win_rate * (1 + sharpe)
        })
    
    res = pd.DataFrame(results)
    
    # Categorize by risk
    low_risk = res[res["m"] <= 1.8]
    medium_risk = res[(res["m"] > 1.8) & (res["m"] <= 2.5)]
    high_risk = res[res["m"] > 2.5]
    
    return {
        "low": low_risk.sort_values("score", ascending=False).iloc[0]["m"] if len(low_risk) > 0 else 1.5,
        "mid": medium_risk.sort_values("score", ascending=False).iloc[0]["m"] if len(medium_risk) > 0 else 2.0,
        "high": high_risk.sort_values("score", ascending=False).iloc[0]["m"] if len(high_risk) > 0 else 3.0,
        "table": res.sort_values("score", ascending=False)
    }

adaptive = get_adaptive_multipliers(df_ml)

# -------------------------------
# SIGNAL ENGINE (REGIME AWARE)
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]
    bet_size = "Large (5-10%)"
elif confidence > 65:
    signal = "✅ BET"
    if regime_data["regime"] == "🟢 HOT":
        target = adaptive["high"]
        bet_size = "Medium (3-5%)"
    else:
        target = adaptive["mid"]
        bet_size = "Small-Medium (2-3%)"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
    if regime_data["regime"] == "🔴 CHOPPY":
        target = adaptive["low"]
        bet_size = "Tiny (0.5-1%)"
    else:
        target = adaptive["mid"]
        bet_size = "Small (1-2%)"
else:
    signal = "❌ SKIP"
    target = None
    bet_size = "None"

# Log prediction
st.session_state.predictions_log.append({
    "timestamp": datetime.now(),
    "signal": signal,
    "target": target,
    "confidence": confidence,
    "regime": regime_data["regime"],
    "actual": None,  # Will be filled when round ends
    "was_correct": None
})

# Keep only last 100 predictions
if len(st.session_state.predictions_log) > 100:
    st.session_state.predictions_log = st.session_state.predictions_log[-100:]

# Calculate recent accuracy
recent_predictions = [p for p in st.session_state.predictions_log if p["was_correct"] is not None]
recent_accuracy = sum(p["was_correct"] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0

# -------------------------------
# UI - TOP DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Signal", signal)
col2.metric("Confidence", f"{confidence:.1f}%")
col3.metric("ML Prob", f"{proba:.2%}")
col4.metric("🎯 Target", f"{target}x" if target else "No Trade")
col5.metric("🧠 Regime", regime_data["regime"])
col6.metric("📊 Recent Accuracy", f"{recent_accuracy:.1%}")

# Learning progress bar
st.progress(min(len(df_ml) / 500, 1.0), text=f"🤖 Learning Progress: {len(df_ml)}/500 rounds for optimal performance")

# -------------------------------
# BET SUGGESTION
# -------------------------------
if target:
    st.info(f"💡 **Suggested Action**: {signal} at {target}x | {bet_size} of bankroll")
    
    # Kelly Criterion calculation
    implied_prob = 1 / target if target else 0
    edge = proba - implied_prob
    if edge > 0:
        kelly = edge / (1 - implied_prob)
        kelly = max(0, min(0.25, kelly))
        st.caption(f"📐 Kelly Criterion suggests: {kelly:.1%} of bankroll")

# -------------------------------
# LAST 10 MULTIPLIERS
# -------------------------------
st.markdown("### 📉 Last 10 Multipliers")

last_10 = df_ml["crash"].tail(10).to_numpy()[::-1]
cols = st.columns(10)

for i, val in enumerate(last_10):
    color = "#00ff00" if val >= 2 else "#ff4444"
    cols[i].markdown(
        f"""
        <div style="
            background-color:{color};
            padding:10px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-weight:bold;">
            {val:.2f}x
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# PREDICTION EXPLANATION
# -------------------------------
with st.expander("🧠 Why did the AI make this decision?"):
    st.markdown(f"**Top Factors Influencing Decision:**")
    for feature, importance in top_features:
        st.write(f"- {feature}: {importance:.2%}")
    
    st.markdown(f"**Context Analysis:**")
    st.write(f"- Volatility: {ctx['volatility']:.2f}")
    st.write(f"- Trend: {ctx['trend']} (10-period avg {ctx['avg_10']:.2f} vs 50-period {ctx['avg_50']:.2f})")
    st.write(f"- Low Streak: {ctx['low_streak']} rounds under 2x")
    st.write(f"- High Streak: {ctx['high_streak']} rounds over 3x")

# -------------------------------
# INSIGHTS
# -------------------------------
with st.expander("📊 AI + Regime Insights"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Volatility: {ctx['volatility']:.2f}")
        st.write(f"Low streak: {ctx['low_streak']}")
        st.write(f"High streak: {ctx['high_streak']}")
        st.write(f"Trend: {ctx['trend']}")
    
    with col2:
        st.write(f"Regime avg: {regime_data['avg']:.2f}")
        st.write(f"Regime std: {regime_data['std']:.2f}")
        st.write(f"Low ratio: {regime_data['low_ratio']:.2%}")
        st.write(f"High ratio: {regime_data['high_ratio']:.2%}")

# -------------------------------
# MULTIPLIER PERFORMANCE TABLE
# -------------------------------
st.subheader("🎯 Adaptive Multiplier Performance")
st.dataframe(adaptive["table"], use_container_width=True)

# -------------------------------
# LEARNING METRICS
# -------------------------------
if st.session_state.training_history:
    st.subheader("📈 Model Learning Progress")
    history_df = pd.DataFrame(st.session_state.training_history)
    
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(history_df.set_index("timestamp")["accuracy"])
        st.caption("Model Accuracy Over Time")
    with col2:
        st.line_chart(history_df.set_index("timestamp")["rounds"])
        st.caption("Training Data Size")

# -------------------------------
# DATA VIEW
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

st.subheader("📈 Crash History")
st.line_chart(df_ml["crash"])

# -------------------------------
# EXPORT FUNCTIONALITY
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Export Trained Model"):
        import joblib
        joblib.dump(st.session_state.model, "crash_ai_model.pkl")
        st.success("Model saved as crash_ai_model.pkl")

with col2:
    if st.button("📥 Export Prediction Log"):
        log_df = pd.DataFrame(st.session_state.predictions_log)
        csv = log_df.to_csv(index=False)
        st.download_button("Download Log", csv, "prediction_log.csv")

# Auto-refresh option for live monitoring
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh dashboard (5 sec)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()
