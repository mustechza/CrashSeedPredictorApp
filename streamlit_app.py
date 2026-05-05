import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Crash AI v4 - Adaptive Trading Engine",
    page_icon="🚀"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .signal-strong {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-skip {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crash AI v4 - Regime Adaptive Engine with Continuous Learning")

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "df": None,
        "model": None,
        "last_training_time": None,
        "rounds_since_training": 0,
        "training_history": [],
        "predictions_log": [],
        "auto_learn": True,
        "last_data_hash": None,
        "model_version": 0,
        "total_predictions": 0,
        "correct_predictions": 0,
        "performance_metrics": [],
        "last_save_time": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -------------------------------
# PERSISTENCE FUNCTIONS
# -------------------------------
def save_model_state():
    """Save model and data to disk for persistence"""
    if st.session_state.model and st.session_state.df is not None:
        try:
            # Create models directory if it doesn't exist
            Path("saved_models").mkdir(exist_ok=True)
            
            # Save model
            model_path = f"saved_models/crash_ai_v{st.session_state.model_version}.pkl"
            joblib.dump(st.session_state.model, model_path)
            
            # Save data
            data_path = f"saved_models/data_v{st.session_state.model_version}.csv"
            st.session_state.df.to_csv(data_path, index=False)
            
            # Save metadata
            metadata = {
                "version": st.session_state.model_version,
                "timestamp": datetime.now().isoformat(),
                "rounds": len(st.session_state.df),
                "accuracy": st.session_state.training_history[-1]["accuracy"] if st.session_state.training_history else 0
            }
            with open("saved_models/metadata.json", "w") as f:
                json.dump(metadata, f)
            
            st.session_state.last_save_time = datetime.now()
            return True
        except Exception as e:
            st.error(f"Failed to save model: {str(e)}")
            return False
    return False

def load_saved_model():
    """Load previously saved model if available"""
    try:
        metadata_path = Path("saved_models/metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            model_path = f"saved_models/crash_ai_v{metadata['version']}.pkl"
            data_path = f"saved_models/data_v{metadata['version']}.csv"
            
            if Path(model_path).exists() and Path(data_path).exists():
                model = joblib.load(model_path)
                df = pd.read_csv(data_path)
                
                st.session_state.model = model
                st.session_state.df = df
                st.session_state.model_version = metadata['version']
                
                st.success(f"✅ Loaded saved model v{metadata['version']} with {len(df)} rounds")
                return True
    except Exception as e:
        st.warning(f"Could not load saved model: {str(e)}")
    return False

# -------------------------------
# IMPROVED MODEL TRAINING
# -------------------------------
def train_or_retrain_model(df, force=False):
    """Train or retrain model with cross-validation"""
    min_rounds = 50
    retrain_threshold = 10
    
    if len(df) < min_rounds:
        st.warning(f"Need at least {min_rounds} rounds. Currently: {len(df)}")
        return None
    
    needs_retraining = (
        st.session_state.model is None or
        force or
        st.session_state.rounds_since_training >= retrain_threshold
    )
    
    if needs_retraining:
        with st.spinner("🔄 Training AI with latest data..."):
            try:
                X_train, X_test, y_train, y_test = prepare_data(df)
                model = RFModel()
                model.train(X_train, y_train)
                
                # Calculate metrics with cross-validation
                cv_scores = cross_val_score(model.model, X_train, y_train, cv=5, scoring='accuracy')
                test_accuracy = model.model.score(X_test, y_test)
                
                # Log training event
                st.session_state.training_history.append({
                    "timestamp": datetime.now(),
                    "rounds": len(df),
                    "accuracy": test_accuracy,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "new_rounds_since_last": st.session_state.rounds_since_training
                })
                
                st.session_state.model = model
                st.session_state.last_training_time = datetime.now()
                st.session_state.rounds_since_training = 0
                st.session_state.model_version += 1
                
                # Auto-save after training
                save_model_state()
                
                return model
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return None
    
    return st.session_state.model

# -------------------------------
# VALIDATION FUNCTIONS
# -------------------------------
def validate_multiplier(value):
    """Validate crash multiplier input"""
    try:
        multiplier = float(value)
        if multiplier < 1.0:
            st.warning("Multiplier cannot be less than 1.0. Setting to 1.0")
            return 1.0
        if multiplier > 1000:
            st.warning("Multiplier seems unusually high. Please verify.")
        return multiplier
    except ValueError:
        st.error("Invalid multiplier value")
        return None

def validate_batch_input(input_string):
    """Validate batch multiplier input"""
    try:
        multipliers = [float(x.strip()) for x in input_string.split(",")]
        multipliers = [max(1.0, m) for m in multipliers]
        return multipliers
    except Exception:
        return None

# -------------------------------
# IMPROVED PREDICTION LOGGING
# -------------------------------
def log_prediction(signal, target, confidence, regime, actual=None):
    """Log prediction with deduplication"""
    # Create hash of current state to avoid duplicates
    current_state = f"{signal}_{target}_{confidence}_{len(st.session_state.df)}"
    current_hash = hashlib.md5(current_state.encode()).hexdigest()
    
    if st.session_state.last_data_hash != current_hash:
        st.session_state.predictions_log.append({
            "timestamp": datetime.now(),
            "signal": signal,
            "target": target,
            "confidence": confidence,
            "regime": regime,
            "actual": actual,
            "was_correct": None,
            "round_number": len(st.session_state.df)
        })
        
        st.session_state.last_data_hash = current_hash
        st.session_state.total_predictions += 1
        
        # Keep only last 200 predictions
        if len(st.session_state.predictions_log) > 200:
            st.session_state.predictions_log = st.session_state.predictions_log[-200:]
        
        return True
    return False

def update_prediction_accuracy(actual_multiplier):
    """Update prediction accuracy when round completes"""
    if st.session_state.predictions_log:
        last_pred = st.session_state.predictions_log[-1]
        if last_pred["actual"] is None:
            last_pred["actual"] = actual_multiplier
            
            # Determine if prediction was correct
            if last_pred["signal"] == "❌ SKIP":
                last_pred["was_correct"] = True  # Skip is always correct
            else:
                # For bet signals, check if crash reached target
                last_pred["was_correct"] = actual_multiplier >= (last_pred["target"] or 0)
            
            if last_pred["was_correct"]:
                st.session_state.correct_predictions += 1
            
            # Calculate rolling accuracy
            recent = [p for p in st.session_state.predictions_log[-50:] if p["was_correct"] is not None]
            if recent:
                rolling_acc = sum(p["was_correct"] for p in recent) / len(recent)
                st.session_state.performance_metrics.append({
                    "timestamp": datetime.now(),
                    "rolling_accuracy": rolling_acc,
                    "total_predictions": st.session_state.total_predictions
                })

# -------------------------------
# ENHANCED CONTEXT FEATURES
# -------------------------------
def get_context(df):
    """Extract enhanced context features"""
    last_10 = df.tail(10)["crash"]
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    all_time = df["crash"]
    
    # Calculate streaks
    recent_crashes = last_10.values
    current_streak = 1
    for i in range(len(recent_crashes)-2, -1, -1):
        if recent_crashes[i] < 2:
            current_streak += 1
        else:
            break
    
    return {
        "volatility": last_20.std(),
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3),
        "current_streak": current_streak,
        "trend": "up" if last_50.mean() < last_20.mean() else "down",
        "avg_10": last_10.mean(),
        "avg_20": last_20.mean(),
        "avg_50": last_50.mean(),
        "max_50": last_50.max(),
        "min_50": last_50.min(),
        "skew": all_time.skew() if len(all_time) > 2 else 0
    }

# -------------------------------
# ENHANCED REGIME DETECTION
# -------------------------------
def detect_regime(df):
    """Enhanced regime detection with more sophisticated logic"""
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    last_100 = df.tail(100)["crash"] if len(df) >= 100 else last_50
    
    avg = last_20.mean()
    std = last_20.std()
    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()
    extreme_ratio = (last_20 > 5).mean()
    
    # Detect regime shifts using multiple timeframes
    short_trend = last_20.mean() - last_50.mean()
    long_trend = last_50.mean() - last_100.mean()
    trend_shift = abs(short_trend) / max(last_50.std(), 0.1)
    
    # Momentum indicator
    momentum = (last_20.mean() / last_50.mean()) - 1 if last_50.mean() > 0 else 0
    
    # Regime classification with confidence
    if std > 3.0 or extreme_ratio > 0.2:
        regime = "🔥 EXTREME VOLATILITY"
        confidence_boost = 20
        risk_level = "Very High"
    elif std > 2.0:
        regime = "⚡ VOLATILE"
        confidence_boost = 10
        risk_level = "High"
    elif low_ratio > 0.7:
        regime = "🔴 EXTREME CHOPPY"
        confidence_boost = -30
        risk_level = "Very Low"
    elif low_ratio > 0.5:
        regime = "🔴 CHOPPY"
        confidence_boost = -15
        risk_level = "Low"
    elif high_ratio > 0.5:
        regime = "🟢 HOT"
        confidence_boost = 20
        risk_level = "High"
    elif high_ratio > 0.3:
        regime = "🟢 WARM"
        confidence_boost = 10
        risk_level = "Medium"
    else:
        regime = "🟡 NORMAL"
        confidence_boost = 0
        risk_level = "Medium"
    
    # Adjust for momentum and trend shifts
    if abs(momentum) > 0.15:
        regime += " (Strong Momentum)"
        confidence_boost += 10 if momentum > 0 else -5
    
    if trend_shift > 2.0:
        regime += " (Trend Shift!)"
        confidence_boost += 15
    
    return {
        "regime": regime,
        "avg": avg,
        "std": std,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "extreme_ratio": extreme_ratio,
        "confidence_boost": confidence_boost,
        "trend_shift": trend_shift,
        "momentum": momentum,
        "risk_level": risk_level
    }

# -------------------------------
# IMPROVED ADAPTIVE MULTIPLIER ENGINE
# -------------------------------
def evaluate_multiplier(df, target, window=100, min_samples=30):
    """Enhanced evaluation with risk-adjusted returns and minimum samples"""
    if len(df) < min_samples:
        return 0, 0, 0
    
    balance = 0
    stake = 1
    wins = 0
    total_trades = 0
    returns = []
    max_drawdown = 0
    peak = 0
    
    start = max(min_samples, len(df) - window)
    
    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]
        
        if crash >= target:
            profit = stake * (target - 1)
            balance += profit
            wins += 1
            returns.append(profit)
        else:
            balance -= stake
            returns.append(-stake)
        
        total_trades += 1
        
        # Track drawdown
        peak = max(peak, balance)
        drawdown = (peak - balance) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    # Risk-adjusted score
    risk_score = (win_rate * sharpe) / (max_drawdown + 0.01) if max_drawdown > 0 else win_rate * sharpe * 100
    
    return balance, win_rate, sharpe, risk_score, max_drawdown

def get_adaptive_multipliers(df):
    """Get optimal multipliers based on risk-adjusted performance"""
    multipliers = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    results = []
    for m in multipliers:
        profit, win_rate, sharpe, risk_score, drawdown = evaluate_multiplier(df, m)
        if profit > 0 or win_rate > 0:  # Only include if there's some data
            results.append({
                "m": m,
                "profit": profit,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "risk_score": risk_score,
                "drawdown": drawdown,
                "score": profit * win_rate * (1 + sharpe) * (1 - drawdown)
            })
    
    if not results:
        return {
            "low": 1.5,
            "mid": 2.0,
            "high": 3.0,
            "table": pd.DataFrame()
        }
    
    res = pd.DataFrame(results)
    
    # Categorize by risk with dynamic thresholds
    low_risk = res[res["m"] <= 1.8]
    medium_risk = res[(res["m"] > 1.8) & (res["m"] <= 2.5)]
    high_risk = res[res["m"] > 2.5]
    
    # Add safety checks
    low_target = low_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(low_risk) > 0 else 1.5
    mid_target = medium_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(medium_risk) > 0 else 2.0
    high_target = high_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(high_risk) > 0 else 3.0
    
    return {
        "low": low_target,
        "mid": mid_target,
        "high": high_target,
        "table": res.sort_values("score", ascending=False)
    }

# -------------------------------
# KELLY CRITERION WITH SAFETY
# -------------------------------
def calculate_kelly(probability, target, max_fraction=0.25):
    """Safe Kelly Criterion calculation"""
    if not target or target <= 1:
        return 0
    
    implied_prob = 1 / target
    edge = probability - implied_prob
    
    if edge <= 0 or implied_prob >= 1:
        return 0
    
    kelly = edge / (1 - implied_prob)
    return max(0, min(max_fraction, kelly))

# -------------------------------
# UI - SIDEBAR
# -------------------------------
st.sidebar.markdown("## 🎮 Add New Round")
st.sidebar.markdown("Each new round helps the AI learn and improve!")

new_rate = st.sidebar.number_input(
    "Crash Multiplier",
    min_value=1.0,
    max_value=100.0,
    value=1.5,
    step=0.1,
    format="%.2f",
    key="live_multiplier"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    add_button = st.button("➕ Add Round", use_container_width=True)
with col2:
    st.session_state.auto_learn = st.checkbox(
        "🤖 Auto-learn",
        value=st.session_state.auto_learn,
        help="Automatically retrain AI after every 10 rounds"
    )

# Handle new round addition
if add_button and st.session_state.df is not None:
    validated_rate = validate_multiplier(new_rate)
    
    if validated_rate:
        now = pd.Timestamp.now()
        
        row = pd.DataFrame([{
            "rate": str(validated_rate),
            "crash": float(validated_rate),
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": f"live_{now.timestamp()}",
            "salt": "live",
            "fetchedAt": now
        }])
        
        # Update prediction accuracy first
        update_prediction_accuracy(validated_rate)
        
        # Add to dataset
        st.session_state.df = pd.concat([st.session_state.df, row], ignore_index=True)
        st.session_state.rounds_since_training += 1
        
        st.sidebar.success(f"✅ Round added! Multiplier: {validated_rate}x")
        
        # Auto-learn if enabled
        if st.session_state.auto_learn and st.session_state.rounds_since_training >= 10:
            st.sidebar.info("🔄 Auto-learning triggered...")
            train_or_retrain_model(st.session_state.df, force=True)
        
        st.rerun()

# -------------------------------
# BATCH UPLOAD FOR HISTORICAL DATA
# -------------------------------
st.sidebar.markdown("## 📁 Data Management")

col_load, col_save = st.sidebar.columns(2)
with col_load:
    file = st.file_uploader("Load JSON", type=["json"], key="file_uploader")
with col_save:
    if st.button("💾 Save Current State", use_container_width=True):
        if save_model_state():
            st.sidebar.success("State saved successfully!")

if file and st.session_state.df is None:
    with st.spinner("Loading historical data..."):
        try:
            st.session_state.df = load_data(file)
            st.success(f"✅ Loaded {len(st.session_state.df)} historical rounds!")
            
            # Try to load saved model first
            if not load_saved_model():
                # Initial training
                train_or_retrain_model(st.session_state.df, force=True)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# -------------------------------
# BULK ADD MULTIPLE ROUNDS
# -------------------------------
with st.sidebar.expander("📊 Batch Add Multiple Rounds"):
    st.markdown("Add multiple rounds at once (comma-separated)")
    batch_multipliers = st.text_input("Multipliers (e.g., 1.5, 2.3, 1.2, 4.5)")
    
    if st.button("Add Batch", use_container_width=True) and st.session_state.df is not None:
        if batch_multipliers:
            multipliers = validate_batch_input(batch_multipliers)
            if multipliers:
                now = pd.Timestamp.now()
                
                new_rows = []
                for i, m in enumerate(multipliers):
                    # Update prediction accuracy for each round (except last)
                    if i < len(multipliers) - 1:
                        update_prediction_accuracy(m)
                    
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
                
                if st.session_state.auto_learn and st.session_state.rounds_since_training >= 10:
                    train_or_retrain_model(st.session_state.df, force=True)
                
                st.rerun()
            else:
                st.sidebar.error("Invalid batch format. Use comma-separated numbers.")

# -------------------------------
# MODEL PERFORMANCE DASHBOARD
# -------------------------------
if st.session_state.df is not None and st.session_state.model is not None:
    st.sidebar.markdown("## 📈 Model Status")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("Total Rounds", len(st.session_state.df))
    with col2:
        st.metric("Rounds since training", st.session_state.rounds_since_training)
    with col3:
        st.metric("Model Version", st.session_state.model_version)
    
    if st.session_state.last_training_time:
        st.sidebar.caption(f"Last trained: {st.session_state.last_training_time.strftime('%H:%M:%S')}")
    
    if st.session_state.last_save_time:
        st.sidebar.caption(f"Last saved: {st.session_state.last_save_time.strftime('%H:%M:%S')}")
    
    col_retrain, col_reset = st.sidebar.columns(2)
    with col_retrain:
        if st.button("🔄 Force Retrain", use_container_width=True):
            train_or_retrain_model(st.session_state.df, force=True)
            st.sidebar.success("Model retrained!")
            st.rerun()
    with col_reset:
        if st.button("🗑️ Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()
    
    # Training history
    if st.session_state.training_history:
        with st.sidebar.expander("📊 Training History"):
            history_df = pd.DataFrame(st.session_state.training_history)
            st.line_chart(history_df.set_index("timestamp")["accuracy"])
            st.caption(f"Latest CV Accuracy: {history_df.iloc[-1]['cv_mean']:.2%} ±{history_df.iloc[-1]['cv_std']:.2%}")

# -------------------------------
# MAIN APPLICATION
# -------------------------------
if st.session_state.df is None:
    st.info("📤 Upload your JSON file or load a saved model to begin. The AI will learn from every round you add!")
    
    # Try to load saved model automatically
    load_saved_model()
    
    if st.session_state.df is None:
        st.stop()

# -------------------------------
# CLEAN DATA
# -------------------------------
try:
    df = clean_data(st.session_state.df)
    df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
    df_ui = df.sort_values("fetchedAt", ascending=False)
except Exception as e:
    st.error(f"Error cleaning data: {str(e)}")
    st.stop()

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
# GET CONTEXT AND REGIME
# -------------------------------
ctx = get_context(df_ml)
regime_data = detect_regime(df_ml)

# -------------------------------
# ML PREDICTION WITH CONFIDENCE
# -------------------------------
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

try:
    proba = model.predict_proba(X_live)[0][1]
except Exception as e:
    st.warning(f"Prediction error: {str(e)}")
    proba = 0.5

# Get feature importance
try:
    feature_importance = dict(zip(FEATURES, model.model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
except:
    top_features = [("feature1", 0), ("feature2", 0)]

# -------------------------------
# ENHANCED CONFIDENCE ENGINE
# -------------------------------
confidence = proba * 50

# Volatility adjustment
if ctx["volatility"] > 3.0:
    confidence += 30
elif ctx["volatility"] > 2.0:
    confidence += 20
elif ctx["volatility"] > 1.5:
    confidence += 10

# Streak adjustments
if ctx["current_streak"] >= 8:
    confidence += 25
elif ctx["current_streak"] >= 5:
    confidence += 15
elif ctx["current_streak"] >= 3:
    confidence += 5

if ctx["low_streak"] >= 6:
    confidence += 20
elif ctx["low_streak"] >= 4:
    confidence += 10

if ctx["high_streak"] >= 5:
    confidence -= 20
elif ctx["high_streak"] >= 3:
    confidence -= 10

# Regime adjustment
confidence += regime_data["confidence_boost"]

# Learning progress boost (diminishing returns)
learning_progress = min(len(df_ml) / 1000, 0.3)
confidence *= (1 + learning_progress)

# Ensure bounds
confidence = max(0, min(100, confidence))

# -------------------------------
# ADAPTIVE MULTIPLIERS
# -------------------------------
adaptive = get_adaptive_multipliers(df_ml)

# -------------------------------
# SIGNAL ENGINE (REGIME AWARE)
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]
    bet_size = "Large (5-10%)"
    risk_color = "strong"
elif confidence > 65:
    signal = "✅ BET"
    if regime_data["regime"] in ["🟢 HOT", "🟢 WARM"]:
        target = adaptive["high"]
        bet_size = "Medium-Large (4-7%)"
    else:
        target = adaptive["mid"]
        bet_size = "Medium (2-4%)"
elif confidence > 50:
    signal = "⚠️ SMALL BET"
    if regime_data["regime"] in ["🔴 CHOPPY", "🔴 EXTREME CHOPPY"]:
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
log_prediction(signal, target, confidence, regime_data["regime"])

# Calculate recent accuracy
recent_predictions = [p for p in st.session_state.predictions_log if p["was_correct"] is not None]
recent_accuracy = sum(p["was_correct"] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0

# -------------------------------
# UI - MAIN DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Signal", signal)
with col2:
    st.metric("Confidence", f"{confidence:.1f}%")
with col3:
    st.metric("ML Probability", f"{proba:.2%}")
with col4:
    st.metric("🎯 Target", f"{target}x" if target else "No Trade")
with col5:
    st.metric("🧠 Regime", regime_data["regime"][:15])
with col6:
    st.metric("📊 Recent Accuracy", f"{recent_accuracy:.1%}")

# Risk level indicator
risk_levels = {"Very Low": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Very High": "⚫"}
st.markdown(f"**Risk Level:** {risk_levels.get(regime_data['risk_level'], '🟡')} {regime_data['risk_level']}")

# Learning progress
progress_value = min(len(df_ml) / 1000, 1.0)
st.progress(progress_value, text=f"🤖 Learning Progress: {len(df_ml)}/1000 rounds for optimal performance")

# -------------------------------
# BET SUGGESTION WITH KELLY
# -------------------------------
if target:
    st.info(f"💡 **Suggested Action**: {signal} at **{target}x** | {bet_size} of bankroll")
    
    # Kelly Criterion calculation
    kelly = calculate_kelly(proba, target)
    if kelly > 0:
        st.caption(f"📐 Kelly Criterion suggests: **{kelly:.1%}** of bankroll (max {calculate_kelly(proba, target, 0.15):.1%} conservative)")
        
        # Risk warning
        if regime_data["risk_level"] in ["High", "Very High"]:
            st.warning(f"⚠️ High risk regime detected. Consider reducing position size to {kelly * 0.5:.1%}")
else:
    st.info("💡 **Suggested Action**: Skip this round - wait for better conditions")

# -------------------------------
# LAST 10 MULTIPLIERS VISUALIZATION
# -------------------------------
st.markdown("### 📉 Last 10 Multipliers")

last_10 = df_ml["crash"].tail(10).to_numpy()[::-1]
cols = st.columns(10)

for i, val in enumerate(last_10):
    if val >= 3:
        color = "#00ff00"
    elif val >= 2:
        color = "#ffff00"
    else:
        color = "#ff4444"
    
    with cols[i]:
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:10px;
                border-radius:10px;
                text-align:center;
                color:white;
                font-weight:bold;
                margin:2px;">
                {val:.2f}x
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------
# PREDICTION EXPLANATION
# -------------------------------
with st.expander("🧠 Why did the AI make this decision?"):
    st.markdown("**Top Factors Influencing Decision:**")
    for feature, importance in top_features:
        st.write(f"- {feature}: {importance:.2%}")
    
    st.markdown("**Context Analysis:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"- Volatility: {ctx['volatility']:.2f}")
        st.write(f"- Trend: {ctx['trend']}")
        st.write(f"- Current Streak: {ctx['current_streak']} rounds")
        st.write(f"- 10-period avg: {ctx['avg_10']:.2f}")
    with col_b:
        st.write(f"- 20-period avg: {ctx['avg_20']:.2f}")
        st.write(f"- 50-period avg: {ctx['avg_50']:.2f}")
        st.write(f"- Low Streak: {ctx['low_streak']} rounds under 2x")
        st.write(f"- High Streak: {ctx['high_streak']} rounds over 3x")
    
    st.markdown("**Regime Analysis:**")
    st.write(f"- Regime: {regime_data['regime']}")
    st.write(f"- Momentum: {regime_data['momentum']:.2%}")
    st.write(f"- Low Ratio: {regime_data['low_ratio']:.1%}")
    st.write(f"- High Ratio: {regime_data['high_ratio']:.1%}")
    st.write(f"- Extreme Ratio (>5x): {regime_data['extreme_ratio']:.1%}")

# -------------------------------
# INSIGHTS DASHBOARD
# -------------------------------
with st.expander("📊 Advanced Analytics"):
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Risk Analysis", "Market Statistics"])
    
    with tab1:
        if st.session_state.performance_metrics:
            perf_df = pd.DataFrame(st.session_state.performance_metrics)
            st.line_chart(perf_df.set_index("timestamp")["rolling_accuracy"])
            st.metric("Total Predictions", st.session_state.total_predictions)
            st.metric("Correct Predictions", st.session_state.correct_predictions)
            if st.session_state.total_predictions > 0:
                st.metric("Overall Accuracy", f"{st.session_state.correct_predictions/st.session_state.total_predictions:.1%}")
    
    with tab2:
        if not adaptive["table"].empty:
            st.subheader("Multiplier Risk-Reward Analysis")
            st.dataframe(adaptive["table"][["m", "win_rate", "sharpe", "drawdown", "risk_score"]], use_container_width=True)
        
        st.subheader("Kelly vs Confidence")
        if target:
            st.metric("Recommended Kelly", f"{kelly:.1%}")
            st.metric("Current Confidence", f"{confidence:.1f}%")
            st.caption("Kelly based on ML probability vs implied odds")
    
    with tab3:
        st.subheader("Market Statistics")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Mean Multiplier", f"{df_ml['crash'].mean():.2f}x")
            st.metric("Median", f"{df_ml['crash'].median():.2f}x")
        with col_s2:
            st.metric("Std Dev", f"{df_ml['crash'].std():.2f}")
            st.metric("Skewness", f"{df_ml['crash'].skew():.2f}")
        with col_s3:
            st.metric("Max", f"{df_ml['crash'].max():.2f}x")
            st.metric("Min", f"{df_ml['crash'].min():.2f}x")

# -------------------------------
# MULTIPLIER PERFORMANCE TABLE
# -------------------------------
if not adaptive["table"].empty:
    st.subheader("🎯 Multiplier Performance Ranking")
    display_cols = ["m", "win_rate", "sharpe", "profit", "drawdown"]
    st.dataframe(adaptive["table"][display_cols].head(10), use_container_width=True)

# -------------------------------
# MODEL LEARNING CURVE
# -------------------------------
if st.session_state.training_history:
    st.subheader("📈 Model Learning Progress")
    history_df = pd.DataFrame(st.session_state.training_history)
    
    if len(history_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(history_df.set_index("timestamp")["accuracy"])
            st.caption("Test Accuracy Over Time")
        with col2:
            if "cv_mean" in history_df.columns:
                st.line_chart(history_df.set_index("timestamp")["cv_mean"])
                st.caption("Cross-Validation Accuracy (5-fold)")

# -------------------------------
# DATA VIEW
# -------------------------------
st.subheader("📊 Latest Rounds")
st.dataframe(df_ui.head(20), use_container_width=True)

# Crash history chart
st.subheader("📈 Crash History")
chart_data = df_ml[["crash"]].tail(200)
st.line_chart(chart_data)

# -------------------------------
# EXPORT FUNCTIONALITY
# -------------------------------
st.subheader("💾 Export Options")
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    if st.button("💾 Export Model", use_container_width=True):
        if save_model_state():
            st.success("Model saved successfully!")
        else:
            st.error("Failed to save model")

with col_exp2:
    if st.button("📥 Export Data", use_container_width=True):
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"crash_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            key="data_download"
        )

with col_exp3:
    if st.button("📊 Export Predictions", use_container_width=True):
        if st.session_state.predictions_log:
            log_df = pd.DataFrame(st.session_state.predictions_log)
            csv = log_df.to_csv(index=False)
            st.download_button(
                "Download Prediction Log",
                csv,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                key="pred_download"
            )
        else:
            st.warning("No predictions to export")

# -------------------------------
# AUTO-REFRESH OPTION
# -------------------------------
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh dashboard (5 sec)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🚀 Crash AI v4 - Continuously learning from every round | Risk Warning: Never bet more than you can afford to lose")
