import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import deque
import hashlib
import json
from pathlib import Path

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

# Page config
st.set_page_config(
    layout="wide",
    page_title="Crash AI v4 - Advanced Learning Engine",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .signal-strong {
        background: linear-gradient(90deg, #ff6b6b, #ff8e53);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .insight-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# ENHANCED SESSION STATE
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
if "auto_learn" not in st.session_state:
    st.session_state.auto_learn = True
if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = {
        "total_profit": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "max_drawdown": 0,
        "peak_balance": 1000,
        "current_balance": 1000,
        "trade_history": []
    }
if "alert_config" not in st.session_state:
    st.session_state.alert_config = {
        "confidence_threshold": 80,
        "profit_target": 20,
        "stop_loss": 10
    }
if "feature_importance_history" not in st.session_state:
    st.session_state.feature_importance_history = []
if "regime_transitions" not in st.session_state:
    st.session_state.regime_transitions = []
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# -------------------------------
# IMPROVED MODEL MANAGEMENT
# -------------------------------
class ModelVersioning:
    """Handle model versioning and rollback"""
    
    def __init__(self, save_dir="models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, version=None):
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        import joblib
        path = self.save_dir / f"crash_model_v{version}.pkl"
        joblib.dump(model, path)
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "accuracy": st.session_state.training_history[-1]["accuracy"] if st.session_state.training_history else 0,
            "rounds": len(st.session_state.df) if st.session_state.df is not None else 0
        }
        with open(self.save_dir / f"metadata_v{version}.json", "w") as f:
            json.dump(metadata, f)
        
        return version
    
    def load_latest_model(self):
        models = list(self.save_dir.glob("crash_model_*.pkl"))
        if not models:
            return None
        
        latest = max(models, key=lambda x: x.stat().st_mtime)
        import joblib
        return joblib.load(latest)

model_versioning = ModelVersioning()

def train_or_retrain_model(df, force=False):
    """Train or retrain model with enhanced tracking"""
    min_rounds = 50
    retrain_threshold = 10
    
    if len(df) < min_rounds:
        st.warning(f"Need at least {min_rounds} rounds. Currently: {len(df)}")
        return None
    
    # Dynamic retraining based on performance degradation
    needs_retraining = (
        st.session_state.model is None or
        force or
        st.session_state.rounds_since_training >= retrain_threshold
    )
    
    # Check if model performance degraded
    if st.session_state.predictions_log and len(st.session_state.predictions_log) >= 20:
        recent_correct = [p for p in st.session_state.predictions_log[-20:] if p["was_correct"] is not None]
        if recent_correct:
            recent_accuracy = sum(p["was_correct"] for p in recent_correct) / len(recent_correct)
            if recent_accuracy < 0.4:  # Below 40% accuracy
                needs_retraining = True
                st.info("⚠️ Performance degradation detected - retraining required")
    
    if needs_retraining:
        with st.spinner("🔄 Advanced training in progress..."):
            try:
                # Prepare data with validation split
                X_train, X_test, y_train, y_test = prepare_data(df)
                
                # Hyperparameter tuning
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                
                base_model = RFModel()
                grid_search = GridSearchCV(
                    base_model.model, 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Use best model
                base_model.model = grid_search.best_estimator_
                accuracy = base_model.model.score(X_test, y_test)
                
                # Store feature importance
                importance_dict = dict(zip(FEATURES, base_model.model.feature_importances_))
                st.session_state.feature_importance_history.append({
                    "timestamp": datetime.now(),
                    "importance": importance_dict
                })
                
                # Log training event
                st.session_state.training_history.append({
                    "timestamp": datetime.now(),
                    "rounds": len(df),
                    "accuracy": accuracy,
                    "new_rounds_since_last": st.session_state.rounds_since_training,
                    "best_params": grid_search.best_params_,
                    "feature_importance": importance_dict
                })
                
                st.session_state.model = base_model
                st.session_state.last_training_time = datetime.now()
                st.session_state.rounds_since_training = 0
                
                # Save model version
                version = model_versioning.save_model(base_model)
                st.success(f"✅ Model trained! Accuracy: {accuracy:.2%} (Version: {version})")
                
                return base_model
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return None
    
    return st.session_state.model

# -------------------------------
# ENHANCED RISK MANAGEMENT
# -------------------------------
class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, balance, max_risk_per_trade=0.05):
        self.balance = balance
        self.max_risk_per_trade = max_risk_per_trade
        self.consecutive_losses = 0
        self.position_sizing_mode = "kelly"  # kelly, fixed, progressive
    
    def calculate_position_size(self, confidence, target, probability, kelly_factor=0.25):
        """Calculate optimal position size based on multiple factors"""
        
        # Base position size using Kelly Criterion
        if target and target > 1:
            implied_prob = 1 / target
            edge = probability - implied_prob
            
            if edge > 0:
                kelly_size = edge / (1 - implied_prob)
                kelly_size = min(kelly_size, self.max_risk_per_trade)
                position_size = kelly_size * kelly_factor
            else:
                position_size = 0
        else:
            position_size = 0
        
        # Adjust for confidence
        confidence_multiplier = confidence / 100
        position_size *= confidence_multiplier
        
        # Adjust for consecutive losses (anti-martingale)
        if self.consecutive_losses > 2:
            position_size *= 0.5
        
        # Max drawdown protection
        if self.balance < 500:  # 50% drawdown from 1000
            position_size *= 0.25
        
        return min(position_size, self.max_risk_per_trade)
    
    def update_after_trade(self, won, profit_loss):
        """Update risk metrics after trade"""
        self.balance += profit_loss
        
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

risk_manager = RiskManager(1000)

# -------------------------------
# IMPROVED CONTEXT FEATURES
# -------------------------------
def get_enhanced_context(df):
    """Extract enhanced context features with market microstructure"""
    last_10 = df.tail(10)["crash"]
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    
    # Volatility clustering detection
    volatility_cluster = last_10.std() / last_50.std() if last_50.std() > 0 else 1
    
    # Pattern recognition
    def detect_pattern(series):
        """Simple pattern detection"""
        increasing = sum(series.diff() > 0) > len(series) * 0.6
        decreasing = sum(series.diff() < 0) > len(series) * 0.6
        if increasing:
            return "uptrend"
        elif decreasing:
            return "downtrend"
        else:
            return "ranging"
    
    pattern = detect_pattern(last_20)
    
    # Momentum indicators
    momentum = (last_10.mean() - last_50.mean()) / last_50.mean() if last_50.mean() > 0 else 0
    
    return {
        "volatility": last_10.std(),
        "volatility_cluster": volatility_cluster,
        "low_streak": sum(last_10 < 2),
        "high_streak": sum(last_10 > 3),
        "trend": "up" if last_50.mean() < last_10.mean() else "down",
        "avg_10": last_10.mean(),
        "avg_50": last_50.mean(),
        "pattern": pattern,
        "momentum": momentum,
        "max_10": last_10.max(),
        "min_10": last_10.min(),
        "range_10": last_10.max() - last_10.min()
    }

# -------------------------------
# ENHANCED REGIME DETECTION
# -------------------------------
def detect_enhanced_regime(df):
    """Multi-factor regime detection with transition tracking"""
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    last_100 = df.tail(100)["crash"]
    
    avg = last_20.mean()
    std = last_20.std()
    low_ratio = (last_20 < 2).mean()
    high_ratio = (last_20 > 3).mean()
    
    # Advanced metrics
    skewness = last_20.skew()
    kurtosis = last_20.kurtosis()
    
    # Detect regime shifts
    prev_regime_avg = last_50.head(20).mean()
    regime_shift = abs(avg - prev_regime_avg) / max(last_50.std(), 0.1)
    
    # Multi-factor regime classification
    scores = {
        "volatile": std > 2.0,
        "choppy": low_ratio > 0.5,
        "hot": high_ratio > 0.3,
        "trending": abs(regime_shift) > 0.5
    }
    
    if scores["volatile"]:
        regime = "⚡ VOLATILE"
        confidence_boost = 10
        color = "#ff6b6b"
    elif scores["choppy"]:
        regime = "🔴 CHOPPY"
        confidence_boost = -20
        color = "#ffa500"
    elif scores["hot"]:
        regime = "🟢 HOT"
        confidence_boost = 15
        color = "#4ecdc4"
    elif scores["trending"]:
        regime = "📈 TRENDING"
        confidence_boost = 5
        color = "#45b7d1"
    else:
        regime = "🟡 NORMAL"
        confidence_boost = 0
        color = "#f9ca24"
    
    # Track regime transitions
    if st.session_state.regime_transitions and st.session_state.regime_transitions[-1]["regime"] != regime:
        st.session_state.regime_transitions.append({
            "timestamp": datetime.now(),
            "from_regime": st.session_state.regime_transitions[-1]["regime"],
            "to_regime": regime,
            "confidence": confidence_boost
        })
    elif not st.session_state.regime_transitions:
        st.session_state.regime_transitions.append({
            "timestamp": datetime.now(),
            "from_regime": None,
            "to_regime": regime,
            "confidence": confidence_boost
        })
    
    return {
        "regime": regime,
        "color": color,
        "avg": avg,
        "std": std,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "confidence_boost": confidence_boost,
        "regime_shift": regime_shift,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "scores": scores
    }

# -------------------------------
# ADVANCED BACKTESTING ENGINE
# -------------------------------
def run_advanced_backtest(df, initial_balance=1000):
    """Run comprehensive backtest with multiple strategies"""
    results = {}
    
    # Strategy 1: Current AI strategy
    balance_ai = initial_balance
    trades_ai = []
    
    # Strategy 2: Conservative (always low multiplier)
    balance_conservative = initial_balance
    trades_conservative = []
    
    # Strategy 3: Aggressive (always high multiplier)
    balance_aggressive = initial_balance
    trades_aggressive = []
    
    for i in range(50, len(df) - 1):
        # Simulate AI strategy (simplified)
        last_20 = df.iloc[i-20:i]["crash"]
        volatility = last_20.std()
        
        # AI decision logic
        if volatility > 2.0:
            target = 1.8
        elif volatility < 1.0:
            target = 3.0
        else:
            target = 2.2
        
        actual = df.iloc[i+1]["crash"]
        
        # AI trade
        if actual >= target:
            profit_ai = 1 * (target - 1)
            balance_ai += profit_ai
            trades_ai.append(profit_ai)
        else:
            balance_ai -= 1
            trades_ai.append(-1)
        
        # Conservative strategy
        if actual >= 1.5:
            balance_conservative += 0.5
            trades_conservative.append(0.5)
        else:
            balance_conservative -= 1
            trades_conservative.append(-1)
        
        # Aggressive strategy
        if actual >= 5.0:
            balance_aggressive += 4
            trades_aggressive.append(4)
        else:
            balance_aggressive -= 1
            trades_aggressive.append(-1)
    
    # Calculate metrics
    for name, balance, trades in [
        ("AI Strategy", balance_ai, trades_ai),
        ("Conservative", balance_conservative, trades_conservative),
        ("Aggressive", balance_aggressive, trades_aggressive)
    ]:
        returns = pd.Series(trades)
        
        results[name] = {
            "final_balance": balance,
            "total_return": balance - initial_balance,
            "return_pct": ((balance - initial_balance) / initial_balance) * 100,
            "win_rate": (returns > 0).mean(),
            "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0,
            "max_drawdown": (returns.cumsum().min()),
            "total_trades": len(trades)
        }
    
    return pd.DataFrame(results).T

# -------------------------------
# REAL-TIME PERFORMANCE TRACKING
# -------------------------------
def update_performance_metrics(actual_multiplier, target, was_bet):
    """Update live performance metrics"""
    metrics = st.session_state.performance_metrics
    
    if was_bet and target:
        if actual_multiplier >= target:
            profit = 1 * (target - 1)
            metrics["winning_trades"] += 1
        else:
            profit = -1
        
        metrics["total_profit"] += profit
        metrics["current_balance"] += profit
        metrics["total_trades"] += 1
        
        # Update peak balance and drawdown
        if metrics["current_balance"] > metrics["peak_balance"]:
            metrics["peak_balance"] = metrics["current_balance"]
        
        drawdown = (metrics["peak_balance"] - metrics["current_balance"]) / metrics["peak_balance"]
        if drawdown > metrics["max_drawdown"]:
            metrics["max_drawdown"] = drawdown
        
        # Store trade
        metrics["trade_history"].append({
            "timestamp": datetime.now(),
            "multiplier": actual_multiplier,
            "target": target,
            "profit": profit,
            "won": profit > 0
        })
        
        # Keep last 1000 trades
        if len(metrics["trade_history"]) > 1000:
            metrics["trade_history"] = metrics["trade_history"][-1000:]

# -------------------------------
# UI COMPONENTS
# -------------------------------
def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart using plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#ff6b6b"},
            "steps": [
                {"range": [0, 33], "color": "#ff4444"},
                {"range": [33, 66], "color": "#ffa500"},
                {"range": [66, 100], "color": "#00ff00"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_equity_curve(trades):
    """Create equity curve from trade history"""
    if not trades:
        return None
    
    cumulative = [0]
    for trade in trades:
        cumulative.append(cumulative[-1] + trade["profit"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumulative,
        mode="lines",
        name="Equity Curve",
        line=dict(color="#00ff00", width=2),
        fill="tozeroy"
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L",
        height=300
    )
    return fig

# -------------------------------
# MAIN APPLICATION
# -------------------------------

# Sidebar - Data Management
st.sidebar.markdown("# 📊 Data Management")

file = st.sidebar.file_uploader("Upload JSON Dataset", type=["json"])
if file and st.session_state.df is None:
    with st.spinner("Loading and validating data..."):
        try:
            st.session_state.df = load_data(file)
            st.success(f"✅ Loaded {len(st.session_state.df)} rounds!")
            
            # Validate data quality
            missing_cols = set(["rate", "crash", "fetchedAt"]) - set(st.session_state.df.columns)
            if missing_cols:
                st.warning(f"Missing columns: {missing_cols}")
            
            train_or_retrain_model(st.session_state.df, force=True)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Live round input
st.sidebar.markdown("## 🎮 Live Round Entry")
new_rate = st.sidebar.number_input("Crash Multiplier", min_value=1.0, step=0.01, key="live_rate")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("➕ Add Round", use_container_width=True) and st.session_state.df is not None:
        now = pd.Timestamp.now()
        new_row = pd.DataFrame([{
            "rate": str(new_rate),
            "crash": new_rate,
            "prepareTime": now,
            "beginTime": now,
            "endTime": now,
            "hash": hashlib.md5(f"{now}_{new_rate}".encode()).hexdigest(),
            "salt": "live",
            "fetchedAt": now
        }])
        
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.session_state.rounds_since_training += 1
        
        # Update performance if this was a predicted round
        if st.session_state.predictions_log and st.session_state.predictions_log[-1]["actual"] is None:
            last_pred = st.session_state.predictions_log[-1]
            last_pred["actual"] = new_rate
            
            if last_pred["signal"] != "❌ SKIP" and last_pred["target"]:
                was_bet = True
                update_performance_metrics(new_rate, last_pred["target"], was_bet)
                last_pred["was_correct"] = new_rate >= last_pred["target"]
        
        st.sidebar.success(f"✅ Added {new_rate}x")
        
        if st.session_state.auto_learn and st.session_state.rounds_since_training >= 10:
            train_or_retrain_model(st.session_state.df, force=True)
        
        st.rerun()

with col2:
    st.session_state.auto_learn = st.checkbox("Auto-learn", st.session_state.auto_learn)

# Batch upload
with st.sidebar.expander("📦 Batch Upload"):
    batch_data = st.text_area("Enter multipliers (one per line)", height=150)
    if st.button("Add Batch") and batch_data and st.session_state.df is not None:
        multipliers = [float(x.strip()) for x in batch_data.split() if x.strip()]
        now = pd.Timestamp.now()
        
        new_rows = []
        for i, m in enumerate(multipliers):
            new_rows.append({
                "rate": str(m),
                "crash": m,
                "prepareTime": now + timedelta(seconds=i*10),
                "beginTime": now + timedelta(seconds=i*10),
                "endTime": now + timedelta(seconds=i*10+5),
                "hash": hashlib.md5(f"{now}_{i}_{m}".encode()).hexdigest(),
                "salt": "batch",
                "fetchedAt": now + timedelta(seconds=i*10)
            })
        
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)
        st.session_state.rounds_since_training += len(multipliers)
        st.sidebar.success(f"✅ Added {len(multipliers)} rounds!")
        st.rerun()

# Main content
if st.session_state.df is None:
    st.info("🚀 **Welcome to Crash AI v4!**\n\nUpload your JSON data to start the advanced learning engine.")
    
    # Demo mode
    if st.button("🎮 Try Demo Mode"):
        demo_data = pd.DataFrame({
            "rate": [str(x) for x in np.random.exponential(2, 100) + 1],
            "crash": np.random.exponential(2, 100) + 1,
            "fetchedAt": pd.date_range(end=datetime.now(), periods=100, freq="5S")
        })
        st.session_state.df = demo_data
        train_or_retrain_model(demo_data, force=True)
        st.rerun()
    
    st.stop()

# Process data
df = clean_data(st.session_state.df)
df_ml = df.sort_values("fetchedAt").reset_index(drop=True)
df_ui = df.sort_values("fetchedAt", ascending=False)

if len(df_ml) < 50:
    st.warning(f"Need more data! {len(df_ml)}/50 rounds")
    st.stop()

# Get model
model = train_or_retrain_model(df_ml)
if model is None:
    st.stop()

# Generate predictions
ctx = get_enhanced_context(df_ml)
regime_data = detect_enhanced_regime(df_ml)

last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]
proba = model.predict_proba(X_live)[0][1]

# Calculate confidence
confidence = proba * 50
confidence += ctx["volatility"] * 5
confidence += regime_data["confidence_boost"]
confidence = np.clip(confidence, 0, 100)

# Adaptive multipliers
adaptive = get_adaptive_multipliers(df_ml)

# Generate signal
if confidence > 75:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]
    action_color = "linear-gradient(90deg, #ff6b6b, #ff8e53)"
elif confidence > 60:
    signal = "✅ BET"
    target = adaptive["mid"]
    action_color = "#4ecdc4"
elif confidence > 50:
    signal = "⚠️ CAUTION"
    target = adaptive["low"]
    action_color = "#ffa500"
else:
    signal = "❌ AVOID"
    target = None
    action_color = "#95a5a6"

# Log prediction
if not st.session_state.predictions_log or st.session_state.predictions_log[-1]["actual"] is not None:
    st.session_state.predictions_log.append({
        "timestamp": datetime.now(),
        "signal": signal,
        "target": target,
        "confidence": confidence,
        "regime": regime_data["regime"],
        "actual": None,
        "was_correct": None
    })

# Calculate position size
if target:
    position_size = risk_manager.calculate_position_size(confidence, target, proba)
else:
    position_size = 0

# -------------------------------
# ADVANCED DASHBOARD
# -------------------------------
st.title("🚀 Crash AI v4 - Advanced Learning Engine")

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div style="text-align:center; padding:1rem; background:{action_color}; border-radius:10px;">
        <div style="font-size:0.8rem;">SIGNAL</div>
        <div class="big-metric">{signal}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("CONFIDENCE", f"{confidence:.1f}%", delta=f"{confidence - 50:.0f}%" if confidence != 50 else None)

with col3:
    st.metric("TARGET", f"{target}x" if target else "N/A", 
              delta=f"{target - adaptive['mid']:.1f}" if target and adaptive['mid'] else None)

with col4:
    st.metric("REGIME", regime_data["regime"])

with col5:
    st.metric("POSITION", f"{position_size:.1%}", delta="Kelly" if position_size > 0 else None)

# Live betting suggestion
if target and position_size > 0:
    st.info(f"""
    💡 **Trading Decision**
    - **Action**: {signal}
    - **Target**: {target}x
    - **Position Size**: {position_size:.1%} of bankroll (${position_size * 1000:.0f})
    - **Expected Value**: ${(proba * (target - 1) - (1 - proba)) * position_size * 1000:.2f}
    """)

# Enhanced visualizations
st.markdown("## 📈 Market Analysis")

col1, col2 = st.columns(2)

with col1:
    # Gauge chart for confidence
    fig_gauge = create_gauge_chart(confidence, "AI Confidence")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Feature importance over time
    if st.session_state.feature_importance_history:
        last_importance = st.session_state.feature_importance_history[-1]["importance"]
        imp_df = pd.DataFrame([last_importance]).T.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(10)
        
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Top 10 Features", color="Importance",
                         color_continuous_scale="Viridis")
        fig_imp.update_layout(height=300)
        st.plotly_chart(fig_imp, use_container_width=True)

# Last 10 rounds visualization
st.markdown("### 🎲 Recent Rounds")
last_15 = df_ml["crash"].tail(15).values
cols = st.columns(15)

for i, (col, val) in enumerate(zip(cols, last_15)):
    color = "#00ff00" if val >= 2 else "#ff4444"
    col.markdown(f"""
    <div style="background:{color}; padding:5px; border-radius:5px; text-align:center; color:white;">
        {val:.1f}x
    </div>
    """, unsafe_allow_html=True)

# Advanced regime analysis
with st.expander("🎯 Advanced Market Analysis"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Current Regime:</strong><br>
            {regime_data['regime']}<br>
            <strong>Avg Multiplier:</strong> {regime_data['avg']:.2f}x<br>
            <strong>Volatility (Std):</strong> {regime_data['std']:.2f}<br>
            <strong>Low Ratio:</strong> {regime_data['low_ratio']:.1%}<br>
            <strong>High Ratio:</strong> {regime_data['high_ratio']:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <strong>📈 Market Statistics:</strong><br>
            <strong>Skewness:</strong> {regime_data['skewness']:.2f}<br>
            <strong>Kurtosis:</strong> {regime_data['kurtosis']:.2f}<br>
            <strong>Regime Shift:</strong> {regime_data['regime_shift']:.2f}<br>
            <strong>Pattern:</strong> {ctx['pattern']}<br>
            <strong>Momentum:</strong> {ctx['momentum']:.2%}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-box">
            <strong>🎯 Risk Metrics:</strong><br>
            <strong>Volatility Cluster:</strong> {ctx['volatility_cluster']:.2f}<br>
            <strong>10-Period Range:</strong> {ctx['range_10']:.2f}<br>
            <strong>Low Streak:</strong> {ctx['low_streak']} rounds<br>
            <strong>High Streak:</strong> {ctx['high_streak']} rounds<br>
            <strong>Current Balance:</strong> ${st.session_state.performance_metrics['current_balance']:.0f}
        </div>
        """, unsafe_allow_html=True)

# Performance metrics dashboard
if st.session_state.performance_metrics["total_trades"] > 0:
    st.markdown("## 💰 Performance Dashboard")
    
    metrics = st.session_state.performance_metrics
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        win_rate = metrics["winning_trades"] / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    with col2:
        st.metric("Total P&L", f"${metrics['total_profit']:.2f}", 
                 delta=f"{metrics['total_profit']:.0f}")
    
    with col3:
        st.metric("Total Trades", metrics["total_trades"])
    
    with col4:
        profit_factor = abs(metrics["total_profit"] / max(1, abs(metrics["total_profit"] - metrics["total_profit"])))
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    
    with col5:
        st.metric("Max DD", f"{metrics['max_drawdown']:.1%}")
    
    # Equity curve
    if metrics["trade_history"]:
        fig_equity = create_equity_curve(metrics["trade_history"])
        if fig_equity:
            st.plotly_chart(fig_equity, use_container_width=True)

# Backtest results
if st.button("🔄 Run Advanced Backtest"):
    with st.spinner("Running comprehensive backtest..."):
        st.session_state.backtest_results = run_advanced_backtest(df_ml)
    
if st.session_state.backtest_results is not None:
    st.markdown("## 📊 Backtest Results")
    st.dataframe(
        st.session_state.backtest_results.style.format({
            'final_balance': '${:.2f}',
            'total_return': '${:.2f}',
            'return_pct': '{:.1f}%',
            'win_rate': '{:.1%}',
            'sharpe_ratio': '{:.2f}',
            'max_drawdown': '${:.2f}'
        }),
        use_container_width=True
    )

# Data tables
with st.expander("📋 Data Explorer"):
    tab1, tab2 = st.tabs(["Recent Rounds", "Model Performance"])
    
    with tab1:
        st.dataframe(df_ui.head(50), use_container_width=True)
        
        # Download button
        csv = df_ui.head(1000).to_csv(index=False)
        st.download_button("📥 Download Data", csv, "crash_data.csv", "text/csv")
    
    with tab2:
        if st.session_state.training_history:
            training_df = pd.DataFrame(st.session_state.training_history)
            st.line_chart(training_df.set_index("timestamp")["accuracy"])
            st.dataframe(training_df.tail(10))

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Live Mode (5s refresh)", False)
if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown
