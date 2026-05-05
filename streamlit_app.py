import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data.loader import load_data
from data.cleaner import clean_data, FEATURES
from training.trainer import prepare_data
from models.random_forest import RFModel

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Crash AI v5 - Advanced Decision Engine",
    page_icon="🚀"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .decision-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .confidence-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f2994a, #f2c94c);
    }
    .confidence-low {
        background: linear-gradient(135deg, #eb3349, #f45c43);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .signal-strong {
        color: #00ff00;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crash AI v5 - Advanced Neural Decision Engine")

# -------------------------------
# ENHANCED SESSION STATE
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
        "total_predictions": 0,
        "correct_predictions": 0,
        "performance_metrics": [],
        "decision_history": deque(maxlen=100),
        "ensemble_predictions": [],
        "market_memory": deque(maxlen=50),
        "confidence_adaptation": 1.0,
        "streak_multiplier": 1.0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -------------------------------
# ADVANCED MARKET MICROSTRUCTURE
# -------------------------------
class MarketMicrostructure:
    """Advanced market microstructure analysis"""
    
    @staticmethod
    def calculate_order_flow_imbalance(df, window=20):
        """Calculate order flow imbalance based on crash patterns"""
        if len(df) < window:
            return 0
        
        recent = df.tail(window)
        # Simulate order flow based on crash sizes
        buy_pressure = (recent['crash'] > recent['crash'].shift(1)).sum()
        sell_pressure = (recent['crash'] < recent['crash'].shift(1)).sum()
        
        if buy_pressure + sell_pressure == 0:
            return 0
        
        return (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure)
    
    @staticmethod
    def calculate_market_depth(df, window=30):
        """Estimate market depth based on volatility clustering"""
        if len(df) < window:
            return 0.5
        
        returns = np.log(df['crash'].tail(window) / df['crash'].tail(window).shift(1)).dropna()
        volatility_cluster = returns.rolling(5).std().mean()
        
        # Normalize to 0-1 range (higher = deeper market)
        depth = 1 / (1 + volatility_cluster * 10)
        return max(0, min(1, depth))
    
    @staticmethod
    def detect_micro_patterns(df):
        """Detect micro-patterns in recent price action"""
        if len(df) < 10:
            return {}
        
        recent = df.tail(20)['crash'].values
        patterns = {
            'double_top': False,
            'double_bottom': False,
            'momentum_divergence': False,
            'volume_cluster': False
        }
        
        # Detect double top/bottom
        if len(recent) >= 10:
            peaks = []
            for i in range(2, len(recent)-2):
                if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                    peaks.append((i, recent[i]))
            
            if len(peaks) >= 2:
                if abs(peaks[-1][1] - peaks[-2][1]) / peaks[-2][1] < 0.05:
                    patterns['double_top'] = True
            
            troughs = []
            for i in range(2, len(recent)-2):
                if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                    troughs.append((i, recent[i]))
            
            if len(troughs) >= 2:
                if abs(troughs[-1][1] - troughs[-2][1]) / troughs[-2][1] < 0.05:
                    patterns['double_bottom'] = True
        
        # Detect momentum divergence
        if len(recent) >= 14:
            price_change = recent[-1] - recent[-7]
            momentum = np.mean(recent[-3:]) - np.mean(recent[-7:-4])
            
            if price_change > 0 and momentum < 0:
                patterns['momentum_divergence'] = True
            elif price_change < 0 and momentum > 0:
                patterns['momentum_divergence'] = True
        
        return patterns

# -------------------------------
# ENSEMBLE PREDICTION ENGINE
# -------------------------------
class EnsemblePredictor:
    """Multi-model ensemble prediction system"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    @staticmethod
    def technical_prediction(df):
        """Technical analysis based prediction"""
        if len(df) < 20:
            return 0.5
        
        recent = df.tail(20)['crash']
        rsi = EnsemblePredictor.calculate_rsi(recent.values)
        macd = EnsemblePredictor.calculate_macd(recent.values)
        bollinger = EnsemblePredictor.calculate_bollinger_position(recent.values)
        
        # Combine technical signals
        tech_score = (rsi * 0.3 + macd * 0.3 + bollinger * 0.4)
        return max(0, min(1, tech_score))
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 0.5
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 1.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to 0-1 (30-70 range)
        normalized = (rsi - 30) / 40
        return max(0, min(1, normalized))
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if len(prices) < slow + signal:
            return 0.5
        
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        if len(macd_line) > 1 and len(signal_line) > 1:
            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                return 0.7
            else:
                return 0.3
        
        return 0.5
    
    @staticmethod
    def calculate_bollinger_position(prices, period=20):
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        current_price = prices[-1]
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        if current_price >= upper_band:
            return 0.2  # Overbought
        elif current_price <= lower_band:
            return 0.8  # Oversold
        else:
            # Position between bands
            position = (current_price - lower_band) / (upper_band - lower_band)
            return 0.5 + (0.5 - position) * 0.5
    
    @staticmethod
    def pattern_recognition(df):
        """Advanced pattern recognition"""
        if len(df) < 30:
            return 0.5
        
        recent = df.tail(30)['crash'].values
        pattern_score = 0.5
        
        # Trend strength
        z_score = stats.zscore(recent)
        trend_strength = abs(np.mean(z_score[-5:]))
        if trend_strength > 1.5:
            pattern_score += 0.1
        
        # Mean reversion potential
        current_deviation = (recent[-1] - np.mean(recent)) / np.std(recent)
        if abs(current_deviation) > 1:
            pattern_score += 0.15 if current_deviation < 0 else -0.1
        
        # Volatility contraction/expansion
        recent_vol = np.std(recent[-10:])
        historical_vol = np.std(recent)
        
        if recent_vol < historical_vol * 0.7:
            pattern_score += 0.1  # Volatility contraction - potential breakout
        elif recent_vol > historical_vol * 1.5:
            pattern_score -= 0.05  # High volatility - caution
        
        return max(0, min(1, pattern_score))

# -------------------------------
# DYNAMIC CONFIDENCE ADJUSTMENT
# -------------------------------
class DynamicConfidenceAdjuster:
    """Adaptive confidence adjustment based on market conditions"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=50)
        self.regime_performance = {}
    
    def adjust_confidence(self, base_confidence, market_regime, recent_accuracy, streak):
        """Dynamically adjust confidence based on multiple factors"""
        adjusted = base_confidence
        
        # Regime-based adjustment
        regime_boost = {
            "🔥 EXTREME VOLATILITY": -0.3,
            "⚡ VOLATILE": -0.15,
            "🔴 EXTREME CHOPPY": -0.4,
            "🔴 CHOPPY": -0.2,
            "🟢 HOT": 0.2,
            "🟢 WARM": 0.1,
            "🟡 NORMAL": 0.0
        }
        
        for regime, boost in regime_boost.items():
            if regime in market_regime:
                adjusted += boost * base_confidence
                break
        
        # Recent performance adjustment
        if recent_accuracy:
            performance_factor = (recent_accuracy - 0.5) * 0.5
            adjusted += performance_factor * base_confidence
        
        # Streak psychology
        if streak > 5:
            adjusted -= 0.15 * base_confidence  # Reduce confidence during long streaks
        elif streak < -3:
            adjusted += 0.1 * base_confidence  # Increase after losses (mean reversion)
        
        # Adaptive scaling based on market depth
        microstructure = MarketMicrostructure()
        market_depth = microstructure.calculate_market_depth(df)
        adjusted *= (0.8 + market_depth * 0.4)
        
        return max(0, min(100, adjusted))

# -------------------------------
# ENHANCED MODEL TRAINING
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
                from sklearn.model_selection import cross_val_score
                
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
                
                return model
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return None
    
    return st.session_state.model

# -------------------------------
# ENHANCED VALIDATION
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
# ENHANCED PREDICTION LOGGING
# -------------------------------
def log_prediction(signal, target, confidence, regime, actual=None, ensemble_scores=None):
    """Log prediction with deduplication"""
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
            "round_number": len(st.session_state.df),
            "ensemble_scores": ensemble_scores
        })
        
        st.session_state.last_data_hash = current_hash
        st.session_state.total_predictions += 1
        
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
            
            # Sophisticated correctness determination
            if last_pred["signal"] == "❌ SKIP":
                last_pred["was_correct"] = True
            else:
                # Dynamic correctness based on confidence
                confidence_factor = last_pred["confidence"] / 100
                target_adjustment = 1 + (1 - confidence_factor) * 0.2
                effective_target = last_pred["target"] * target_adjustment
                
                last_pred["was_correct"] = actual_multiplier >= effective_target
            
            if last_pred["was_correct"]:
                st.session_state.correct_predictions += 1
                st.session_state.confidence_adaptation *= 1.05
            else:
                st.session_state.confidence_adaptation *= 0.95
            
            # Update rolling accuracy
            recent = [p for p in st.session_state.predictions_log[-50:] if p["was_correct"] is not None]
            if recent:
                rolling_acc = sum(p["was_correct"] for p in recent) / len(recent)
                st.session_state.performance_metrics.append({
                    "timestamp": datetime.now(),
                    "rolling_accuracy": rolling_acc,
                    "total_predictions": st.session_state.total_predictions
                })

# -------------------------------
# ADVANCED CONTEXT FEATURES
# -------------------------------
def get_context(df):
    """Extract enhanced context features"""
    last_10 = df.tail(10)["crash"]
    last_20 = df.tail(20)["crash"]
    last_50 = df.tail(50)["crash"]
    all_time = df["crash"]
    
    # Calculate streaks
    recent_crashes = last_10.values
    current_streak = 0
    for i in range(len(recent_crashes)-1, -1, -1):
        if recent_crashes[i] < 2:
            current_streak += 1
        else:
            break
    
    # Calculate additional metrics
    returns = np.log(df['crash'] / df['crash'].shift(1)).dropna()
    
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
        "skew": all_time.skew() if len(all_time) > 2 else 0,
        "kurtosis": all_time.kurtosis() if len(all_time) > 3 else 0,
        "sharpe_ratio": returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0,
        "var_95": np.percentile(returns, 5) if len(returns) > 0 else 0
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
    
    # Advanced statistical measures
    hurst_exponent = calculate_hurst_exponent(last_50.values) if len(last_50) >= 20 else 0.5
    
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
        strategy = "Ultra-conservative, wait for stability"
    elif std > 2.0:
        regime = "⚡ VOLATILE"
        confidence_boost = 10
        risk_level = "High"
        strategy = "Small positions, tight stops"
    elif low_ratio > 0.7:
        regime = "🔴 EXTREME CHOPPY"
        confidence_boost = -30
        risk_level = "Very Low"
        strategy = "Avoid trading, high uncertainty"
    elif low_ratio > 0.5:
        regime = "🔴 CHOPPY"
        confidence_boost = -15
        risk_level = "Low"
        strategy = "Low probability setups only"
    elif high_ratio > 0.5:
        regime = "🟢 HOT"
        confidence_boost = 20
        risk_level = "High"
        strategy = "Aggressive but disciplined"
    elif high_ratio > 0.3:
        regime = "🟢 WARM"
        confidence_boost = 10
        risk_level = "Medium"
        strategy = "Normal trading, standard size"
    else:
        regime = "🟡 NORMAL"
        confidence_boost = 0
        risk_level = "Medium"
        strategy = "Balanced approach"
    
    # Add Hurst exponent insight
    if hurst_exponent > 0.6:
        regime += " (Trending)"
        confidence_boost += 10
    elif hurst_exponent < 0.4:
        regime += " (Mean-Reverting)"
        confidence_boost += 5
    
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
        "risk_level": risk_level,
        "strategy": strategy,
        "hurst_exponent": hurst_exponent
    }

def calculate_hurst_exponent(price_series):
    """Calculate Hurst exponent for trend detection"""
    if len(price_series) < 20:
        return 0.5
    
    lags = range(2, min(20, len(price_series) // 2))
    tau = []
    
    for lag in lags:
        diff = np.log(price_series[lag:]) - np.log(price_series[:-lag])
        var = np.var(diff)
        tau.append(var)
    
    try:
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = m[0] / 2
        return max(0, min(1, hurst))
    except:
        return 0.5

# -------------------------------
# ADVANCED MULTIPLIER OPTIMIZATION
# -------------------------------
def evaluate_multiplier_advanced(df, target, window=100, min_samples=30):
    """Advanced evaluation with risk-adjusted returns and market regime"""
    if len(df) < min_samples:
        return 0, 0, 0, 0, 0, 0
    
    balance = 0
    stake = 1
    wins = 0
    total_trades = 0
    returns = []
    max_drawdown = 0
    peak = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    start = max(min_samples, len(df) - window)
    
    for i in range(start, len(df) - 1):
        crash = df.iloc[i + 1]["crash"]
        
        if crash >= target:
            profit = stake * (target - 1)
            balance += profit
            wins += 1
            returns.append(profit)
            consecutive_losses = 0
        else:
            balance -= stake
            returns.append(-stake)
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        total_trades += 1
        
        # Track drawdown
        peak = max(peak, balance)
        drawdown = (peak - balance) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
    sortino = np.mean(returns) / np.std([r for r in returns if r < 0]) if len(returns) > 0 else 0
    
    # Advanced metrics
    profit_factor = abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else 0
    risk_score = (win_rate * sharpe * sortino) / (max_drawdown + 0.01) if max_drawdown > 0 else win_rate * sharpe * sortino * 100
    
    return balance, win_rate, sharpe, sortino, risk_score, max_drawdown, profit_factor, max_consecutive_losses

def get_adaptive_multipliers_advanced(df):
    """Get optimal multipliers based on advanced risk metrics"""
    multipliers = [1.2, 1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    results = []
    for m in multipliers:
        (profit, win_rate, sharpe, sortino, risk_score, 
         drawdown, profit_factor, max_losses) = evaluate_multiplier_advanced(df, m)
        
        if profit > 0 or win_rate > 0:
            # Comprehensive scoring
            score = (profit * win_rate * (1 + sharpe) * (1 + sortino) * 
                    (1 - drawdown) * profit_factor / (1 + max_losses/10))
            
            results.append({
                "m": m,
                "profit": profit,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "sortino": sortino,
                "risk_score": risk_score,
                "drawdown": drawdown,
                "profit_factor": profit_factor,
                "max_losses": max_losses,
                "score": score
            })
    
    if not results:
        return {
            "low": 1.5,
            "mid": 2.0,
            "high": 3.0,
            "ultra_conservative": 1.3,
            "table": pd.DataFrame()
        }
    
    res = pd.DataFrame(results)
    
    # Risk-based categorization
    ultra_conservative = res[res["m"] <= 1.5]
    low_risk = res[res["m"] <= 1.8]
    medium_risk = res[(res["m"] > 1.8) & (res["m"] <= 2.5)]
    high_risk = res[res["m"] > 2.5]
    
    return {
        "ultra_conservative": ultra_conservative.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(ultra_conservative) > 0 else 1.3,
        "low": low_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(low_risk) > 0 else 1.5,
        "mid": medium_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(medium_risk) > 0 else 2.0,
        "high": high_risk.sort_values("risk_score", ascending=False).iloc[0]["m"] if len(high_risk) > 0 else 3.0,
        "table": res.sort_values("score", ascending=False)
    }

# -------------------------------
# ENHANCED KELLY CRITERION
# -------------------------------
def calculate_kelly_advanced(probability, target, confidence, streak_adjustment=1.0):
    """Advanced Kelly Criterion with confidence and streak adjustment"""
    if not target or target <= 1:
        return 0
    
    implied_prob = 1 / target
    edge = probability - implied_prob
    
    if edge <= 0 or implied_prob >= 1:
        return 0
    
    # Base Kelly
    kelly = edge / (1 - implied_prob)
    
    # Confidence adjustment
    confidence_factor = confidence / 100
    kelly *= confidence_factor
    
    # Streak adjustment (reduce during long winning streaks)
    kelly *= streak_adjustment
    
    # Cap at reasonable levels
    max_cap = 0.25
    conservative_cap = 0.15
    
    if confidence < 60:
        return max(0, min(conservative_cap, kelly * 0.5))
    elif confidence < 75:
        return max(0, min(conservative_cap, kelly * 0.75))
    else:
        return max(0, min(max_cap, kelly))

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
st.sidebar.markdown("## 📁 Historical Data")
file = st.sidebar.file_uploader("Upload JSON (once)", type=["json"], key="file_uploader")

if file and st.session_state.df is None:
    with st.spinner("Loading historical data..."):
        try:
            st.session_state.df = load_data(file)
            st.success(f"✅ Loaded {len(st.session_state.df)} historical rounds!")
            
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
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Rounds", len(st.session_state.df))
    with col2:
        st.metric("Rounds since training", st.session_state.rounds_since_training)
    
    if st.session_state.last_training_time:
        st.sidebar.caption(f"Last trained: {st.session_state.last_training_time.strftime('%H:%M:%S')}")
    
    col_retrain, col_reset = st.sidebar.columns(2)
    with col_retrain:
        if st.button("🔄
