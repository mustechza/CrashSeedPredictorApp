import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy, fallback if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("SciPy not available. Some advanced statistics will be disabled.")

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
        if len(returns) < 5:
            return 0.5
            
        volatility_cluster = returns.rolling(5).std().mean()
        
        # Normalize to 0-1 range (higher = deeper market)
        depth = 1 / (1 + volatility_cluster * 10) if not np.isnan(volatility_cluster) else 0.5
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
        if len(deltas) < period:
            return 0.5
            
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
        
        # Trend strength using z-score if scipy available
        if SCIPY_AVAILABLE and len(recent) >= 5:
            try:
                z_score = stats.zscore(recent)
                trend_strength = abs(np.mean(z_score[-5:])) if len(z_score) >= 5 else 0
                if trend_strength > 1.5:
                    pattern_score += 0.1
            except:
                pass
        
        # Mean reversion potential
        if len(recent) >= 5:
            current_deviation = (recent[-1] - np.mean(recent)) / (np.std(recent) + 0.001)
            if abs(current_deviation) > 1:
                pattern_score += 0.15 if current_deviation < 0 else -0.1
        
        # Volatility contraction/expansion
        if len(recent) >= 10:
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
    
    def adjust_confidence(self, base_confidence, market_regime, recent_accuracy, streak, df=None):
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
        if df is not None:
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
    
    context = {
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
    }
    
    # Add returns-based metrics if available
    if len(returns) > 0:
        context["sharpe_ratio"] = returns.mean() / (returns.std() + 0.001)
        context["var_95"] = np.percentile(returns, 5) if len(returns) > 0 else 0
    else:
        context["sharpe_ratio"] = 0
        context["var_95"] = 0
    
    return context

# -------------------------------
# ENHANCED REGIME DETECTION
# -------------------------------
def calculate_hurst_exponent(price_series):
    """Calculate Hurst exponent for trend detection"""
    if len(price_series) < 20:
        return 0.5
    
    lags = range(2, min(20, len(price_series) // 2))
    tau = []
    
    for lag in lags:
        diff = np.log(price_series[lag:]) - np.log(price_series[:-lag])
        var = np.var(diff)
        if not np.isnan(var) and var > 0:
            tau.append(var)
    
    if len(tau) < 2:
        return 0.5
        
    try:
        m = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        hurst = m[0] / 2
        return max(0, min(1, hurst))
    except:
        return 0.5

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

# -------------------------------
# ADVANCED MULTIPLIER OPTIMIZATION
# -------------------------------
def evaluate_multiplier_advanced(df, target, window=100, min_samples=30):
    """Advanced evaluation with risk-adjusted returns and market regime"""
    if len(df) < min_samples:
        return 0, 0, 0, 0, 0, 0, 0, 0
    
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
    
    if total_trades == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
        
    win_rate = wins / total_trades
    returns_array = np.array(returns)
    
    sharpe = returns_array.mean() / (returns_array.std() + 0.001)
    
    # Sortino ratio (downside deviation)
    negative_returns = returns_array[returns_array < 0]
    sortino = returns_array.mean() / (negative_returns.std() + 0.001) if len(negative_returns) > 0 else 0
    
    # Advanced metrics
    positive_sum = sum(r for r in returns if r > 0)
    negative_sum = abs(sum(r for r in returns if r < 0))
    profit_factor = positive_sum / negative_sum if negative_sum > 0 else 0
    
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
            score = (profit * win_rate * (1 + abs(sharpe)) * (1 + abs(sortino)) * 
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
                "score": score if not np.isnan(score) else 0
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
    res = res[res['score'].notna()].sort_values("score", ascending=False)
    
    # Risk-based categorization
    ultra_conservative = res[res["m"] <= 1.5]
    low_risk = res[res["m"] <= 1.8]
    medium_risk = res[(res["m"] > 1.8) & (res["m"] <= 2.5)]
    high_risk = res[res["m"] > 2.5]
    
    return {
        "ultra_conservative": ultra_conservative.iloc[0]["m"] if len(ultra_conservative) > 0 else 1.3,
        "low": low_risk.iloc[0]["m"] if len(low_risk) > 0 else 1.5,
        "mid": medium_risk.iloc[0]["m"] if len(medium_risk) > 0 else 2.0,
        "high": high_risk.iloc[0]["m"] if len(high_risk) > 0 else 3.0,
        "table": res
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
            if 'cv_mean' in history_df.columns:
                st.caption(f"Latest CV Accuracy: {history_df.iloc[-1]['cv_mean']:.2%} ±{history_df.iloc[-1]['cv_std']:.2%}")

# -------------------------------
# MAIN APPLICATION
# -------------------------------
if st.session_state.df is None:
    st.info("📤 Upload your JSON file to begin. The AI will learn from every round you add!")
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
# ENSEMBLE PREDICTION
# -------------------------------
# ML prediction
last_row = df_ml.iloc[[-1]]
X_live = last_row[FEATURES]

try:
    ml_proba = model.predict_proba(X_live)[0][1]
except Exception as e:
    st.warning(f"ML Prediction error: {str(e)}")
    ml_proba = 0.5

# Technical prediction
tech_proba = EnsemblePredictor.technical_prediction(df_ml)

# Pattern recognition
pattern_proba = EnsemblePredictor.pattern_recognition(df_ml)

# Ensemble weighted average
ensemble_proba = (ml_proba * 0.5 + tech_proba * 0.3 + pattern_proba * 0.2)
proba = ensemble_proba

# Get feature importance
try:
    feature_importance = dict(zip(FEATURES, model.model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
except:
    top_features = [("feature1", 0), ("feature2", 0)]

# -------------------------------
# ENHANCED CONFIDENCE ENGINE
# -------------------------------
base_confidence = proba * 50

# Volatility adjustment
if ctx["volatility"] > 3.0:
    base_confidence += 30
elif ctx["volatility"] > 2.0:
    base_confidence += 20
elif ctx["volatility"] > 1.5:
    base_confidence += 10

# Streak adjustments
if ctx["current_streak"] >= 8:
    base_confidence += 25
elif ctx["current_streak"] >= 5:
    base_confidence += 15
elif ctx["current_streak"] >= 3:
    base_confidence += 5

if ctx["low_streak"] >= 6:
    base_confidence += 20
elif ctx["low_streak"] >= 4:
    base_confidence += 10

if ctx["high_streak"] >= 5:
    base_confidence -= 20
elif ctx["high_streak"] >= 3:
    base_confidence -= 10

# Regime adjustment
base_confidence += regime_data["confidence_boost"]

# Dynamic confidence adjustment
confidence_adjuster = DynamicConfidenceAdjuster()
recent_accuracy = None
if st.session_state.performance_metrics:
    recent_accuracy = st.session_state.performance_metrics[-1]["rolling_accuracy"]

confidence = confidence_adjuster.adjust_confidence(
    base_confidence, 
    regime_data["regime"], 
    recent_accuracy, 
    ctx["current_streak"],
    df_ml
)

# Learning progress boost (diminishing returns)
learning_progress = min(len(df_ml) / 1000, 0.3)
confidence *= (1 + learning_progress)

# Apply confidence adaptation from historical performance
confidence *= st.session_state.confidence_adaptation

# Ensure bounds
confidence = max(0, min(100, confidence))

# -------------------------------
# ADAPTIVE MULTIPLIERS
# -------------------------------
adaptive = get_adaptive_multipliers_advanced(df_ml)

# -------------------------------
# SIGNAL ENGINE (REGIME AWARE)
# -------------------------------
if confidence > 80:
    signal = "🔥 STRONG BET"
    target = adaptive["high"]
    bet_size = "Large (5-10%)"
    signal_color = "strong"
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
        target = adaptive["ultra_conservative"]
        bet_size = "Tiny (0.5-1%)"
    else:
        target = adaptive["mid"]
        bet_size = "Small (1-2%)"
else:
    signal = "❌ SKIP"
    target = None
    bet_size = "None"
    signal_color = "skip"

# Log prediction
ensemble_scores = {
    "ml": ml_proba,
    "technical": tech_proba,
    "pattern": pattern_proba
}
log_prediction(signal, target, confidence, regime_data["regime"], ensemble_scores=ensemble_scores)

# Calculate recent accuracy
recent_predictions = [p for p in st.session_state.predictions_log if p["was_correct"] is not None]
recent_accuracy = sum(p["was_correct"] for p in recent_predictions) / len(recent_predictions) if recent_predictions else 0

# -------------------------------
# UI - MAIN DASHBOARD
# -------------------------------
st.markdown("## 🔥 LIVE AI DECISION")

# Dynamic card styling
confidence_class = "confidence-high" if confidence > 70 else "confidence-medium" if confidence > 50 else "confidence-low"

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

# Ensemble breakdown
with st.expander("🎯 Ensemble Model Breakdown"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("ML Model", f"{ml_proba:.2%}")
    with col_b:
        st.metric("Technical Analysis", f"{tech_proba:.2%}")
    with col_c:
        st.metric("Pattern Recognition", f"{pattern_proba:.2%}")

# Risk level indicator
risk_levels = {"Very Low": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Very High": "⚫"}
st.markdown(f"**Risk Level:** {risk_levels.get(regime_data['risk_level'], '🟡')} {regime_data['risk_level']}")

# Strategy recommendation
st.info(f"📋 **Regime Strategy:** {regime_data['strategy']}")

# Learning progress
progress_value = min(len(df_ml) / 1000, 1.0)
st.progress(progress_value, text=f"🤖 Learning Progress: {len(df_ml)}/1000 rounds for optimal performance")

# -------------------------------
# BET SUGGESTION WITH KELLY
# -------------------------------
if target:
    # Calculate streak adjustment
    streak_adj = max(0.5, min(1.5, 1 - (ctx["current_streak"] - 3) * 0.1))
    kelly = calculate_kelly_advanced(proba, target, confidence, streak_adj)
    
    st.success(f"💡 **Suggested Action**: {signal} at **{target}x** | {bet_size} of bankroll")
    
    if kelly > 0:
        st.caption(f"📐 Kelly Criterion suggests: **{kelly:.1%}** of bankroll (max {calculate_kelly_advanced(proba, target, confidence, 0.5):.1%} conservative)")
        
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
    st.markdown("**Ensemble Model Contributions:**")
    st.write(f"- Machine Learning: {ml_proba:.2%} (50% weight)")
    st.write(f"- Technical Analysis: {tech_proba:.2%} (30% weight)")
    st.write(f"- Pattern Recognition: {pattern_proba:.2%} (20% weight)")
    
    st.markdown("**Top Features Influencing ML Model:**")
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
    st.write(f"- Hurst Exponent: {regime_data['hurst_exponent']:.2f} ({'Trending' if regime_data['hurst_exponent'] > 0.6 else 'Mean-Reverting' if regime_data['hurst_exponent'] < 0.4 else 'Random'})")
    st.write(f"- Low Ratio: {regime_data['low_ratio']:.1%}")
    st.write(f"- High Ratio: {regime_data['high_ratio']:.1%}")
    st.write(f"- Extreme Ratio (>5x): {regime_data['extreme_ratio']:.1%}")
    
    st.markdown("**Confidence Adjustments:**")
    st.write(f"- Base Confidence: {base_confidence:.1f}%")
    st.write(f"- Regime Boost: {regime_data['confidence_boost']:.1f}%")
    st.write(f"- Performance Factor: {recent_accuracy if recent_accuracy else 0:.1%}")
    st.write(f"- Market Depth: {MarketMicrostructure.calculate_market_depth(df_ml):.2f}")
    st.write(f"- Adaptation Factor: {st.session_state.confidence_adaptation:.2f}")

# -------------------------------
# INSIGHTS DASHBOARD
# -------------------------------
with st.expander("📊 Advanced Analytics"):
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Risk Analysis", "Market Statistics", "Micro Patterns"])
    
    with tab1:
        if st.session_state.performance_metrics:
            perf_df = pd.DataFrame(st.session_state.performance_metrics)
            st.line_chart(perf_df.set_index("timestamp")["rolling_accuracy"])
            st.metric("Total Predictions", st.session_state.total_predictions)
            st.metric("Correct Predictions", st.session_state.correct_predictions)
            if st.session_state.total_predictions > 0:
                st.metric("Overall Accuracy", f"{st.session_state.correct_predictions/st.session_state.total_predictions:.1%}")
        
        # Ensemble performance
        st.subheader("Ensemble Model Performance")
        ensemble_df = pd.DataFrame(st.session_state.predictions_log[-50:])
        if len(ensemble_df) > 0:
            st.write("Recent ensemble contributions tracked")
    
    with tab2:
        if not adaptive["table"].empty:
            st.subheader("Multiplier Risk-Reward Analysis")
            display_cols = ["m", "win_rate", "sharpe", "sortino", "drawdown", "profit_factor"]
            st.dataframe(adaptive["table"][display_cols].head(10), use_container_width=True)
        
        st.subheader("Kelly vs Confidence")
        if target:
            st.metric("Recommended Kelly", f"{kelly:.1%}")
            st.metric("Current Confidence", f"{confidence:.1f}%")
            st.metric("Streak Adjustment", f"{streak_adj:.2f}")
            st.caption("Kelly based on ensemble probability vs implied odds")
    
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
        
        st.subheader("Advanced Metrics")
        st.metric("Hurst Exponent", f"{regime_data['hurst_exponent']:.3f}")
        st.metric("Sharpe Ratio (10-period)", f"{ctx.get('sharpe_ratio', 0):.2f}")
        st.metric("VaR 95%", f"{ctx.get('var_95', 0):.2%}")
    
    with tab4:
        st.subheader("Micro Pattern Detection")
        patterns = MarketMicrostructure.detect_micro_patterns(df_ml)
        for pattern, detected in patterns.items():
            status = "✅ Detected" if detected else "❌ Not Detected"
            st.write(f"- {pattern.replace('_', ' ').title()}: {status}")
        
        st.subheader("Order Flow Analysis")
        order_flow = MarketMicrostructure.calculate_order_flow_imbalance(df_ml)
        st.metric("Order Flow Imbalance", f"{order_flow:.2f}")
        if order_flow > 0.3:
            st.info("🟢 Positive order flow - buying pressure detected")
        elif order_flow < -0.3:
            st.warning("🔴 Negative order flow - selling pressure detected")

# -------------------------------
# MULTIPLIER PERFORMANCE TABLE
# -------------------------------
if not adaptive["table"].empty:
    st.subheader("🎯 Multiplier Performance Ranking")
    display_cols = ["m", "win_rate", "sharpe", "sortino", "profit", "drawdown", "profit_factor"]
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
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if st.button("📥 Export Data", use_container_width=True):
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"crash_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            key="data_download"
        )

with col_exp2:
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
st.caption("🚀 Crash AI v5 - Advanced Ensemble Decision Engine | Risk Warning: Never bet more than you can afford to lose")
