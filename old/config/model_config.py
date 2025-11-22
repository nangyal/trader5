"""
Model Training Configuration
XGBoost hyperparameters and training settings
"""

# ============================================================================
# XGBoost Model Parameters
# ============================================================================

XGBOOST_PARAMS = {
    'n_estimators': 500,        # Number of boosting rounds (trees)
    'max_depth': 6,             # Maximum tree depth
    'learning_rate': 0.1,       # Step size shrinkage (eta)
    'subsample': 0.8,           # Fraction of samples for each tree
    'colsample_bytree': 0.8,    # Fraction of features for each tree
    'gamma': 0,                 # Minimum loss reduction for split
    'min_child_weight': 1,      # Minimum sum of instance weight in child
    'tree_method': 'hist',      # Tree construction algorithm (GPU optimized)
    'device': 'cuda:0',         # Device: 'cuda:0' for GPU, 'cpu' for CPU
    'random_state': 42,         # Reproducibility
    'eval_metric': 'mlogloss'   # Evaluation metric
}

# CPU Fallback parameters (if GPU unavailable)
XGBOOST_PARAMS_CPU = {
    **XGBOOST_PARAMS,
    'device': 'cpu',
    'tree_method': 'hist'
}

# ============================================================================
# Hyperparameter Optimization
# ============================================================================

# Enable/disable hyperparameter search
OPTIMIZE_HYPERPARAMS = False  # Set to True for RandomizedSearchCV

# RandomizedSearchCV parameter distributions
PARAM_DISTRIBUTIONS = {
    'n_estimators': [300, 500, 800],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

# RandomizedSearchCV settings
RANDOMIZED_SEARCH_CONFIG = {
    'n_iter': 10,           # Number of parameter combinations to try
    'cv': 3,                # Cross-validation folds
    'scoring': 'accuracy',  # Scoring metric
    'random_state': 42,
    'n_jobs': 1,            # Parallel jobs (1 for GPU, -1 for CPU)
    'verbose': 1
}

# ============================================================================
# Training Configuration
# ============================================================================

# Train/test split
TEST_SIZE = 0.2             # 20% of data for testing
RANDOM_STATE = 42           # Seed for reproducibility

# Class imbalance handling
HANDLE_CLASS_IMBALANCE = True  # Use balanced class weights

# Minimum samples per pattern class
MIN_PATTERNS_PER_CLASS = 30    # Filter out rare patterns

# ============================================================================
# Feature Engineering
# ============================================================================

# Feature groups to include
FEATURE_GROUPS = {
    'price_features': True,         # OHLC-based features
    'trend_indicators': True,       # SMA, EMA, MACD
    'momentum_indicators': True,    # RSI, Stochastic, Williams %R
    'volatility_indicators': True,  # ATR, Bollinger Bands
    'volume_indicators': True,      # OBV, Volume SMA
    'pattern_features': True        # Support/Resistance, Trend strength
}

# Technical indicator periods
INDICATOR_PERIODS = {
    'sma_periods': [10, 20, 50],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'stochastic_period': 14,
    'williams_period': 14,
    'atr_period': 14,
    'bbands_period': 20,
    'adx_period': 14
}

# ============================================================================
# Model Output Settings
# ============================================================================

# Model save path
MODEL_SAVE_PATH = 'enhanced_forex_pattern_model.pkl'

# Output directory for charts and reports
OUTPUT_DIR = 'outputs'

# Charts to generate
GENERATE_CHARTS = {
    'feature_importance': True,
    'confusion_matrix': True,
    'learning_curve': False,      # Disabled (slow)
    'roc_curves': False           # Disabled (multi-class complex)
}

# Feature importance chart settings
FEATURE_IMPORTANCE_TOP_N = 20   # Top N features to display

# ============================================================================
# Validation Settings
# ============================================================================

# Early stopping (disabled in XGBoost 2.0+)
EARLY_STOPPING = False
EARLY_STOPPING_ROUNDS = 50

# Verbose training output
VERBOSE_TRAINING = True

# ============================================================================
# Data Preprocessing
# ============================================================================

# Handle missing values
FILL_METHOD = 'ffill_bfill'  # Forward fill then backward fill

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_QUANTILES = (0.01, 0.99)  # Remove bottom 1% and top 1%

# Data type optimization
OPTIMIZE_DTYPES = True  # Convert float64 → float32 for GPU

# ============================================================================
# Prediction Thresholds
# ============================================================================

# Minimum probability threshold for predictions
MIN_PREDICTION_PROBABILITY = 0.6  # 60% confidence required

# Pattern-specific probability thresholds
PATTERN_PROBABILITY_THRESHOLDS = {
    'ascending_triangle': 0.7,
    'descending_triangle': 0.7,
    'symmetrical_triangle': 0.7,
    'double_top': 0.75,
    'double_bottom': 0.75,
    'head_and_shoulders': 0.75,
    'cup_and_handle': 0.7,
    'wedge_rising': 0.7,
    'wedge_falling': 0.7,
    'flag_bullish': 0.7,
    'flag_bearish': 0.7,
    'no_pattern': 0.5
}
