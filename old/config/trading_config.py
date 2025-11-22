"""
Trading Configuration
Live trading, backtesting, and risk management settings
"""

# ============================================================================
# General Trading Settings
# ============================================================================

# Trading mode
TRADING_MODE = 'demo'  # 'demo' or 'live' (use demo for safety!)

# Default trading symbol
DEFAULT_SYMBOL = 'DOGEUSDT'

# Supported symbols
SUPPORTED_SYMBOLS = [
    'DOGEUSDT',
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT'
]

# ============================================================================
# Risk Management
# ============================================================================

# Base risk per trade (% of capital)
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Tiered risk management (compound strategy)
USE_TIERED_RISK = True

RISK_TIERS = [
    {'max_capital_ratio': 2.0, 'risk': 0.02},      # <2x initial: 2%
    {'max_capital_ratio': 3.0, 'risk': 0.015},     # 2-3x initial: 1.5%
    {'max_capital_ratio': 5.0, 'risk': 0.01},      # 3-5x initial: 1%
    {'max_capital_ratio': float('inf'), 'risk': 0.0075}  # >5x initial: 0.75%
]

# Position sizing
MAX_POSITION_SIZE_USD = 5000    # Max position size in USD
MIN_POSITION_SIZE_USD = 50      # Min position size in USD

# Maximum concurrent trades
MAX_CONCURRENT_TRADES = 20

# Maximum daily trades
MAX_DAILY_TRADES = 10

# ============================================================================
# Stop Loss & Take Profit Targets
# ============================================================================

# Pattern-specific SL/TP percentages
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.015,    # -1.5% stop loss
        'tp_pct': 0.03      # +3.0% take profit
    },
    'descending_triangle': {
        'sl_pct': 0.015,
        'tp_pct': 0.03
    },
    'symmetrical_triangle': {
        'sl_pct': 0.02,
        'tp_pct': 0.04
    },
    'double_top': {
        'sl_pct': 0.02,
        'tp_pct': 0.04
    },
    'double_bottom': {
        'sl_pct': 0.02,
        'tp_pct': 0.04
    },
    'head_and_shoulders': {
        'sl_pct': 0.025,
        'tp_pct': 0.05
    },
    'cup_and_handle': {
        'sl_pct': 0.02,
        'tp_pct': 0.045
    },
    'wedge_rising': {
        'sl_pct': 0.018,
        'tp_pct': 0.036
    },
    'wedge_falling': {
        'sl_pct': 0.018,
        'tp_pct': 0.036
    },
    'flag_bullish': {
        'sl_pct': 0.015,
        'tp_pct': 0.03
    },
    'flag_bearish': {
        'sl_pct': 0.015,
        'tp_pct': 0.03
    }
}

# Trailing stop settings
USE_TRAILING_STOP = False
TRAILING_STOP_ACTIVATION_PCT = 0.015  # Activate after +1.5% profit
TRAILING_STOP_DISTANCE_PCT = 0.01     # Trail by 1%

# ============================================================================
# Trading Strategy
# ============================================================================

# Strategy type
STRATEGY_TYPE = 'long_only_aligned'  # 'long_only_aligned', 'long_short', 'long_only'

# Trend alignment settings (for long_only_aligned strategy)
TREND_ALIGNMENT = {
    'enable': True,
    'lookback_period': 20,              # Bars for trend calculation
    'use_ema_filter': True,             # Use EMA(50) as trend filter
    'ema_period': 50,
    'min_slope': 0.0001,                # Minimum slope for trend detection
    'bullish_patterns': [
        'ascending_triangle',
        'symmetrical_triangle',
        'double_bottom',
        'cup_and_handle',
        'flag_bullish',
        'wedge_falling'
    ],
    'bearish_patterns': [
        'descending_triangle',
        'double_top',
        'head_and_shoulders',
        'flag_bearish',
        'wedge_rising'
    ]
}

# Volatility filter
VOLATILITY_FILTER = {
    'enable': True,
    'atr_period': 14,
    'min_atr_pct': 0.5,     # Min 0.5% ATR (skip low volatility)
    'max_atr_pct': 5.0      # Max 5.0% ATR (skip extreme volatility)
}

# Pattern filters
PATTERN_FILTERS = {
    'min_probability': 0.7,     # Min 70% ML confidence
    'min_strength': 0.6,        # Min 60% pattern strength score
    'blacklist_patterns': []    # Patterns to skip (e.g., ['cup_and_handle'])
}

# ============================================================================
# Backtesting Configuration
# ============================================================================

# Initial capital for backtesting
BACKTEST_INITIAL_CAPITAL = 10000  # $10,000

# Backtesting mode
BACKTEST_MODE = 'realistic'  # 'realistic' or 'optimistic'

# Slippage simulation (for realistic mode)
BACKTEST_SLIPPAGE = {
    'enable': True,
    'percent': 0.001,       # 0.1% slippage
    'fixed': 0.0            # Fixed slippage in USD (0 = disabled)
}

# Commission simulation
BACKTEST_COMMISSION = {
    'enable': True,
    'percent': 0.001,       # 0.1% commission (Binance spot)
    'fixed': 0.0            # Fixed commission per trade (0 = disabled)
}

# Backtest output settings
BACKTEST_OUTPUT = {
    'save_trades': True,            # Save trades to CSV
    'plot_equity_curve': True,      # Generate equity curve chart
    'plot_drawdown': True,          # Generate drawdown chart
    'print_summary': True,          # Print results summary
    'save_path': 'backtest_results.csv'
}

# ============================================================================
# Live Trading Configuration
# ============================================================================

# Order execution settings
ORDER_EXECUTION = {
    'order_type': 'MARKET',     # 'MARKET' or 'LIMIT'
    'timeout': 30,              # Order timeout in seconds
    'max_retries': 3,           # Max retry attempts
    'retry_delay': 2            # Delay between retries (seconds)
}

# Trade monitoring
TRADE_MONITORING = {
    'check_interval': 1,        # Check trades every N seconds
    'update_balance_interval': 30,  # Update balance every N seconds
    'print_status_interval': 60     # Print status every N seconds
}

# Safety limits
SAFETY_LIMITS = {
    'max_loss_per_day_pct': 0.05,       # Max 5% daily loss
    'max_drawdown_pct': 0.15,           # Max 15% drawdown (stop trading)
    'min_balance_usd': 100,             # Min balance to continue trading
    'emergency_stop_loss_pct': 0.10     # Emergency SL if normal SL fails
}

# ============================================================================
# WebSocket Configuration
# ============================================================================

# WebSocket settings for live data
WEBSOCKET_CONFIG = {
    'use_mainnet_data': True,       # Use mainnet for real-time data
    'auto_reconnect': True,         # Auto-reconnect on disconnect
    'reconnect_delay': 5,           # Delay before reconnect (seconds)
    'max_reconnect_attempts': 3,    # Max reconnection attempts
    'ping_interval': 60,            # Keepalive ping interval
    'ping_timeout': 10              # Ping timeout
}

# Kline (candlestick) stream timeframes
KLINE_TIMEFRAMES = ['1m', '5m', '15m', '1h']

# Candle buffer sizes (deque maxlen)
CANDLE_BUFFER_SIZES = {
    '1m': 1000,
    '5m': 500,
    '15m': 300,
    '1h': 200,
    '4h': 100
}

# ============================================================================
# Data Management
# ============================================================================

# Historical data loading
HISTORICAL_DATA = {
    'load_on_start': True,      # Load historical candles on startup
    'save_to_csv': False,       # Save loaded data to CSV
    'csv_path': 'data/live/'    # Path for saved data
}

# Minimum data points before trading
MIN_DATA_POINTS = 100  # Wait for 100 candles before starting

# ============================================================================
# Logging & Alerts
# ============================================================================

# Trade logging
TRADE_LOGGING = {
    'enable': True,
    'log_file': 'trades.log',
    'log_level': 'INFO',        # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'save_json': True,          # Save trades to JSON
    'json_file': 'trades.json'
}

# Alert system
ALERTS = {
    'enable': True,
    'alert_threshold': 0.75,    # Alert on patterns with 75%+ strength
    'log_file': 'pattern_alerts.json',
    'console_output': True
}

# Performance tracking
PERFORMANCE_TRACKING = {
    'enable': True,
    'track_by_pattern': True,       # Track P&L by pattern type
    'track_by_timeframe': True,     # Track P&L by timeframe
    'save_interval': 10,            # Save stats every N trades
    'stats_file': 'performance_stats.json'
}

# ============================================================================
# API Rate Limiting
# ============================================================================

# Binance API rate limits
API_RATE_LIMITS = {
    'requests_per_minute': 1200,    # Binance spot limit
    'orders_per_10s': 100,          # Order rate limit
    'orders_per_day': 200000,       # Daily order limit
    'weight_per_minute': 1200       # Request weight limit
}

# Rate limiter settings
RATE_LIMITER = {
    'enable': True,
    'buffer_pct': 0.8,              # Use 80% of limit (safety margin)
    'cooldown_on_limit': 60         # Cooldown period if limit hit (seconds)
}

# ============================================================================
# Paper Trading (Simulation)
# ============================================================================

# Paper trading settings (simulate without real orders)
PAPER_TRADING = {
    'enable': False,                # Enable paper trading mode
    'initial_balance': 10000,       # Starting balance
    'track_slippage': True,         # Simulate slippage
    'track_commission': True        # Simulate commission
}
