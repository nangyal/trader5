"""
Configuration file for live trading and backtesting
Contains all trading parameters, risk management settings, and API credentials
"""

import os


# =============================================================================
# API Configuration
# =============================================================================
class ApiConfig:
    """Binance API configuration"""
    
    # Environment: 'demo' or 'production'
    ENVIRONMENT = 'demo'
    
    # API credentials (use environment variables for security)
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Connection settings
    CONNECTION_CONFIG = {
        'timeout': 30,  # Request timeout in seconds
        'max_retries': 3,
        'backoff_factor': 1
    }
    
    @staticmethod
    def get_api_credentials():
        """Get API credentials from environment or config"""
        api_key = os.getenv('BINANCE_API_KEY', ApiConfig.API_KEY)
        api_secret = os.getenv('BINANCE_API_SECRET', ApiConfig.API_SECRET)
        
        if not api_key or not api_secret:
            print("⚠️  Warning: API credentials not set!")
            print("   Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        
        return api_key, api_secret


# =============================================================================
# Trading Configuration
# =============================================================================
class TradingConfig:
    """Trading strategy and risk management settings"""
    
    # Default trading pair
    DEFAULT_SYMBOL = 'ETHUSDT'
    
    # Base risk per trade (can be overridden by tiered risk)
    RISK_PER_TRADE = 0.02  # 2%
    
    # Tiered risk management (safer compounding)
    USE_TIERED_RISK = True
    RISK_TIERS = [
        {'max_capital_ratio': 2.0, 'risk': 0.020},   # $10k-20k: 2.0% risk
        {'max_capital_ratio': 3.0, 'risk': 0.015},   # $20k-30k: 1.5% risk
        {'max_capital_ratio': 5.0, 'risk': 0.010},   # $30k-50k: 1.0% risk
        {'max_capital_ratio': float('inf'), 'risk': 0.0075}  # $50k+: 0.75% risk
    ]
    
    # Maximum concurrent trades
    MAX_CONCURRENT_TRADES = 20
    
    # Maximum trades per symbol (None = unlimited)
    MAX_TRADES_PER_SYMBOL = 5  # Each coin can have max 5 positions
    
    # Pattern targets (stop loss and take profit percentages)
    PATTERN_TARGETS = {
        # Bullish patterns
        'ascending_triangle': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'ascending_triangle_breakout': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'symmetrical_triangle': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'symmetrical_triangle_breakout': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'cup_and_handle': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'cup_and_handle_breakout': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'double_bottom': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'double_bottom_breakout': {'sl_pct': 0.015, 'tp_pct': 0.03},
        
        # Bearish patterns (blacklisted in LONG-ONLY strategy)
        'descending_triangle': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'descending_triangle_breakdown': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'head_and_shoulders': {'sl_pct': 0.015, 'tp_pct': 0.03},
        'head_and_shoulders_breakdown': {'sl_pct': 0.015, 'tp_pct': 0.03},
        
        # Default
        'default': {'sl_pct': 0.015, 'tp_pct': 0.03}
    }
    
    # Trend alignment settings
    TREND_ALIGNMENT = {
        'use_ema_filter': True,  # Filter trades using EMA
        'ema_period': 50,  # EMA period for trend filter
        'lookback_period': 50,  # Candles to analyze for trend
        
        # Pattern classification
        'bullish_patterns': [
            'ascending_triangle',
            'ascending_triangle_breakout',
            'symmetrical_triangle',
            'symmetrical_triangle_breakout',
            'cup_and_handle',
            'cup_and_handle_breakout',
            'double_bottom',
            'double_bottom_breakout'
        ],
        'bearish_patterns': [
            'descending_triangle',
            'descending_triangle_breakdown',
            'head_and_shoulders',
            'head_and_shoulders_breakdown',
            'wedge',
            'wedge_breakdown'
        ]
    }
    
    # Volatility filter settings
    VOLATILITY_FILTER = {
        'enable': True,
        'atr_period': 14,  # ATR period
        'min_atr_pct': 0.5,  # Minimum ATR% (0.5% = skip low volatility)
        'max_atr_pct': 10.0  # Maximum ATR% (skip excessive volatility)
    }
    
    # Pattern filters
    PATTERN_FILTERS = {
        'min_probability': 0.7,  # 70% minimum pattern probability
        'min_strength': 0.7,     # 70% minimum pattern strength
        
        # Blacklist patterns (never trade these)
        'blacklist_patterns': [
            'descending_triangle',
            'descending_triangle_breakdown',
            'head_and_shoulders',
            'head_and_shoulders_breakdown'
        ]
    }


# =============================================================================
# Model Configuration
# =============================================================================
class ModelConfig:
    """ML model settings"""
    
    # Model file path
    MODEL_SAVE_PATH = 'enhanced_forex_pattern_model.pkl'
    
    # Model parameters
    MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }


# =============================================================================
# Pattern Configuration
# =============================================================================
class PatternConfig:
    """Pattern detection settings"""
    
    # Pattern detection windows
    DETECTION_WINDOWS = {
        'min_window': 20,
        'max_window': 100,
        'default_window': 50
    }
    
    # Pattern strength thresholds
    STRENGTH_THRESHOLDS = {
        'weak': 0.3,
        'moderate': 0.5,
        'strong': 0.7,
        'very_strong': 0.85
    }


# Export configuration instances
api_config = ApiConfig()
trading_config = TradingConfig()
model_config = ModelConfig()
pattern_config = PatternConfig()


if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"\nAPI Environment: {api_config.ENVIRONMENT}")
    print(f"Default Symbol: {trading_config.DEFAULT_SYMBOL}")
    print(f"Base Risk: {trading_config.RISK_PER_TRADE * 100}%")
    print(f"Tiered Risk: {trading_config.USE_TIERED_RISK}")
    print(f"EMA Filter: {trading_config.TREND_ALIGNMENT['use_ema_filter']}")
    print(f"ATR Filter: {trading_config.VOLATILITY_FILTER['enable']}")
    print(f"Model Path: {model_config.MODEL_SAVE_PATH}")
