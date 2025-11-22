"""
Configuration Package
Centralized configuration management for the trading system

Usage:
    from config import model_config, trading_config, pattern_config
    
    # Access settings
    risk = trading_config.RISK_PER_TRADE
    params = model_config.XGBOOST_PARAMS
"""

# Import all config modules
from . import model_config
from . import pattern_config
from . import trading_config
from . import api_config
from . import data_config

__all__ = [
    'model_config',
    'pattern_config', 
    'trading_config',
    'api_config',
    'data_config'
]

# Version info
__version__ = '3.0.0'
__author__ = 'AI Assistant'
__date__ = '2025-11-12'
