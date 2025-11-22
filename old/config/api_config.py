"""
API Configuration
Binance API settings and credentials
"""

# ============================================================================
# IMPORTANT: API Security
# ============================================================================
# 
# ⚠️  NEVER commit real API keys to version control!
# 
# For production use:
# 1. Store keys in environment variables
# 2. Use a .env file (add to .gitignore)
# 3. Use a secrets management service
# 
# This file contains DEMO/TESTNET credentials only.
# ============================================================================

import os

# ============================================================================
# Binance API Endpoints
# ============================================================================

# API URLs
API_URLS = {
    'mainnet': 'https://api.binance.com',
    'testnet': 'https://testnet.binance.vision',
    'futures_mainnet': 'https://fapi.binance.com',
    'futures_testnet': 'https://testnet.binancefuture.com'
}

# WebSocket URLs
WEBSOCKET_URLS = {
    'mainnet': 'wss://stream.binance.com:443',
    'testnet': 'wss://testnet.binance.vision',
    'futures_mainnet': 'wss://fstream.binance.com',
    'futures_testnet': 'wss://stream.binancefuture.com'
}

# ============================================================================
# API Credentials
# ============================================================================

# Current environment (CHANGE THIS FOR PRODUCTION!)
ENVIRONMENT = 'testnet'  # 'testnet' or 'mainnet'

# Demo/Testnet API credentials (safe to commit)
TESTNET_API_KEY = '9fapGQP3fbdv4Df9bNDPAjEsxtgDQLw59X9gdZRsZPHiNZwooOTaSRQdIZwrTon2'
TESTNET_API_SECRET = 'Yl7c2AqSkL50v10Sv0qe9ZEClNo65KbBc57cmqzuxnYtfi6HiOZicSQBuXN5UrCk'

# Mainnet API credentials (LOAD FROM ENVIRONMENT!)
MAINNET_API_KEY = os.getenv('BINANCE_API_KEY', '')
MAINNET_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# ============================================================================
# API Configuration Helpers
# ============================================================================

def get_api_credentials(environment=None):
    """
    Get API credentials based on environment
    
    Args:
        environment (str): 'testnet' or 'mainnet' (default: from config)
    
    Returns:
        tuple: (api_key, api_secret)
    """
    env = environment or ENVIRONMENT
    
    if env == 'testnet':
        return TESTNET_API_KEY, TESTNET_API_SECRET
    elif env == 'mainnet':
        if not MAINNET_API_KEY or not MAINNET_API_SECRET:
            raise ValueError(
                "Mainnet API credentials not set! "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )
        return MAINNET_API_KEY, MAINNET_API_SECRET
    else:
        raise ValueError(f"Unknown environment: {env}")


def get_api_url(environment=None):
    """
    Get API URL based on environment
    
    Args:
        environment (str): 'testnet' or 'mainnet' (default: from config)
    
    Returns:
        str: API base URL
    """
    env = environment or ENVIRONMENT
    return API_URLS.get(env, API_URLS['testnet'])


def get_websocket_url(environment=None):
    """
    Get WebSocket URL based on environment
    
    Args:
        environment (str): 'testnet' or 'mainnet' (default: from config)
    
    Returns:
        str: WebSocket base URL
    """
    env = environment or ENVIRONMENT
    return WEBSOCKET_URLS.get(env, WEBSOCKET_URLS['testnet'])

# ============================================================================
# API Client Configuration
# ============================================================================

# Connection settings
CONNECTION_CONFIG = {
    'timeout': 30,              # Request timeout (seconds)
    'recv_window': 5000,        # Binance recv_window parameter (ms)
    'max_retries': 3,           # Max retry attempts
    'retry_delay': 2,           # Delay between retries (seconds)
    'retry_backoff': 1.5        # Backoff multiplier for retries
}

# Session settings
SESSION_CONFIG = {
    'pool_connections': 10,     # Connection pool size
    'pool_maxsize': 10,         # Max pool size
    'max_retries_adapter': 3    # Adapter-level retries
}

# ============================================================================
# API Features
# ============================================================================

# Enable/disable API features
API_FEATURES = {
    'spot_trading': True,       # Spot trading enabled
    'futures_trading': False,   # Futures trading disabled
    'margin_trading': False,    # Margin trading disabled
    'websocket_streams': True,  # WebSocket data streams
    'user_data_stream': False   # User data stream (requires auth)
}

# ============================================================================
# Demo Mode Configuration
# ============================================================================

# Use demo/testnet by default
USE_DEMO_MODE = True

# Demo mode safety checks
DEMO_SAFETY = {
    'max_order_value_usd': 10000,   # Max order value in demo
    'confirm_large_orders': False,  # No confirmation needed in demo
    'print_warnings': True          # Print demo mode warnings
}

# ============================================================================
# Testing & Development
# ============================================================================

# Test connection on startup
TEST_CONNECTION_ON_START = True

# Verify API permissions
VERIFY_PERMISSIONS = {
    'check_on_start': True,
    'required_permissions': [
        'SPOT_ACCOUNT_READ',
        'SPOT_ACCOUNT_TRADE'
    ]
}

# ============================================================================
# Logging
# ============================================================================

# API request logging
API_LOGGING = {
    'log_requests': False,      # Log all API requests
    'log_responses': False,     # Log all API responses
    'log_errors': True,         # Log errors only
    'log_file': 'api_requests.log'
}

# ============================================================================
# IP Whitelist (for production)
# ============================================================================

# Enable IP whitelist check
USE_IP_WHITELIST = False

# Allowed IP addresses (empty = disabled)
IP_WHITELIST = [
    # Add your server IPs here
    # '123.456.789.0',
]

# ============================================================================
# API Key Rotation (for production)
# ============================================================================

# Automatic API key rotation
KEY_ROTATION = {
    'enable': False,
    'rotation_interval_days': 90,   # Rotate keys every 90 days
    'warn_before_days': 7           # Warn 7 days before expiry
}
