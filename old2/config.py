import os
from pathlib import Path

# General
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
STAT_DIR = ROOT / 'stat'
OLD_DIR = ROOT / 'old'
MODEL_DIR = ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'enhanced_forex_pattern_model.pkl'

# Choose source: 'backtest' or 'websocket'
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'websocket')  # Switch to realtime mode

# Coins / pairs to process - optimized for best performers only
COINS = [
    'BTCUSDT', 'ETHUSDT'  # Focus on high-volume pairs
]

# Which timeframe buckets should be produced from tick-level data
# Only use profitable timeframes based on backtest analysis
TIMEFRAMES = ['15s', '30s', '1min']

# Choose trading logic name: trading_logic1 or trading_logic2
TRADING_LOGIC = os.environ.get('TRADING_LOGIC', 'trading_logic1')

# Multiprocessing
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 4))

# Backtest specifics
BACKTEST_INITIAL_CAPITAL = float(os.environ.get('BACKTEST_INITIAL_CAPITAL', 200.0))

# Use advanced (copied) classifier with GPU support
USE_ADVANCED_CLASSIFIER = os.environ.get('USE_ADVANCED_CLASSIFIER', '1') == '1'

# Websocket settings (example uses Binance aggregated trade stream)
BINANCE_WS = 'wss://stream.binance.com:9443/ws'

# Binance API credentials (Demo mode)
BINANCE_API_KEY = '9fapGQP3fbdv4Df9bNDPAjEsxtgDQLw59X9gdZRsZPHiNZwooOTaSRQdIZwrTon2'
BINANCE_API_SECRET = 'Yl7c2AqSkL50v10Sv0qe9ZEClNo65KbBc57cmqzuxnYtfi6HiOZicSQBuXN5UrCk'
BINANCE_DEMO_MODE = True  # Set to False for live trading

def ensure_dirs():
    STAT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
    ensure_dirs()
