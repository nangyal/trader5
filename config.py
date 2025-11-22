"""
Konfigurációs beállítások a crypto trading keretrendszerhez
"""
import os
from pathlib import Path

# ============================================================================
# ÁLTALÁNOS BEÁLLÍTÁSOK
# ============================================================================

# Könyvtár struktúra
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
STAT_DIR = ROOT / 'stat'
OLD_DIR = ROOT / 'old'
MODEL_DIR = ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'enhanced_forex_pattern_model.pkl'

# ============================================================================
# ADATFORRÁS VÁLASZTÁSA
# ============================================================================

# Válassz adatforrást: 'backtest' vagy 'websocket'
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'backtest')

# ============================================================================
# KERESKEDÉSI PÁROK
# ============================================================================

# Csak a legjobb teljesítményű párok (optimalizálva)
COINS = [
    'BTCUSDT',
    'ETHUSDT'
]

# ============================================================================
# IDŐKERETEK (TIMEFRAMES)
# ============================================================================

# Backtest és WebSocket esetén is ezeket az időkereteket használja
TIMEFRAMES = ['15s', '30s', '1min']

# ============================================================================
# MULTIPROCESSING BEÁLLÍTÁSOK
# ============================================================================

NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 1))  # Single worker debug

# ============================================================================
# BACKTEST SPECIFIKUS BEÁLLÍTÁSOK
# ============================================================================

# Kezdő tőke backtest esetén (USDT)
BACKTEST_INITIAL_CAPITAL = float(os.environ.get('BACKTEST_INITIAL_CAPITAL', 200.0))

# CSV file elérési út sablon
# Példa: data/BTCUSDT/1min/monthly/BTCUSDT-2025-01.csv
BACKTEST_DATA_PATH_TEMPLATE = str(DATA_DIR / '{coin}' / '1min' / 'monthly')

# ============================================================================
# WEBSOCKET BEÁLLÍTÁSOK (Binance)
# ============================================================================

# Binance WebSocket URL
BINANCE_WS = 'wss://stream.binance.com:9443/ws'

# Binance API credentials (DEMO MODE)
BINANCE_API_KEY = '9fapGQP3fbdv4Df9bNDPAjEsxtgDQLw59X9gdZRsZPHiNZwooOTaSRQdIZwrTon2'
BINANCE_API_SECRET = 'Yl7c2AqSkL50v10Sv0qe9ZEClNo65KbBc57cmqzuxnYtfi6HiOZicSQBuXN5UrCk'

# DEMO mode engedélyezése (True = testnet, False = mainnet)
BINANCE_DEMO_MODE = True

# ============================================================================
# RISK MANAGEMENT & TRADING LOGIKA BEÁLLÍTÁSOK
# ============================================================================

# Kockázat per trade (% of capital)
RISK_PER_TRADE = 0.02  # 2%

# Használj tiered risk management-et?
USE_TIERED_RISK = True

# Tiered risk szintek - ULTRA MEGA AGGRESSIVE (200%+ target!)
RISK_TIERS = [
    {'max_capital_ratio': 2.0, 'risk': 0.15},      # <2x: 15% (EXTREME!)
    {'max_capital_ratio': 3.0, 'risk': 0.12},      # 2-3x: 12%
    {'max_capital_ratio': 5.0, 'risk': 0.10},      # 3-5x: 10%
    {'max_capital_ratio': float('inf'), 'risk': 0.08}  # >5x: 8%
]

# Maximum egyidejű tradeek - ULTRA AGGRESSIVE!
MAX_CONCURRENT_TRADES = 100  # 100 parallel trade!

# ============================================================================
# PATTERN TARGETS (STOP LOSS & TAKE PROFIT)
# ============================================================================

# OPTIMIZED pattern targets - 2% TP PROVEN TO WORK!
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.005,  # -0.5% (proven)
        'tp_pct': 0.020   # +2.0% (proven, 1:4 R/R)
    },
    'descending_triangle': {
        'sl_pct': 0.005,
        'tp_pct': 0.020
    },
    'symmetrical_triangle': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'double_top': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'double_bottom': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'head_and_shoulders': {
        'sl_pct': 0.007,
        'tp_pct': 0.028
    },
    'cup_and_handle': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'wedge_rising': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'wedge_falling': {
        'sl_pct': 0.006,
        'tp_pct': 0.024
    },
    'flag_bullish': {
        'sl_pct': 0.005,
        'tp_pct': 0.020
    },
    'flag_bearish': {
        'sl_pct': 0.005,
        'tp_pct': 0.020
    }
}

# ============================================================================
# TREND ALIGNMENT BEÁLLÍTÁSOK (LONG-ONLY STRATEGY)
# ============================================================================

TREND_ALIGNMENT = {
    'enable': False,  # Kikapcsolva több trade-hez!
    'lookback_period': 20,
    'use_ema_filter': False,  # EMA filter kikapcsolva
    'ema_period': 50,
    # OPTIMIZED: Több pattern = több trade opportunity
    'bullish_patterns': ['ascending_triangle', 'symmetrical_triangle', 'cup_and_handle', 'flag_bullish', 'double_bottom'],
    'bearish_patterns': ['descending_triangle', 'wedge_falling', 'double_top']  # Ezeket is használjuk LONG-ba downtrend-ben!
}

# ============================================================================
# VOLATILITY FILTER
# ============================================================================

VOLATILITY_FILTER = {
    'enable': False,  # KIKAPCSOLVA - még több trade!
    'atr_period': 14,
    'min_atr_pct': 0.1,  # 0.1% (ultra low)
}

# ============================================================================
# PATTERN FILTERS
# ============================================================================

PATTERN_FILTERS = {
    'min_probability': 0.5,   # Min 50% ML konfidencia (csökkentve 0.7-ről)
    'min_strength': 0.5,      # Min 50% pattern erősség (csökkentve 0.7-ről)
    'blacklist_patterns': []  # Kizárt patternek
}

# ============================================================================
# LOGGING & CSV
# ============================================================================

# CSV trade log file
TRADES_LOG_FILE = ROOT / 'trades_log.csv'

# Console logging level
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# ============================================================================
# EXCEL STATISZTIKÁK
# ============================================================================

# Excel report könyvtár
EXCEL_OUTPUT_DIR = STAT_DIR

# Excel file név sablon
EXCEL_FILENAME_TEMPLATE = 'backtest_report_{timestamp}.xlsx'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Biztosítja, hogy az összes szükséges könyvtár létezik"""
    STAT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"✅ Könyvtárak létrehozva/ellenőrizve")


if __name__ == '__main__':
    ensure_dirs()
    print("\n=== KONFIGURÁCIÓ ===")
    print(f"Adatforrás: {DATA_SOURCE}")
    print(f"Coinok: {', '.join(COINS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Backtest tőke: ${BACKTEST_INITIAL_CAPITAL}")
    print(f"Demo mode: {BINANCE_DEMO_MODE}")
