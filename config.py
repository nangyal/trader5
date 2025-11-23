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

# Válassz adatforrást: 'backtest', 'backtest_hedging', 'websocket' vagy 'websocket_hedging'
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
TIMEFRAMES = ['15s', '30s', '1min', '5min',  '15min',  '30min']

# ============================================================================
# MULTIPROCESSING BEÁLLÍTÁSOK
# ============================================================================

NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 28))  # Single worker debug

# ============================================================================
# BACKTEST SPECIFIKUS BEÁLLÍTÁSOK
# ============================================================================

# Kezdő tőke backtest esetén (USDT)
BACKTEST_INITIAL_CAPITAL = float(os.environ.get('BACKTEST_INITIAL_CAPITAL', 200.0))

# CSV file elérési út sablon
# Példa: /home/nangyal/Desktop/v4/data/BTCUSDT/15s/monthly/BTCUSDT-trades-2025-10_15s.csv
BACKTEST_DATA_PATH_TEMPLATE = '/home/nangyal/Desktop/v4/data/{coin}/{timeframe}/monthly/{coin}-trades-2025-10_{timeframe}.csv'

# ============================================================================
# WEBSOCKET BEÁLLÍTÁSOK (Binance)
# ============================================================================

# Binance WebSocket URL
BINANCE_WS = 'wss://stream.binance.com:9443/ws'

# Binance API credentials (DEMO MODE)
BINANCE_API_KEY = '9fapGQP3fbdv4Df9bNDPAjEsxtgDQLw59X9gdZRsZPHiNZwooOTaSRQdIZwrTon2'
BINANCE_API_SECRET = 'Yl7c2AqSkL50v10Sv0qe9ZEClNo65KbBc57cmqzuxnYtfi6HiOZicSQBuXN5UrCk'

# DEMO mode engedélyezése (True = demo, False = mainnet)
BINANCE_DEMO_MODE = True

# Trading hours - csak nappal kereskedik (UTC időzóna)
# FIGYELEM: DEMO módban automatikusan kikapcsolva (24/7 trading)
TRADING_HOURS = {
    'enable': False,  # Kikapcsolva ha DEMO mode aktív
    'start_hour': 6,   # 06:00 UTC (8:00 magyar idő)
    'end_hour': 20,    # 20:00 UTC (22:00 magyar idő)
    'timezone': 'UTC'
}

# ============================================================================
# RISK MANAGEMENT & TRADING LOGIKA BEÁLLÍTÁSOK
# ============================================================================

# Kockázat per trade (% of capital)
RISK_PER_TRADE = 0.02  # 2%

# Használj tiered risk management-et?
USE_TIERED_RISK = True

# Tiered risk szintek - REALISTIC SETTINGS
RISK_TIERS = [
    {'max_capital_ratio': 2.0, 'risk': 0.05},      # <2x: 5%
    {'max_capital_ratio': 3.0, 'risk': 0.04},      # 2-3x: 4%
    {'max_capital_ratio': 5.0, 'risk': 0.03},      # 3-5x: 3%
    {'max_capital_ratio': float('inf'), 'risk': 0.02}  # >5x: 2%
]

# Maximum egyidejű tradeek - REALISTIC (adjusted to capital limits)
MAX_CONCURRENT_TRADES = 3  # 3 parallel trades max (3 × 30% = 90% max capital usage)

# Maximum position size per trade (% of current capital)
MAX_POSITION_SIZE_PCT = 0.30  # 30% - allows 90% total usage, 10% buffer for safety

# ============================================================================
# TRADING COSTS (COMMISSIONS & FEES)
# ============================================================================

# Trading commission settings
TRADING_COMMISSION = {
    'enable': True,
    'percent': 0.001,  # 0.1% per trade (Binance standard spot/futures fee)
    'calculate_both_sides': True,  # True = charge on entry AND exit (0.2% total)
}

# Slippage simulation (optional, more relevant for high volatility)
SLIPPAGE = {
    'enable': False,  # Disabled by default
    'percent': 0.0005,  # 0.05% slippage
}

# ============================================================================
# PATTERN TARGETS (STOP LOSS & TAKE PROFIT)
# ============================================================================

# OPTIMIZED pattern targets - ADJUSTED for commission costs
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.008,  # -0.8% (wider to survive commission + noise)
        'tp_pct': 0.012   # +1.2% (more achievable, 1:1.5 R/R)
    },
    'descending_triangle': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'symmetrical_triangle': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'double_top': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'double_bottom': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'head_and_shoulders': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'cup_and_handle': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'wedge_rising': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'wedge_falling': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'flag_bullish': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    },
    'flag_bearish': {
        'sl_pct': 0.008,
        'tp_pct': 0.012
    }
}

# ============================================================================
# TREND ALIGNMENT BEÁLLÍTÁSOK (LONG-ONLY STRATEGY)
# ============================================================================

TREND_ALIGNMENT = {
    'enable': False,  # DISABLED - túl szigorú! (2328 → 126 trades-re csökkentette!)
    'lookback_period': 20,
    'use_ema_filter': False,  # DISABLED - EMA filter túl szigorú!
    'ema_period': 50,
    # OPTIMIZED: Több pattern = több trade opportunity
    'bullish_patterns': ['ascending_triangle', 'symmetrical_triangle', 'cup_and_handle', 'flag_bullish', 'double_bottom'],
    'bearish_patterns': ['descending_triangle', 'wedge_falling', 'double_top']  # Ezeket is használjuk LONG-ba downtrend-ben!
}

# ============================================================================
# VOLATILITY FILTER
# ============================================================================

VOLATILITY_FILTER = {
    'enable': False,  # DISABLED - túl szigorú! (2328 → 126 trades-re csökkentette!)
    'atr_period': 14,
    'min_atr_pct': 0.005,  # 0.5% (decimal format, NOT percentage)
}

# ============================================================================
# PATTERN FILTERS
# ============================================================================

PATTERN_FILTERS = {
    'min_probability': 0.60,   # Min 60% ML konfidencia (higher quality signals)
    'min_strength': 0.55,      # Min 55% pattern erősség
    'blacklist_patterns': []  # Kizárt patternek
}

# ============================================================================
# HEDGING BEÁLLÍTÁSOK (backtest_hedging módhoz)
# ============================================================================

HEDGING = {
    'enable': True,
    'hedge_threshold': 0.18,  # 18% drawdown után aktivál (emelve 15%-ról)
    'hedge_recovery_threshold': 0.08,  # 8% drawdown alatt zárja (emelve 5%-ról, gyorsabb close)
    'hedge_ratio': 0.35,  # Hedge mérete: 35% of total exposure (csökkentve 50%-ról)
    'dynamic_hedge': True,  # Dinamikus threshold volatilitás alapján
    'volatility_window': 20,  # Volatilitás számítási ablak
    'min_hedge_threshold': 0.12,  # Min threshold (alacsony volatilitás, emelve 10%-ról)
    'max_hedge_threshold': 0.25,  # Max threshold (magas volatilitás)
    'drawdown_basis': 'equity',  # 'capital' vagy 'equity' (unrealized PnL-lel)
    
    # Coin-specific overrides (optional)
    'coin_overrides': {
        'ETHUSDT': {
            'hedge_threshold': 0.22,  # ETHUSDT-nél magasabb threshold (22%)
            'hedge_recovery_threshold': 0.10,  # Gyorsabb recovery close
            'hedge_ratio': 0.30,  # Kisebb hedge ratio
        },
        'BTCUSDT': {
            'hedge_threshold': 0.16,  # BTCUSDT jól teljesített, alacsonyabb threshold
            'hedge_ratio': 0.40,  # Nagyobb hedge ratio
        }
    }
}

# ============================================================================
# ADVANCED PROFIT/LOSS STRATEGIES
# ============================================================================

# Trailing Stop Loss - profitot követő stop loss
TRAILING_STOP = {
    'enable': False,  # DISABLED - causes premature exits before TP
    'activation_pct': 0.010,  # +1.0% profit után aktiválódik
    'trail_pct': 0.005,  # 0.5% trailing distance
}

# Partial Take Profit - részleges profitzárás
PARTIAL_TP = {
    'enable': True,
    'levels': [
        {'pct': 0.008, 'close_ratio': 0.50},  # +0.8% → close 50% (cumulative)
        {'pct': 0.012, 'close_ratio': 0.75},  # +1.2% → close 75% (cumulative)
        {'pct': 0.018, 'close_ratio': 1.00},  # +1.8% → close 100% (cumulative)
    ]
}

# Breakeven Stop - profit esetén SL → entry price
BREAKEVEN_STOP = {
    'enable': True,
    'activation_pct': 0.008,  # +0.8% profit után SL → breakeven
    'buffer_pct': 0.001,  # +0.1% buffer (entry + buffer)
}

# ML Confidence Based Position Sizing
ML_CONFIDENCE_WEIGHTING = {
    'enable': True,
    'tiers': [
        {'min_prob': 0.80, 'multiplier': 1.5},  # 80%+ → 1.5x position
        {'min_prob': 0.70, 'multiplier': 1.2},  # 70-80% → 1.2x position
        {'min_prob': 0.65, 'multiplier': 1.0},  # 65-70% → 1.0x position
    ]
}

# Losing Streak Protection
LOSING_STREAK_PROTECTION = {
    'enable': True,
    'reduce_risk_after': 3,  # 3 vesztő trade után risk csökkentés
    'risk_multiplier': 0.5,  # Risk → 50%
    'stop_trading_after': 5,  # 5 vesztő trade után STOP
    'cooldown_candles': 60,  # 60 candle pause (1 óra @ 1min)
}

# Pattern Performance Filter - rossz pattern-ek kiszűrése
PATTERN_PERFORMANCE_FILTER = {
    'enable': True,
    'min_trades': 10,  # Min trade szám aPattern Stats alapján
    'min_win_rate': 0.40,  # Min 40% win rate
    'min_profit_factor': 1.0,  # Min 1.0 profit factor
    'auto_blacklist': True,  # Automatikus blacklist rossz pattern-eknek
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
