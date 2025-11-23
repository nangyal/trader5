# Developer Documentation - Crypto Trading Framework v5

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Trading Logic](#trading-logic)
6. [WebSocket Live Trading](#websocket-live-trading)
7. [Backtesting](#backtesting)
8. [Pattern Detection](#pattern-detection)
9. [Risk Management](#risk-management)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## Overview

**Crypto Trading Framework v5** is a comprehensive algorithmic trading system with:
- ✅ **Live WebSocket Trading** (Binance MAINNET + DEMO API)
- ✅ **Backtesting Engine** with historical data
- ✅ **ML Pattern Recognition** (XGBoost classifier)
- ✅ **Multi-coin/Multi-timeframe** support
- ✅ **Shared Capital Pool** management
- ✅ **Advanced Risk Management**

### Key Features
- **Real-time pattern detection** with 80%+ ML confidence
- **Conservative filters**: EMA50 trend, ATR volatility, probability thresholds
- **Shared capital pool** across all traders
- **Paper trading mode** (DEMO) for testing
- **24/7 operation** (configurable trading hours)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRADING FRAMEWORK                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   start.py   │───▶│  config.py   │───▶│ WebSocket /  │  │
│  │  (launcher)  │    │  (settings)  │    │  Backtest    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │          │
│                                                   ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         websocket_live_trading.py                    │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  LiveWebSocketTrader                           │  │  │
│  │  │  • Binance WS connection (MAINNET)             │  │  │
│  │  │  • Real-time kline streaming                   │  │  │
│  │  │  • Historical data pre-load (1000 candles)     │  │  │
│  │  │  • Shared capital pool ($7046.58)              │  │  │
│  │  │  • Pattern detection on candle close           │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         trading_logic.py                             │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  TradingLogic (shared by all modes)            │  │  │
│  │  │  • should_open_trade()                         │  │  │
│  │  │  • calculate_pattern_targets() (SL/TP)         │  │  │
│  │  │  • calculate_position_size()                   │  │  │
│  │  │  • open_trade() / close_trade()                │  │  │
│  │  │  • check_trade_exit() (SL/TP/Trailing)         │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   forex_pattern_classifier.py (ML Model)             │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  EnhancedForexPatternClassifier                │  │  │
│  │  │  • XGBoost model (~30-35 indicators)           │  │  │
│  │  │  • Pattern recognition (11 patterns)           │  │  │
│  │  │  • Probability scoring (0-1)                   │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. **start.py** - Main Entry Point
```python
# Launch modes
DATA_SOURCE=backtest python start.py      # Historical backtesting
DATA_SOURCE=backtest_hedging python start.py  # Backtest with hedging
DATA_SOURCE=websocket python start.py     # Live WebSocket trading
```

**Flow:**
1. Load `config.py` settings
2. Create directories (`data/`, `stat/`, `models/`)
3. Route to appropriate trading mode
4. Initialize traders and start execution

### 2. **config.py** - Configuration Hub
```python
# Key settings
COINS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15s', '30s', '1min', '5min', '15min', '30min']
MAX_CONCURRENT_TRADES = 3
MAX_POSITION_SIZE_PCT = 0.33  # 33% per trade
BINANCE_DEMO_MODE = True  # Paper trading
TRADING_HOURS = {'enable': False, 'start_hour': 6, 'end_hour': 20}
```

**Categories:**
- General settings (paths, data source)
- Trading pairs & timeframes
- Risk management (capital, position sizing)
- Pattern targets (SL/TP percentages)
- Advanced strategies (trailing stop, partial TP)
- WebSocket/Binance API credentials

### 3. **websocket_live_trading.py** - Live Trading Engine

#### Class: `LiveWebSocketTrader`

**Initialization:**
```python
trader = LiveWebSocketTrader(
    coins=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1m', '5m', '15m', '30m'],
    api_key='YOUR_API_KEY',
    api_secret='YOUR_SECRET',
    demo_mode=True
)
await trader.initialize()
await trader.run()
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `initialize()` | Load capital, historical data (1000 candles) |
| `load_historical_klines()` | Fetch from Binance API, resample 15s/30s |
| `run()` | Main WebSocket loop, handle messages |
| `handle_ws_message()` | Parse kline streams, route to `process_kline()` |
| `process_kline()` | Update OHLCV, detect candle close |
| `process_closed_candle()` | ML prediction, pattern detection, trade execution |
| `get_account_balance()` | Fetch USDT/BTC/ETH/BNB balances |
| `sync_trader_capital()` | Update all traders with shared capital |
| `print_status()` | 30-second status updates |

**WebSocket Connection:**
- **URL**: `wss://stream.binance.com:9443/stream`
- **Streams**: Multi-coin/timeframe (e.g., `btcusdt@kline_1m`, `ethusdt@kline_5m`)
- **Data**: Real-time MAINNET prices
- **API**: DEMO mode for orders (paper trading)

**Data Flow:**
```
WebSocket → parse kline → update DataFrame → candle close? 
    → ML prediction → pattern detected? → filters pass? → open trade
```

### 4. **trading_logic.py** - Core Trading Logic

#### Class: `TradingLogic`

**Shared by:**
- ✅ WebSocket live trading
- ✅ Backtesting
- ✅ Backtesting with hedging

**Trade Lifecycle:**

```python
# 1. Check if trade should open
should_open = trader.should_open_trade(pattern, probability, strength)

# 2. Calculate targets
direction, entry, sl, tp = trader.calculate_pattern_targets(
    pattern, current_candle, df_ohlcv
)

# 3. Calculate position size
position_size = trader.calculate_position_size(entry, sl, capital)

# 4. Open trade
trade = trader.open_trade(
    coin='BTCUSDT',
    pattern='ascending_triangle',
    entry_price=98500.00,
    stop_loss=98000.00,
    take_profit=100000.00,
    position_size=0.0715,  # ~$7046 worth
    probability=0.807,
    strength=0.807,
    timeframe='1min',
    entry_time=datetime.now()
)

# 5. Monitor exit
should_close, exit_price, reason, ratio = trader.check_trade_exit(
    trade, current_candle
)

# 6. Close trade
pnl = trader.close_trade(trade, exit_price, reason, datetime.now(), ratio)
```

**Filters:**

| Filter | Purpose | Threshold |
|--------|---------|-----------|
| **Pattern Probability** | ML confidence | ≥65% |
| **Pattern Strength** | Pattern quality | ≥65% |
| **EMA50 Trend** | Price > EMA50 | Required for LONG |
| **ATR Volatility** | Min market movement | ≥0.3% |
| **Max Concurrent Trades** | Risk limit | ≤3 trades |
| **Position Size** | Capital protection | ≤33% per trade |

### 5. **forex_pattern_classifier.py** - ML Model

#### Class: `EnhancedForexPatternClassifier`

**Model:** XGBoost multi-class classifier

**Patterns Detected (11):**
1. `ascending_triangle` - Bullish breakout
2. `descending_triangle` - Bearish breakdown
3. `symmetrical_triangle` - Consolidation
4. `double_top` - Reversal pattern
5. `double_bottom` - Reversal pattern
6. `head_and_shoulders` - Major reversal
7. `cup_and_handle` - Bullish continuation
8. `wedge_rising` - Trend continuation
9. `wedge_falling` - Trend continuation
10. `flag_bullish` - Continuation
11. `flag_bearish` - Continuation

**Indicators (30-35):**
- **Moving Averages**: SMA 10/20/50, EMA 12/26/50
- **Oscillators**: RSI, Stochastic, CCI
- **Trend**: MACD, ADX, Parabolic SAR
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, VWAP, Volume MA

**Usage:**
```python
classifier = EnhancedForexPatternClassifier()
classifier.load_model('models/enhanced_forex_pattern_model.pkl')

predictions, probabilities = classifier.predict(df_ohlcv.iloc[-60:])
pattern = predictions[-1]  # Last candle
probability = np.max(probabilities[-1])  # Max class probability
```

---

## Configuration

### Environment Variables
```bash
# Choose data source
export DATA_SOURCE=websocket  # or 'backtest', 'backtest_hedging'

# Backtest capital (optional)
export BACKTEST_INITIAL_CAPITAL=200.0

# CPU workers for backtesting (optional)
export NUM_WORKERS=28
```

### Key Config Parameters

#### Trading Pairs & Timeframes
```python
COINS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15s', '30s', '1min', '5min', '15min', '30min']
```

#### Risk Management
```python
RISK_PER_TRADE = 0.02  # 2% base risk
MAX_CONCURRENT_TRADES = 3  # Max parallel trades
MAX_POSITION_SIZE_PCT = 0.33  # 33% max per trade
```

#### Pattern Targets (SL/TP)
```python
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.005,  # -0.5% stop loss
        'tp_pct': 0.020   # +2.0% take profit (1:4 R/R)
    },
    # ... other patterns
}
```

#### Trend Alignment
```python
TREND_ALIGNMENT = {
    'enable': True,
    'use_ema_filter': True,
    'ema_period': 50,
    'bullish_patterns': ['ascending_triangle', 'cup_and_handle', ...],
    'bearish_patterns': ['descending_triangle', 'double_top', ...]
}
```

#### Advanced Strategies
```python
# Trailing Stop Loss
TRAILING_STOP = {
    'enable': True,
    'activation_pct': 0.010,  # Activate at +1.0% profit
    'trail_pct': 0.005  # Trail 0.5% below peak
}

# Partial Take Profit
PARTIAL_TP = {
    'enable': True,
    'levels': [
        {'pct': 0.015, 'close_ratio': 0.50},  # +1.5% → close 50%
        {'pct': 0.025, 'close_ratio': 0.30},  # +2.5% → close 30%
    ]
}

# Breakeven Stop
BREAKEVEN_STOP = {
    'enable': True,
    'activation_pct': 0.008,  # Move SL to breakeven at +0.8%
    'buffer_pct': 0.001  # +0.1% buffer
}
```

#### Trading Hours
```python
TRADING_HOURS = {
    'enable': False,  # Disabled in DEMO mode (24/7)
    'start_hour': 6,   # 06:00 UTC (8:00 CET)
    'end_hour': 20,    # 20:00 UTC (22:00 CET)
    'timezone': 'UTC'
}
```

---

## Trading Logic

### Entry Logic Flow

```
1. Candle closes
    ↓
2. ML Model predicts pattern
    ↓
3. Pattern probability ≥65%? ──NO──▶ Skip
    ↓ YES
4. Pattern strength ≥65%? ──NO──▶ Skip
    ↓ YES
5. Max trades reached? ──YES──▶ Skip
    ↓ NO
6. EMA50 filter check
    • Price > EMA50? ──NO──▶ Skip (trend failed)
    ↓ YES
7. ATR volatility check
    • ATR ≥0.3%? ──NO──▶ Skip (low volatility)
    ↓ YES
8. Calculate position size
    • Valid size? ──NO──▶ Skip (invalid size)
    ↓ YES
9. OPEN TRADE ✅
```

### Exit Logic Flow

```
Every candle (for active trades):

1. Check Stop Loss
    • Current price ≤ SL? ──YES──▶ CLOSE (Stop Loss)
    ↓ NO
2. Check Take Profit
    • Current price ≥ TP? ──YES──▶ CLOSE (Take Profit)
    ↓ NO
3. Check Trailing Stop
    • Enabled & activated? ──YES──▶ Update trailing SL
    • Price ≤ Trailing SL? ──YES──▶ CLOSE (Trailing Stop)
    ↓ NO
4. Check Partial TP
    • Price at level? ──YES──▶ CLOSE partial (50%/30%/20%)
    ↓ NO
5. Check Breakeven
    • Profit ≥ activation? ──YES──▶ Move SL to breakeven
    ↓ NO
6. Continue monitoring ⏳
```

### Position Sizing

```python
def calculate_position_size(entry_price, stop_loss, capital):
    """
    Risk-based position sizing
    
    Formula:
    risk_amount = capital × RISK_PER_TRADE (2%)
    risk_per_unit = |entry_price - stop_loss|
    position_size = risk_amount / risk_per_unit
    
    Max position = min(calculated, capital × MAX_POSITION_SIZE_PCT)
    """
    risk_amount = capital * config.RISK_PER_TRADE
    risk_per_unit = abs(entry_price - stop_loss)
    position_size = risk_amount / risk_per_unit
    
    # Cap at 33% of capital
    max_position = (capital * config.MAX_POSITION_SIZE_PCT) / entry_price
    return min(position_size, max_position)
```

**Example:**
- Capital: $7046.58
- Entry: $98,500
- Stop Loss: $98,000 (-0.5%)
- Risk per trade: 2% = $140.93
- Risk per unit: $500
- Position size: $140.93 / $500 = **0.2819 BTC** (~$27,767)
- Max 33%: $2,325.37 / $98,500 = **0.0236 BTC** (~$2,325)
- **Final: 0.0236 BTC** (capped at 33%)

---

## WebSocket Live Trading

### Setup & Initialization

```python
# 1. Create trader instance
trader = LiveWebSocketTrader(
    coins=config.COINS,
    timeframes=['1m', '5m', '15m', '30m'],
    api_key=config.BINANCE_API_KEY,
    api_secret=config.BINANCE_API_SECRET,
    demo_mode=config.BINANCE_DEMO_MODE
)

# 2. Initialize (async)
await trader.initialize()
# - Loads account balance from API
# - Sets shared_capital = $7046.58
# - Loads 1000 historical candles per coin/timeframe
# - Generates 15s/30s from 1s data via resample

# 3. Run WebSocket
await trader.run()
# - Connects to wss://stream.binance.com:9443/stream
# - Subscribes to multi-coin/timeframe kline streams
# - Processes messages in real-time
# - Prints status every 30 seconds
```

### WebSocket Message Flow

**1. Raw WebSocket Message:**
```json
{
  "stream": "btcusdt@kline_1m",
  "data": {
    "e": "kline",
    "E": 1763854884033,
    "s": "BTCUSDT",
    "k": {
      "t": 1763854860000,  // Open time
      "T": 1763854919999,  // Close time
      "s": "BTCUSDT",
      "i": "1m",           // Interval
      "f": 5551498767,
      "L": 5551500190,
      "o": "98500.00",     // Open
      "c": "98520.50",     // Close
      "h": "98550.00",     // High
      "l": "98480.00",     // Low
      "v": "12.5",         // Volume
      "x": true            // Is candle closed?
    }
  }
}
```

**2. Parse & Normalize:**
```python
stream = message['stream']
coin = stream.split('@')[0].upper()  # 'BTCUSDT'
binance_tf = stream.split('@')[1].split('_')[1]  # '1m'
normalized_tf = '1min'  # Internal format
kline = message['data']['k']
```

**3. Process Kline:**
```python
await process_kline(coin, normalized_tf, kline)
# - Creates DataFrame row: {open, high, low, close, volume}
# - Checks if timestamp exists → update OR append
# - If candle closed (kline['x'] == True):
#     → process_closed_candle()
```

**4. Pattern Detection (on candle close):**
```python
await process_closed_candle(coin, timeframe, df_ohlcv)
# - Check if enough data (≥60 candles)
# - Check trading hours (if enabled)
# - Check exit conditions for active trades
# - Run ML prediction on last 60 candles
# - Check filters (probability, strength, EMA50, ATR)
# - Calculate targets (SL/TP)
# - Calculate position size
# - Open trade (DEMO or LIVE)
```

### Shared Capital Pool

```python
# All traders share same capital pool
self.shared_capital = 7046.58  # Initial from API
self.initial_capital = 7046.58

# Each trader references shared capital
for trader in self.traders.values():
    trader.capital = self.shared_capital

# On trade open/close:
self.shared_capital = trader.capital  # Update pool
self.sync_trader_capital()  # Sync all traders
```

**Example Scenario:**
```
Initial: $7046.58

BTCUSDT opens trade:
  - Position: 0.0236 BTC @ $98,500 = $2,325
  - Shared capital: $7046.58 → $4,721.58
  - ETHUSDT trader.capital: $4,721.58 (synced)

BTCUSDT closes trade (+$50 profit):
  - Shared capital: $4,721.58 → $4,771.58
  - ETHUSDT trader.capital: $4,771.58 (synced)

Both traders always use updated shared capital
MAX_CONCURRENT_TRADES = 3 prevents over-leveraging
```

### Status Updates (Every 30s)

```
================================================================================
📊 LIVE TRADING STATUS - 2025-11-23 00:42:20
================================================================================
💰 Shared capital: $7046.58 USDT
📈 Összes P&L: $0.00 USDT (+0.00%)
🔄 Aktív kereskedések: 0
✅ Lezárt kereskedések: 0
================================================================================
```

### Debug Logging

```
📌 RAW CANDLE CLOSE: BTCUSDT 1min | x=True | timestamp=2025-11-22 23:41:00
🔔 CANDLE CLOSED: BTCUSDT 1min | Processing trading logic...
🔍 process_closed_candle called: BTCUSDT 1min | Candles: 1000

🔍 PATTERN DETECTED: BTCUSDT 1min
   Pattern: ascending_triangle
   Probability: 0.807
   Strength: 0.807
   ⛔ SKIP: Trend/direction check failed
      - Price below EMA50 or bearish pattern or wrong trend alignment
```

---

## Backtesting

### Run Backtest
```bash
DATA_SOURCE=backtest python start.py
```

**Data Source:**
```python
BACKTEST_DATA_PATH_TEMPLATE = 
  '/home/nangyal/Desktop/v4/data/{coin}/{timeframe}/monthly/{coin}-trades-2025-10_{timeframe}.csv'
```

**Features:**
- ✅ Historical CSV data (October 2025)
- ✅ Multi-coin/timeframe parallel processing
- ✅ Same `TradingLogic` as live trading
- ✅ Excel report generation (`stat/backtest_report_*.xlsx`)
- ✅ Trade log CSV (`trades_log.csv`)

### Backtest with Hedging
```bash
DATA_SOURCE=backtest_hedging python start.py
```

**Additional Features:**
- ✅ Drawdown-based hedging
- ✅ Dynamic hedge thresholds (volatility-adjusted)
- ✅ Coin-specific hedge overrides
- ✅ Recovery-based hedge closing

**Hedging Config:**
```python
HEDGING = {
    'enable': True,
    'hedge_threshold': 0.18,  # 18% drawdown → activate
    'hedge_recovery_threshold': 0.08,  # 8% → close hedge
    'hedge_ratio': 0.35,  # 35% of exposure
    'dynamic_hedge': True,  # Adjust by volatility
    'coin_overrides': {
        'BTCUSDT': {'hedge_threshold': 0.16},
        'ETHUSDT': {'hedge_threshold': 0.22}
    }
}
```

---

## Pattern Detection

### Pattern Recognition Process

```python
# 1. Calculate 30-35 technical indicators
df_indicators = calculate_indicators(df_ohlcv)

# 2. ML model prediction
predictions, probabilities = classifier.predict(df_indicators.iloc[-60:])
pattern = predictions[-1]  # 'ascending_triangle'
probability = np.max(probabilities[-1])  # 0.807

# 3. Pattern strength (same as probability for now)
strength = probability

# 4. Filter checks
if probability < 0.65:  # Min 65% confidence
    skip("Probability too low")

if strength < 0.65:  # Min 65% strength
    skip("Strength too low")

# 5. Trend alignment
direction, entry, sl, tp = calculate_pattern_targets(pattern, candle, df)
if direction == 'skip':
    skip("Trend/direction check failed")

# 6. Position sizing
size = calculate_position_size(entry, sl, capital)
if size <= 0:
    skip("Invalid position size")

# 7. TRADE OPENED ✅
```

### Pattern Targets

**Example: Ascending Triangle**
```python
{
    'sl_pct': 0.005,  # -0.5% from entry
    'tp_pct': 0.020   # +2.0% from entry (1:4 R/R)
}

Entry: $98,500
Stop Loss: $98,500 × (1 - 0.005) = $98,007.50
Take Profit: $98,500 × (1 + 0.020) = $100,470.00
```

**Risk/Reward Ratios:**

| Pattern | SL% | TP% | R/R Ratio |
|---------|-----|-----|-----------|
| ascending_triangle | 0.5% | 2.0% | 1:4 |
| descending_triangle | 0.5% | 2.0% | 1:4 |
| symmetrical_triangle | 0.6% | 2.4% | 1:4 |
| double_top/bottom | 0.6% | 2.4% | 1:4 |
| head_and_shoulders | 0.7% | 2.8% | 1:4 |
| cup_and_handle | 0.6% | 2.4% | 1:4 |
| wedges | 0.6% | 2.4% | 1:4 |
| flags | 0.5% | 2.0% | 1:4 |

---

## Risk Management

### Capital Protection Layers

**1. Shared Capital Pool**
- All traders share $7046.58 USDT
- Updates on every trade open/close
- Prevents isolated capital depletion

**2. Max Concurrent Trades**
- Limit: 3 trades globally
- Prevents over-leveraging
- ~$2,325 per trade (33% × 3 = 99%)

**3. Position Size Limits**
- Max 33% per trade
- Risk-based sizing (2% capital risk)
- Caps at smaller of: risk-based OR 33%

**4. Pattern Filters**
- ML probability ≥65%
- Pattern strength ≥65%
- EMA50 trend filter
- ATR volatility filter (≥0.3%)

**5. Stop Loss**
- Always active
- Pattern-specific (0.5%-0.7%)
- Conservative levels

**6. Advanced Exit Strategies**
- Trailing stop (locks profits)
- Partial TP (reduces exposure)
- Breakeven stop (eliminates risk)

### Losing Streak Protection

```python
LOSING_STREAK_PROTECTION = {
    'enable': True,
    'reduce_risk_after': 3,  # 3 losses → 50% risk
    'risk_multiplier': 0.5,
    'stop_trading_after': 5,  # 5 losses → STOP
    'cooldown_candles': 60   # 1 hour pause
}
```

### ML Confidence Weighting

```python
ML_CONFIDENCE_WEIGHTING = {
    'enable': True,
    'tiers': [
        {'min_prob': 0.80, 'multiplier': 1.5},  # 80%+ → 1.5x size
        {'min_prob': 0.70, 'multiplier': 1.2},  # 70-80% → 1.2x
        {'min_prob': 0.65, 'multiplier': 1.0},  # 65-70% → 1.0x
    ]
}
```

---

## Deployment

### Production Setup (LIVE Mode)

**1. Switch to LIVE mode:**
```python
# config.py
BINANCE_DEMO_MODE = False  # ⚠️ REAL MONEY!
TRADING_HOURS = {'enable': True, 'start_hour': 6, 'end_hour': 20}
```

**2. Update API keys (MAINNET):**
```python
BINANCE_API_KEY = 'YOUR_MAINNET_API_KEY'
BINANCE_API_SECRET = 'YOUR_MAINNET_SECRET'
```

**3. Enable real order execution:**
```python
# websocket_live_trading.py
# Uncomment TODO sections for:
# - client.create_order() for BUY
# - client.create_order() for SELL
```

**4. Run with monitoring:**
```bash
nohup DATA_SOURCE=websocket python start.py > trader.log 2>&1 &
tail -f trader.log
```

**5. Set up alerts:**
- Monitor `trader.log` for errors
- Track shared capital changes
- Alert on max drawdown exceeded
- Slack/Telegram integration (TODO)

### DEMO Mode (Current)

```python
BINANCE_DEMO_MODE = True  # Paper trading
# Uses:
# - MAINNET WebSocket for REAL prices
# - DEMO API for balance/orders (no real execution)
# - Safe testing environment
```

### System Requirements

- **Python**: 3.10+
- **RAM**: 2GB+ (ML model + data buffers)
- **CPU**: Multi-core recommended (parallel backtesting)
- **Network**: Stable internet (WebSocket reliability)
- **OS**: Linux/macOS (tested on Ubuntu)

### Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `python-binance` - Binance API/WebSocket
- `websockets` - Async WebSocket client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `xgboost` - ML model
- `scikit-learn` - ML utilities
- `openpyxl` - Excel reports

---

## Troubleshooting

### Common Issues

**1. No Pattern Detections**
```
Symptom: Running for hours, no trades opened
Cause: Conservative filters (EMA50, probability)
Solution: Check filter thresholds in config.py
```

**2. WebSocket Disconnects**
```
Symptom: ❌ WebSocket connection lost
Cause: Network issues or Binance rate limits
Solution: Auto-reconnect implemented, check network
```

**3. Candles Not Closing**
```
Symptom: No "🔔 CANDLE CLOSED" logs
Cause: kline['x'] = False (candle not closed yet)
Solution: Wait for full candle duration (1min, 5min, etc.)
```

**4. Pattern Skipped (EMA50)**
```
Symptom: Pattern detected but skipped (trend failed)
Cause: Price below EMA50 or wrong trend alignment
Solution: Market in downtrend, wait for uptrend
```

**5. Shared Capital Not Syncing**
```
Symptom: Traders have different capital values
Cause: sync_trader_capital() not called
Solution: Fixed - syncs after every trade open/close
```

### Debug Mode

**Enable detailed logging:**
```python
# websocket_live_trading.py
# Already enabled:
# - 📌 RAW CANDLE CLOSE
# - 🔔 CANDLE CLOSED
# - 🔍 process_closed_candle called
# - 🔍 PATTERN DETECTED
# - ⛔ SKIP reasons
```

**Check specific timeframes:**
```python
# Only log 1min candles
if timeframe == '1min':
    print(f"DEBUG: {coin} {timeframe} ...")
```

### Performance Optimization

**1. Reduce Candle Buffer**
```python
self.max_candles = 200  # Default, reduce if memory issue
```

**2. Limit Timeframes**
```python
TIMEFRAMES = ['1min', '5min']  # Focus on fewer TFs
```

**3. Parallel Backtesting**
```python
NUM_WORKERS = 28  # Adjust based on CPU cores
```

### Logs & Monitoring

**Log Files:**
- `/tmp/websocket_test.log` - Live trading output
- `trades_log.csv` - All trades (backtest)
- `stat/backtest_report_*.xlsx` - Backtest results

**Status Checks:**
```bash
# Check process
ps aux | grep "python start.py"

# Check WebSocket connection
tail -f /tmp/websocket_test.log | grep "WebSocket"

# Check patterns
tail -f /tmp/websocket_test.log | grep "PATTERN DETECTED"

# Check trades
tail -f /tmp/websocket_test.log | grep "DEMO OPEN"
```

---

## API Reference

### LiveWebSocketTrader

```python
class LiveWebSocketTrader:
    def __init__(coins, timeframes, api_key, api_secret, demo_mode=True)
    async def initialize()
    async def run()
    async def load_historical_klines()
    async def get_account_balance() -> float
    def sync_trader_capital()
    async def print_status()
    async def handle_ws_message(message)
    async def process_kline(coin, timeframe, kline)
    async def process_closed_candle(coin, timeframe, df_ohlcv)
```

### TradingLogic

```python
class TradingLogic:
    def __init__(config)
    def should_open_trade(pattern, probability, strength) -> bool
    def calculate_pattern_targets(pattern, candle, df) -> (direction, entry, sl, tp)
    def calculate_position_size(entry, sl, capital) -> float
    def open_trade(**kwargs) -> dict
    def close_trade(trade, exit_price, reason, time, ratio=1.0) -> float
    def check_trade_exit(trade, candle) -> (should_close, price, reason, ratio)
```

### EnhancedForexPatternClassifier

```python
class EnhancedForexPatternClassifier:
    def load_model(path)
    def predict(df_ohlcv) -> (predictions, probabilities)
```

---

## Future Enhancements

### Planned Features
- [ ] Real order execution (LIVE mode)
- [ ] Telegram/Slack notifications
- [ ] Web dashboard (real-time monitoring)
- [ ] Advanced hedging strategies
- [ ] Multi-exchange support (Bybit, OKX)
- [ ] Sentiment analysis integration
- [ ] Auto-retraining ML model
- [ ] Database storage (PostgreSQL/MongoDB)

### Experimental
- [ ] Reinforcement Learning agent
- [ ] Portfolio optimization
- [ ] Cross-pair correlation analysis
- [ ] Options/futures trading
- [ ] MEV/arbitrage detection

---

## Contributing

### Code Style
- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** for all classes/methods
- **Comments** for complex logic

### Testing
```bash
# Run backtest (fast validation)
DATA_SOURCE=backtest python start.py

# Run WebSocket (live validation)
timeout 300 bash -c 'DATA_SOURCE=websocket python start.py'
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-indicator

# Commit changes
git add .
git commit -m "feat: add RSI divergence detector"

# Push and create PR
git push origin feature/new-indicator
```

---

## License

Proprietary - © 2025 Nangyal Trading Systems

---

## Support

**Issues:** Create GitHub issue with:
- Error logs
- Config settings
- Steps to reproduce

**Questions:** Contact developer

---

**Last Updated:** November 23, 2025  
**Version:** 5.0  
**Status:** Production (DEMO Mode)
