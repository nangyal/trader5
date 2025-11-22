# ⚙️ Configuration Files

Ez a mappa tartalmazza a kereskedési rendszer összes konfigurációs fájlját, témakörök szerint rendszerezve.

---

## 📁 Fájlok Áttekintése

### 1. `model_config.py` - ML Modell Beállítások

**Tartalom:**
- XGBoost hyperparaméterek
- Feature engineering beállítások
- Training/validation konfiguráció
- Model mentési útvonalak

**Példa használat:**
```python
from config import model_config

# XGBoost paraméterek
params = model_config.XGBOOST_PARAMS
model = xgb.XGBClassifier(**params)

# Feature groups
if model_config.FEATURE_GROUPS['momentum_indicators']:
    extractor.add_momentum_features()
```

**Kulcs paraméterek:**
- `XGBOOST_PARAMS` - ML model hyperparaméterek
- `OPTIMIZE_HYPERPARAMS` - Automatikus tuning be/ki
- `FEATURE_GROUPS` - Mely feature-ök legyenek használva
- `MIN_PREDICTION_PROBABILITY` - Minimum előrejelzési bizalom (0.6)

---

### 2. `pattern_config.py` - Pattern Detekció

**Tartalom:**
- Pattern detekciós küszöbök
- Pattern-specifikus paraméterek
- Adaptive window méretek
- Pattern strength scoring weights

**Példa használat:**
```python
from config import pattern_config

# Pattern threshold
threshold = pattern_config.PATTERN_DETECTION_THRESHOLDS['ascending_triangle']

# Window size by timeframe
window = pattern_config.ADAPTIVE_WINDOWS['1h']['triangle']

# Check if bullish
is_bullish = 'ascending' in pattern_config.BULLISH_PATTERNS
```

**Kulcs paraméterek:**
- `PATTERN_DETECTION_THRESHOLDS` - Min confidence minden pattern-re
- `ADAPTIVE_WINDOWS` - Timeframe-specifikus window méretek
- `BULLISH_PATTERNS` / `BEARISH_PATTERNS` - Pattern osztályozás
- `STRENGTH_WEIGHTS` - Pattern strength score súlyok

---

### 3. `trading_config.py` - Kereskedési Beállítások

**Tartalom:**
- Risk management (2% risk, tiered compounding)
- Stop loss / Take profit targets
- Trading stratégia (LONG-only aligned)
- Backtest konfiguráció
- WebSocket beállítások

**Példa használat:**
```python
from config import trading_config

# Risk per trade
risk_pct = trading_config.RISK_PER_TRADE  # 0.02 (2%)

# Get SL/TP for pattern
targets = trading_config.PATTERN_TARGETS['ascending_triangle']
sl_pct = targets['sl_pct']  # 0.015 (-1.5%)
tp_pct = targets['tp_pct']  # 0.03 (+3.0%)

# Max concurrent trades
max_trades = trading_config.MAX_CONCURRENT_TRADES  # 2
```

**Kulcs paraméterek:**
- `RISK_PER_TRADE` - Alapértelmezett rizikó (0.02 = 2%)
- `RISK_TIERS` - Tiered compounding stratégia
- `PATTERN_TARGETS` - SL/TP minden pattern-re
- `STRATEGY_TYPE` - 'long_only_aligned', 'long_short', stb.
- `MAX_CONCURRENT_TRADES` - Maximum párhuzamos pozíciók (2)

---

### 4. `api_config.py` - Binance API

**Tartalom:**
- API kulcsok (testnet/mainnet)
- API endpoint URL-ek
- Connection settings
- Rate limiting

**Példa használat:**
```python
from config import api_config

# Get credentials
api_key, api_secret = api_config.get_api_credentials()

# Get API URL
url = api_config.get_api_url()  # testnet vagy mainnet

# Connection timeout
timeout = api_config.CONNECTION_CONFIG['timeout']  # 30 sec
```

**Kulcs paraméterek:**
- `ENVIRONMENT` - 'testnet' vagy 'mainnet'
- `TESTNET_API_KEY` / `TESTNET_API_SECRET` - Demo kulcsok
- `CONNECTION_CONFIG` - Timeout, retry settings
- `API_RATE_LIMITS` - Binance rate limit értékek

**⚠️ BIZTONSÁGI FIGYELMEZTETÉS:**
- **NE** commitold a mainnet API kulcsokat!
- Használj environment változókat: `BINANCE_API_KEY`, `BINANCE_API_SECRET`

---

### 5. `data_config.py` - Adatkezelés

**Tartalom:**
- Data paths (CSV, cache, export)
- Data preprocessing rules
- Data validation checks
- Binance data download settings

**Példa használat:**
```python
from config import data_config

# Default training data
csv_path = data_config.DEFAULT_TRAINING_DATA

# Outlier detection
if data_config.OUTLIER_DETECTION['enable']:
    q_low, q_high = data_config.OUTLIER_DETECTION['quantile_range']

# Missing data strategy
strategy = data_config.MISSING_DATA['strategy']  # 'ffill_bfill'
```

**Kulcs paraméterek:**
- `DEFAULT_TRAINING_DATA` - Alapértelmezett CSV fájl
- `DATA_CLEANING` - Invalid OHLC, duplicates, gaps
- `OUTLIER_DETECTION` - Kiugró értékek kezelése
- `MISSING_DATA` - Hiányzó adatok kezelése

---

## 🚀 Használati Példák

### Teljes Training Pipeline

```python
from config import model_config, data_config, pattern_config

# 1. Load data
df = pd.read_csv(data_config.DEFAULT_TRAINING_DATA)

# 2. Clean data
if data_config.OUTLIER_DETECTION['enable']:
    df = remove_outliers(df)

# 3. Create pattern labels
detector = AdvancedPatternDetector()
for pattern_name, threshold in pattern_config.PATTERN_DETECTION_THRESHOLDS.items():
    # Detect patterns with threshold
    ...

# 4. Train model
classifier = EnhancedForexPatternClassifier()
model = classifier.train(
    df, 
    labels,
    optimize_hyperparams=model_config.OPTIMIZE_HYPERPARAMS
)

# 5. Save model
classifier.save_model(model_config.MODEL_SAVE_PATH)
```

### Live Trading Setup

```python
from config import api_config, trading_config

# 1. Get API credentials
api_key, api_secret = api_config.get_api_credentials()

# 2. Initialize trader
trader = BinanceLiveTrader(
    api_key=api_key,
    api_secret=api_secret,
    symbol=trading_config.DEFAULT_SYMBOL,
    risk_per_trade=trading_config.RISK_PER_TRADE
)

# 3. Enable tiered risk
trader.use_tiered_risk = trading_config.USE_TIERED_RISK
trader.risk_tiers = trading_config.RISK_TIERS

# 4. Set max concurrent trades
trader.max_concurrent = trading_config.MAX_CONCURRENT_TRADES

# 5. Start trading
trader.run()
```

### Backtest Configuration

```python
from config import trading_config

# Setup backtest engine
backtester = BacktestingEngine(
    initial_capital=trading_config.BACKTEST_INITIAL_CAPITAL,
    risk_per_trade=trading_config.RISK_PER_TRADE
)

# Enable slippage & commission
if trading_config.BACKTEST_SLIPPAGE['enable']:
    backtester.slippage_pct = trading_config.BACKTEST_SLIPPAGE['percent']

if trading_config.BACKTEST_COMMISSION['enable']:
    backtester.commission_pct = trading_config.BACKTEST_COMMISSION['percent']

# Run backtest
results = backtester.run_backtest(df, predictions, probabilities)
```

---

## 🔧 Módosítási Útmutató

### 1. Risk Beállítások Módosítása

Fájl: `trading_config.py`

```python
# Alap rizikó 2%-ról 1%-ra
RISK_PER_TRADE = 0.01  # Változtasd 0.02-ről 0.01-re

# Tiered rizikó kikapcsolása
USE_TIERED_RISK = False

# Max concurrent trades 2-ről 5-re
MAX_CONCURRENT_TRADES = 5
```

### 2. Pattern Detekció Finomhangolása

Fájl: `pattern_config.py`

```python
# Szigorúbb ascending triangle detekció
PATTERN_DETECTION_THRESHOLDS = {
    'ascending_triangle': 0.75,  # 0.65-ről 0.75-re
    ...
}

# Nagyobb window 1h timeframe-re
ADAPTIVE_WINDOWS = {
    '1h': {
        'triangle': 150,  # 100-ról 150-re
        ...
    }
}
```

### 3. Model Hyperparaméterek

Fájl: `model_config.py`

```python
# Több fa jobb accuracy-ért (lassabb training)
XGBOOST_PARAMS = {
    'n_estimators': 1000,  # 500-ról 1000-re
    'max_depth': 8,        # 6-ról 8-ra
    ...
}

# Hyperparameter search bekapcsolása
OPTIMIZE_HYPERPARAMS = True
```

### 4. API Environment Váltás

Fájl: `api_config.py`

```python
# Testnet → Mainnet (VIGYÁZZ!)
ENVIRONMENT = 'mainnet'  # 'testnet'-ről 'mainnet'-re

# Előtte set-eld az env változókat:
# export BINANCE_API_KEY="your_real_api_key"
# export BINANCE_API_SECRET="your_real_api_secret"
```

---

## 📊 Paraméter Összefoglaló Táblázat

| Kategória | Paraméter | Alapértelmezett | Hol található |
|-----------|-----------|----------------|---------------|
| **Risk** | Base risk | 2% | `trading_config.RISK_PER_TRADE` |
| **Risk** | Max trades | 2 | `trading_config.MAX_CONCURRENT_TRADES` |
| **SL/TP** | Ascending △ SL | -1.5% | `trading_config.PATTERN_TARGETS` |
| **SL/TP** | Ascending △ TP | +3.0% | `trading_config.PATTERN_TARGETS` |
| **Model** | Trees (n_estimators) | 500 | `model_config.XGBOOST_PARAMS` |
| **Model** | Max depth | 6 | `model_config.XGBOOST_PARAMS` |
| **Model** | Learning rate | 0.1 | `model_config.XGBOOST_PARAMS` |
| **Pattern** | Asc △ threshold | 0.65 | `pattern_config.PATTERN_DETECTION_THRESHOLDS` |
| **Pattern** | 1h window | 100 bars | `pattern_config.ADAPTIVE_WINDOWS` |
| **API** | Environment | testnet | `api_config.ENVIRONMENT` |
| **API** | Timeout | 30s | `api_config.CONNECTION_CONFIG` |
| **Data** | Min rows | 100 | `data_config.DATA_VALIDATION` |
| **Data** | Outlier method | quantile | `data_config.OUTLIER_DETECTION` |

---

## ✅ Best Practices

### 1. Version Control

```bash
# Add config to git
git add config/

# SOHA ne commitold a mainnet API kulcsokat!
# Ellenőrizd .gitignore-ban:
echo "config/api_config.py" >> .gitignore  # HA mainnet kulcsokat írtál bele
```

### 2. Environment Variables

Mainnet használatához:

```bash
# .env fájl (add to .gitignore!)
BINANCE_API_KEY=your_real_api_key
BINANCE_API_SECRET=your_real_api_secret

# Load in Python
from dotenv import load_dotenv
load_dotenv()

# api_config.py automatikusan használja:
MAINNET_API_KEY = os.getenv('BINANCE_API_KEY', '')
```

### 3. Backup Configs

```bash
# Mentsd el az aktuális config-ot production előtt
cp -r config/ config_backup_$(date +%Y%m%d)/
```

### 4. Testing Changes

Új config módosítás után:

```python
# Test model config
from config import model_config
print(model_config.XGBOOST_PARAMS)

# Test pattern config
from config import pattern_config
print(pattern_config.ADAPTIVE_WINDOWS['1h'])

# Test trading config
from config import trading_config
print(trading_config.PATTERN_TARGETS)
```

---

## 🔍 Troubleshooting

### ImportError: No module named 'config'

```bash
# Ellenőrizd, hogy a config mappa jó helyen van
ls config/__init__.py

# Futtasd a scriptet a root directory-ból
cd ~/Desktop/patterns_deepseek/test4-binance
python enhanced_main.py
```

### API kulcsok nem működnek

```python
# Ellenőrizd az environment-et
from config import api_config
print(api_config.ENVIRONMENT)  # 'testnet' vagy 'mainnet'?

# Test connection
from config.api_config import get_api_credentials
api_key, api_secret = get_api_credentials()
print(f"API Key (first 10): {api_key[:10]}...")
```

### Config változások nem jelennek meg

```python
# Reload module (Jupyter/iPython-ban)
import importlib
from config import trading_config
importlib.reload(trading_config)

# Vagy restart Python interpreter
```

---

## 📞 Support

Ha kérdésed van a config fájlokkal kapcsolatban:

1. Nézd meg a `DEVELOPER.md` dokumentációt
2. Ellenőrizd az inline kommenteket a config fájlokban
3. Nézd meg a példa kódokat fent

---

**Készítette:** AI Assistant  
**Utolsó frissítés:** 2025-11-12  
**Verzió:** 3.0

---

Happy Configuration! ⚙️🚀
