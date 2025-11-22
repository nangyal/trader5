# Crypto Trading Framework

Moduláris kereskedési keretrendszer backtest és real-time websocket támogatással.

## Struktúra

```
├── config.py              # Központi konfiguráció (adatforrás, coin lista, timeframe-ek)
├── start.py               # Program belépési pont (backtest vagy websocket)
├── training.py            # Modell tanítása (classifier training)
│
├── /classes
│   ├── training.py        # BacktestRunner - CSV tick adatok feldolgozása, multiprocessing
│   ├── classification.py  # Classifier wrapper (Simple vagy GPU-enhanced)
│   └── realtime.py        # RealtimeRunner - websocket adatok feldolgozása
│
├── /trading_logics
│   ├── trading_logic1.py  # HedgingBacktestEngine wrapper (utils/backtest_engine.py használva)
│   └── trading_logic2.py  # Példa SMA crossover logika
│
├── /utils
│   ├── trade_logger.py      # Trade naplózás CSV-be + PnL USDT-ben
│   ├── backtest_engine.py   # HedgingBacktestEngine (másolt az /old-ból)
│   ├── classifier.py        # SimpleClassifier implementáció
│   └── advanced_classifier.py # GPU-enabled classifier wrapper
│
├── /data                  # CSV fájlok tick formátumban (coin/timeframe struktúra)
├── /stat                  # Excel statisztikák a backtest eredményekről
├── /models                # Tanított modellek (.pkl fájlok)
└── /old                   # Korábbi verzió (referencia célra)
```

## Funkciók

### 1. Kétféle Adatforrás
- **Backtest**: Korábbi tick adatok CSV fájlokból
- **Websocket**: Real-time adatok Binance websocket-en keresztül

### 2. Timeframe Generálás
- Tick szintű adatokból automatikus OHLCV resample
- Konfigurálható timeframe-ek: `15s`, `30s`, `1min`
- Több coin szinkronizált időkezelése

### 3. Multiprocessing
- Párhuzamos feldolgozás több coin-ra
- Konfigurálható worker szám
- Pickle-safe worker implementáció

### 4. Választható Trading Logic
- Plugin rendszer - könnyű új logika hozzáadása
- Config fájlban állítható melyik logika fusson
- Minden logika ugyanazt az interfészt használja

### 5. Naplózás és Statisztika
- CSV log minden trade-ről (PnL USDT-ben)
- Részletes Excel report backtest után
- Konzol kimenet fejlesztéshez és analízishez

### 6. GPU Támogatás
- GPU-accelerált classifier (XGBoost gpu_hist)
- Automatikus CUDA device detection
- Fallback SimpleClassifier CPU-ra

## Használat

### Modell Tanítása

```bash
python training.py
```

Ez betölti a historikus adatokat, feature-öket készít, és tanítja a modellt.

### Backtest Futtatása

```bash
# Alapértelmezett beállításokkal
python start.py

# Vagy környezeti változókkal
export DATA_SOURCE=backtest
export TRADING_LOGIC=trading_logic1
export NUM_WORKERS=4
export BACKTEST_INITIAL_CAPITAL=200
python start.py
```

### Real-time Trading (Websocket)

```bash
export DATA_SOURCE=websocket
export TRADING_LOGIC=trading_logic2
python start.py
```

## Konfiguráció

A `config.py` fájlban vagy környezeti változókkal:

- `DATA_SOURCE`: `'backtest'` vagy `'websocket'`
- `COINS`: Lista a feldolgozandó coin-okról (pl. `['BTCUSDT', 'ETHUSDT']`)
- `TIMEFRAMES`: Lista a timeframe-ekről (pl. `['15s', '30s', '1min']`)
- `TRADING_LOGIC`: Trading logic neve (pl. `'trading_logic1'`)
- `NUM_WORKERS`: Worker processek száma
- `BACKTEST_INITIAL_CAPITAL`: Kezdő tőke USDT-ben
- `USE_ADVANCED_CLASSIFIER`: GPU classifier használata (`'1'` vagy `'0'`)

## Adatok

A CSV fájlok az alábbi struktúrában:
```
data/
  BTCUSDT/
    15s/
    1min/
    30s/
    5min/
    monthly/
  ETHUSDT/
    ...
```

Tick formátum oszlopok: `time`, `price`, `qty` (vagy standard OHLCV)

## Eredmények

- **trades_log.csv**: Minden trade részletei (coin, timeframe, entry/exit, PnL USDT)
- **stat/bt_report_YYYYMMDD_HHMMSS.xlsx**: Backtest összefoglaló statisztikák
- Konzol output: Real-time progress és debug információk

## GPU Támogatás Beállítása

1. Telepítsd a CUDA-t és cuDNN-t
2. Telepítsd az XGBoost-ot GPU támogatással:
   ```bash
   pip install xgboost --upgrade
   ```
3. Ellenőrizd a GPU elérhetőségét:
   ```bash
   python -c "import xgboost as xgb; print(xgb.core._LIB.XGBoosterGetNumFeature)"
   ```

## Új Trading Logic Hozzáadása

1. Hozz létre `trading_logics/trading_logic_X.py` fájlt
2. Implementáld a `TradingLogic` osztályt:
   ```python
   class TradingLogic:
       def __init__(self, classifier):
           self.classifier = classifier
       
       def run_backtest(self, coin, timeframe, ohlc_df, initial_capital):
           # Logika implementációja
           return {
               'trades': [...],
               'total_pnl': ...,
               'final_capital': ...,
               'total_trades': ...
           }
   ```
3. Állítsd be a `config.py`-ban: `TRADING_LOGIC = 'trading_logic_X'`

## Megjegyzések

- A framework nem függ az `/old` könyvtártól (minden szükséges kód át lett másolva)
- Multiprocessing használat miatt minden worker külön példányosítja a classifier-t
- GPU használat esetén ügyelj a memória limit-ekre (több worker = több GPU memória)
- A websocket runner még prototípus, de bővíthető tetszőleges exchange API-val
