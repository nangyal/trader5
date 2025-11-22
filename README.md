# Crypto Trading Framework v5

## Áttekintés

Professzionális crypto trading keretrendszer, ami **két különböző adatforrással** dolgozik, de **ugyanazt a kereskedési logikát** használja mindkettőnél:

1. **Backtest mód**: Korábbi tick adatok alapján történő szimuláció (CSV fájlokból)
2. **WebSocket mód**: Valós idejű kereskedés Binance WebSocket-en keresztül

## Fő jellemzők

✅ **Egységes trading logika** - Backtest és WebSocket ugyanazt a kódot használja  
✅ **Multiprocessing** - Több coin párhuzamos feldolgozása  
✅ **Multi-timeframe** - 15s, 30s, 1min timeframe-ek egyidejű elemzése  
✅ **ML-alapú pattern detection** - XGBoost modell használata  
✅ **Részletes Excel riportok** - Backtest eredmények statisztikái  
✅ **CSV trade logging** - Minden trade naplózva  
✅ **Risk management** - Tiered compounding, stop loss, take profit  

## Fájl struktúra

```
v5/
├── config.py                 # Központi konfiguráció
├── start.py                  # Fő indító script
├── trading_logic.py          # Közös trading logika (backtest ÉS websocket is ezt használja!)
├── backtest.py               # Backtest modul (CSV tick adatokkal)
├── websocket_trading.py      # WebSocket live trading
├── excel_stats.py            # Excel riport generátor
├── data/                     # CSV tick adatok
│   └── BTCUSDT/
│       └── 1min/
│           └── monthly/
│               └── *.csv
├── stat/                     # Excel statisztikák
├── models/                   # ML modell
│   └── enhanced_forex_pattern_model.pkl
├── old/                      # Régebbi verziók (referencia)
└── trades_log.csv            # Trade napló
```

## Használat

### 1. Backtest mód indítása

```bash
# config.py-ban állítsd be:
DATA_SOURCE = 'backtest'

# Indítás:
python start.py
```

**Mit csinál:**
- Betölti a CSV tick adatokat a `data/` könyvtárból
- Minden coin-ra külön processzben fut (multiprocessing)
- Minden timeframe-en (15s, 30s, 1min) elemez
- Pattern-eket detektál ML modellel
- Szimulált trade-eket nyit és zár
- Excel riportot generál a `stat/` könyvtárba
- CSV-be logolja a trade-eket

### 2. WebSocket mód indítása

```bash
# config.py-ban állítsd be:
DATA_SOURCE = 'websocket'

# Indítás:
python start.py
```

**Mit csinál:**
- Csatlakozik a Binance WebSocket-hez (minden coin-ra külön)
- Valós időben dolgozza fel a tick-eket
- Real-time OHLCV candle-eket generál (15s, 30s, 1min)
- Pattern detection minden timeframe-en
- Valós trade megnyitás/zárás
- **DEMO MODE**: Testnet használata (biztonságos tesztelés)

## Konfiguráció

### config.py - Fő beállítások

```python
# Adatforrás választása
DATA_SOURCE = 'backtest'  # vagy 'websocket'

# Coinok
COINS = ['BTCUSDT', 'ETHUSDT']

# Timeframes
TIMEFRAMES = ['15s', '30s', '1min']

# Backtest
BACKTEST_INITIAL_CAPITAL = 200.0  # USDT

# WebSocket
BINANCE_DEMO_MODE = True  # True = testnet, False = mainnet

# Risk Management
RISK_PER_TRADE = 0.02  # 2% kockázat per trade
USE_TIERED_RISK = True  # Tiered compounding

# Pattern Filters
PATTERN_FILTERS = {
    'min_probability': 0.7,  # Min 70% ML konfidencia
    'min_strength': 0.7,     # Min 70% pattern erősség
}
```

## Trading Logika

### Közös logika (backtest ÉS websocket)

A `trading_logic.py` modul tartalmazza a **teljes trading logikát**, amit **mindkét mód használ**:

1. **Pattern detection** - ML modell + pattern strength scorer
2. **Trend alignment** - LONG-ONLY stratégia (csak bullish patterns uptrend-ben)
3. **Volatility filter** - ATR alapú, alacsony volatilitás kizárása
4. **Position sizing** - Risk-alapú, tiered compounding
5. **Entry/Exit logika** - Stop loss és take profit számítás
6. **Trade management** - Aktív trade-ek monitoring

### Pattern targets (Stop Loss & Take Profit)

```python
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.008,  # -0.8% SL
        'tp_pct': 0.012   # +1.2% TP
    },
    'symmetrical_triangle': {
        'sl_pct': 0.010,  # -1.0% SL
        'tp_pct': 0.015   # +1.5% TP
    },
    # ... stb.
}
```

## CSV Adatforrás

### Tick adatok formátuma

A backtest CSV fájloknak tartalmazniuk kell:
- `time` vagy `timestamp` oszlop (milliszekundum vagy másodperc timestamp)
- `price` oszlop (tick ár)
- `qty` vagy `amount` oszlop (mennyiség)

Példa elérési út:
```
data/BTCUSDT/1min/monthly/BTCUSDT-2025-01.csv
```

### Timeframe generálás

A backtest **tick adatokból** generál OHLCV candle-eket:
- Tick adatok → **resample** → 15s/30s/1min candle-ek
- Minden timeframe-en külön pattern detection

## WebSocket Működés

### Real-time candle építés

1. WebSocket tick-eket kap Binance-tól
2. Tick-eket buffer-eli (deque)
3. Minden tick frissíti az aktuális candle-t
4. Candle lezárása → pattern detection
5. Pattern találat → trade megnyitás (ha kritériumok teljesülnek)

### Demo vs Live Mode

**DEMO MODE (javasolt tesztelésre):**
```python
BINANCE_DEMO_MODE = True
```
- Binance Testnet használata
- Nincs valódi pénz mozgás
- Biztonságos gyakorlás

**LIVE MODE (élő kereskedés):**
```python
BINANCE_DEMO_MODE = False
```
- Valódi Binance API
- **Valódi pénz kereskedés!**
- **ÓVATOSAN használd!**

## Excel Statisztikák

A backtest mód részletes Excel riportot generál:

### Sheets:
1. **Summary** - Összefoglaló statisztikák
2. **Detailed Results** - Minden coin részletes eredménye
3. **Per Coin Stats** - Coin-onként aggregált adatok
4. **Per Timeframe Stats** - Timeframe-enként aggregált adatok
5. **Top Performers** - Legjobb és legrosszabb eredmények

## Multiprocessing

### Backtest
- Minden coin külön processzben fut
- CPU core-ok száma alapján párhuzamosítás
- `NUM_WORKERS` config beállítás

```python
NUM_WORKERS = 4  # 4 processz egyidejűleg
```

### WebSocket
- Minden coin külön thread-ben fut
- Közös WebSocket manager
- Real-time tick distribution

## Logging

### Console logging
Minden fontos esemény kiírásra kerül:
- Pattern detection
- Trade open/close
- P&L számítás
- Egyenleg változások

### CSV trade log
Minden trade naplózva a `trades_log.csv` fájlban:
- Entry/exit időpont
- Ár, pozíció méret
- Pattern, timeframe
- P&L (USDT)

## Követelmények

### Python csomagok
```bash
pip install pandas numpy talib scikit-learn xgboost joblib
pip install openpyxl  # Excel íráshoz
pip install websocket-client python-binance  # WebSocket-hez
```

### ML Model
A `models/enhanced_forex_pattern_model.pkl` fájl szükséges a pattern detection-höz.

Ha nincs meg, generáld le:
```bash
cd old/
python forex_pattern_classifier.py
```

## Fejlesztési Notes

### Ugyanaz a trading logika mindenhol!

**FONTOS:** A `trading_logic.py` modul a központi agya a rendszernek.  
**Mind a backtest, mind a websocket ezt használja!**

Ha változtatni akarsz a trading stratégián:
1. Módosítsd a `trading_logic.py`-t
2. Automatikusan mindkét mód frissül

### Pattern detection flow

```
CSV tick / WebSocket tick
    ↓
OHLCV candle generálás (timeframe-enként)
    ↓
ML Model prediction
    ↓
Pattern strength scoring
    ↓
Filters (probability, strength, trend, volatility)
    ↓
Position sizing
    ↓
Trade open (közös TradingLogic)
    ↓
Monitor aktív trade-ek
    ↓
Trade close (SL/TP hit)
```

## Gyakori Problémák

### Nincs CSV adat
```
❌ BTCUSDT: Nincs adat
```
**Megoldás:** Ellenőrizd a `data/BTCUSDT/1min/monthly/` könyvtárat, hogy vannak-e CSV fájlok.

### Model betöltési hiba
```
❌ Model betöltési hiba
```
**Megoldás:** Futtasd le a model training script-et az `old/` könyvtárban.

### WebSocket nem csatlakozik
```
❌ WebSocket nem csatlakozott!
```
**Megoldás:** Ellenőrizd az internet kapcsolatot és a Binance API elérhetőségét.

## Bővítési Lehetőségek

- [ ] További timeframe-ek (5min, 15min)
- [ ] Több coin egyidejű kereskedése (portfólió)
- [ ] Trailing stop implementálása
- [ ] Advanced risk management (drawdown protection)
- [ ] Telegram/Discord alert-ek
- [ ] Real-time dashboard (Flask/Streamlit)

## Licenc

Ez a projekt privát használatra készült. Ne oszd meg az API kulcsokat!

## Kontakt

Ha kérdésed van, írj!
