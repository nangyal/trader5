# Gyors Kezdés - Quick Start Guide

## 1. Telepítés

```bash
# Python csomagok telepítése
pip install pandas numpy talib scikit-learn xgboost joblib openpyxl websocket-client python-binance
```

## 2. Könyvtár struktúra ellenőrzése

```bash
cd /home/nangyal/Desktop/v5

# Szükséges könyvtárak létrehozása
python config.py
```

Kimenet:
```
✅ Könyvtárak létrehozva/ellenőrizve
Adatforrás: backtest
...
```

## 3. ML Model ellenőrzése

Ellenőrizd, hogy a model file létezik:
```bash
ls models/enhanced_forex_pattern_model.pkl
```

Ha **nincs meg**, generáld le a régi kódból:
```bash
cd old/
python forex_pattern_classifier.py
# Várj, amíg a training befejeződik (pár perc)
```

## 4. CSV Adatok előkészítése (Backtest módhoz)

### Példa adatok letöltése

A backtest mód CSV tick adatokat vár a következő struktúrában:

```
data/
├── BTCUSDT/
│   └── 1min/
│       └── monthly/
│           ├── BTCUSDT-2025-01.csv
│           ├── BTCUSDT-2025-02.csv
│           └── ...
└── ETHUSDT/
    └── 1min/
        └── monthly/
            ├── ETHUSDT-2025-01.csv
            └── ...
```

### CSV formátum

A CSV fájloknak tartalmazniuk kell ezeket az oszlopokat:
- `time` vagy `timestamp` - időbélyeg (milliszekundum vagy másodperc)
- `price` - tick ár
- `qty` vagy `amount` - mennyiség

Példa sor:
```csv
timestamp,price,qty
1704067200000,42150.5,0.05
1704067200100,42151.0,0.12
...
```

### Adatok beszerzése Binance-ról (opcionális)

Ha nincsenek meg az adatok, tölthetsz le Binance-ról:

```bash
# Példa script Binance adatok letöltésére
cd old/
python download_monthly.py
```

## 5. Backtest futtatása

### 5.1 Config beállítása

Nyisd meg a `config.py` fájlt és ellenőrizd:

```python
DATA_SOURCE = 'backtest'  # BACKTEST mód
COINS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15s', '30s', '1min']
BACKTEST_INITIAL_CAPITAL = 200.0
```

### 5.2 Futtatás

```bash
python start.py
```

### 5.3 Kimenet

```
================================================================================
🚀 CRYPTO TRADING FRAMEWORK
================================================================================

📋 Konfiguráció:
   Adatforrás: backtest
   Coinok: BTCUSDT, ETHUSDT
   Timeframes: 15s, 30s, 1min
   Workers: 4

💰 Backtest Beállítások:
   Kezdő tőke: $200.0
   ...

================================================================================
BACKTEST MÓD
================================================================================

[Worker 1] 🚀 BTCUSDT backtest indítása...
[Worker 2] 🚀 ETHUSDT backtest indítása...
...
```

### 5.4 Eredmények

A backtest befejeztével:
1. **Console-ra** kiíródnak az eredmények
2. **CSV log** létrejön: `trades_log.csv`
3. **Excel riport** generálódik: `stat/backtest_report_YYYYMMDD_HHMMSS.xlsx`

## 6. WebSocket mód (Live trading)

### 6.1 Config beállítása

```python
DATA_SOURCE = 'websocket'  # WEBSOCKET mód
BINANCE_DEMO_MODE = True   # DEMO mode (biztonságos!)
```

### 6.2 Futtatás

```bash
python start.py
```

### 6.3 DEMO vs LIVE

**DEMO MODE** (ajánlott tesztelésre):
- Binance Testnet
- Nincs valódi pénz mozgás
- API kulcs: már benne van a config-ban

**LIVE MODE** (éles kereskedés):
```python
BINANCE_DEMO_MODE = False
```
⚠️ **FIGYELEM: Valódi pénz kereskedés! Csak saját felelősségre!**

### 6.4 Kimenet

```
================================================================================
🚀 CRYPTO TRADING FRAMEWORK
================================================================================
...
WEBSOCKET LIVE TRADING MÓD
================================================================================

⚠️  FIGYELEM: Live trading mód!
✅ DEMO/TESTNET mód - biztonságos tesztelés

[BTCUSDT] 🚀 WebSocket Trading indítása...
[BTCUSDT] 💰 Binance USDT egyenleg: $1000.00
[BTCUSDT] ✅ Trading logic inicializálva
[BTCUSDT] 📥 Historikus candle-ek betöltése...
[BTCUSDT] ✅ 500 candle betöltve
[BTCUSDT] ✅ WebSocket csatlakozva
[BTCUSDT] 🚀 WebSocket elindítva
[BTCUSDT] ✅ WebSocket Trading fut! Várakozás tick-ekre...

[BTCUSDT] 📊 Status:
   Capital: $1000.00
   Active trades: 0
   Total trades: 0
   ...
```

## 7. Eredmények elemzése

### 7.1 Excel riport megnyitása

```bash
# Nyisd meg a legfrissebb Excel riportot
cd stat/
ls -ltr  # Legfrissebb fájl alul
```

### 7.2 Excel sheet-ek

- **Summary** - Összefoglaló statisztikák
- **Detailed Results** - Coin-onkénti részletek
- **Per Coin Stats** - Coin statisztikák
- **Per Timeframe Stats** - Timeframe statisztikák
- **Top Performers** - Legjobb/legrosszabb eredmények

### 7.3 CSV trade log

```bash
# Trade log megtekintése
cat trades_log.csv
```

Oszlopok:
```
timestamp, coin, action, direction, pattern, timeframe,
entry_price, exit_price, stop_loss, take_profit,
position_size, probability, strength, exit_reason, pnl_usdt, total_pnl
```

## 8. Hibaelhárítás

### 8.1 "No module named 'talib'"

```bash
# Ubuntu/Debian:
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# macOS:
brew install ta-lib
pip install TA-Lib

# Windows:
# Töltsd le a pre-built wheel-t:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXXm-win_amd64.whl
```

### 8.2 "Model betöltési hiba"

```bash
# Generáld újra a modelt
cd old/
python forex_pattern_classifier.py
```

### 8.3 "Nincs CSV adat"

Ellenőrizd a data/ könyvtár struktúrát:
```bash
tree data/
```

Ha üres, töltsd le az adatokat vagy másold át a régi adatokat.

### 8.4 "WebSocket nem csatlakozik"

- Ellenőrizd az internet kapcsolatot
- Binance API lehet ideiglenesen lezárva
- Próbáld újra pár perc múlva

## 9. Config testreszabása

### 9.1 Több coin hozzáadása

```python
COINS = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'ADAUSDT',
    # ... stb.
]
```

### 9.2 Timeframe-ek változtatása

```python
TIMEFRAMES = ['15s', '30s', '1min', '5min']
```

### 9.3 Risk management

```python
RISK_PER_TRADE = 0.01  # 1% kockázat (konzervatívabb)
USE_TIERED_RISK = True  # Tiered compounding BE

RISK_TIERS = [
    {'max_capital_ratio': 2.0, 'risk': 0.01},   # Módosítva 1%-ra
    {'max_capital_ratio': 3.0, 'risk': 0.008},
    # ...
]
```

### 9.4 Pattern filter-ek

```python
PATTERN_FILTERS = {
    'min_probability': 0.8,  # Szigorúbb (80%)
    'min_strength': 0.8,     # Szigorúbb (80%)
    'blacklist_patterns': ['wedge_rising']  # Kizárt pattern-ek
}
```

## 10. Fejlesztés

### 10.1 Trading logika módosítása

A **teljes trading logika** a `trading_logic.py`-ban van.

**Mindkét mód (backtest ÉS websocket) ezt használja!**

Példa: TP/SL arányok módosítása:
```python
# config.py
PATTERN_TARGETS = {
    'ascending_triangle': {
        'sl_pct': 0.005,  # Szorosabb SL: -0.5%
        'tp_pct': 0.020   # Nagyobb TP: +2.0%
    },
}
```

### 10.2 Új pattern hozzáadása

1. Add hozzá a config-hoz:
```python
PATTERN_TARGETS['my_new_pattern'] = {
    'sl_pct': 0.010,
    'tp_pct': 0.015
}
```

2. Módosítsd a trend alignment-et:
```python
TREND_ALIGNMENT = {
    'bullish_patterns': [
        'ascending_triangle',
        'my_new_pattern',  # Új pattern
    ]
}
```

## 11. Következő lépések

- [x] Backtest futtatása történelmi adatokon
- [x] Eredmények elemzése Excel-ben
- [ ] Fine-tuning: Config paraméterek optimalizálása
- [ ] Demo WebSocket tesztelése
- [ ] Live WebSocket (óvatosan!)

## Kérdések?

Nézd meg a `README.md`-t részletesebb információkért!
