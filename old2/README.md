Crypto trading framework

Structure
- config.py - configuration for data source, coin list and timeframes
- start.py - program entry point (choose between backtest or websocket via config.DATA_SOURCE)
- classes/ - core classes
  - training.py - backtest runner using CSV tick data and resample to timeframes
  - classification.py - wrapper for the old EnhancedForexPatternClassifier
  - realtime.py - very small prototype of websocket ingestion and processing
- trading_logics/ - trading logic plug-ins
   - `trading_logic1.py` - wraps `HedgingBacktestEngine` which has been copied into `utils` to avoid direct imports from `/old`
  - trading_logic2.py - example SMA crossover logic
- utils/ - small utilities
  - trade_logger.py - logs trades to CSV with PnL in USDT

Run backtest example (use config or export environment vars):

```bash
python start.py
```

Set datasource:

```bash
export DATA_SOURCE=backtest
export TRADING_LOGIC=trading_logic1
python start.py
```

Realtime (experimental prototype):

```bash
export DATA_SOURCE=websocket
export TRADING_LOGIC=trading_logic2
python start.py
```

Notes
- Backtest runner will search for CSV files under `data/<coin>/`. It will attempt to convert tick (price/qty/time) to OHLC for the configured timeframes. It then calls the configured trading logic's `run_backtest` method.
- Realtime engine is a prototype: it uses Binance public WS and resamples incoming trade ticks — you can customize sources.
 - The framework can optionally use a pre-trained model saved with the old classifier; place `enhanced_forex_pattern_model.pkl` under `models/` (or update `MODEL_PATH` in `config.py`).
 - `HedgingBacktestEngine` implementation was copied locally to `utils/backtest_engine.py` to avoid importing engine code from `/old`.
