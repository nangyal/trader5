import pandas as pd
import os
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import importlib
from pathlib import Path
import config
from utils.trade_logger import TradeLogger
from typing import List


def find_csv_for_coin(coin):
    base = config.DATA_DIR / coin
    if not base.exists():
        return []
    csvs = []
    # search recursively for csv files
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(root, f))
    return csvs


def load_tick_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # identify time column
    for col in ['time', 'datetime', 'timestamp', 'date']:
        if col in df.columns:
            try:
                df['time'] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                if df['time'].isna().all():
                    df['time'] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                df['time'] = pd.to_datetime(df[col], errors='coerce')
            break

    if 'time' in df.columns:
        df = df.dropna(subset=['time']).set_index('time').sort_index()

    return df


def tick_to_ohlcv(df, tf):
    # map config timeframes to pandas offsets
    tf_map = {'15s': '15S', '30s': '30S', '1min': '1T'}
    freq = tf_map.get(tf, tf)
    if 'price' in df.columns and 'qty' in df.columns:
        # Use pandas ohlc() helper and combine with aggregated volume
        ohlc_price = df['price'].resample(freq).ohlc()
        vol = df['qty'].resample(freq).sum().rename('volume')
        ohlc = pd.concat([ohlc_price, vol], axis=1)
    else:
        ohlc = df.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    ohlc = ohlc.dropna()
    return ohlc


def process_coin_worker(coin, timeframes, trading_logic_module_name, model_path, initial_capital):
    """Worker function executed inside Pool. Returns a list of dictionaries (summaries).

    This function doesn't reference the BacktestRunner object, so it avoids pickling
    issues. It imports the requested trading logic inside the worker.
    """
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    print(f'🚀 Worker [{coin}]: STARTED', flush=True)
    
    import importlib
    from classes.classification import Classifier
    from utils.trade_logger import TradeLogger

    results = []

    csv_files = find_csv_for_coin(coin)
    if not csv_files:
        print(f'  No csv in data for {coin}', flush=True)
        return []

    print(f'📁 Worker [{coin}]: Found {len(csv_files)} CSV files', flush=True)
    
    # instantiate classifier and trading logic per worker
    clf = Classifier()
    try:
        # CRITICAL: Enable GPU mode in worker processes
        print(f'🔧 Worker [{coin}]: Loading model with GPU...', flush=True)
        clf.load_model_if_exists(Path(model_path), force_gpu=False)  # CPU mode to avoid deadlock
        print(f'✅ Worker [{coin}]: Model loaded', flush=True)
    except Exception as e:
        print(f'⚠️  Worker [{coin}]: Could not load model: {e}', flush=True)
        pass

    print(f'📦 Worker [{coin}]: Importing trading logic...', flush=True)
    try:
        module = importlib.import_module('trading_logics.' + trading_logic_module_name)
        TradingLogicClass = module.TradingLogic
    except Exception as e:
        print(f'❌ Worker [{coin}]: Import error for trading logic: {e}', flush=True)
        return []

    trading_logic_instance = TradingLogicClass(clf)
    print(f'✅ Worker [{coin}]: Trading logic ready', flush=True)

    # Process each CSV file for this coin
    print(f'🔄 Worker [{coin}]: Starting CSV processing loop...', flush=True)
    for idx, csv_path in enumerate(csv_files, 1):
        print(f'📄 Worker [{coin}]: Processing CSV {idx}/{len(csv_files)}: {os.path.basename(csv_path)}', flush=True)
        
        df = load_tick_data(csv_path)
        if df is None or df.empty:
            print(f'⚠️  Worker [{coin}]: Empty/invalid CSV: {csv_path}', flush=True)
            continue

        print(f'📊 Worker [{coin}]: Loaded {len(df)} ticks from {os.path.basename(csv_path)}', flush=True)
        
        # Process each timeframe
        for tf in timeframes:
            print(f'⏱️  Worker [{coin}]: Processing timeframe {tf}...', flush=True)
            
            ohlc = tick_to_ohlcv(df, tf)
            if ohlc is None or len(ohlc) < 50:
                print(f'⚠️  Worker [{coin}]: Insufficient data for {tf} (got {len(ohlc) if ohlc is not None else 0} bars)', flush=True)
                continue

            print(f'📈 Worker [{coin}]: Generated {len(ohlc)} OHLC bars for {tf}', flush=True)
            
            # Run backtest via trading logic
            print(f'🚀 Worker [{coin}]: Running backtest for {tf}...', flush=True)
            try:
                summary = trading_logic_instance.run_backtest(coin, tf, ohlc, initial_capital)
                if summary:
                    summary['coin'] = coin
                    summary['timeframe'] = tf
                    summary['csv_file'] = os.path.basename(csv_path)
                    results.append(summary)
                    print(f'✅ Worker [{coin}]: Backtest complete for {tf} - {summary.get("num_trades", 0)} trades', flush=True)
                else:
                    print(f'⚠️  Worker [{coin}]: No summary returned for {tf}', flush=True)
            except Exception as e:
                print(f'❌ Worker [{coin}]: Backtest error for {tf}: {e}', flush=True)
                import traceback
                traceback.print_exc()

    print(f'🏁 Worker [{coin}]: COMPLETED - {len(results)} results', flush=True)
    return results


class BacktestRunner:
    def __init__(self, config_obj, trading_logic, logger: TradeLogger):
        self.config = config_obj
        self.trading_logic = trading_logic
        self.logger = logger

    def process_coin(self, coin):
        # keep compatibility with worker version, use local worker for single-threaded runs
        return process_coin_worker(
            coin,
            self.config.TIMEFRAMES,
            self.config.TRADING_LOGIC,
            str(self.config.MODEL_PATH),
            self.config.BACKTEST_INITIAL_CAPITAL
        )

    def run(self):
        coins = self.config.COINS
        print('BacktestRunner: running for coins:', coins)
        n_workers = min(self.config.NUM_WORKERS, max(1, cpu_count()))
        results = []
        if n_workers <= 1:
            for coin in coins:
                res = self.process_coin(coin)
                if res:
                    results.extend(res)
        else:
            # Use module-level worker to avoid pickling the BacktestRunner instance
            print(f'Starting {n_workers} worker processes...')
            args = [
                (coin, self.config.TIMEFRAMES, self.config.TRADING_LOGIC, str(self.config.MODEL_PATH), self.config.BACKTEST_INITIAL_CAPITAL)
                for coin in coins
            ]
            print(f'Worker args prepared for: {[a[0] for a in args]}')
            
            with Pool(processes=n_workers) as p:
                print(f'Pool created with {n_workers} processes')
                print(f'Submitting {len(args)} tasks...')
                outputs = p.starmap(process_coin_worker, args)
                print(f'All tasks completed!')
            
            for out in outputs:
                if out:
                    results.extend(out)

        # time synchronization step (create combined OHLC for each timeframe across coins)
        print('Attempting synchronization across coins for timeframes: ', self.config.TIMEFRAMES)
        for tf in self.config.TIMEFRAMES:
            coin_ohlc = {}
            for coin in coins:
                csvs = find_csv_for_coin(coin)
                if not csvs:
                    continue
                # take the first CSV for synchronization purpose
                df = load_tick_data(csvs[0])
                if df is None:
                    continue
                ohlc = tick_to_ohlcv(df, tf)
                if ohlc is not None:
                    coin_ohlc[coin] = ohlc

            if len(coin_ohlc) < 2:
                continue
            # compute intersection of timestamps
            idx = None
            for df in coin_ohlc.values():
                if idx is None:
                    idx = df.index
                else:
                    idx = idx.intersection(df.index)

            if idx is None or len(idx) == 0:
                print('No common timestamps for timeframe', tf)
                continue

            print(f"Synchronized index size for {tf}: {len(idx)}")
            # reindex all dfs and create a combined dataframe
            combined = {}
            for coin, df in coin_ohlc.items():
                combined[coin] = df.reindex(idx)

            # If trading logic implements multicoin, call it
            mfunc = getattr(self.trading_logic, 'run_multicoin', None)
            if callable(mfunc):
                print('Calling trading logic multicoin for', tf)
                try:
                    mresults = mfunc(tf, combined, self.config.BACKTEST_INITIAL_CAPITAL)
                    # You can log multicoin results separately
                    if mresults and 'trades' in mresults:
                        for trade in mresults['trades']:
                            self.logger.log_trade('MULTI', tf, trade)
                except Exception as e:
                    print('Multi coin logic error:', e)

        # Create an Excel report
        if results:
            import pandas as pd
            # use only the summary rows for Excel
            summary_rows = [r for r in results if 'trade_record' not in r]
            if summary_rows:
                df = pd.DataFrame(summary_rows)
            else:
                df = pd.DataFrame(results)
            excel_file = Path(self.config.STAT_DIR) / f'bt_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            df.to_excel(excel_file, index=False)
            print('Backtest summary saved to', excel_file)
        # Also log any raw trades returned from workers
        for item in results:
            if isinstance(item, dict) and 'trade_record' in item:
                trade = item['trade_record']
                coin = trade.pop('_coin', None)
                timeframe = trade.pop('_timeframe', None)
                self.logger.log_trade(coin, timeframe, trade)
        
