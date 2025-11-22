"""
Backtest modul - CSV tick adatok alapján történő kereskedés szimuláció
Multiprocessing-et használ több coin párhuzamos feldolgozására
"""
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, Manager
import traceback
import warnings
warnings.filterwarnings('ignore')

# Imports
import config
from trading_logic import TradingLogic


def resample_tick_to_timeframe(df_tick, timeframe):
    """
    Átszámol tick adatokat OHLCV formátumra adott timeframe-re
    
    Args:
        df_tick: DataFrame tick adatokkal (time, price, qty)
        timeframe: str, pl. '15s', '30s', '1min'
        
    Returns:
        DataFrame: OHLCV formátumban
    """
    # Timeframe konverzió pandas offset-re
    tf_map = {
        '15s': '15S',
        '30s': '30S',
        '1min': '1T',
        '5min': '5T'
    }
    
    freq = tf_map.get(timeframe, '1T')
    
    # Resample
    ohlcv = df_tick.resample(freq).agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum'
    })
    
    # Rename columns
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = ohlcv.dropna()
    
    return ohlcv


def load_timeframe_data(coin, timeframe, data_path_template):
    """
    Betölt CSV adatokat egy coinra és timeframe-re
    
    Args:
        coin: str, pl. 'BTCUSDT'
        timeframe: str, pl. '15s', '30s', '1min'
        data_path_template: str, path sablon
        
    Returns:
        DataFrame vagy None
    """
    try:
        # Get CSV file path
        csv_path = Path(data_path_template.format(coin=coin, timeframe=timeframe))
        
        if not csv_path.exists():
            print(f"  ❌ {coin} {timeframe}: Nincs adat ({csv_path})")
            return None
        
        print(f"  📥 {coin} {timeframe}: CSV betöltése...")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"    ❌ Hiba: {csv_path.name} - {e}")
            return None
        
        # Parse time column
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        
        if not time_cols:
            print(f"  ❌ {coin}: Nincs time oszlop")
            return None
        
        time_col = time_cols[0]
        
        # Convert timestamp
        if df[time_col].dtype == 'int64':
            # Try milliseconds first
            df['time'] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
            
            # Check if reasonable (after 2020)
            if df['time'].iloc[0].year < 2020:
                # Probably seconds
                df['time'] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
        else:
            df['time'] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Drop invalid
        df = df.dropna(subset=['time'])
        df = df.set_index('time').sort_index()
        
        # Check required columns (price, qty)
        if 'price' not in df.columns or 'qty' not in df.columns:
            # Try to find price column
            price_cols = [c for c in df.columns if 'price' in c.lower() or 'close' in c.lower()]
            qty_cols = [c for c in df.columns if 'qty' in c.lower() or 'amount' in c.lower() or 'volume' in c.lower()]
            
            if price_cols and qty_cols:
                df = df.rename(columns={price_cols[0]: 'price', qty_cols[0]: 'qty'})
            else:
                print(f"  ❌ {coin}: Nincs price/qty oszlop")
                return None
        
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        
        print(f"  ✅ {coin} {timeframe}: {len(df):,} adatpont betöltve ({df.index[0]} - {df.index[-1]})")
        
        return df
        
    except Exception as e:
        print(f"  ❌ {coin} {timeframe}: Hiba - {e}")
        traceback.print_exc()
        return None


def run_single_coin_backtest(args):
    """
    Egy coin backtestje (párhuzamos futáshoz)
    
    Args:
        args: tuple (coin, timeframes, shared_state, worker_id)
        
    Returns:
        dict: Eredmények
    """
    coin, timeframes, data_path_template, model_path, worker_id = args
    
    try:
        print(f"\n[Worker {worker_id}] 🚀 {coin} backtest indítása...")
        
        # Load ML model
        from old.forex_pattern_classifier import EnhancedForexPatternClassifier, PatternStrengthScorer
        
        classifier = EnhancedForexPatternClassifier()
        try:
            classifier.load_model(str(model_path))
            print(f"  ✅ Model betöltve: {model_path}")
        except Exception as e:
            print(f"  ❌ Model betöltési hiba: {e}")
            return {
                'coin': coin,
                'status': 'model_load_error',
                'total_trades': 0,
                'return_pct': 0.0
            }
        
        scorer = PatternStrengthScorer()
        
        # Initialize trading logic
        trading = TradingLogic(config)
        
        # Process each timeframe
        timeframe_results = {}
        
        for timeframe in timeframes:
            print(f"  📊 {coin} - {timeframe} timeframe feldolgozása...")
            
            # Load timeframe-specific data
            df_tick = load_timeframe_data(coin, timeframe, data_path_template)
            
            if df_tick is None or len(df_tick) < 1000:
                print(f"    ⚠️  Nincs elég adat")
                continue
            
            # Resample tick data to OHLCV
            df_ohlcv = resample_tick_to_timeframe(df_tick, timeframe)
            
            if len(df_ohlcv) < 100:
                print(f"    ⚠️  Kevés candle ({len(df_ohlcv)})")
                continue
            
            print(f"    Candles: {len(df_ohlcv):,} ({df_ohlcv.index[0]} - {df_ohlcv.index[-1]})")
            
            # Run predictions
            print(f"    🔮 Prediction futtatása {len(df_ohlcv):,} candle-ra...")
            try:
                predictions, probabilities = classifier.predict(df_ohlcv)
                print(f"    ✅ Prediction kész")
            except Exception as e:
                print(f"    ❌ Prediction hiba: {e}")
                continue
            
            # Calculate pattern strength (simplified for performance)
            pattern_strengths = []
            for i in range(len(predictions)):
                if predictions[i] != 'no_pattern':
                    # OLD2 optimization: Use probability as strength proxy (10x faster)
                    strength = probabilities[i][np.argmax(probabilities[i])]
                    pattern_strengths.append(strength)
                else:
                    pattern_strengths.append(0.0)
            
            pattern_strengths = np.array(pattern_strengths)
            
            # Count quality signals
            prob_max = np.max(probabilities, axis=1)
            quality_signals = 0
            
            for i in range(len(predictions)):
                if predictions[i] != 'no_pattern':
                    if prob_max[i] >= config.PATTERN_FILTERS['min_probability'] and \
                       pattern_strengths[i] >= config.PATTERN_FILTERS['min_strength']:
                        quality_signals += 1
            
            print(f"    Quality signals: {quality_signals}")
            
            # Simulate trading
            for i in range(len(df_ohlcv)):
                current_candle = df_ohlcv.iloc[i]
                pattern = predictions[i]
                pattern_prob = prob_max[i]
                pattern_strength = pattern_strengths[i]
                
                # OPTIMIZED: Engedélyezzük a descending patternt is (LONG strategy downtrend-ben)
                # Az old2-ben ez 42% win rate-et hozott!
                
                # Check if we should open trade
                if trading.should_open_trade(pattern, pattern_prob, pattern_strength):
                    # Get recent data for trend calc
                    recent_data = df_ohlcv.iloc[max(0, i-30):i+1]
                    
                    entry_price = current_candle['close']
                    
                    # Calculate targets
                    sl, tp, direction, params = trading.calculate_pattern_targets(
                        pattern, entry_price, current_candle, recent_data
                    )
                    
                    if direction == 'skip':
                        continue
                    
                    # Calculate position size
                    position_size = trading.calculate_position_size(
                        entry_price, sl, trading.capital
                    )
                    
                    if position_size <= 0:
                        continue
                    
                    # Open trade
                    trade = trading.open_trade(
                        coin=coin,
                        pattern=pattern,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        position_size=position_size,
                        probability=pattern_prob,
                        strength=pattern_strength,
                        timeframe=timeframe,
                        entry_time=current_candle.name
                    )
                
                # Check active trades for exit
                for trade in list(trading.active_trades):
                    should_close, exit_price, exit_reason, partial_ratio = trading.check_trade_exit(trade, current_candle)
                    
                    if should_close:
                        pnl = trading.close_trade(trade, exit_price, exit_reason, current_candle.name, partial_ratio or 1.0)
                
                # Decrement cooldown counter
                if trading.cooldown_until_candle > 0:
                    trading.cooldown_until_candle -= 1
            
            # Store timeframe results
            timeframe_results[timeframe] = {
                'candles': len(df_ohlcv),
                'quality_signals': quality_signals,
                'trades': len(trading.closed_trades)
            }
        
        # Get final statistics
        stats = trading.get_statistics()
        
        print(f"\n  ✅ {coin} befejezve:")
        print(f"     Total trades: {stats['total_trades']}")
        print(f"     Win rate: {stats['win_rate']*100:.1f}%")
        print(f"     Return: {stats['return_pct']:.2f}%")
        print(f"     Final capital: ${stats['final_capital']:.2f}")
        
        return {
            'coin': coin,
            'status': 'completed',
            'timeframes': timeframe_results,
            **stats
        }
        
    except Exception as e:
        print(f"\n  ❌ {coin} hiba: {e}")
        traceback.print_exc()
        return {
            'coin': coin,
            'status': f'error: {str(e)}',
            'total_trades': 0,
            'return_pct': 0.0
        }


def run_backtest(coins, timeframes, num_workers=None):
    """
    Fő backtest függvény - multiprocessing
    
    Args:
        coins: list of coin strings
        timeframes: list of timeframe strings
        num_workers: int vagy None (auto)
        
    Returns:
        list: Eredmények minden coinra
    """
    print("\n" + "="*80)
    print("🚀 BACKTEST INDÍTÁSA (MULTIPROCESSING)")
    print("="*80)
    
    print(f"\nCoinok: {len(coins)}")
    for coin in coins:
        print(f"  • {coin}")
    
    print(f"\nTimeframes: {timeframes}")
    print(f"Kezdő tőke: ${config.BACKTEST_INITIAL_CAPITAL}")
    
    # CPU count
    if num_workers is None:
        num_workers = min(cpu_count(), len(coins))
    
    print(f"Workers: {num_workers}")
    
    # Prepare args
    args_list = []
    for idx, coin in enumerate(coins):
        args_list.append((
            coin,
            timeframes,
            config.BACKTEST_DATA_PATH_TEMPLATE,
            config.MODEL_PATH,
            idx + 1
        ))
    
    # Run parallel
    print(f"\n{'='*80}")
    print("PÁRHUZAMOS FELDOLGOZÁS INDÍTÁSA")
    print(f"{'='*80}\n")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_coin_backtest, args_list)
    
    print(f"\n{'='*80}")
    print("✅ BACKTEST BEFEJEZVE")
    print(f"{'='*80}\n")
    
    # Print summary
    successful = [r for r in results if r['status'] == 'completed']
    
    if successful:
        total_trades = sum(r['total_trades'] for r in successful)
        avg_return = np.mean([r['return_pct'] for r in successful])
        
        print(f"Sikeres backtestek: {len(successful)}/{len(coins)}")
        print(f"Összes trade: {total_trades}")
        print(f"Átlagos hozam: {avg_return:.2f}%")
    else:
        print("⚠️  Nincs sikeres backtest")
    
    return results


if __name__ == '__main__':
    # Test run
    config.ensure_dirs()
    
    results = run_backtest(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        num_workers=config.NUM_WORKERS
    )
    
    print("\n=== EREDMÉNYEK ===")
    for result in results:
        print(f"\n{result['coin']}: {result['status']}")
        if result['status'] == 'completed':
            print(f"  Trades: {result['total_trades']}, Return: {result['return_pct']:.2f}%")
