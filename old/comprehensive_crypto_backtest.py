"""
Comprehensive Multi-Coin Multi-Timeframe Backtest with Hedging
Tests all available crypto pairs on all timeframes
Generates Excel report with detailed analysis

PARALLEL EXECUTION: Uses all 28 CPU cores for maximum speed
"""

import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import pandas as pd
import numpy as np
from forex_pattern_classifier import (
    EnhancedForexPatternClassifier,
    PatternStrengthScorer
)
from backtest_with_hedging import HedgingBacktestEngine
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
import traceback

# Configuration
DATA_ROOT = 'data'

# All available coins
# COINS = [
#     'ADAUSDT', 'ATOMUSDT', 'AVAXUSDC', 'BNBUSDC', 'BTCUSDC',
#     'DOGEUSDT', 'DOTUSDT', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC'
# ]

# COINS = [
#     'ETHUSDC'
# ]

COINS = [
    'AAVEUSDT',
    'APTUSDT',
    'ARUSDT',
    'ATOMUSDT',
    'AVAXUSDT',
    'BCHUSDT',
    'BNBUSDT',
    'BTCUSDT',
    'CAKEUSDT',
    'COMPUSDT',
    'DASHUSDT',
    'DOTUSDT',
    'ENSUSDT',
    'ETHUSDT',
    'FILUSDT',
    'FXSUSDT',
    'GMXUSDT',
    'ICPUSDT',
    'INJUSDT',
    'LDOUSDT',
    'LINKUSDT',
    'LTCUSDT',
    'NEARUSDT',
    'NEOUSDT',
    'PENDLEUSDT',
    'SNXUSDT',
    'SUIUSDT',
    'TONUSDT',
    'TWTUSDT',
    'XMRUSDT',
    'XRPUSDT',
    'ZECUSDT',
]

# Timeframes to test (excluding daily/monthly tick data)
# TIMEFRAMES = ['1h', '1min', '5min', '5s', '15min', '15s', '30min', '30s']
TIMEFRAMES = [ '1min', '15s', '30s']
# Hedging parameters
HEDGING_CONFIG = {
    'enable_hedging': True,
    'hedge_threshold': 0.15,      # Activate hedge at 15% drawdown
    'hedge_ratio': 0.5,           # Hedge 50% of exposure
    'dynamic_hedge': True,        # Dynamic hedge adjustment
    'hedge_recovery_threshold': 0.05  # Close hedge at 5% recovery
}

# Backtest parameters - SHARED CAPITAL ACROSS ALL COINS
BACKTEST_CONFIG = {
    'initial_capital': 200,       # TOTAL capital for ALL coins combined
    'risk_per_trade': 0.02,       # 2% risk per trade
    'max_position_pct': 0.10,     # Max 10% of capital per trade
    'max_concurrent_trades': 9,   # Max 9 trades across ALL coins
    'min_probability': 0.7,       # STRICT threshold
    'min_strength': 0.7,          # STRICT threshold
    'use_fixed_risk': True,       # Use FIXED initial capital for risk calculation (no compounding)
}


def load_timeframe_data(coin, timeframe):
    """Load data for specific coin and timeframe"""
    base_path = os.path.join(DATA_ROOT, coin, timeframe)
    
    # Try multiple possible locations (prioritize monthly for full month data)
    possible_paths = [
        os.path.join(base_path, 'monthly'),     # monthly subfolder (full month data)
        os.path.join(base_path, 'daily'),       # daily subfolder (daily chunks)
        base_path,                              # direct path
    ]
    
    csv_files = []
    for path in possible_paths:
        if os.path.exists(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if csv_files:
                file_path = path
                break
    
    if not csv_files:
        return None
    
    # Load and combine all CSV files
    all_dfs = []
    for csv_file in csv_files:
        csv_path = os.path.join(file_path, csv_file)
        
        try:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error loading {csv_path}: {str(e)}")
            continue
    
    if not all_dfs:
        return None
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Detect time column
    time_col = None
    for col in ['time', 'datetime', 'timestamp', 'date']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        # Try parsing timestamps
        if df[time_col].dtype == 'int64':
            # Try milliseconds first (most common)
            df['time'] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
            
            # Check if dates are reasonable (after 2020)
            if df['time'].iloc[0].year < 2020:
                # Probably seconds, not milliseconds
                df['time'] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
        else:
            df['time'] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['time'])
        df.set_index('time', inplace=True)
    
    # Sort by time
    df = df.sort_index()
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return None
    
    return df


def resample_tick_data(df, target_timeframe):
    """Resample tick data to target timeframe"""
    if 'price' in df.columns and 'qty' in df.columns:
        # Tick data format
        ohlcv = df.resample(target_timeframe).agg({
            'price': ['first', 'max', 'min', 'last'],
            'qty': 'sum'
        })
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv = ohlcv.dropna()
        return ohlcv
    
    return df


def run_single_backtest(coin, timeframe, df, classifier):
    """Run backtest for single coin/timeframe combination"""
    
    print(f"\n  Processing {coin} - {timeframe}...")
    print(f"    Candles: {len(df)}")
    print(f"    Date range: {df.index[0]} to {df.index[-1]}")
    
    # Generate predictions
    predictions, probabilities = classifier.predict(df)
    
    # Calculate pattern strength
    scorer = PatternStrengthScorer()
    pattern_strengths = []
    
    for i in range(len(predictions)):
        if predictions[i] != 'no_pattern':
            strength = scorer.calculate_pattern_strength(df, predictions[i], i, window=50)
            pattern_strengths.append(strength)
        else:
            pattern_strengths.append(0.0)
    
    pattern_strengths = np.array(pattern_strengths)
    
    # Count quality signals
    prob_max = np.max(probabilities, axis=1)
    quality_signals = 0
    
    for i in range(len(predictions)):
        if predictions[i] != 'no_pattern':
            if prob_max[i] >= BACKTEST_CONFIG['min_probability'] and \
               pattern_strengths[i] >= BACKTEST_CONFIG['min_strength']:
                quality_signals += 1
    
    print(f"    Quality signals: {quality_signals}")
    
    # Run backtest WITH hedging
    engine = HedgingBacktestEngine(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        risk_per_trade=BACKTEST_CONFIG['risk_per_trade'],
        max_position_pct=BACKTEST_CONFIG['max_position_pct'],
        enable_hedging=HEDGING_CONFIG['enable_hedging'],
        hedge_threshold=HEDGING_CONFIG['hedge_threshold'],
        hedge_ratio=HEDGING_CONFIG['hedge_ratio'],
        dynamic_hedge=HEDGING_CONFIG['dynamic_hedge'],
        hedge_recovery_threshold=HEDGING_CONFIG['hedge_recovery_threshold'],
        use_fixed_risk=BACKTEST_CONFIG.get('use_fixed_risk', False)
    )
    
    results = engine.run_backtest(
        df, 
        predictions, 
        probabilities,
        pattern_strength_scores=pattern_strengths
    )
    
    if results and results.get('total_trades', 0) > 0:
        print(f"    ✅ Trades: {results.get('total_trades', 0)}, " +
              f"Win Rate: {results.get('win_rate', 0)*100:.1f}%, " +
              f"Return: {results.get('return_pct', 0):.2f}%")
        
        return {
            'coin': coin,
            'timeframe': timeframe,
            'candles': len(df),
            'date_start': str(df.index[0]),
            'date_end': str(df.index[-1]),
            'price_start': df['close'].iloc[0],
            'price_end': df['close'].iloc[-1],
            'price_change_pct': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
            'quality_signals': quality_signals,
            'total_trades': results.get('total_trades', 0),
            'main_trades': results.get('main_trades', 0),
            'hedge_trades': results.get('hedge_trades', 0),
            'winning_trades': results.get('winning_trades', 0),
            'losing_trades': results.get('losing_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'total_pnl': results.get('total_pnl', 0),
            'return_pct': results.get('return_pct', 0),
            'final_capital': results.get('final_capital', BACKTEST_CONFIG['initial_capital']),
            'max_drawdown': results.get('max_drawdown', 0),
            'profit_factor': results.get('profit_factor', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'avg_win': results.get('avg_win', 0),
            'avg_loss': results.get('avg_loss', 0),
            'largest_win': results.get('largest_win', 0),
            'largest_loss': results.get('largest_loss', 0),
            'status': 'Completed'
        }
    else:
        print(f"    ⚠️  No trades executed")
        
        return {
            'coin': coin,
            'timeframe': timeframe,
            'candles': len(df),
            'date_start': str(df.index[0]),
            'date_end': str(df.index[-1]),
            'price_start': df['close'].iloc[0],
            'price_end': df['close'].iloc[-1],
            'price_change_pct': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
            'quality_signals': quality_signals,
            'total_trades': 0,
            'main_trades': 0,
            'hedge_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'return_pct': 0,
            'final_capital': BACKTEST_CONFIG['initial_capital'],
            'max_drawdown': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'status': 'No Trades'
        }


# Global classifier for each worker process
_process_classifier = None

def init_worker():
    """Initialize worker process with loaded model (called once per process)"""
    global _process_classifier
    _process_classifier = EnhancedForexPatternClassifier()
    _process_classifier.load_model('enhanced_forex_pattern_model.pkl')
    print(f"  Worker {os.getpid()}: Model loaded ✓")


def process_combination_wrapper(args):
    """Wrapper function for parallel processing"""
    coin, timeframe, current, total = args
    
    try:
        print(f"\n[{current}/{total}] Processing {coin} - {timeframe} (Worker: {os.getpid()})")
        
        # Use pre-loaded classifier from worker initialization
        global _process_classifier
        
        # Load data
        df = load_timeframe_data(coin, timeframe)
        
        if df is None or len(df) < 100:
            print(f"  ⚠️  {coin} {timeframe}: Insufficient data (need min 100 candles)")
            return {
                'coin': coin,
                'timeframe': timeframe,
                'candles': len(df) if df is not None else 0,
                'status': 'Insufficient Data',
                'total_trades': 0,
                'return_pct': 0
            }
        
        # Run backtest with pre-loaded classifier
        result = run_single_backtest(coin, timeframe, df, _process_classifier)
        return result
        
    except Exception as e:
        print(f"  ❌ {coin} {timeframe}: Error - {str(e)}")
        traceback.print_exc()
        return {
            'coin': coin,
            'timeframe': timeframe,
            'status': f'Error: {str(e)}',
            'total_trades': 0,
            'return_pct': 0
        }


def main():
    print("\n" + "="*100)
    print("COMPREHENSIVE MULTI-COIN MULTI-TIMEFRAME BACKTEST WITH HEDGING")
    print("="*100)
    
    # Detect CPU count
    n_cpus = cpu_count()
    print(f"\n� System Info:")
    print(f"  Available CPUs: {n_cpus}")
    print(f"  Using all {n_cpus} cores for parallel processing")
    
    print(f"\n📋 Configuration:")
    print(f"  Coins: {len(COINS)}")
    print(f"  Timeframes: {len(TIMEFRAMES)}")
    print(f"  Total combinations: {len(COINS) * len(TIMEFRAMES)}")
    print(f"\n  💰 SHARED CAPITAL MODEL:")
    print(f"  Total Capital: ${BACKTEST_CONFIG['initial_capital']:,} (shared across ALL coins)")
    print(f"  Max Position Size: {BACKTEST_CONFIG['max_position_pct']*100}% = ${BACKTEST_CONFIG['initial_capital'] * BACKTEST_CONFIG['max_position_pct']:.2f} per trade")
    print(f"  Max Concurrent Trades: {BACKTEST_CONFIG['max_concurrent_trades']} (across ALL coins)")
    print(f"  Risk per Trade: {BACKTEST_CONFIG['risk_per_trade']*100}%")
    print(f"  Min Probability: {BACKTEST_CONFIG['min_probability']}")
    print(f"  Min Strength: {BACKTEST_CONFIG['min_strength']}")
    print(f"\n  Hedging Enabled: {HEDGING_CONFIG['enable_hedging']}")
    print(f"  Hedge Threshold: {HEDGING_CONFIG['hedge_threshold']*100}%")
    print(f"  Hedge Ratio: {HEDGING_CONFIG['hedge_ratio']*100}%")
    
    # Create all combinations
    combinations = []
    current = 0
    for coin in COINS:
        for timeframe in TIMEFRAMES:
            current += 1
            combinations.append((coin, timeframe, current, len(COINS) * len(TIMEFRAMES)))
    
    print(f"\n🚀 Starting parallel backtest on {n_cpus} CPUs...")
    print(f"  This will be MUCH faster than sequential execution!")
    print(f"  Loading model once per worker process...")
    
    # Run parallel processing with model pre-loading
    with Pool(processes=n_cpus, initializer=init_worker) as pool:
        all_results = pool.map(process_combination_wrapper, combinations)
    
    print(f"\n✅ All {len(combinations)} combinations completed!")
    
    # Create comprehensive report
    print(f"\n{'='*100}")
    print("GENERATING COMPREHENSIVE EXCEL REPORT")
    print(f"{'='*100}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Fill missing values with 0
    numeric_cols = ['total_trades', 'winning_trades', 'losing_trades', 'win_rate', 
                    'total_pnl', 'return_pct', 'max_drawdown', 'profit_factor', 
                    'sharpe_ratio', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
                    'main_trades', 'hedge_trades', 'quality_signals', 'candles']
    
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = results_df[col].fillna(0)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = f'comprehensive_backtest_report_{timestamp}.xlsx'
    
    # Create Excel writer
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Sheet 1: Full Results
        results_df.to_excel(writer, sheet_name='Full Results', index=False)
        
        # Sheet 2: Summary by Coin
        if len(results_df[results_df['total_trades'] > 0]) > 0:
            coin_summary = results_df[results_df['total_trades'] > 0].groupby('coin').agg({
                'total_trades': 'sum',
                'winning_trades': 'sum',
                'losing_trades': 'sum',
                'win_rate': 'mean',
                'return_pct': 'sum',
                'total_pnl': 'sum',
                'max_drawdown': 'max',
                'profit_factor': 'mean',
                'sharpe_ratio': 'mean'
            }).round(2)
            coin_summary.to_excel(writer, sheet_name='Summary by Coin')
        
        # Sheet 3: Summary by Timeframe
        if len(results_df[results_df['total_trades'] > 0]) > 0:
            tf_summary = results_df[results_df['total_trades'] > 0].groupby('timeframe').agg({
                'total_trades': 'sum',
                'winning_trades': 'sum',
                'losing_trades': 'sum',
                'win_rate': 'mean',
                'return_pct': 'sum',
                'total_pnl': 'sum',
                'max_drawdown': 'max',
                'profit_factor': 'mean',
                'sharpe_ratio': 'mean'
            }).round(2)
            tf_summary.to_excel(writer, sheet_name='Summary by Timeframe')
        
        # Sheet 4: Top Performers
        if len(results_df[results_df['total_trades'] > 0]) > 0:
            top_performers = results_df[results_df['total_trades'] > 0].nlargest(20, 'return_pct')[
                ['coin', 'timeframe', 'total_trades', 'win_rate', 'return_pct', 
                 'max_drawdown', 'profit_factor', 'sharpe_ratio']
            ]
            top_performers.to_excel(writer, sheet_name='Top 20 Performers', index=False)
        
        # Sheet 5: Worst Performers
        if len(results_df[results_df['total_trades'] > 0]) > 0:
            worst_performers = results_df[results_df['total_trades'] > 0].nsmallest(20, 'return_pct')[
                ['coin', 'timeframe', 'total_trades', 'win_rate', 'return_pct', 
                 'max_drawdown', 'profit_factor', 'sharpe_ratio']
            ]
            worst_performers.to_excel(writer, sheet_name='Worst 20 Performers', index=False)
        
        # Sheet 6: Statistics
        total_trades = results_df.get('total_trades', pd.Series([0])).sum()
        winning_trades = results_df.get('winning_trades', pd.Series([0])).sum()
        losing_trades = results_df.get('losing_trades', pd.Series([0])).sum()
        total_pnl = results_df.get('total_pnl', pd.Series([0])).sum()
        return_pct_mean = results_df.get('return_pct', pd.Series([0])).mean()
        return_pct_max = results_df.get('return_pct', pd.Series([0])).max()
        return_pct_min = results_df.get('return_pct', pd.Series([0])).min()
        max_dd_mean = results_df.get('max_drawdown', pd.Series([0])).mean()
        
        stats_data = {
            'Metric': [
                'Total Combinations Tested',
                'Combinations with Trades',
                'Combinations with No Trades',
                'Total Trades Executed',
                'Total Winning Trades',
                'Total Losing Trades',
                'Overall Win Rate',
                'Total P&L',
                'Average Return per Combination',
                'Best Return',
                'Worst Return',
                'Average Max Drawdown'
            ],
            'Value': [
                len(results_df),
                len(results_df[results_df.get('total_trades', 0) > 0]) if 'total_trades' in results_df.columns else 0,
                len(results_df[results_df.get('total_trades', 0) == 0]) if 'total_trades' in results_df.columns else len(results_df),
                total_trades,
                winning_trades,
                losing_trades,
                f"{(winning_trades / total_trades * 100):.2f}%" if total_trades > 0 else '0%',
                f"${total_pnl:,.2f}",
                f"{return_pct_mean:.2f}%",
                f"{return_pct_max:.2f}%",
                f"{return_pct_min:.2f}%",
                f"{max_dd_mean:.2f}%"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Overall Statistics', index=False)
    
    print(f"\n✅ Excel report saved: {excel_filename}")
    
    # Print summary
    print(f"\n{'='*100}")
    print("BACKTEST SUMMARY")
    print(f"{'='*100}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Combinations Tested: {len(results_df)}")
    
    # Safe access to columns
    total_trades_col = results_df.get('total_trades', pd.Series([0] * len(results_df)))
    trades_executed = total_trades_col.sum()
    
    print(f"  Combinations with Trades: {(total_trades_col > 0).sum()}")
    print(f"  Total Trades: {int(trades_executed)}")
    
    if trades_executed > 0 and 'winning_trades' in results_df.columns and 'total_pnl' in results_df.columns:
        winning_trades_sum = results_df['winning_trades'].sum()
        overall_win_rate = winning_trades_sum / trades_executed
        print(f"  Overall Win Rate: {overall_win_rate*100:.2f}%")
        print(f"  Total P&L: ${results_df['total_pnl'].sum():,.2f}")
        
        trades_mask = total_trades_col > 0
        if trades_mask.any() and 'return_pct' in results_df.columns:
            print(f"  Average Return: {results_df[trades_mask]['return_pct'].mean():.2f}%")
            
            # Best performer
            best_idx = results_df[trades_mask]['return_pct'].idxmax()
            best = results_df.loc[best_idx]
            print(f"\n🏆 Best Performer:")
            print(f"  {best['coin']} - {best['timeframe']}")
            print(f"  Return: +{best['return_pct']:.2f}%")
            print(f"  Trades: {int(best['total_trades'])}, Win Rate: {best.get('win_rate', 0)*100:.1f}%")
            
            # Worst performer
            worst_idx = results_df[trades_mask]['return_pct'].idxmin()
            worst = results_df.loc[worst_idx]
            print(f"\n⚠️  Worst Performer:")
            print(f"  {worst['coin']} - {worst['timeframe']}")
            print(f"  Return: {worst['return_pct']:.2f}%")
            print(f"  Trades: {int(worst['total_trades'])}, Win Rate: {worst.get('win_rate', 0)*100:.1f}%")
    else:
        print(f"\n  ⚠️  NO TRADES executed across all combinations")
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE BACKTEST COMPLETE")
    print(f"{'='*100}\n")
    
    return excel_filename


if __name__ == "__main__":
    excel_file = main()
    print(f"\n📄 Full report: {excel_file}")
