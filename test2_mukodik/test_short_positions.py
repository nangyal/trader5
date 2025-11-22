"""
Test SHORT Position Implementation for Bearish Patterns
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("TESTING SHORT POSITION FOR BEARISH PATTERNS")
print("="*80)

# Load September data
print("\n📂 Loading September 2025 data...")
df_ticks = pd.read_csv('data/DOGEUSDT-trades-2025-09.csv')
df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='ms')
df_ticks = df_ticks.set_index('datetime')

df_sept = df_ticks.resample('1H').agg({
    'price': ['first', 'max', 'min', 'last'],
    'qty': 'sum'
})
df_sept.columns = ['open', 'high', 'low', 'close', 'volume']
df_sept = df_sept.dropna()

print(f"✓ Loaded {len(df_sept)} hourly candles")

# Load model and predict
print("\n🤖 Making predictions...")
classifier = EnhancedForexPatternClassifier()
classifier.load_model('enhanced_forex_pattern_model.pkl')
predictions, probabilities = classifier.predict(df_sept)

# Count patterns
print("\n📊 Pattern Distribution:")
pattern_counts = pd.Series(predictions).value_counts()
for pattern, count in pattern_counts.items():
    print(f"  {pattern}: {count}")

# Test with LONG + SHORT
print("\n" + "="*80)
print("BACKTEST WITH LONG + SHORT POSITIONS")
print("="*80)

engine = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results = engine.run_backtest(df_sept, predictions, probabilities)

if results:
    print("\n📊 DETAILED ANALYSIS:")
    print("="*80)
    
    trades_df = results['trades_df']
    
    # Check positions by direction
    print("\nTrades by Direction:")
    direction_counts = trades_df['direction'].value_counts()
    for direction, count in direction_counts.items():
        dir_trades = trades_df[trades_df['direction'] == direction]
        dir_pnl = dir_trades['pnl'].sum()
        dir_win_rate = (dir_trades['pnl'] > 0).sum() / len(dir_trades) * 100
        print(f"  {direction.upper()}: {count} trades, ${dir_pnl:,.2f} P&L, {dir_win_rate:.2f}% win rate")
    
    # Pattern-specific with direction
    print("\n📊 Pattern Performance by Direction:")
    for pattern in trades_df['pattern'].unique():
        pattern_trades = trades_df[trades_df['pattern'] == pattern]
        
        long_trades = pattern_trades[pattern_trades['direction'] == 'long']
        short_trades = pattern_trades[pattern_trades['direction'] == 'short']
        
        print(f"\n{pattern}:")
        if len(long_trades) > 0:
            print(f"  LONG: {len(long_trades)} trades, ${long_trades['pnl'].sum():,.2f} P&L")
        if len(short_trades) > 0:
            print(f"  SHORT: {len(short_trades)} trades, ${short_trades['pnl'].sum():,.2f} P&L")
    
    # Descending Triangle specific
    desc_trades = trades_df[trades_df['pattern'].str.contains('descending', case=False, na=False)]
    if len(desc_trades) > 0:
        print("\n" + "="*80)
        print("DESCENDING TRIANGLE ANALYSIS (NOW SHORT!)")
        print("="*80)
        print(f"Total Trades: {len(desc_trades)}")
        print(f"Direction: {desc_trades['direction'].unique()}")
        print(f"Winning Trades: {(desc_trades['pnl'] > 0).sum()}")
        print(f"Losing Trades: {(desc_trades['pnl'] < 0).sum()}")
        print(f"Win Rate: {(desc_trades['pnl'] > 0).sum() / len(desc_trades) * 100:.2f}%")
        print(f"Total P&L: ${desc_trades['pnl'].sum():,.2f}")
        print(f"Average P&L: ${desc_trades['pnl'].mean():,.2f}")
        
        print(f"\nExit Reasons:")
        print(desc_trades['exit_reason'].value_counts())
        
        print(f"\nBest 5 Trades:")
        best = desc_trades.nlargest(5, 'pnl')[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']]
        print(best)
        
        print(f"\nWorst 5 Trades:")
        worst = desc_trades.nsmallest(5, 'pnl')[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']]
        print(worst)
    
    # Compare with LONG-ONLY
    print("\n" + "="*80)
    print("COMPARISON: LONG-ONLY vs LONG+SHORT")
    print("="*80)
    
    long_only_return = 259.31  # From previous test
    long_short_return = results['return_pct']
    
    print(f"\nLONG-ONLY (skip bearish): {long_only_return:.2f}%")
    print(f"LONG+SHORT (trade bearish): {long_short_return:.2f}%")
    print(f"Difference: {long_short_return - long_only_return:+.2f}%")
    
    if long_short_return > long_only_return:
        print(f"\n✅ SHORT positions IMPROVED results by {long_short_return - long_only_return:.2f}%")
    else:
        print(f"\n⚠️ SHORT positions DECREASED results by {long_only_return - long_short_return:.2f}%")

# Test with Hedging
print("\n" + "="*80)
print("TESTING LONG+SHORT WITH HEDGING")
print("="*80)

engine_hedging = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=True,
    hedge_threshold=0.15,
    hedge_ratio=0.5
)

results_hedging = engine_hedging.run_backtest(df_sept, predictions, probabilities)

if results_hedging:
    print("\n📊 HEDGING RESULTS:")
    print(f"Return: {results_hedging['return_pct']:.2f}%")
    print(f"Total P&L: ${results_hedging['total_pnl']:,.2f}")
    print(f"Max Drawdown: {results_hedging['max_drawdown']:.2f}%")

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
