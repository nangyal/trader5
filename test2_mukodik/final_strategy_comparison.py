"""
Final Comparison: LONG-ONLY vs LONG+SHORT Strategy
September 2025 DOGEUSDT
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("FINAL STRATEGY COMPARISON")
print("="*80)

# Load data
df_ticks = pd.read_csv('data/DOGEUSDT-trades-2025-09.csv')
df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='ms')
df_ticks = df_ticks.set_index('datetime')

df_sept = df_ticks.resample('1H').agg({
    'price': ['first', 'max', 'min', 'last'],
    'qty': 'sum'
})
df_sept.columns = ['open', 'high', 'low', 'close', 'volume']
df_sept = df_sept.dropna()

# Predictions
classifier = EnhancedForexPatternClassifier()
classifier.load_model('enhanced_forex_pattern_model.pkl')
predictions, probabilities = classifier.predict(df_sept)

# Test 1: LONG+SHORT (current code)
print("\n" + "="*80)
print("TEST 1: LONG+SHORT STRATEGY")
print("="*80)

engine_bidirectional = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results_bidirectional = engine_bidirectional.run_backtest(df_sept, predictions, probabilities)

# Summary
print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)

comparison = {
    'Strategy': ['LONG-ONLY (skip bearish)', 'LONG+SHORT (trade bearish)'],
    'Return (%)': [259.31, results_bidirectional['return_pct']],
    'Final Capital': [35930.78, results_bidirectional['final_capital']],
    'Total P&L': [25930.78, results_bidirectional['total_pnl']],
    'Win Rate (%)': [50.81, results_bidirectional['win_rate'] * 100],
    'Max Drawdown (%)': [64.39, results_bidirectional['max_drawdown']],
    'Profit Factor': [1.28, results_bidirectional['profit_factor']],
    'Sharpe Ratio': [1.02, results_bidirectional['sharpe_ratio']],
    'Total Trades': [185, results_bidirectional['total_trades']]
}

df_comparison = pd.DataFrame(comparison)
print("\n", df_comparison.to_string(index=False))

# Decision
long_only_return = 259.31
bidirectional_return = results_bidirectional['return_pct']
difference = bidirectional_return - long_only_return

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if bidirectional_return > long_only_return:
    print(f"✅ WINNER: LONG+SHORT Strategy")
    print(f"   Improvement: {difference:+.2f}%")
    print(f"   Reason: SHORT positions on bearish patterns are profitable")
else:
    print(f"✅ WINNER: LONG-ONLY Strategy")
    print(f"   Better by: {abs(difference):.2f}%")
    print(f"   Reason: SHORT positions on bearish patterns are NOT profitable")

# Detailed pattern analysis
trades_df = results_bidirectional['trades_df']

print("\n" + "="*80)
print("PATTERN-BY-PATTERN BREAKDOWN")
print("="*80)

for pattern in ['ascending_triangle', 'descending_triangle', 'cup_and_handle']:
    pattern_trades = trades_df[trades_df['pattern'] == pattern]
    if len(pattern_trades) > 0:
        long_trades = pattern_trades[pattern_trades['direction'] == 'long']
        short_trades = pattern_trades[pattern_trades['direction'] == 'short']
        
        print(f"\n{pattern.upper()}:")
        print(f"  Total: {len(pattern_trades)} trades, ${pattern_trades['pnl'].sum():,.2f} P&L")
        
        if len(long_trades) > 0:
            long_wr = (long_trades['pnl'] > 0).sum() / len(long_trades) * 100
            print(f"  LONG:  {len(long_trades)} trades, ${long_trades['pnl'].sum():,.2f} P&L, {long_wr:.1f}% WR")
        
        if len(short_trades) > 0:
            short_wr = (short_trades['pnl'] > 0).sum() / len(short_trades) * 100
            print(f"  SHORT: {len(short_trades)} trades, ${short_trades['pnl'].sum():,.2f} P&L, {short_wr:.1f}% WR")

# Recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if bidirectional_return > long_only_return:
    print("✅ Use LONG+SHORT Strategy (current code)")
    print("   Keep SHORT positions for bearish patterns")
else:
    print("✅ Use LONG-ONLY Strategy")
    print("   Skip bearish patterns (Descending Triangle, Wedge)")
    print("\n   To revert to LONG-ONLY, change line 81-86 in backtest_with_hedging.py:")
    print("   Replace 'direction = short' with 'return 0, 0, skip, None'")

print("\n" + "="*80)
