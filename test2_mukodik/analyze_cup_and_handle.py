"""
Analyze Cup & Handle Pattern Performance
Determine if it should be kept or removed
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("CUP & HANDLE PATTERN ANALYSIS")
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

# Current backtest (with Cup & Handle)
print("\n" + "="*80)
print("CURRENT BACKTEST (ALL PATTERNS)")
print("="*80)

engine_current = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results_current = engine_current.run_backtest(df_sept, predictions, probabilities)

# Analyze Cup & Handle trades
trades_df = results_current['trades_df']
cup_trades = trades_df[trades_df['pattern'].str.contains('cup', case=False, na=False)]

print("\n" + "="*80)
print("CUP & HANDLE DETAILED ANALYSIS")
print("="*80)

if len(cup_trades) > 0:
    print(f"\nTotal Trades: {len(cup_trades)}")
    print(f"Winning Trades: {(cup_trades['pnl'] > 0).sum()}")
    print(f"Losing Trades: {(cup_trades['pnl'] < 0).sum()}")
    print(f"Win Rate: {(cup_trades['pnl'] > 0).sum() / len(cup_trades) * 100:.2f}%")
    print(f"Total P&L: ${cup_trades['pnl'].sum():,.2f}")
    print(f"Average P&L: ${cup_trades['pnl'].mean():,.2f}")
    print(f"Average Win: ${cup_trades[cup_trades['pnl'] > 0]['pnl'].mean():,.2f}")
    print(f"Average Loss: ${abs(cup_trades[cup_trades['pnl'] < 0]['pnl'].mean()):,.2f}")
    
    print(f"\nExit Reasons:")
    print(cup_trades['exit_reason'].value_counts())
    
    # Trend when Cup & Handle appears
    cup_indices = cup_trades['entry_index'].values
    trends = []
    for idx in cup_indices:
        if idx >= 20:
            closes = df_sept['close'].values[idx-20:idx]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            trend = 'up' if slope > 0 else 'down'
            trends.append(trend)
    
    if trends:
        trend_counts = pd.Series(trends).value_counts()
        print(f"\nTrend Analysis:")
        print(f"  Uptrend: {trend_counts.get('up', 0)} ({trend_counts.get('up', 0)/len(trends)*100:.1f}%)")
        print(f"  Downtrend: {trend_counts.get('down', 0)} ({trend_counts.get('down', 0)/len(trends)*100:.1f}%)")
    
    print(f"\nBest 5 Trades:")
    best = cup_trades.nlargest(5, 'pnl')[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']]
    print(best)
    
    print(f"\nWorst 5 Trades:")
    worst = cup_trades.nsmallest(5, 'pnl')[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']]
    print(worst)

# Test WITHOUT Cup & Handle
print("\n" + "="*80)
print("BACKTEST WITHOUT CUP & HANDLE")
print("="*80)

# Filter out Cup & Handle predictions
predictions_filtered = np.array([p if 'cup' not in p.lower() else 'no_pattern' for p in predictions])

engine_filtered = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results_filtered = engine_filtered.run_backtest(df_sept, predictions_filtered, probabilities)

# Comparison
print("\n" + "="*80)
print("COMPARISON: WITH vs WITHOUT CUP & HANDLE")
print("="*80)

comparison = pd.DataFrame({
    'Metric': [
        'Final Capital',
        'Total Return (%)',
        'Total P&L',
        'Total Trades',
        'Win Rate (%)',
        'Max Drawdown (%)',
        'Profit Factor',
        'Sharpe Ratio',
        'Cup & Handle P&L'
    ],
    'With Cup & Handle': [
        f"${results_current['final_capital']:,.2f}",
        f"{results_current['return_pct']:.2f}%",
        f"${results_current['total_pnl']:,.2f}",
        results_current['total_trades'],
        f"{results_current['win_rate']*100:.2f}%",
        f"{results_current['max_drawdown']:.2f}%",
        f"{results_current['profit_factor']:.2f}",
        f"{results_current['sharpe_ratio']:.2f}",
        f"${cup_trades['pnl'].sum():,.2f}" if len(cup_trades) > 0 else "$0.00"
    ],
    'Without Cup & Handle': [
        f"${results_filtered['final_capital']:,.2f}",
        f"{results_filtered['return_pct']:.2f}%",
        f"${results_filtered['total_pnl']:,.2f}",
        results_filtered['total_trades'],
        f"{results_filtered['win_rate']*100:.2f}%",
        f"{results_filtered['max_drawdown']:.2f}%",
        f"{results_filtered['profit_factor']:.2f}",
        f"{results_filtered['sharpe_ratio']:.2f}",
        "$0.00"
    ]
})

print("\n", comparison.to_string(index=False))

# Calculate improvement
return_diff = results_filtered['return_pct'] - results_current['return_pct']
pnl_diff = results_filtered['total_pnl'] - results_current['total_pnl']

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if return_diff > 0:
    print(f"✅ BETTER WITHOUT Cup & Handle")
    print(f"   Return improvement: {return_diff:+.2f}%")
    print(f"   P&L improvement: ${pnl_diff:+,.2f}")
    print(f"\n💡 RECOMMENDATION: REMOVE Cup & Handle pattern")
else:
    print(f"✅ BETTER WITH Cup & Handle")
    print(f"   Return better by: {abs(return_diff):.2f}%")
    print(f"   P&L better by: ${abs(pnl_diff):,.2f}")
    print(f"\n💡 RECOMMENDATION: KEEP Cup & Handle pattern")

# Show what Cup & Handle contributed
if len(cup_trades) > 0:
    contribution_pct = (cup_trades['pnl'].sum() / results_current['total_pnl'] * 100) if results_current['total_pnl'] != 0 else 0
    print(f"\nCup & Handle Contribution: {contribution_pct:.2f}% of total P&L")
    print(f"Cup & Handle Drag: ${cup_trades['pnl'].sum():,.2f} ({len(cup_trades)} trades, {(cup_trades['pnl'] > 0).sum() / len(cup_trades) * 100:.1f}% WR)")

print("\n" + "="*80)
