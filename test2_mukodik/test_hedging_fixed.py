"""
Test Fixed Hedging Implementation
Compare before/after bug fixes
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("TESTING FIXED HEDGING IMPLEMENTATION")
print("="*80)

# Load September tick data and resample
print("\n📂 Loading September 2025 tick data...")
df_ticks = pd.read_csv('data/DOGEUSDT-trades-2025-09.csv')
df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='ms')
df_ticks = df_ticks.set_index('datetime')

print(f"✓ Loaded {len(df_ticks):,} tick records")

# Resample to 1-hour
print("\n⏱️ Resampling to 1-hour candlesticks...")
df_sept = df_ticks.resample('1H').agg({
    'price': ['first', 'max', 'min', 'last'],
    'qty': 'sum'
})
df_sept.columns = ['open', 'high', 'low', 'close', 'volume']
df_sept = df_sept.dropna()

print(f"✓ Created {len(df_sept)} hourly candles")

# Load model and predict
print("\n🤖 Loading model and making predictions...")
classifier = EnhancedForexPatternClassifier()
classifier.load_model('enhanced_forex_pattern_model.pkl')

predictions, probabilities = classifier.predict(df_sept)

print(f"✓ Made {len(predictions)} predictions")
print(f"✓ Found {(predictions != 'no_pattern').sum()} pattern signals")

# Test 1: Baseline (no hedging)
print("\n" + "="*80)
print("TEST 1: BASELINE (NO HEDGING)")
print("="*80)

engine_baseline = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_hedging=False
)

results_baseline = engine_baseline.run_backtest(df_sept, predictions, probabilities)

# Test 2: Fixed Hedging
print("\n" + "="*80)
print("TEST 2: FIXED HEDGING")
print("="*80)

engine_hedging = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_hedging=True,
    hedge_threshold=0.15,
    hedge_ratio=0.5
)

results_hedging = engine_hedging.run_backtest(df_sept, predictions, probabilities)

# Compare results
print("\n" + "="*80)
print("COMPARISON: BASELINE vs FIXED HEDGING")
print("="*80)

comparison = pd.DataFrame({
    'Metric': [
        'Final Capital',
        'Total Return (%)',
        'Total P&L',
        'Main Trades',
        'Hedge Trades',
        'Win Rate (%)',
        'Max Drawdown (%)',
        'Profit Factor',
        'Sharpe Ratio'
    ],
    'Baseline': [
        f"${results_baseline['final_capital']:,.2f}",
        f"{results_baseline['return_pct']:.2f}%",
        f"${results_baseline['total_pnl']:,.2f}",
        results_baseline['main_trades'],
        0,
        f"{results_baseline['win_rate']*100:.2f}%",
        f"{results_baseline['max_drawdown']:.2f}%",
        f"{results_baseline['profit_factor']:.2f}",
        f"{results_baseline['sharpe_ratio']:.2f}"
    ],
    'Fixed Hedging': [
        f"${results_hedging['final_capital']:,.2f}",
        f"{results_hedging['return_pct']:.2f}%",
        f"${results_hedging['total_pnl']:,.2f}",
        results_hedging['main_trades'],
        results_hedging['hedge_trades'],
        f"{results_hedging['win_rate']*100:.2f}%",
        f"{results_hedging['max_drawdown']:.2f}%",
        f"{results_hedging['profit_factor']:.2f}",
        f"{results_hedging['sharpe_ratio']:.2f}"
    ]
})

print("\n", comparison.to_string(index=False))

# Calculate improvement
return_improvement = results_hedging['return_pct'] - results_baseline['return_pct']
dd_improvement = results_baseline['max_drawdown'] - results_hedging['max_drawdown']

print("\n" + "="*80)
print("IMPROVEMENTS WITH FIXED HEDGING")
print("="*80)
print(f"Return Improvement: {return_improvement:+.2f}%")
print(f"Drawdown Reduction: {dd_improvement:.2f}%")
print(f"Hedge P&L Contribution: ${results_hedging['hedge_pnl']:,.2f}")
print(f"Hedge Trade Count: {results_hedging['hedge_trades']}")

# Analyze hedge trades
if results_hedging['hedge_trades'] > 0:
    hedge_df = pd.DataFrame(engine_hedging.hedge_trades)
    print("\n" + "="*80)
    print("HEDGE TRADE ANALYSIS")
    print("="*80)
    print(f"\nTotal Hedge Trades: {len(hedge_df)}")
    print(f"Winning Hedge Trades: {(hedge_df['pnl'] > 0).sum()}")
    print(f"Losing Hedge Trades: {(hedge_df['pnl'] < 0).sum()}")
    print(f"Hedge Win Rate: {(hedge_df['pnl'] > 0).sum() / len(hedge_df) * 100:.2f}%")
    print(f"\nAverage Hedge P&L: ${hedge_df['pnl'].mean():,.2f}")
    print(f"Total Hedge P&L: ${hedge_df['pnl'].sum():,.2f}")
    
    print("\nHedge Exit Reasons:")
    print(hedge_df['exit_reason'].value_counts())
    
    print("\nFirst 5 Hedge Trades:")
    print(hedge_df[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']].head())

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
