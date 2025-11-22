"""
Test Fixed Pattern Logic - Skip Bearish Patterns in LONG-ONLY Backtest
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("TESTING FIXED PATTERN LOGIC")
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

# Test 1: OLD LOGIC (before fix)
print("\n" + "="*80)
print("COMPARISON: BEFORE vs AFTER FIX")
print("="*80)

print("\n❌ BEFORE (Old Logic - Long bearish patterns in downtrend):")
print("  - Descending Triangle in downtrend → LONG (WRONG!)")
print("  - Result: -$15,799 P&L, 28.71% win rate")

# Test 2: NEW LOGIC (after fix)
print("\n✅ AFTER (Fixed Logic - Skip bearish patterns):")

engine_fixed = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results_fixed = engine_fixed.run_backtest(df_sept, predictions, probabilities)

if results_fixed:
    print("\n📊 RESULTS WITH FIXED LOGIC:")
    print("="*80)
    
    trades_df = results_fixed['trades_df']
    
    # Check which patterns were actually traded
    print("\nPatterns Actually Traded:")
    traded_patterns = trades_df['pattern'].value_counts()
    for pattern, count in traded_patterns.items():
        print(f"  {pattern}: {count} trades")
    
    # Verify no bearish patterns
    bearish_traded = trades_df[trades_df['pattern'].str.contains('descending|wedge', case=False, na=False)]
    
    if len(bearish_traded) == 0:
        print("\n✅ SUCCESS: No bearish patterns traded (all skipped)")
    else:
        print(f"\n⚠️ WARNING: {len(bearish_traded)} bearish pattern trades found!")
    
    # Calculate improvement
    old_return = 30.61  # From previous test
    new_return = results_fixed['return_pct']
    old_pnl = 3061.33
    new_pnl = results_fixed['total_pnl']
    
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"Return:   {old_return:.2f}% → {new_return:.2f}% ({new_return - old_return:+.2f}%)")
    print(f"P&L:      ${old_pnl:,.2f} → ${new_pnl:,.2f} (${new_pnl - old_pnl:+,.2f})")
    print(f"Max DD:   84.28% → {results_fixed['max_drawdown']:.2f}% ({results_fixed['max_drawdown'] - 84.28:+.2f}%)")
    print(f"Win Rate: 40.56% → {results_fixed['win_rate']*100:.2f}% ({results_fixed['win_rate']*100 - 40.56:+.2f}%)")
    
    # Saved losses
    saved_losses = 15799.44  # Descending triangle losses from before
    print(f"\n💰 Avoided Losses from Descending Triangle: ${saved_losses:,.2f}")
    print(f"💰 Net Improvement: ${new_pnl - old_pnl:,.2f}")

# Test with Hedging
print("\n" + "="*80)
print("TESTING FIXED LOGIC WITH HEDGING")
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
    old_hedging_return = 52.15  # From previous hedging test
    new_hedging_return = results_hedging['return_pct']
    
    print("\n📊 HEDGING RESULTS COMPARISON:")
    print("="*80)
    print(f"Return:   {old_hedging_return:.2f}% → {new_hedging_return:.2f}% ({new_hedging_return - old_hedging_return:+.2f}%)")
    print(f"P&L:      $5,214.67 → ${results_hedging['total_pnl']:,.2f} (${results_hedging['total_pnl'] - 5214.67:+,.2f})")
    print(f"Hedge P&L: $6,430.41 → ${results_hedging['hedge_pnl']:,.2f}")

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
print("\n🎯 Fix Applied: Bearish patterns (descending triangle, wedge) now SKIPPED")
print("   in LONG-ONLY backtest instead of being traded in wrong direction")
