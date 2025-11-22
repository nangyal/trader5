"""
Test Zone Recovery V2 - Simplified and Correct
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_zone_recovery_v2 import ZoneRecoveryBacktestEngine


def resample_ticks_to_hourly(csv_file):
    """Resample tick/trade data to hourly candlesticks"""
    print(f"Loading tick data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('datetime')
    
    print("Resampling to 1-hour candlesticks...")
    ohlcv = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum'
    })
    
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = ohlcv.dropna()
    
    print(f"Resampled data shape: {ohlcv.shape}")
    return ohlcv


print("=" * 80)
print("ZONE RECOVERY V2 TEST - SEPTEMBER 2025")
print("=" * 80)

df_sept = resample_ticks_to_hourly('data/DOGEUSDT-trades-2025-09.csv')

classifier = EnhancedForexPatternClassifier()
classifier.load_model('enhanced_forex_pattern_model.pkl')

predictions, probabilities = classifier.predict(df_sept)

strength_scores = np.array([
    PatternStrengthScorer.calculate_pattern_strength(df_sept, predictions[i], i) 
    if predictions[i] != 'no_pattern' else 0.0
    for i in range(len(df_sept))
])

# Test WITHOUT Zone Recovery (baseline)
print("\n" + "=" * 80)
print("BASELINE - WITHOUT ZONE RECOVERY")
print("=" * 80)

backtester_baseline = ZoneRecoveryBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_zone_recovery=False
)

results_baseline = backtester_baseline.run_backtest(df_sept, predictions, probabilities, strength_scores)

# Test WITH Zone Recovery
print("\n" + "=" * 80)
print("WITH ZONE RECOVERY V2")
print("=" * 80)

backtester_zone = ZoneRecoveryBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_zone_recovery=True,
    recovery_zone_size=0.01,  # 1% zones
    max_recovery_zones=5,
    recovery_position_size=200  # $200 per zone
)

results_zone = backtester_zone.run_backtest(df_sept, predictions, probabilities, strength_scores)

# Compare
print("\n" + "=" * 80)
print("COMPARISON - SEPTEMBER 2025")
print("=" * 80)

print(f"\n{'Metric':<30} {'Baseline':<20} {'Zone Recovery':<20} {'Difference':<15}")
print("-" * 85)

metrics = [
    ('Final Capital', 'final_capital', '$'),
    ('Return %', 'return_pct', '%'),
    ('Total P&L', 'total_pnl', '$'),
    ('Total Trades', 'total_trades', ''),
    ('Win Rate %', 'win_rate', '%'),
    ('Profit Factor', 'profit_factor', ''),
    ('Max Drawdown %', 'max_drawdown', '%'),
    ('Sharpe Ratio', 'sharpe_ratio', ''),
]

for label, key, unit in metrics:
    base_val = results_baseline[key]
    zone_val = results_zone[key]
    
    if unit == '%' and key == 'win_rate':
        base_val *= 100
        zone_val *= 100
    
    diff = zone_val - base_val
    
    if unit == '$':
        print(f"{label:<30} ${base_val:>17,.2f} ${zone_val:>17,.2f} ${diff:>13,.2f}")
    elif unit == '%':
        print(f"{label:<30} {base_val:>18.2f}% {zone_val:>18.2f}% {diff:>13.2f}%")
    else:
        print(f"{label:<30} {base_val:>19.2f} {zone_val:>19.2f} {diff:>14.2f}")

print("-" * 85)

improvement = results_zone['return_pct'] - results_baseline['return_pct']
print(f"\n{'🟢' if improvement > 0 else '🔴'} Return Improvement: {improvement:+.2f}%")

if results_zone.get('recovery_pnl', 0) != 0:
    print(f"\n📊 Zone Recovery Details:")
    print(f"   Recovery Trades: {results_zone['recovery_trades']}")
    print(f"   Recovery P&L: ${results_zone['recovery_pnl']:,.2f}")
    print(f"   Main P&L: ${results_zone['main_pnl']:,.2f}")
    
    if results_zone['recovery_pnl'] > 0:
        print(f"   ✅ Recovery trades are PROFITABLE")
    else:
        print(f"   ❌ Recovery trades are LOSING money")

print("\n" + "=" * 80)
