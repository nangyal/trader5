"""
Debug Zone Recovery - Simple test to verify logic
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_zone_recovery import ZoneRecoveryBacktestEngine


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


# Test on September only
print("=" * 80)
print("ZONE RECOVERY DEBUG TEST - SEPTEMBER 2025")
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

# Test with Zone Recovery
print("\n" + "=" * 80)
print("WITH ZONE RECOVERY")
print("=" * 80)

backtester_zone = ZoneRecoveryBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_zone_recovery=True,
    recovery_zone_size=0.01,  # 1% zones
    max_recovery_zones=5,
    recovery_position_multiplier=0.5
)

results_zone = backtester_zone.run_backtest(df_sept, predictions, probabilities, strength_scores)

# Test WITHOUT Zone Recovery  
print("\n" + "=" * 80)
print("WITHOUT ZONE RECOVERY (for comparison)")
print("=" * 80)

backtester_no_zone = ZoneRecoveryBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    take_profit_multiplier=2.0,
    enable_zone_recovery=False  # Disabled
)

results_no_zone = backtester_no_zone.run_backtest(df_sept, predictions, probabilities, strength_scores)

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"\nWithout Zone Recovery:")
print(f"  Final Capital: ${results_no_zone['final_capital']:,.2f}")
print(f"  Return: {results_no_zone['return_pct']:.2f}%")
print(f"  Total P&L: ${results_no_zone['total_pnl']:,.2f}")
print(f"  Trades: {results_no_zone['total_trades']}")
print(f"  Win Rate: {results_no_zone['win_rate']*100:.2f}%")

print(f"\nWith Zone Recovery:")
print(f"  Final Capital: ${results_zone['final_capital']:,.2f}")
print(f"  Return: {results_zone['return_pct']:.2f}%")
print(f"  Total P&L: ${results_zone['total_pnl']:,.2f}")
print(f"  Main P&L: ${results_zone['main_pnl']:,.2f}")
print(f"  Recovery P&L: ${results_zone['recovery_pnl']:,.2f}")
print(f"  Trades: {results_zone['total_trades']} (Main: {results_zone['main_trades']}, Recovery: {results_zone['recovery_trades']})")
print(f"  Win Rate: {results_zone['win_rate']*100:.2f}%")

improvement = results_zone['return_pct'] - results_no_zone['return_pct']
print(f"\n{'🟢' if improvement > 0 else '🔴'} Improvement: {improvement:+.2f}%")

if results_zone['recovery_pnl'] < 0:
    print(f"\n⚠️  WARNING: Recovery trades are LOSING money!")
    print(f"   Recovery contribution: ${results_zone['recovery_pnl']:,.2f}")
    print(f"   This means the recovery strategy is making things WORSE")
