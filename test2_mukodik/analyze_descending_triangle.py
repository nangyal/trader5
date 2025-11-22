"""
Analyze why Descending Triangle always loses money
Deep dive into pattern logic and market conditions
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine

print("="*80)
print("ANALYZING DESCENDING TRIANGLE PATTERN")
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

print(f"✓ Made {len(predictions)} predictions")

# Analyze descending triangle patterns
print("\n" + "="*80)
print("DESCENDING TRIANGLE PATTERN ANALYSIS")
print("="*80)

descending_indices = [i for i, p in enumerate(predictions) if 'descending' in p.lower()]
print(f"\nTotal Descending Triangle signals: {len(descending_indices)}")

if len(descending_indices) == 0:
    print("⚠️ No descending triangle patterns found!")
    exit()

# Analyze each descending triangle signal
print("\nAnalyzing each signal...")
print("-"*80)

for idx, i in enumerate(descending_indices[:10]):  # First 10 for analysis
    pattern = predictions[i]
    prob = probabilities[i].max()
    
    # Get trend
    if i >= 20:
        closes = df_sept['close'].values[i-20:i]
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        trend = 'UP' if slope > 0 else 'DOWN'
        trend_strength = abs(slope) / closes[-1] * 100
    else:
        trend = 'UNKNOWN'
        trend_strength = 0
    
    current_price = df_sept.iloc[i]['close']
    
    # Calculate SL/TP
    sl_pct = 0.015
    tp_pct = 0.03
    stop_loss = current_price * (1 - sl_pct)
    take_profit = current_price * (1 + tp_pct)
    
    # Check what happened after signal
    max_profit_pct = 0
    max_loss_pct = 0
    hit_sl = False
    hit_tp = False
    
    for j in range(i+1, min(i+50, len(df_sept))):  # Check next 50 hours
        high = df_sept.iloc[j]['high']
        low = df_sept.iloc[j]['low']
        
        if low <= stop_loss:
            hit_sl = True
            break
        if high >= take_profit:
            hit_tp = True
            break
        
        max_profit_pct = max(max_profit_pct, (high - current_price) / current_price * 100)
        max_loss_pct = min(max_loss_pct, (low - current_price) / current_price * 100)
    
    print(f"\nSignal #{idx+1} at index {i}:")
    print(f"  Pattern: {pattern}")
    print(f"  Probability: {prob:.3f}")
    print(f"  Entry Price: ${current_price:.5f}")
    print(f"  Trend: {trend} ({trend_strength:.3f}% strength)")
    print(f"  Stop Loss: ${stop_loss:.5f} (-{sl_pct*100}%)")
    print(f"  Take Profit: ${take_profit:.5f} (+{tp_pct*100}%)")
    print(f"  Result: {'✓ TP HIT' if hit_tp else '✗ SL HIT' if hit_sl else '⏱️ PENDING'}")
    print(f"  Max Profit: +{max_profit_pct:.2f}%")
    print(f"  Max Loss: {max_loss_pct:.2f}%")

# Now check the LOGIC issue
print("\n" + "="*80)
print("PATTERN LOGIC ANALYSIS")
print("="*80)

print("\n🔍 Checking pattern entry logic...")
print("\nCurrent logic (line 81 in backtest_with_hedging.py):")
print("  if (trend == 'up' and is_bullish) or (trend == 'down' and is_bearish):")
print("      direction = 'long'")
print("\nDescending Triangle = BEARISH pattern")
print("Logic says: Enter LONG when trend is DOWN and pattern is BEARISH")

# Count trend alignment
trends_when_descending = []
for i in descending_indices:
    if i >= 20:
        closes = df_sept['close'].values[i-20:i]
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        trend = 'up' if slope > 0 else 'down'
        trends_when_descending.append(trend)

trend_counts = pd.Series(trends_when_descending).value_counts()
print(f"\nTrend when Descending Triangle appears:")
print(f"  Uptrend: {trend_counts.get('up', 0)} ({trend_counts.get('up', 0)/len(trends_when_descending)*100:.1f}%)")
print(f"  Downtrend: {trend_counts.get('down', 0)} ({trend_counts.get('down', 0)/len(trends_when_descending)*100:.1f}%)")

print("\n⚠️ THE PROBLEM:")
print("="*80)
print("Descending Triangle is a BEARISH continuation pattern")
print("It appears when price is FALLING (downtrend)")
print("Current logic enters LONG during DOWNTREND")
print("This is BACKWARDS - we're buying into a falling market!")
print("\n💡 SOLUTION:")
print("Descending Triangle should be traded SHORT, not LONG")
print("OR skip descending triangle in LONG-ONLY backtest")
print("OR wait for breakout confirmation (not just pattern detection)")

# Run backtest to confirm
print("\n" + "="*80)
print("BACKTEST CONFIRMATION")
print("="*80)

engine = HedgingBacktestEngine(
    initial_capital=10000,
    risk_per_trade=0.02,
    enable_hedging=False
)

results = engine.run_backtest(df_sept, predictions, probabilities)

if results:
    print("\nPattern-specific results:")
    print(results['pattern_performance'])
    
    # Detailed analysis
    trades_df = results['trades_df']
    desc_trades = trades_df[trades_df['pattern'].str.contains('descending', case=False, na=False)]
    
    print(f"\n📊 Descending Triangle Trade Analysis:")
    print(f"  Total Trades: {len(desc_trades)}")
    print(f"  Winning Trades: {(desc_trades['pnl'] > 0).sum()}")
    print(f"  Losing Trades: {(desc_trades['pnl'] < 0).sum()}")
    print(f"  Win Rate: {(desc_trades['pnl'] > 0).sum() / len(desc_trades) * 100:.2f}%")
    print(f"  Total P&L: ${desc_trades['pnl'].sum():,.2f}")
    print(f"  Average P&L: ${desc_trades['pnl'].mean():,.2f}")
    
    # Check exit reasons
    print(f"\n  Exit Reasons:")
    print(desc_trades['exit_reason'].value_counts())
    
    # Show worst trades
    print(f"\n  Worst 5 Trades:")
    worst_trades = desc_trades.nsmallest(5, 'pnl')[['entry_index', 'entry_price', 'exit_price', 'exit_reason', 'pnl']]
    print(worst_trades)

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
