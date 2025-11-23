#!/usr/bin/env python3
"""
Test ATR (Average True Range) calculation
"""
import pandas as pd
import numpy as np
import config
from trading_logic import TradingLogic

print("="*60)
print("ATR CALCULATION TEST")
print("="*60)

logic = TradingLogic(config, initial_capital=7046.58)

# Create sample candle data with known True Range values
data = {
    'high':   [100, 102, 105, 103, 108, 110, 107, 109, 112, 115, 118, 120, 122, 119, 121],
    'low':    [98,  99,  102, 100, 104, 107, 105, 106, 109, 112, 115, 117, 118, 116, 118],
    'close':  [99,  101, 104, 101, 107, 109, 106, 108, 111, 114, 117, 119, 120, 117, 120],
}

recent_data = pd.DataFrame(data)

# Calculate ATR manually for verification
high = recent_data['high']
low = recent_data['low']
close_prev = recent_data['close'].shift(1)

tr1 = high - low
tr2 = abs(high - close_prev)
tr3 = abs(low - close_prev)

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

print("\n📊 Sample Data (last 5 candles):")
print(recent_data.tail())

print("\n📐 True Range Calculation:")
print(f"TR (last 5): {tr.tail().values}")
print(f"ATR-14 (last value): {atr.iloc[-1]:.4f}")

# Test calculate_pattern_targets with ATR filter
current_candle = {
    'high': recent_data['high'].iloc[-1],
    'low': recent_data['low'].iloc[-1],
    'close': recent_data['close'].iloc[-1],
}

sl, tp, direction, params = logic.calculate_pattern_targets(
    'ascending_triangle',
    entry_price=120,
    candle_data=current_candle,
    recent_data=recent_data
)

print(f"\n🎯 Pattern Target Results:")
print(f"Direction: {direction}")
print(f"Stop Loss: ${sl:.2f}")
print(f"Take Profit: ${tp:.2f}")

# Calculate ATR percentage
atr_pct = (atr.iloc[-1] / 120) if not pd.isna(atr.iloc[-1]) else 0
min_atr_pct = config.VOLATILITY_FILTER['min_atr_pct']

print(f"\n📈 Volatility Check:")
print(f"ATR: {atr.iloc[-1]:.4f}")
print(f"ATR % (decimal): {atr_pct:.4f} ({atr_pct*100:.2f}%)")
print(f"Min ATR % (decimal): {min_atr_pct:.4f} ({min_atr_pct*100:.2f}%)")
print(f"ATR Filter: {'PASS ✅' if atr_pct >= min_atr_pct else 'FAIL ❌'}")

if direction == 'skip' and atr_pct < min_atr_pct:
    print(f"\n✅ CORRECT: Trade skipped due to low volatility")
elif direction != 'skip' and atr_pct >= min_atr_pct:
    print(f"\n✅ CORRECT: Trade allowed (sufficient volatility)")
else:
    print(f"\n❓ Check ATR filter logic")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
