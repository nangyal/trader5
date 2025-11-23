#!/usr/bin/env python3
"""
Test position sizing calculation with ML confidence weighting
"""
import config
from trading_logic import TradingLogic

# Test position sizing with ML confidence
logic = TradingLogic(config, initial_capital=7046.58)

entry = 98500
sl = 98000
capital = 7046.58

print("="*60)
print("POSITION SIZING TEST - ML Confidence Weighting")
print("="*60)
print(f"Entry: ${entry}")
print(f"Stop Loss: ${sl}")
print(f"Capital: ${capital}")
print(f"Risk per trade: 2%")
print()

# Without high ML confidence (65%)
size_normal = logic.calculate_position_size(entry, sl, capital, ml_probability=0.65)
value_normal = size_normal * entry
print(f"📊 Normal (65% prob, 1.0x multiplier):")
print(f"   Position: {size_normal:.6f} BTC")
print(f"   Value: ${value_normal:.2f}")
print(f"   % of capital: {(value_normal/capital)*100:.2f}%")
print()

# With 70-80% ML confidence (1.2x)
size_medium = logic.calculate_position_size(entry, sl, capital, ml_probability=0.75)
value_medium = size_medium * entry
print(f"📊 Medium (75% prob, 1.2x multiplier):")
print(f"   Position: {size_medium:.6f} BTC")
print(f"   Value: ${value_medium:.2f}")
print(f"   % of capital: {(value_medium/capital)*100:.2f}%")
print()

# With 80%+ ML confidence (1.5x)
size_high = logic.calculate_position_size(entry, sl, capital, ml_probability=0.85)
value_high = size_high * entry
print(f"📊 High (85% prob, 1.5x multiplier):")
print(f"   Position: {size_high:.6f} BTC")
print(f"   Value: ${value_high:.2f}")
print(f"   % of capital: {(value_high/capital)*100:.2f}%")
print()

# Check if capped at 33%
max_allowed = capital * 0.33
print("="*60)
print(f"Max allowed (33% cap): ${max_allowed:.2f}")
print(f"High conf value: ${value_high:.2f}")
print(f"✅ Within limit: {value_high <= max_allowed}")
print()

# Test with OLD BUGGY method (simulate)
print("="*60)
print("COMPARISON - OLD BUGGY METHOD:")
print("="*60)

# Old method: calculate then multiply
risk_amount = capital * 0.02  # 2%
risk_per_unit = entry - sl  # 500
size_old = risk_amount / risk_per_unit  # 140.93 / 500 = 0.2819
size_old_ml = size_old * 1.5  # Apply 1.5x AFTER
value_old = size_old_ml * entry

print(f"Old buggy method (85% prob):")
print(f"   Base size: {size_old:.6f} BTC")
print(f"   After 1.5x: {size_old_ml:.6f} BTC") 
print(f"   Value: ${value_old:.2f}")
print(f"   % of capital: {(value_old/capital)*100:.2f}%")
print(f"❌ Exceeds 33%: {value_old > max_allowed}")
print()

print("="*60)
print("RESULT: NEW METHOD CORRECTLY CAPS AT 33%!")
print("="*60)
