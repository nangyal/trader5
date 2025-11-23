#!/usr/bin/env python3
"""
Test Partial TP with HIGH reaching target but CLOSE below
This tests the CRITICAL FIX for using HIGH instead of CLOSE
"""
import config
from trading_logic import TradingLogic

print("="*60)
print("PARTIAL TP - HIGH vs CLOSE TEST")
print("="*60)

logic = TradingLogic(config, initial_capital=7046.58)

# Create test trade
trade = {
    'coin': 'BTCUSDT',
    'pattern': 'ascending_triangle',
    'entry_price': 100000.0,
    'stop_loss': 99500.0,
    'take_profit': 102000.0,
    'position_size': 0.0236,
    'position_value': 2360.0,
    'probability': 0.80,
    'strength': 0.80,
    'timeframe': '1min',
    'direction': 'long',
    'status': 'open',
    'pnl': 0.0,
    'trailing_stop': None,
    'partial_closed': 0.0,
    'breakeven_activated': False,
}

print(f"\n💼 Trade Setup:")
print(f"Entry: ${trade['entry_price']:.2f}")
print(f"Partial TP levels:")
for level in config.PARTIAL_TP['levels']:
    target_price = trade['entry_price'] * (1 + level['pct'])
    print(f"  {level['pct']*100:.1f}%: ${target_price:.2f} (close {level['close_ratio']*100:.0f}%)")

# SCENARIO 1: High reaches TP, Close below
print(f"\n{'='*60}")
print("SCENARIO 1: HIGH reaches 1.5%, CLOSE only 1.2%")
print("="*60)

# Partial TP 1.5% = $101,500
# Candle: High=$101,600 (reaches!), Close=$101,200 (below!)
candle_1 = {
    'open': 100500.0,
    'high': 101600.0,   # 1.6% profit - REACHES 1.5% target!
    'low': 100500.0,
    'close': 101200.0,  # 1.2% profit - BELOW 1.5% target!
}

partial_tp_price = trade['entry_price'] * 1.015  # $101,500
close_profit = (candle_1['close'] - trade['entry_price']) / trade['entry_price'] * 100
high_profit = (candle_1['high'] - trade['entry_price']) / trade['entry_price'] * 100

print(f"\n📊 Candle data:")
print(f"   High: ${candle_1['high']:.2f} ({high_profit:.1f}% profit)")
print(f"   Close: ${candle_1['close']:.2f} ({close_profit:.1f}% profit)")
print(f"   Partial TP target: ${partial_tp_price:.2f} (1.5%)")

should_close, exit_price, reason, ratio = logic.check_trade_exit(trade, candle_1)

print(f"\n✅ Result:")
print(f"   Should close: {should_close}")
print(f"   Exit price: ${exit_price:.2f}" if exit_price else "   Exit price: None")
print(f"   Reason: {reason}")
print(f"   Ratio: {ratio:.2f}" if ratio else "   Ratio: None")

if should_close and reason == 'partial_tp_1.5%':
    print(f"\n✅ CORRECT! Triggered using HIGH")
    print(f"   Exit price ${exit_price:.2f} should be ${partial_tp_price:.2f}")
    if abs(exit_price - partial_tp_price) < 0.01:
        print(f"   ✅ Exit price CORRECT!")
    else:
        print(f"   ❌ Exit price WRONG! Should be partial_tp_price, not current_price")
elif not should_close:
    print(f"\n❌ WRONG! Should have triggered (HIGH reached 1.5%)")
else:
    print(f"\n❌ WRONG! Wrong reason: {reason}")

# SCENARIO 2: Neither HIGH nor CLOSE reach target
print(f"\n{'='*60}")
print("SCENARIO 2: Neither HIGH nor CLOSE reach 1.5%")
print("="*60)

# Reset trade
trade['partial_closed'] = 0.0

candle_2 = {
    'open': 100500.0,
    'high': 101400.0,   # 1.4% profit - BELOW 1.5%
    'low': 100500.0,
    'close': 101100.0,  # 1.1% profit - BELOW 1.5%
}

close_profit2 = (candle_2['close'] - trade['entry_price']) / trade['entry_price'] * 100
high_profit2 = (candle_2['high'] - trade['entry_price']) / trade['entry_price'] * 100

print(f"\n📊 Candle data:")
print(f"   High: ${candle_2['high']:.2f} ({high_profit2:.1f}% profit)")
print(f"   Close: ${candle_2['close']:.2f} ({close_profit2:.1f}% profit)")
print(f"   Partial TP target: ${partial_tp_price:.2f} (1.5%)")

should_close2, exit_price2, reason2, ratio2 = logic.check_trade_exit(trade, candle_2)

print(f"\n✅ Result:")
print(f"   Should close: {should_close2}")
print(f"   Reason: {reason2}")

if not should_close2:
    print(f"\n✅ CORRECT! Not triggered (HIGH below target)")
else:
    print(f"\n❌ WRONG! Should not trigger (neither HIGH nor CLOSE reached 1.5%)")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
