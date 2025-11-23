#!/usr/bin/env python3
"""
Test exit logic - SL/TP order and Partial TP
"""
import config
from trading_logic import TradingLogic
import pandas as pd

print("="*60)
print("EXIT LOGIC TEST")
print("="*60)

logic = TradingLogic(config, initial_capital=7046.58)

# Create test trade
trade = {
    'coin': 'BTCUSDT',
    'pattern': 'ascending_triangle',
    'entry_price': 98500.0,
    'stop_loss': 98000.0,  # -0.5%
    'take_profit': 100470.0,  # +2.0%
    'position_size': 0.0236,
    'position_value': 2325.0,
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

print(f"Entry: ${trade['entry_price']}")
print(f"Stop Loss: ${trade['stop_loss']} (-0.5%)")
print(f"Take Profit: ${trade['take_profit']} (+2.0%)")
print()

# Test 1: Volatile candle hits BOTH SL and TP
print("="*60)
print("TEST 1: Volatile Candle (hits both SL and TP)")
print("="*60)

# Temporarily disable Partial TP to test Regular TP/SL
original_partial_enable = config.PARTIAL_TP['enable']
config.PARTIAL_TP['enable'] = False
logic_no_partial = TradingLogic(config, initial_capital=7046.58)

# Bullish candle: Open 98400, High 100600 (TP!), Low 97800 (SL!), Close 99000
candle_bullish = {
    'open': 98400.0,
    'high': 100600.0,  # Above TP
    'low': 97800.0,    # Below SL
    'close': 99000.0
}

print(f"Bullish Candle: O:{candle_bullish['open']} H:{candle_bullish['high']} L:{candle_bullish['low']} C:{candle_bullish['close']}")

should_close, exit_price, reason, ratio = logic_no_partial.check_trade_exit(trade, candle_bullish)
print(f"Result: should_close={should_close}, exit_price={exit_price}, reason={reason}")
print(f"Expected: TP hit first (bullish candle)")
print(f"✅ CORRECT!" if reason == 'take_profit' else f"❌ WRONG! Got {reason}")
print()

# Bearish candle: Open 98600, Low 97800 (SL!), High 100600 (TP!), Close 98200
candle_bearish = {
    'open': 98600.0,
    'high': 100600.0,  # Above TP
    'low': 97800.0,    # Below SL
    'close': 98200.0
}

# Reset trade
trade['status'] = 'open'
trade['partial_closed'] = 0.0

print(f"Bearish Candle: O:{candle_bearish['open']} H:{candle_bearish['high']} L:{candle_bearish['low']} C:{candle_bearish['close']}")

should_close, exit_price, reason, ratio = logic_no_partial.check_trade_exit(trade, candle_bearish)
print(f"Result: should_close={should_close}, exit_price={exit_price}, reason={reason}")
print(f"Expected: SL hit first (bearish candle)")
print(f"✅ CORRECT!" if reason == 'stop_loss' else f"❌ WRONG! Got {reason}")
print()

# Test 2: Partial TP sequence
print("="*60)
print("TEST 2: Partial TP Sequence")
print("="*60)

# Re-enable Partial TP for Test 2
config.PARTIAL_TP['enable'] = original_partial_enable

# Create NEW logic instance for Test 2 with fresh state
logic_partial = TradingLogic(config, initial_capital=7046.58)

# Reset trade
trade['status'] = 'open'
trade['partial_closed'] = 0.0
trade['position_size'] = 0.0236
trade['trailing_stop'] = None  # Reset trailing stop

# Candle 1: +1.5% profit
candle_1 = {
    'open': 99000.0,
    'high': 99978.0,  # +1.5% from entry
    'low': 99000.0,
    'close': 99978.0
}

print(f"Candle 1: Price +1.5% (${candle_1['close']})")
should_close, exit_price, reason, ratio = logic_partial.check_trade_exit(trade, candle_1)
print(f"Result: {reason}, ratio={ratio:.2f}")
print(f"Expected: partial_tp_1.5%, ratio=0.50")
print(f"✅ CORRECT!" if reason == 'partial_tp_1.5%' and abs(ratio - 0.50) < 0.01 else f"❌ WRONG!")
print(f"Partial closed: {trade['partial_closed']:.2f}")
print()

# Simulate partial close
if should_close and ratio < 1.0:
    # Close partial
    close_size = trade['position_size'] * ratio
    trade['position_size'] -= close_size
    print(f"Closed {ratio*100:.0f}% → Remaining: {trade['position_size']:.6f} BTC")
print()

# Candle 2: +2.5% profit
# Target price = 98500 × 1.025 = 100962.50
candle_2 = {
    'open': 99978.0,
    'high': 100963.0,  # +2.5% from entry (rounded up to ensure >= 2.5%)
    'low': 99978.0,
    'close': 100963.0
}

print(f"Candle 2: Price +2.5% (${candle_2['close']})")
should_close, exit_price, reason, ratio = logic_partial.check_trade_exit(trade, candle_2)
print(f"Result: {reason}, ratio={ratio:.2f}")
print(f"Expected: partial_tp_2.5%, ratio=0.25 (cumulative 75%, incremental 25%)")
print(f"Partial closed: {trade['partial_closed']:.2f} (cumulative)")

# Old buggy: partial_closed=0.50 < close_ratio=0.30 → FALSE → SKIP ❌
# New fixed: partial_closed=0.50 < close_ratio=0.75 → TRUE → EXECUTE ✅
print(f"✅ CORRECT!" if should_close and abs(ratio - 0.25) < 0.01 else f"❌ WRONG! Should be 0.25 (75%-50%)")
print()

# Candle 3: +4.0% profit
# Target price = 98500 × 1.04 = 102460
candle_3 = {
    'open': 100963.0,
    'high': 102460.0,  # +4.0% from entry
    'low': 100963.0,
    'close': 102460.0
}

print(f"Candle 3: Price +4.0% (${candle_3['close']})")
should_close, exit_price, reason, ratio = logic_partial.check_trade_exit(trade, candle_3)
print(f"Result: {reason}, ratio={ratio:.2f}")
print(f"Expected: partial_tp_4.0%, ratio=1.00 (final level triggers full close)")
print(f"✅ CORRECT!" if should_close and abs(ratio - 1.0) < 0.01 else f"❌ WRONG!")
print()

print("="*60)
print("ALL TESTS COMPLETED")
print("="*60)
