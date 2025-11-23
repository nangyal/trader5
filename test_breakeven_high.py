#!/usr/bin/env python3
"""
Test Breakeven Stop activation using HIGH vs CLOSE

CRITICAL BUG #21: Breakeven Stop should activate using HIGH (max profit),
not CLOSE (final price).

SCENARIO: Breakeven activates at 0.8% profit, moves SL to entry + 0.1%

Example:
- Entry: $100,000
- Candle: HIGH=$100,900 (+0.9%), CLOSE=$100,700 (+0.7%)
  - OLD: profit_pct=0.7% < 0.8% → NOT activated ❌
  - NEW: max_profit=0.9% >= 0.8% → ACTIVATED ✅
  
Expected:
✅ Breakeven uses HIGH for activation check
✅ SL moved to entry + buffer when HIGH reaches activation
"""

import config
from trading_logic import TradingLogic
import pandas as pd

def test_breakeven_high_vs_close():
    """Test breakeven activates using HIGH, not CLOSE"""
    print("\n=== TEST: BREAKEVEN STOP HIGH vs CLOSE ===\n")
    
    # Initialize trading logic
    trading_logic = TradingLogic(config, initial_capital=10000)
    
    print(f"Breakeven Stop Config:")
    print(f"  Activation: {config.BREAKEVEN_STOP['activation_pct']*100}%")
    print(f"  Buffer: {config.BREAKEVEN_STOP['buffer_pct']*100}%")
    print()
    
    # Create active trade
    trade = {
        'coin': 'BTCUSDT',
        'entry_price': 100000,
        'position_size': 0.1,
        'direction': 'long',
        'stop_loss': 99000,  # -1%
        'take_profit': 104000,
        'pattern': 'test',
        'probability': 0.8,
        'strength': 0.8,
        'timeframe': '1min',
        'status': 'open',
        'pnl': 0.0,
        'partial_closed': 0,
        'trailing_stop': None,
        'breakeven_activated': False
    }
    trading_logic.active_trades.append(trade)
    
    # SCENARIO 1: HIGH=+0.9%, CLOSE=+0.7% (HIGH above, CLOSE below 0.8%)
    print("SCENARIO 1: HIGH=+0.9%, CLOSE=+0.7%")
    print(f"Entry: ${trade['entry_price']:,}")
    print(f"Initial SL: ${trade['stop_loss']:,} (-1.0%)")
    
    candle_1 = {
        'open': 100500,
        'high': 100900,  # +0.9%
        'low': 100300,
        'close': 100700,  # +0.7%
        'volume': 100
    }
    
    should_exit, exit_price, exit_type, partial_pct = trading_logic.check_trade_exit(
        trade, candle_1
    )
    
    print(f"Candle: HIGH=${candle_1['high']:,} (+{((candle_1['high']/trade['entry_price'])-1)*100:.1f}%), "
          f"CLOSE=${candle_1['close']:,} (+{((candle_1['close']/trade['entry_price'])-1)*100:.1f}%)")
    
    # Check if breakeven activated
    expected_new_sl = trade['entry_price'] * (1 + config.BREAKEVEN_STOP['buffer_pct'])
    
    if trade['breakeven_activated']:
        print(f"✅ CORRECT! Breakeven activated using HIGH")
        print(f"   New SL: ${trade['stop_loss']:,}")
        print(f"   Expected: ${expected_new_sl:,}")
        
        if abs(trade['stop_loss'] - expected_new_sl) < 0.01:
            print(f"   ✅ SL correctly moved to entry + buffer")
        else:
            print(f"   ❌ WRONG SL! Expected ${expected_new_sl:,}, got ${trade['stop_loss']:,}")
    else:
        print(f"❌ CRITICAL FAILURE! Breakeven NOT activated (should use HIGH=0.9%)")
        print(f"   OLD logic would use CLOSE=0.7% < 0.8% → not activated")
    
    # SCENARIO 2: Test that breakeven doesn't activate twice
    print("\n\nSCENARIO 2: Even higher profit, but already activated")
    
    candle_2 = {
        'open': 101000,
        'high': 101500,  # +1.5%
        'low': 100900,
        'close': 101200,
        'volume': 100
    }
    
    old_sl = trade['stop_loss']
    should_exit, exit_price, exit_type, partial_pct = trading_logic.check_trade_exit(
        trade, candle_2
    )
    
    print(f"Candle: HIGH=${candle_2['high']:,} (+1.5%)")
    print(f"Old SL: ${old_sl:,}")
    print(f"Current SL: ${trade['stop_loss']:,}")
    
    if trade['stop_loss'] == old_sl:
        print(f"✅ CORRECT! SL unchanged (already activated)")
    else:
        print(f"❌ WRONG! SL changed again (should only activate once)")
    
    print("\n" + "="*60)
    print("BREAKEVEN STOP HIGH LOGIC TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_breakeven_high_vs_close()
