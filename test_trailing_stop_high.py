#!/usr/bin/env python3
"""
Test Trailing Stop activation using HIGH vs CLOSE

CRITICAL BUG #20: Trailing Stop should activate using HIGH (max profit),
not CLOSE (final price). This ensures proper risk protection.

SCENARIO: Trailing Stop activates at 1.0% profit, trails by 0.5%

Example:
- Entry: $100
- Candle 1: HIGH=$101.50 (+1.5%), CLOSE=$101.00 (+1.0%)
  - OLD: profit_pct=1.0% → trail_price=$101.00*0.995=$100.495 ❌
  - NEW: max_profit=1.5% → trail_price=$101.50*0.995=$100.9925 ✅
  
Expected:
✅ Trailing stop uses HIGH for activation check
✅ Trailing stop calculated from HIGH (max profit)
✅ Better risk protection (higher trailing stop)
"""

import config
from trading_logic import TradingLogic
import pandas as pd

def test_trailing_stop_high_vs_close():
    """Test trailing stop activates using HIGH, not CLOSE"""
    print("\n=== TEST: TRAILING STOP HIGH vs CLOSE ===\n")
    
    # Initialize trading logic with trailing stop
    trading_logic = TradingLogic(config, initial_capital=10000)
    
    print(f"Trailing Stop Config:")
    print(f"  Activation: {config.TRAILING_STOP['activation_pct']*100}%")
    print(f"  Trail: {config.TRAILING_STOP['trail_pct']*100}%")
    print()
    
    # Create active trade
    trade = {
        'coin': 'BTCUSDT',
        'entry_price': 100000,
        'position_size': 0.1,
        'direction': 'long',
        'stop_loss': 98000,
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
    
    # SCENARIO 1: HIGH reaches activation, CLOSE also above
    # BUT: HIGH also reaches Partial TP 1.5% → will exit instead of updating trail!
    print("SCENARIO 1: HIGH=+1.5%, CLOSE=+1.2%")
    print("  NOTE: HIGH=+1.5% triggers Partial TP (priority 2), so trailing stop NOT updated")
    print(f"Entry: ${trade['entry_price']:,}")
    
    candle_1 = {
        'open': 101000,
        'high': 101500,  # +1.5% → Partial TP!
        'low': 100800,
        'close': 101200,  # +1.2%
        'volume': 100
    }
    
    should_exit, exit_price, exit_type, partial_pct = trading_logic.check_trade_exit(
        trade, candle_1
    )
    
    print(f"Candle: HIGH=${candle_1['high']:,} (+{((candle_1['high']/trade['entry_price'])-1)*100:.1f}%), "
          f"CLOSE=${candle_1['close']:,} (+{((candle_1['close']/trade['entry_price'])-1)*100:.1f}%)")
    
    # Check if Partial TP triggered
    if should_exit and exit_type == 'partial_tp_1.5%':
        print(f"✅ CORRECT! Partial TP triggered (priority 2), exit_price=${exit_price:,.0f}")
        print(f"  Trailing stop NOT updated (Partial TP takes priority)")
    else:
        print(f"❌ WRONG! Expected Partial TP exit, got should_exit={should_exit}, type={exit_type}")
    
    # SCENARIO 2: HIGH reaches activation, CLOSE below
    print("\n\nSCENARIO 2: HIGH=+1.2%, CLOSE=+0.8% (HIGH above, CLOSE below 1.0%)")
    
    # Reset trade
    trade['trailing_stop'] = None
    
    candle_2 = {
        'open': 100500,
        'high': 101200,  # +1.2%
        'low': 100300,
        'close': 100800,  # +0.8%
        'volume': 100
    }
    
    should_exit, exit_price, exit_type, partial_pct = trading_logic.check_trade_exit(
        trade, candle_2
    )
    
    print(f"Candle: HIGH=${candle_2['high']:,} (+{((candle_2['high']/trade['entry_price'])-1)*100:.1f}%), "
          f"CLOSE=${candle_2['close']:,} (+{((candle_2['close']/trade['entry_price'])-1)*100:.1f}%)")
    
    # OLD logic would NOT activate (profit_pct=0.8% < 1.0%)
    # NEW logic SHOULD activate (HIGH=1.2% > 1.0%)
    if trade['trailing_stop'] is not None:
        expected_trail = candle_2['high'] * (1 - config.TRAILING_STOP['trail_pct'])
        print(f"Expected trail_price: ${candle_2['high']}*{1-config.TRAILING_STOP['trail_pct']} = ${expected_trail:,.2f}")
        print(f"Actual trailing_stop: ${trade['trailing_stop']:,.2f}")
        
        if abs(trade['trailing_stop'] - expected_trail) < 0.01:
            print("✅ CORRECT! Trailing stop activated using HIGH (CLOSE below threshold)")
        else:
            print(f"❌ WRONG! Expected ${expected_trail:,.2f}, got ${trade['trailing_stop']:,.2f}")
    else:
        print("❌ CRITICAL FAILURE! Trailing stop NOT activated (should use HIGH=1.2%)")
    
    # SCENARIO 3: Update trailing stop (higher HIGH)
    print("\n\nSCENARIO 3: Even higher HIGH updates trailing stop")
    
    candle_3 = {
        'open': 101500,
        'high': 102000,  # +2.0%
        'low': 101300,
        'close': 101800,  # +1.8%
        'volume': 100
    }
    
    old_trail = trade['trailing_stop']
    should_exit, exit_price, exit_type, partial_pct = trading_logic.check_trade_exit(
        trade, candle_3
    )
    
    print(f"Candle: HIGH=${candle_3['high']:,} (+{((candle_3['high']/trade['entry_price'])-1)*100:.1f}%)")
    print(f"Old trailing_stop: ${old_trail:,.2f}")
    print(f"New trailing_stop: ${trade['trailing_stop']:,.2f}")
    
    expected_trail = candle_3['high'] * (1 - config.TRAILING_STOP['trail_pct'])
    if trade['trailing_stop'] > old_trail and abs(trade['trailing_stop'] - expected_trail) < 0.01:
        print(f"✅ CORRECT! Trailing stop updated to ${expected_trail:,.2f}")
    else:
        print(f"❌ WRONG! Expected ${expected_trail:,.2f}, got ${trade['trailing_stop']:,.2f}")
    
    print("\n" + "="*60)
    print("TRAILING STOP HIGH LOGIC TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_trailing_stop_high_vs_close()
