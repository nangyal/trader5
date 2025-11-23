#!/usr/bin/env python3
"""
Test Partial TP final level (Bug #22)

CRITICAL BUG #22: When final Partial TP level (cumulative 1.0) is reached,
should return partial_ratio=1.0 to trigger FULL close, not incremental ratio.

SCENARIO: 3-level Partial TP (50%, 75%, 100%)
- Level 1: +1.5% close 50% → partial_ratio=0.5
- Level 2: +2.5% close 75% (cumulative) → partial_ratio=0.25 (incremental)
- Level 3: +4.0% close 100% (cumulative) → partial_ratio=1.0 (FULL CLOSE)

Expected:
✅ Final level returns partial_ratio=1.0
✅ Trade status changed to 'closed'
✅ Trade removed from active_trades
"""

import config
from trading_logic import TradingLogic

def test_partial_tp_final_level():
    """Test final Partial TP level triggers full close"""
    print("\n=== TEST: PARTIAL TP FINAL LEVEL (1.0) ===\n")
    
    logic = TradingLogic(config, initial_capital=10000)
    
    # Create trade
    trade = {
        'coin': 'BTCUSDT',
        'pattern': 'test',
        'entry_price': 100000,
        'stop_loss': 99000,
        'take_profit': 104000,
        'position_size': 1.0,
        'original_position_size': 1.0,
        'position_value': 100000,
        'probability': 0.8,
        'strength': 0.8,
        'timeframe': '1min',
        'direction': 'long',
        'status': 'open',
        'pnl': 0.0,
        'trailing_stop': None,
        'partial_closed': 0.0,
        'breakeven_activated': False,
    }
    logic.active_trades.append(trade)
    
    print(f"Entry: ${trade['entry_price']:,}")
    print(f"Position: {trade['position_size']} BTC")
    print(f"Partial TP Levels: 1.5% (50%), 2.5% (75%), 4.0% (100%)")
    print()
    
    # Level 1: +1.5%
    print("LEVEL 1: +1.5% → Close 50%")
    candle_1 = {'open': 100000, 'high': 101500, 'low': 100500, 'close': 101500, 'volume': 100}
    should_exit, exit_price, exit_type, partial_ratio = logic.check_trade_exit(trade, candle_1)
    
    print(f"  should_exit: {should_exit}")
    print(f"  exit_type: {exit_type}")
    print(f"  partial_ratio: {partial_ratio}")
    print(f"  partial_closed: {trade['partial_closed']}")
    
    if should_exit and partial_ratio == 0.5:
        # Close 50%
        pnl_1 = logic.close_trade(trade, exit_price, exit_type, None, partial_ratio)
        print(f"  ✅ Closed 50%, remaining: {trade['position_size']} BTC")
        print(f"  Trade status: {trade['status']}")
        print(f"  Active trades: {len(logic.active_trades)}")
    else:
        print(f"  ❌ FAILED!")
        return
    
    # Level 2: +2.5%
    print("\nLEVEL 2: +2.5% → Close 75% (cumulative), 25% incremental")
    candle_2 = {'open': 101500, 'high': 102500, 'low': 101500, 'close': 102500, 'volume': 100}
    should_exit, exit_price, exit_type, partial_ratio = logic.check_trade_exit(trade, candle_2)
    
    print(f"  should_exit: {should_exit}")
    print(f"  exit_type: {exit_type}")
    print(f"  partial_ratio: {partial_ratio}")
    print(f"  partial_closed: {trade['partial_closed']}")
    
    if should_exit and partial_ratio == 0.25:
        # Close 25%
        pnl_2 = logic.close_trade(trade, exit_price, exit_type, None, partial_ratio)
        print(f"  ✅ Closed 25%, remaining: {trade['position_size']} BTC")
        print(f"  Trade status: {trade['status']}")
        print(f"  Active trades: {len(logic.active_trades)}")
    else:
        print(f"  ❌ FAILED!")
        return
    
    # Level 3: +4.0% (FINAL)
    print("\nLEVEL 3: +4.0% → Close 100% (cumulative), FULL CLOSE")
    candle_3 = {'open': 102500, 'high': 104000, 'low': 102500, 'close': 104000, 'volume': 100}
    should_exit, exit_price, exit_type, partial_ratio = logic.check_trade_exit(trade, candle_3)
    
    print(f"  should_exit: {should_exit}")
    print(f"  exit_type: {exit_type}")
    print(f"  partial_ratio: {partial_ratio}")
    print(f"  partial_closed: {trade['partial_closed']}")
    
    # CRITICAL CHECK: partial_ratio should be 1.0 (FULL CLOSE), not 0.25
    if should_exit and partial_ratio == 1.0:
        print(f"  ✅ CORRECT! Final level returns partial_ratio=1.0")
        
        # Close trade
        pnl_3 = logic.close_trade(trade, exit_price, exit_type, None, partial_ratio)
        print(f"  Remaining position: {trade['position_size']} BTC")
        print(f"  Trade status: {trade['status']}")
        print(f"  Active trades: {len(logic.active_trades)}")
        
        # Validate full close
        if trade['status'] == 'closed':
            print(f"  ✅ Trade status = 'closed'")
        else:
            print(f"  ❌ WRONG! Trade status = '{trade['status']}' (should be 'closed')")
        
        if len(logic.active_trades) == 0:
            print(f"  ✅ Trade removed from active_trades")
        else:
            print(f"  ❌ WRONG! Trade still in active_trades (count: {len(logic.active_trades)})")
        
        if trade['position_size'] < 0.00000001:
            print(f"  ✅ Position size ~0 ({trade['position_size']:.10f})")
        else:
            print(f"  ❌ WRONG! Position size not zero: {trade['position_size']}")
    else:
        print(f"  ❌ CRITICAL FAILURE!")
        print(f"     Expected partial_ratio=1.0 (FULL CLOSE)")
        print(f"     Got partial_ratio={partial_ratio}")
        print(f"     OLD BUG: Would return 0.25, leaving trade open with 0 position!")
    
    print("\n" + "="*60)
    print("PARTIAL TP FINAL LEVEL TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_partial_tp_final_level()
