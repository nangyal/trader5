#!/usr/bin/env python3
"""
Test Partial TP PnL accumulation (Bug #24)

CRITICAL BUG #24: trade['pnl'] was not accumulated across partial closes,
only the last partial close PnL was stored.

Example:
- Entry: $100,000, Size: 1.0 BTC
- Level 1 (+1.5%): Close 50% @ $101,500 → PnL = +$750
- Level 2 (+2.5%): Close 25% @ $102,500 → PnL = +$625
- Level 3 (+4.0%): Close 25% @ $104,000 → PnL = +$1,000

OLD BUG: trade['pnl'] = $1,000 (only last partial) ❌
NEW FIX: trade['pnl'] = $750 + $625 + $1,000 = $2,375 ✅

Expected:
✅ trade['pnl'] accumulates across all partial closes
✅ total_pnl also correctly sums all PnL
"""

import config
from trading_logic import TradingLogic

def test_partial_pnl_accumulation():
    """Test PnL accumulation across Partial TP levels"""
    print("\n=== TEST: PARTIAL TP PNL ACCUMULATION ===\n")
    
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
    print()
    
    # Level 1: +1.5% → Close 50%
    print("LEVEL 1: +1.5% → Close 50% @ $101,500")
    exit_price_1 = 101500
    partial_ratio_1 = 0.5
    close_size_1 = 1.0 * 0.5  # 0.5 BTC
    expected_pnl_1 = (exit_price_1 - trade['entry_price']) * close_size_1
    
    pnl_1 = logic.close_trade(trade, exit_price_1, 'partial_tp_1.5%', None, partial_ratio_1)
    
    print(f"  Close size: {close_size_1} BTC")
    print(f"  PnL: ${pnl_1:,.2f}")
    print(f"  Expected: ${expected_pnl_1:,.2f}")
    print(f"  trade['pnl']: ${trade['pnl']:,.2f}")
    print(f"  Remaining: {trade['position_size']} BTC")
    
    if abs(trade['pnl'] - expected_pnl_1) < 0.01:
        print(f"  ✅ CORRECT! trade['pnl'] = ${trade['pnl']:,.2f}")
    else:
        print(f"  ❌ WRONG! Expected ${expected_pnl_1:,.2f}, got ${trade['pnl']:,.2f}")
    print()
    
    # Level 2: +2.5% → Close 25% (of original)
    print("LEVEL 2: +2.5% → Close 25% @ $102,500")
    exit_price_2 = 102500
    partial_ratio_2 = 0.25
    close_size_2 = 1.0 * 0.25  # 0.25 BTC (of original)
    expected_pnl_2 = (exit_price_2 - trade['entry_price']) * close_size_2
    expected_cumulative_pnl = expected_pnl_1 + expected_pnl_2
    
    pnl_2 = logic.close_trade(trade, exit_price_2, 'partial_tp_2.5%', None, partial_ratio_2)
    
    print(f"  Close size: {close_size_2} BTC")
    print(f"  PnL: ${pnl_2:,.2f}")
    print(f"  Expected: ${expected_pnl_2:,.2f}")
    print(f"  trade['pnl']: ${trade['pnl']:,.2f}")
    print(f"  Expected cumulative: ${expected_cumulative_pnl:,.2f}")
    print(f"  Remaining: {trade['position_size']} BTC")
    
    if abs(trade['pnl'] - expected_cumulative_pnl) < 0.01:
        print(f"  ✅ CORRECT! Cumulative PnL = ${trade['pnl']:,.2f}")
    else:
        print(f"  ❌ WRONG! Expected ${expected_cumulative_pnl:,.2f}, got ${trade['pnl']:,.2f}")
    print()
    
    # Level 3: +4.0% → Close remaining (FULL CLOSE)
    print("LEVEL 3: +4.0% → Close 25% (FULL CLOSE) @ $104,000")
    exit_price_3 = 104000
    partial_ratio_3 = 1.0  # FULL CLOSE
    close_size_3 = trade['position_size']  # Remaining 0.25 BTC
    expected_pnl_3 = (exit_price_3 - trade['entry_price']) * close_size_3
    expected_total_pnl = expected_pnl_1 + expected_pnl_2 + expected_pnl_3
    
    pnl_3 = logic.close_trade(trade, exit_price_3, 'partial_tp_4.0%', None, partial_ratio_3)
    
    print(f"  Close size: {close_size_3} BTC")
    print(f"  PnL: ${pnl_3:,.2f}")
    print(f"  Expected: ${expected_pnl_3:,.2f}")
    print(f"  trade['pnl']: ${trade['pnl']:,.2f}")
    print(f"  Expected TOTAL: ${expected_total_pnl:,.2f}")
    print(f"  Trade status: {trade['status']}")
    
    if abs(trade['pnl'] - expected_total_pnl) < 0.01:
        print(f"  ✅ CORRECT! Total accumulated PnL = ${trade['pnl']:,.2f}")
    else:
        print(f"  ❌ WRONG! Expected ${expected_total_pnl:,.2f}, got ${trade['pnl']:,.2f}")
        print(f"     OLD BUG: Would only have ${expected_pnl_3:,.2f} (last partial only)")
    
    # Validate total_pnl
    print()
    print(f"total_pnl: ${logic.total_pnl:,.2f}")
    if abs(logic.total_pnl - expected_total_pnl) < 0.01:
        print(f"✅ total_pnl CORRECT!")
    else:
        print(f"❌ total_pnl WRONG! Expected ${expected_total_pnl:,.2f}")
    
    print("\n" + "="*60)
    print("PARTIAL TP PNL ACCUMULATION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_partial_pnl_accumulation()
