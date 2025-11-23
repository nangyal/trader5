#!/usr/bin/env python3
"""
Test shared capital pool with concurrent trades
Simulates BTCUSDT and ETHUSDT opening trades simultaneously
"""
import config
from trading_logic import TradingLogic

print("="*60)
print("SHARED CAPITAL POOL TEST")
print("="*60)

# Simulate shared capital pool
shared_capital = 7046.58
initial_capital = shared_capital

# Create traders for BTCUSDT and ETHUSDT
btc_trader = TradingLogic(config, initial_capital=shared_capital)
eth_trader = TradingLogic(config, initial_capital=shared_capital)

print(f"\n💰 Initial shared capital: ${shared_capital:.2f}")
print(f"   BTC trader capital: ${btc_trader.capital:.2f}")
print(f"   ETH trader capital: ${eth_trader.capital:.2f}")

# Simulate position sizing for BOTH traders
btc_entry = 98500.0
btc_sl = 98000.0
btc_position = btc_trader.calculate_position_size(
    btc_entry, btc_sl, shared_capital, ml_probability=0.80
)
btc_value = btc_position * btc_entry

eth_entry = 3500.0
eth_sl = 3480.0
eth_position = eth_trader.calculate_position_size(
    eth_entry, eth_sl, shared_capital, ml_probability=0.75
)
eth_value = eth_position * eth_entry

print(f"\n📊 Position Sizing (both using ${shared_capital:.2f}):")
print(f"   BTC: {btc_position:.6f} BTC = ${btc_value:.2f} ({btc_value/shared_capital*100:.1f}%)")
print(f"   ETH: {eth_position:.6f} ETH = ${eth_value:.2f} ({eth_value/shared_capital*100:.1f}%)")
print(f"   Total allocated: ${btc_value + eth_value:.2f} ({(btc_value + eth_value)/shared_capital*100:.1f}%)")

# Check if over-allocated
if btc_value + eth_value > shared_capital:
    print(f"   ❌ OVER-ALLOCATED! Both calculated from same capital!")
else:
    print(f"   ✅ Safe allocation")

# SCENARIO 1: Sequential opening (CORRECT)
print(f"\n{'='*60}")
print("SCENARIO 1: Sequential Trade Opening (CORRECT)")
print("="*60)

# Reset
btc_trader.capital = shared_capital
eth_trader.capital = shared_capital

# BTC opens first
btc_trader.capital = shared_capital
btc_trade = btc_trader.open_trade(
    coin='BTCUSDT',
    pattern='ascending_triangle',
    entry_price=btc_entry,
    stop_loss=btc_sl,
    take_profit=btc_entry * 1.02,
    position_size=btc_position,
    probability=0.80,
    strength=0.80,
    timeframe='1min'
)
shared_capital = btc_trader.capital  # Update shared pool
print(f"1️⃣ BTC trade opened: ${btc_value:.2f}")
print(f"   Shared capital after: ${shared_capital:.2f}")

# ETH opens second (with updated capital)
eth_trader.capital = shared_capital  # Sync
eth_position_updated = eth_trader.calculate_position_size(
    eth_entry, eth_sl, shared_capital, ml_probability=0.75
)
eth_value_updated = eth_position_updated * eth_entry

eth_trade = eth_trader.open_trade(
    coin='ETHUSDT',
    pattern='ascending_triangle',
    entry_price=eth_entry,
    stop_loss=eth_sl,
    take_profit=eth_entry * 1.02,
    position_size=eth_position_updated,
    probability=0.75,
    strength=0.75,
    timeframe='1min'
)
shared_capital = eth_trader.capital  # Update shared pool
print(f"2️⃣ ETH trade opened: ${eth_value_updated:.2f}")
print(f"   Shared capital after: ${shared_capital:.2f}")

total_allocated = btc_value + eth_value_updated
print(f"\n💼 Total capital used: ${total_allocated:.2f}")
print(f"   Remaining: ${shared_capital:.2f}")
print(f"   % of initial capital: {total_allocated/initial_capital*100:.1f}%")

if shared_capital >= 0:
    print(f"   ✅ CORRECT: Capital managed properly")
else:
    print(f"   ❌ WRONG: Negative capital!")

# SCENARIO 2: What if we DON'T sync? (WRONG)
print(f"\n{'='*60}")
print("SCENARIO 2: Without Sync (WRONG - demonstrates the bug)")
print("="*60)

# Reset
shared_capital = initial_capital
btc_trader2 = TradingLogic(config, initial_capital=shared_capital)
eth_trader2 = TradingLogic(config, initial_capital=shared_capital)

# Both calculate position with SAME capital (not updated)
btc_pos2 = btc_trader2.calculate_position_size(btc_entry, btc_sl, shared_capital, ml_probability=0.80)
eth_pos2 = eth_trader2.calculate_position_size(eth_entry, eth_sl, shared_capital, ml_probability=0.75)

btc_val2 = btc_pos2 * btc_entry
eth_val2 = eth_pos2 * eth_entry

print(f"⚠️  Both traders calculate with ${shared_capital:.2f}")
print(f"   BTC would use: ${btc_val2:.2f}")
print(f"   ETH would use: ${eth_val2:.2f}")
print(f"   Total: ${btc_val2 + eth_val2:.2f} ({(btc_val2 + eth_val2)/initial_capital*100:.1f}%)")

if btc_val2 + eth_val2 > shared_capital:
    print(f"   ❌ OVER-ALLOCATION! This is the bug we fixed!")
else:
    print(f"   ✅ Within limits (lucky)")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
print("\n💡 Key takeaway:")
print("   - MUST sync capital BEFORE each calculate_position_size()")
print("   - MUST update shared_capital AFTER each trade open/close")
print("   - MUST sync to all traders after shared_capital update")
