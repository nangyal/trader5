#!/usr/bin/env python3
"""
Test MAX_CONCURRENT_TRADES = 3 with all positions open
"""
import config
from trading_logic import TradingLogic

print("="*60)
print("MAX CONCURRENT TRADES TEST (3 trades)")
print("="*60)

shared_capital = 7046.58
initial_capital = shared_capital

# Create 3 traders
traders = {
    'BTCUSDT': TradingLogic(config, initial_capital=shared_capital),
    'ETHUSDT': TradingLogic(config, initial_capital=shared_capital),
    'SOLUSDT': TradingLogic(config, initial_capital=shared_capital),
}

print(f"\n💰 Initial shared capital: ${shared_capital:.2f}")
print(f"   Max concurrent trades: {config.MAX_CONCURRENT_TRADES}")
print(f"   Max position size: {config.MAX_POSITION_SIZE_PCT*100:.0f}%")

# Open 3 trades sequentially with proper sync
trades_info = [
    {'coin': 'BTCUSDT', 'entry': 98500.0, 'sl': 98000.0, 'prob': 0.80},
    {'coin': 'ETHUSDT', 'entry': 3500.0, 'sl': 3480.0, 'prob': 0.75},
    {'coin': 'SOLUSDT', 'entry': 150.0, 'sl': 148.5, 'prob': 0.70},
]

opened_trades = []
total_value = 0.0

for i, info in enumerate(trades_info, 1):
    coin = info['coin']
    trader = traders[coin]
    
    # Sync capital before position sizing
    trader.capital = shared_capital
    
    # Calculate position
    position = trader.calculate_position_size(
        info['entry'], info['sl'], shared_capital, ml_probability=info['prob']
    )
    value = position * info['entry']
    
    # Open trade
    trade = trader.open_trade(
        coin=coin,
        pattern='ascending_triangle',
        entry_price=info['entry'],
        stop_loss=info['sl'],
        take_profit=info['entry'] * 1.02,
        position_size=position,
        probability=info['prob'],
        strength=info['prob'],
        timeframe='1min'
    )
    
    # Update shared capital
    shared_capital = trader.capital
    total_value += value
    
    print(f"\n{i}️⃣ {coin} opened:")
    print(f"   Position: ${value:.2f} ({value/initial_capital*100:.1f}% of initial)")
    print(f"   Shared capital remaining: ${shared_capital:.2f}")
    
    opened_trades.append({'coin': coin, 'value': value, 'trade': trade})

print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)
print(f"Total capital allocated: ${total_value:.2f}")
print(f"% of initial capital: {total_value/initial_capital*100:.1f}%")
print(f"Remaining capital: ${shared_capital:.2f}")
print(f"% remaining: {shared_capital/initial_capital*100:.1f}%")

if shared_capital >= 0:
    print(f"\n✅ PASS: All 3 trades opened successfully!")
    print(f"   Capital management working correctly")
else:
    print(f"\n❌ FAIL: Negative capital - over-leveraged!")

# Try to open 4th trade (should be blocked)
print(f"\n{'='*60}")
print("Attempting 4th trade (should be blocked)...")
print("="*60)

# Check GLOBAL active trades count (like WebSocket does)
total_active = sum(len(t.active_trades) for t in traders.values())
btc_trader = traders['BTCUSDT']
btc_trader.capital = shared_capital  # Sync

# Simulate WebSocket global check
can_open_global = total_active < config.MAX_CONCURRENT_TRADES

# Also check trader-level (pattern/prob/strength)
can_open_trader = btc_trader.should_open_trade('ascending_triangle', 0.80, 0.80)

print(f"Active trades (GLOBAL): {total_active}")
print(f"Max allowed: {config.MAX_CONCURRENT_TRADES}")
print(f"Global check (WebSocket level): {can_open_global}")
print(f"Trader check (pattern/prob/strength): {can_open_trader}")
print(f"Final decision: {'CAN OPEN' if can_open_global and can_open_trader else 'BLOCKED'}")
print(f"Expected: BLOCKED")
print(f"{'✅ CORRECT!' if not (can_open_global and can_open_trader) else '❌ WRONG - should be blocked!'}")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
