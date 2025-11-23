#!/usr/bin/env python3
"""
Test losing streak protection and cooldown decrement
"""
import config
from trading_logic import TradingLogic

print("="*60)
print("COOLDOWN DECREMENT TEST")
print("="*60)

logic = TradingLogic(config, initial_capital=7046.58)

# Simulate 5 consecutive losses
print("\n📉 Simulating 5 consecutive losses...")
for i in range(5):
    logic.consecutive_losses += 1
    print(f"Loss {i+1}: consecutive_losses = {logic.consecutive_losses}")

# Check if cooldown activated
if logic.consecutive_losses >= logic.losing_streak_protection['stop_trading_after']:
    logic.cooldown_until_candle = logic.losing_streak_protection['cooldown_candles']
    print(f"\n🔴 COOLDOWN ACTIVATED: {logic.cooldown_until_candle} candles")

# Test should_open_trade during cooldown
can_trade = logic.should_open_trade('ascending_triangle', 0.80, 0.80)
print(f"\n❓ Can trade during cooldown? {can_trade}")
print(f"Expected: False")
print(f"✅ CORRECT!" if not can_trade else f"❌ WRONG!")

# Simulate 60 candles passing (decrement cooldown)
print(f"\n⏳ Simulating 60 candles passing...")
for i in range(60):
    logic.decrement_cooldown()
    if (i + 1) % 10 == 0:
        print(f"Candle {i+1}: cooldown = {logic.cooldown_until_candle}, consecutive_losses = {logic.consecutive_losses}")

# After cooldown, should be able to trade
can_trade_after = logic.should_open_trade('ascending_triangle', 0.80, 0.80)
print(f"\n✅ After cooldown - Can trade? {can_trade_after}")
print(f"Expected: True")
print(f"Consecutive losses reset? {logic.consecutive_losses == 0}")
print(f"✅ CORRECT!" if can_trade_after and logic.consecutive_losses == 0 else f"❌ WRONG!")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
