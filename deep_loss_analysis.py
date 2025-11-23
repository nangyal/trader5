"""
Deep Loss Analysis - Find why avg loss is 0.78% instead of 0.50%
Analyze exit reasons and position-level PnL distribution
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("DEEP LOSS DISTRIBUTION ANALYSIS")
print("="*80)

# Try to load actual trade data from backtest results
# The trading_logic saves closed_trades, so we need to check if there's a JSON export

# For now, let's do mathematical analysis based on aggregated stats
df_dict = pd.read_excel('stat/backtest_report_20251123_023022.xlsx', sheet_name=None)
df_detail = df_dict['Detailed Results']
df_patterns = df_dict['Pattern Stats']

print("\n1. STATISTICAL ANALYSIS")
print("-"*80)

for idx, row in df_detail.iterrows():
    coin = row['Coin']
    total = row['Total Trades']
    wins = row['Winning Trades']
    losses = row['Losing Trades']
    avg_win = row['Avg Win (USDT)']
    avg_loss = row['Avg Loss (USDT)']
    total_pnl = row['Total P&L (USDT)']
    
    print(f"\n{coin}:")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Avg Win: ${avg_win:.4f}, Avg Loss: ${avg_loss:.4f}")
    
    # Calculate total win/loss amounts
    total_wins = wins * avg_win
    total_losses = losses * avg_loss
    
    print(f"  Total from wins: ${total_wins:.2f}")
    print(f"  Total from losses: ${total_losses:.2f}")
    print(f"  Net P&L: ${total_wins + total_losses:.2f}")
    print(f"  Reported P&L: ${total_pnl:.2f}")
    
    # Check consistency
    diff = abs((total_wins + total_losses) - total_pnl)
    if diff > 0.01:
        print(f"  ⚠️  WARNING: P&L mismatch ${diff:.2f}")
    else:
        print(f"  ✅ P&L consistent")

print("\n2. LOSS DISTRIBUTION SIMULATION")
print("-"*80)

# Simulate what could cause 0.78% avg loss
# Theory: Most losses are -0.5% (SL), but some are worse

print("\nTheoretical scenarios:")

# Scenario 1: All losses at SL (-0.5%)
print("\n  Scenario 1: All losses hit regular SL (-0.5%)")
print("    Expected avg loss: -0.50%")
print("    Actual avg loss: -0.78%")
print("    Difference: +0.28% worse")
print("    Conclusion: ❌ Not all losses are regular SL hits")

# Scenario 2: Mix of regular SL and worse exits
print("\n  Scenario 2: Mix of exit types")

# Let's calculate what mix would give -0.78%
# If x% are -0.5% SL, and (100-x)% are worse (let's say -2.0% extreme)
# -0.78 = x * -0.5 + (100-x) * -2.0
# -0.78 = -0.5x - 2.0(100-x)
# -0.78 = -0.5x - 200 + 2.0x
# -0.78 + 200 = 1.5x
# x = 199.22 / 1.5 = 132.8% (impossible!)

# Try with -1.5% as worse loss
# -0.78 = x * -0.5 + (100-x) * -1.5
# -0.78 = -0.5x - 150 + 1.5x
# -0.78 + 150 = 1.0x
# x = 149.22% (still impossible!)

# Try with -1.0% as worse loss
# -0.78 = x * -0.5 + (100-x) * -1.0
# -0.78 = -0.5x - 100 + 1.0x
# -0.78 + 100 = 0.5x
# x = 99.22 / 0.5 = 198.44% (impossible!)

# WAIT! The issue is we're using percentages wrong
# Let's use dollar amounts with avg position size ~$36

avg_pos_size = 36.0
sl_pct = 0.005  # 0.5%
avg_loss_pct = 0.0078  # 0.78%

sl_dollar = avg_pos_size * sl_pct  # $0.18
avg_loss_dollar = avg_pos_size * avg_loss_pct  # $0.28

print(f"\n  Position size analysis:")
print(f"    Avg position: ${avg_pos_size:.2f}")
print(f"    SL (-0.5%): ${sl_dollar:.2f}")
print(f"    Actual avg loss: ${avg_loss_dollar:.2f}")
print(f"    Extra loss: ${avg_loss_dollar - sl_dollar:.2f}")

# But we know actual avg loss is $0.279 (from report)
actual_avg_loss = 0.279

print(f"\n  Actual from report:")
print(f"    Avg loss: ${actual_avg_loss:.3f}")
print(f"    Implied position size: ${actual_avg_loss / 0.005:.2f} (if -0.5% SL)")
print(f"    OR implied SL%: {(actual_avg_loss / avg_pos_size)*100:.3f}% (if $36 pos)")

# Calculate required worse exit percentage
# If some trades exit at worse prices, what % would explain this?

print("\n3. POSSIBLE EXPLANATIONS")
print("-"*80)

print("\n  A) Position size varies with ML confidence:")
print("     80%+ prob → 1.5x position = $54")
print("     70-80% prob → 1.2x position = $43.2")
print("     65-70% prob → 1.0x position = $36")
print("     If SL hits on high-confidence trades more often:")
print("       $54 × 0.5% = $0.27 (close to actual $0.279!)")

print("\n  B) Some exits are WORSE than SL:")
print("     - Trailing stop hit after partial close?")
print("     - Breakeven stop at entry + buffer?")
print("     - Cooldown forcing exit at market?")

print("\n  C) Partial close dynamics:")
print("     Example: 50% close @ +1.5%, then SL hit on remaining 50%")
print("     - Close 50% @ $101.50 → +$0.75")
print("     - SL 50% @ $99.50 → -$0.25")
print("     - Net: +$0.50 (counted as WIN)")
print("     This should IMPROVE stats, not worsen!")

print("\n  D) Gap/slippage in backtest:")
print("     SL set @ $99.50, but candle LOW = $99.20")
print("     Exit @ $99.20 instead of $99.50")
print("     Loss: $100 - $99.20 = -$0.80 (60% worse!)")

print("\n4. MOST LIKELY CAUSE")
print("-"*80)

print("\n  HYPOTHESIS: ML confidence weighting creates LARGER positions")
print("  that hit SL, resulting in LARGER dollar losses")
print()
print("  Distribution estimate:")
print("    - 30% of losses: 1.5x position (high confidence) → $0.405 loss")
print("    - 40% of losses: 1.2x position (med confidence) → $0.324 loss")
print("    - 30% of losses: 1.0x position (low confidence) → $0.270 loss")
print()
print("    Weighted avg: 0.3×$0.405 + 0.4×$0.324 + 0.3×$0.270 = $0.331")
print("    Still higher than $0.279 actual...")

print("\n  ALTERNATIVE: Position size calculation uses VARYING risk amounts")
print("  Risk amount = capital × tiered_risk × ml_multiplier × streak_protection")
print()
print("  If capital varies (growing/shrinking during backtest):")
print("    - Early trades (low capital): smaller positions")
print("    - Later trades (high capital): LARGER positions")
print("    - If later trades MORE LIKELY to lose → higher avg loss!")

print("\n5. RECOMMENDED INVESTIGATION")
print("-"*80)
print()
print("  To find the root cause, we need:")
print("    1. Export actual closed_trades with full details")
print("    2. Analyze exit_reason distribution (SL vs trailing vs other)")
print("    3. Check position_size distribution for wins vs losses")
print("    4. Verify OHLC execution (is exit_price always matching SL exactly?)")
print("    5. Look for any partial close trades that ended in SL")
print()
print("  Create a modified backtest that logs:")
print("    - Every exit: reason, price, position_size, entry_price")
print("    - Calculate actual loss% for each trade")
print("    - Build histogram of loss percentages")
print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("Current findings:")
print("  ✅ Avg WIN: 1.49% (99.4% efficient - EXCELLENT!)")
print("  ❌ Avg LOSS: 0.78% (155% of theoretical 0.5% - NEEDS FIX)")
print("  📊 R:R Ratio: 1.92 (should be 4.0)")
print()
print("Most likely causes:")
print("  1. ML confidence weighting → larger losing positions")
print("  2. Tiered risk compounding → varying position sizes")
print("  3. Capital growth during backtest → later trades larger")
print("  4. Possible OHLC execution slippage in backtest")
print()
print("Action items:")
print("  [ ] Export detailed trade logs from next backtest")
print("  [ ] Analyze exit_reason distribution")
print("  [ ] Check if SL price execution is exact or has slippage")
print("  [ ] Verify partial close trades don't worsen avg loss")
print()
