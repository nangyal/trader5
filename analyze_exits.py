"""
Analyze exit reasons and PnL distribution
"""
import pandas as pd
import numpy as np

# Load latest backtest report
df_dict = pd.read_excel('stat/backtest_report_20251123_023022.xlsx', sheet_name=None)

# Get detailed results
df_detail = df_dict['Detailed Results']

print("="*80)
print("EXIT REASON ANALYSIS")
print("="*80)

# We need to get the actual trades from backtest results
# The Excel doesn't have exit reasons per trade, only aggregated stats

# So let's calculate what we can from aggregated data
for idx, row in df_detail.iterrows():
    coin = row['Coin']
    total = row['Total Trades']
    wins = row['Winning Trades']
    losses = row['Losing Trades']
    avg_win = row['Avg Win (USDT)']
    avg_loss = row['Avg Loss (USDT)']
    win_rate = row['Win Rate (%)']
    
    print(f"\n{coin}:")
    print(f"  Total trades: {total}")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Avg win: ${avg_win:.4f}")
    print(f"  Avg loss: ${avg_loss:.4f}")
    print(f"  R:R ratio: {abs(avg_win/avg_loss):.2f}")
    
    # Calculate expected pattern
    # Theoretical: TP 2.0%, SL 0.5% = 4:1 R:R
    # Actual: varies by coin
    
    # If Partial TP working:
    # Expected avg win: ~1.5% (50% @ 1.5%, 50% @ trailing ~1.5%)
    # Expected avg loss: -0.5% (SL)
    
    print(f"  Theoretical R:R (2.0% TP / 0.5% SL): 4.00")
    print(f"  Actual R:R: {abs(avg_win/avg_loss):.2f}")
    print(f"  R:R efficiency: {(abs(avg_win/avg_loss) / 4.0)*100:.1f}%")

print("\n" + "="*80)
print("PATTERN STATS")
print("="*80)

df_patterns = df_dict['Pattern Stats']
print(df_patterns.to_string())

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Calculate some metrics
for idx, row in df_patterns.iterrows():
    pattern = row['Pattern']
    total = row['Total Trades']
    wins = row['Winning']
    losses = row['Losing']
    avg_pnl = row['Avg P&L']
    avg_win = row['Avg Win']
    avg_loss = row['Avg Loss']
    win_rate = row['Win Rate (%)']
    profit_factor = row['Profit Factor']
    
    # Expected vs Actual
    print(f"\n{pattern}:")
    print(f"  Total: {total}, Win Rate: {win_rate:.2f}%")
    print(f"  Avg Win: ${avg_win:.4f}, Avg Loss: ${avg_loss:.4f}")
    print(f"  Profit Factor: {profit_factor:.3f}")
    
    # Estimate avg profit %
    # Assuming avg position size ~$36
    avg_win_pct = (avg_win / 36) * 100 if avg_win > 0 else 0
    avg_loss_pct = (abs(avg_loss) / 36) * 100 if avg_loss < 0 else 0
    
    print(f"  Estimated avg win%: {avg_win_pct:.2f}% (assuming $36 pos size)")
    print(f"  Estimated avg loss%: {avg_loss_pct:.2f}%")
    
    # Compare to theoretical
    theoretical_win = 1.5  # Partial TP avg
    theoretical_loss = 0.5  # SL
    
    print(f"  Theoretical win%: {theoretical_win:.2f}%")
    print(f"  Theoretical loss%: {theoretical_loss:.2f}%")
    print(f"  Win efficiency: {(avg_win_pct/theoretical_win)*100:.1f}%")
    print(f"  Loss control: {(avg_loss_pct/theoretical_loss)*100:.1f}%")
