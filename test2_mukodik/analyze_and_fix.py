#!/usr/bin/env python3
"""
Detailed analysis script with CLI logging
Analyzes trades, finds problems, and suggests fixes
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import (
    load_and_preprocess_data,
    AdvancedPatternDetector,
    EnhancedForexPatternClassifier,
    create_labels_from_data,
    BacktestingEngine,
    PatternStrengthScorer
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 DETAILED TRADING ANALYSIS & DIAGNOSTICS")
print("=" * 80)

# Load data
print("\n📂 STEP 1: Loading Data...")
print("-" * 80)
DATA_FILE = 'data/DOGEUSDT-1h-2025-08.csv'  # V2.3: Use hourly candles instead of tick data
df = load_and_preprocess_data(DATA_FILE, sample_size=None)  # Use all data
print(f"✅ Loaded {len(df)} rows")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")
print(f"   Price range: ${df['close'].min():.6f} - ${df['close'].max():.6f}")
print(f"   Avg volume: {df['volume'].mean():.0f}")

# Create patterns
print("\n🎯 STEP 2: Detecting Patterns...")
print("-" * 80)
detector = AdvancedPatternDetector()
pattern_labels = create_labels_from_data(df, detector)

# Analyze pattern distribution
pattern_counts = pd.Series(pattern_labels).value_counts()
print("\n📊 Pattern Distribution:")
for pattern, count in pattern_counts.items():
    pct = count / len(pattern_labels) * 100
    if pattern != 'no_pattern':
        print(f"   {pattern:25s}: {count:5d} ({pct:5.2f}%)")

total_patterns = len(pattern_labels) - pattern_counts.get('no_pattern', 0)
print(f"\n   Total patterns found: {total_patterns} ({total_patterns/len(df)*100:.2f}%)")

# Train classifier
print("\n🤖 STEP 3: Training Classifier...")
print("-" * 80)

# Remove patterns with too few samples
pattern_series = pd.Series(pattern_labels)
pattern_counts_check = pattern_series.value_counts()
valid_patterns = pattern_counts_check[pattern_counts_check >= 10].index.tolist()

print(f"   Filtering out patterns with <10 samples...")
filtered_labels = [p if p in valid_patterns else 'no_pattern' for p in pattern_labels]
print(f"   Patterns kept: {len([p for p in valid_patterns if p != 'no_pattern'])}")

classifier = EnhancedForexPatternClassifier()
classifier.train(df, pd.Series(filtered_labels), optimize_hyperparams=False)
predictions, probabilities = classifier.predict(df)
print("✅ Model trained and predictions made")

# Calculate pattern strengths
print("\n💪 STEP 4: Calculating Pattern Strengths...")
print("-" * 80)
strength_scores = []
for i in range(len(df)):
    if predictions[i] != 'no_pattern':
        score = PatternStrengthScorer.calculate_pattern_strength(df, predictions[i], i)
        strength_scores.append(score)
    else:
        strength_scores.append(0.0)

avg_strength = np.mean([s for s in strength_scores if s > 0])
print(f"✅ Average pattern strength: {avg_strength:.3f}")

# Run backtest with detailed logging
print("\n💰 STEP 5: Running Backtest (REVERSED SIGNALS)...")
print("=" * 80)

backtester = BacktestingEngine(initial_capital=10000, risk_per_trade=0.02)

# Manually run backtest with detailed logging
capital = 10000
trades = []
trade_log = []

print("\n🔄 Processing trades...")
print("-" * 80)

trade_count = 0
for i in range(len(df)):
    pattern = predictions[i]
    
    if pattern == 'no_pattern':
        continue
    
    # Check pattern strength - LOWERED for stricter pattern detection
    if strength_scores[i] < 0.5:  # Down from 0.6
        continue
    
    # Check probability - LOWERED for stricter pattern detection  
    pattern_prob = np.max(probabilities[i])
    if pattern_prob < 0.5:  # Down from 0.6
        continue
    
    current_row = df.iloc[i]
    entry_price = current_row['close']
    
    # Pass recent data for trend calculation (V2.3)
    recent_data = df.iloc[max(0, i-30):i+1]
    
    # Calculate targets
    sl, tp, direction = backtester.calculate_pattern_targets(
        pattern, entry_price, current_row['high'], current_row['low'], recent_data
    )
    
    # Skip misaligned setups (V2.3)
    if direction == 'skip':
        continue
    
    # Calculate position size
    risk_amount = capital * 0.02
    
    if direction == 'long':
        risk_per_unit = entry_price - sl
    else:
        risk_per_unit = sl - entry_price
    
    position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    
    if position_size <= 0:
        continue
    
    trade_count += 1
    
    # Log trade entry
    trade_info = {
        'trade_id': trade_count,
        'entry_index': i,
        'entry_time': current_row.name,
        'pattern': pattern,
        'direction': direction,
        'entry_price': entry_price,
        'sl': sl,
        'tp': tp,
        'position_size': position_size,
        'risk_amount': risk_amount,
        'probability': pattern_prob,
        'strength': strength_scores[i],
        'status': 'open'
    }
    
    # Look ahead to see exit
    exit_found = False
    for j in range(i+1, min(i+100, len(df))):
        future_row = df.iloc[j]
        
        if direction == 'long':
            # Check stop loss
            if future_row['low'] <= sl:
                pnl = -risk_amount
                trade_info['exit_price'] = sl
                trade_info['exit_reason'] = 'STOP_LOSS'
                trade_info['exit_time'] = future_row.name
                trade_info['exit_index'] = j
                trade_info['pnl'] = pnl
                trade_info['bars_held'] = j - i
                capital += pnl
                exit_found = True
                break
            # Check take profit
            elif future_row['high'] >= tp:
                pnl = risk_amount * 2.0
                trade_info['exit_price'] = tp
                trade_info['exit_reason'] = 'TAKE_PROFIT'
                trade_info['exit_time'] = future_row.name
                trade_info['exit_index'] = j
                trade_info['pnl'] = pnl
                trade_info['bars_held'] = j - i
                capital += pnl
                exit_found = True
                break
        else:  # short
            # Check stop loss
            if future_row['high'] >= sl:
                pnl = -risk_amount
                trade_info['exit_price'] = sl
                trade_info['exit_reason'] = 'STOP_LOSS'
                trade_info['exit_time'] = future_row.name
                trade_info['exit_index'] = j
                trade_info['pnl'] = pnl
                trade_info['bars_held'] = j - i
                capital += pnl
                exit_found = True
                break
            # Check take profit
            elif future_row['low'] <= tp:
                pnl = risk_amount * 2.0
                trade_info['exit_price'] = tp
                trade_info['exit_reason'] = 'TAKE_PROFIT'
                trade_info['exit_time'] = future_row.name
                trade_info['exit_index'] = j
                trade_info['pnl'] = pnl
                trade_info['bars_held'] = j - i
                capital += pnl
                exit_found = True
                break
    
    if exit_found:
        trades.append(trade_info)
        trade_log.append(trade_info)
        
        # Log first 10 trades
        if trade_count <= 10:
            result = "✅ WIN" if trade_info['pnl'] > 0 else "❌ LOSS"
            print(f"\nTrade #{trade_count} {result}")
            print(f"  Pattern: {pattern:20s} Direction: {direction.upper()}")
            print(f"  Entry:  ${entry_price:.6f} @ {trade_info['entry_time']}")
            print(f"  Exit:   ${trade_info['exit_price']:.6f} @ {trade_info['exit_time']}")
            print(f"  SL: ${sl:.6f} | TP: ${tp:.6f}")
            print(f"  P&L: ${trade_info['pnl']:+.2f} ({trade_info['exit_reason']})")
            print(f"  Bars held: {trade_info['bars_held']}")
            print(f"  Capital: ${capital:.2f}")

print(f"\n\n✅ Processed {trade_count} trades")

# Analyze results
print("\n" + "=" * 80)
print("📈 BACKTEST RESULTS ANALYSIS")
print("=" * 80)

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)
    
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    
    total_pnl = df_trades['pnl'].sum()
    win_rate = len(wins) / len(df_trades) * 100
    
    print(f"\n💰 Overall Performance:")
    print(f"   Initial Capital:  $10,000.00")
    print(f"   Final Capital:    ${capital:,.2f}")
    print(f"   Total P&L:        ${total_pnl:+,.2f} ({total_pnl/10000*100:+.2f}%)")
    print(f"   Total Trades:     {len(df_trades)}")
    print(f"   Winning Trades:   {len(wins)} ({win_rate:.1f}%)")
    print(f"   Losing Trades:    {len(losses)} ({100-win_rate:.1f}%)")
    
    if len(wins) > 0:
        print(f"   Avg Win:          ${wins['pnl'].mean():+.2f}")
        print(f"   Largest Win:      ${wins['pnl'].max():+.2f}")
    
    if len(losses) > 0:
        print(f"   Avg Loss:         ${losses['pnl'].mean():.2f}")
        print(f"   Largest Loss:     ${losses['pnl'].min():.2f}")
    
    if len(losses) > 0 and len(wins) > 0:
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum())
        print(f"   Profit Factor:    {profit_factor:.2f}")
    
    # Analyze by pattern
    print(f"\n📊 Performance by Pattern:")
    print("-" * 80)
    for pattern in df_trades['pattern'].unique():
        pattern_trades = df_trades[df_trades['pattern'] == pattern]
        pattern_wins = pattern_trades[pattern_trades['pnl'] > 0]
        pattern_pnl = pattern_trades['pnl'].sum()
        pattern_wr = len(pattern_wins) / len(pattern_trades) * 100 if len(pattern_trades) > 0 else 0
        
        print(f"   {pattern:25s}: {len(pattern_trades):3d} trades | "
              f"WR: {pattern_wr:5.1f}% | P&L: ${pattern_pnl:+8.2f}")
    
    # Analyze by direction
    print(f"\n🎯 Performance by Direction:")
    print("-" * 80)
    for direction in ['long', 'short']:
        dir_trades = df_trades[df_trades['direction'] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades['pnl'] > 0]
            dir_pnl = dir_trades['pnl'].sum()
            dir_wr = len(dir_wins) / len(dir_trades) * 100
            
            print(f"   {direction.upper():5s}: {len(dir_trades):3d} trades | "
                  f"WR: {dir_wr:5.1f}% | P&L: ${dir_pnl:+8.2f}")
    
    # Show worst performers
    print(f"\n❌ Worst Performing Patterns (need reversal):")
    print("-" * 80)
    pattern_performance = df_trades.groupby('pattern')['pnl'].agg(['sum', 'count', 'mean'])
    pattern_performance = pattern_performance.sort_values('sum')
    
    for idx, (pattern, row) in enumerate(pattern_performance.head(5).iterrows()):
        if row['sum'] < 0:
            print(f"   {idx+1}. {pattern:25s}: ${row['sum']:+8.2f} "
                  f"({int(row['count'])} trades, avg: ${row['mean']:+.2f})")
    
    # Price movement analysis
    print(f"\n📉 Price Movement Analysis:")
    print("-" * 80)
    
    actual_movements = []
    for trade in trades[:20]:  # Analyze first 20 trades
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        direction = trade['direction']
        
        if direction == 'long':
            expected_move = 'UP'
            actual_move = 'UP' if exit_price > entry_price else 'DOWN'
        else:
            expected_move = 'DOWN'
            actual_move = 'DOWN' if exit_price < entry_price else 'UP'
        
        correct = expected_move == actual_move
        actual_movements.append({
            'pattern': trade['pattern'],
            'direction': direction,
            'expected': expected_move,
            'actual': actual_move,
            'correct': correct,
            'pnl': trade['pnl']
        })
    
    correct_predictions = sum(1 for m in actual_movements if m['correct'])
    print(f"   Price moved as expected: {correct_predictions}/{len(actual_movements)} "
          f"({correct_predictions/len(actual_movements)*100:.1f}%)")
    
    # Detailed trade log
    print(f"\n📝 Sample Losing Trades (first 5):")
    print("-" * 80)
    losing_trades = [t for t in trades if t['pnl'] < 0][:5]
    for i, trade in enumerate(losing_trades, 1):
        print(f"\n   Loss #{i}:")
        print(f"      Pattern: {trade['pattern']} ({trade['direction'].upper()})")
        print(f"      Entry: ${trade['entry_price']:.6f} → Exit: ${trade['exit_price']:.6f}")
        print(f"      Expected: {'UP' if trade['direction']=='long' else 'DOWN'}")
        
        move_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
        print(f"      Actual move: {move_pct:+.2f}%")
        print(f"      P&L: ${trade['pnl']:.2f}")

else:
    print("\n⚠️  No trades executed!")
    print("   Possible reasons:")
    print("   - No patterns detected")
    print("   - Pattern strength too low (<0.6)")
    print("   - Pattern probability too low (<0.6)")

# Save detailed log
print("\n\n💾 Saving detailed trade log...")
if len(trades) > 0:
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('backtest_detailed_log.csv', index=False)
    print(f"✅ Saved to: backtest_detailed_log.csv ({len(trades)} trades)")

print("\n" + "=" * 80)
print("🎯 RECOMMENDATIONS")
print("=" * 80)

if len(trades) > 0 and total_pnl < 0:
    print("""
❌ SYSTEM IS LOSING MONEY - SIGNALS MIGHT STILL BE WRONG

Possible Issues:
1. Reversal logic might not be complete
2. Stop-loss/Take-profit ratios might be wrong
3. Pattern detection might be inaccurate
4. Market might be ranging (not trending)

Next Steps:
1. Check if price movements match expected directions
2. Analyze which specific patterns are losing
3. Consider adjusting SL/TP ratios
4. Try different reversal strategies
    """)
elif len(trades) > 0 and total_pnl > 0:
    print(f"""
✅ SYSTEM IS PROFITABLE!

Total Return: {total_pnl/10000*100:+.2f}%
Win Rate: {win_rate:.1f}%

Keep monitoring and consider:
- Optimizing position sizing
- Fine-tuning entry/exit rules
- Adding more filters for pattern quality
    """)
else:
    print("""
⚠️  NO TRADES - SYSTEM TOO CONSERVATIVE

Consider:
- Lowering pattern strength threshold (currently 0.6)
- Lowering probability threshold (currently 0.6)
- Checking if patterns are being detected at all
    """)

print("\n" + "=" * 80)
print("Analysis complete! Check backtest_detailed_log.csv for full details.")
print("=" * 80)
