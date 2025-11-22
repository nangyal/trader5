"""
Compare all three strategies: Normal, Hedging, and Zone Recovery
Test on September and October 2025 data
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_zone_recovery import ZoneRecoveryBacktestEngine
from backtest_with_hedging import HedgingBacktestEngine
import os
from datetime import datetime


def resample_ticks_to_hourly(csv_file):
    """Resample tick/trade data to hourly candlesticks"""
    print(f"Loading tick data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('datetime')
    
    print("Resampling to 1-hour candlesticks...")
    ohlcv = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum'
    })
    
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = ohlcv.dropna()
    
    print(f"Resampled data shape: {ohlcv.shape}")
    return ohlcv


def compare_all_strategies(month_name, csv_file, model_path):
    """Compare all three strategies on a specific month"""
    print("\n" + "="*90)
    print(f"STRATEGY COMPARISON - {month_name.upper()}")
    print("="*90)
    
    # Load data
    df_month = resample_ticks_to_hourly(csv_file)
    
    # Load model
    classifier = EnhancedForexPatternClassifier()
    classifier.load_model(model_path)
    
    # Make predictions
    predictions, probabilities = classifier.predict(df_month)
    
    # Calculate pattern strength
    strength_scores = np.array([
        PatternStrengthScorer.calculate_pattern_strength(df_month, predictions[i], i) 
        if predictions[i] != 'no_pattern' else 0.0
        for i in range(len(df_month))
    ])
    
    # Strategy 1: Normal (baseline)
    print(f"\n{'='*90}")
    print(f"STRATEGY 1: NORMAL (BASELINE) - {month_name.upper()}")
    print(f"{'='*90}")
    
    backtester_normal = BacktestingEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0
    )
    results_normal = backtester_normal.run_backtest(
        df_month, predictions, probabilities, strength_scores
    )
    
    # Strategy 2: Hedging
    print(f"\n{'='*90}")
    print(f"STRATEGY 2: HEDGING - {month_name.upper()}")
    print(f"{'='*90}")
    
    backtester_hedge = HedgingBacktestEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0,
        enable_hedging=True,
        hedge_threshold=0.15,
        hedge_ratio=0.5
    )
    results_hedge = backtester_hedge.run_backtest(
        df_month, predictions, probabilities, strength_scores
    )
    
    # Strategy 3: Zone Recovery
    print(f"\n{'='*90}")
    print(f"STRATEGY 3: ZONE RECOVERY - {month_name.upper()}")
    print(f"{'='*90}")
    
    backtester_zone = ZoneRecoveryBacktestEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0,
        enable_zone_recovery=True,
        recovery_zone_size=0.01,
        max_recovery_zones=5,
        recovery_position_multiplier=0.5
    )
    results_zone = backtester_zone.run_backtest(
        df_month, predictions, probabilities, strength_scores
    )
    
    # Create output directory
    output_dir = f"{month_name}_all_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save equity curves
    if results_normal:
        backtester_normal.plot_equity_curve(
            os.path.join(output_dir, f'equity_normal_{month_name}.png')
        )
    if results_hedge:
        backtester_hedge.plot_equity_curve(
            os.path.join(output_dir, f'equity_hedge_{month_name}.png')
        )
    if results_zone:
        backtester_zone.plot_equity_curve(
            os.path.join(output_dir, f'equity_zone_{month_name}.png')
        )
    
    # Print comprehensive comparison
    print(f"\n{'='*90}")
    print(f"COMPREHENSIVE COMPARISON - {month_name.upper()}")
    print(f"{'='*90}")
    
    if all([results_normal, results_hedge, results_zone]):
        print(f"\n{'Metric':<25} {'Normal':<20} {'Hedging':<20} {'Zone Recovery':<20}")
        print("-"*90)
        
        metrics = [
            ('Final Capital', 'final_capital', '$'),
            ('Return %', 'return_pct', '%'),
            ('Total P&L', 'total_pnl', '$'),
            ('Win Rate %', 'win_rate', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Max Drawdown %', 'max_drawdown', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Total Trades', 'total_trades', ''),
        ]
        
        for label, key, unit in metrics:
            normal_val = results_normal[key]
            hedge_val = results_hedge[key]
            zone_val = results_zone[key]
            
            if unit == '%' and key not in ['return_pct', 'max_drawdown']:
                normal_val *= 100
                hedge_val *= 100
                zone_val *= 100
            
            if unit == '$':
                print(f"{label:<25} ${normal_val:>17,.2f} ${hedge_val:>17,.2f} ${zone_val:>17,.2f}")
            elif unit == '%':
                print(f"{label:<25} {normal_val:>18.2f}% {hedge_val:>18.2f}% {zone_val:>18.2f}%")
            else:
                print(f"{label:<25} {normal_val:>19.2f} {hedge_val:>19.2f} {zone_val:>19.2f}")
        
        print("-"*90)
        
        # Find best strategy
        returns = {
            'Normal': results_normal['return_pct'],
            'Hedging': results_hedge['return_pct'],
            'Zone Recovery': results_zone['return_pct']
        }
        best_strategy = max(returns, key=returns.get)
        best_return = returns[best_strategy]
        
        print(f"\n🏆 Best Strategy: {best_strategy} ({best_return:+.2f}%)")
        
        # Calculate improvements
        hedge_improvement = results_hedge['return_pct'] - results_normal['return_pct']
        zone_improvement = results_zone['return_pct'] - results_normal['return_pct']
        
        print(f"\n📊 Improvements over Normal:")
        print(f"   Hedging: {hedge_improvement:+.2f}% ({(hedge_improvement/abs(results_normal['return_pct'])*100) if results_normal['return_pct'] != 0 else 0:+.1f}%)")
        print(f"   Zone Recovery: {zone_improvement:+.2f}% ({(zone_improvement/abs(results_normal['return_pct'])*100) if results_normal['return_pct'] != 0 else 0:+.1f}%)")
        
        # Drawdown comparison
        drawdowns = {
            'Normal': results_normal['max_drawdown'],
            'Hedging': results_hedge['max_drawdown'],
            'Zone Recovery': results_zone['max_drawdown']
        }
        lowest_dd = min(drawdowns, key=drawdowns.get)
        
        print(f"\n🛡️ Lowest Drawdown: {lowest_dd} ({drawdowns[lowest_dd]:.2f}%)")
        
        # Additional metrics
        print(f"\n📈 Strategy Details:")
        print(f"   Hedging: {results_hedge.get('hedge_trades', 0)} hedge trades")
        print(f"   Zone Recovery: {results_zone.get('recovery_trades', 0)} recovery trades")
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print("="*90)
    
    return results_normal, results_hedge, results_zone


def main():
    """Main comparison function"""
    print("="*90)
    print("COMPLETE STRATEGY COMPARISON")
    print("Testing: Normal vs Hedging vs Zone Recovery")
    print("="*90)
    
    model_path = 'enhanced_forex_pattern_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Test September
    print("\n" + "🗓️  SEPTEMBER 2025 ".center(90, "="))
    sept_normal, sept_hedge, sept_zone = compare_all_strategies(
        'september',
        'data/DOGEUSDT-trades-2025-09.csv',
        model_path
    )
    
    # Test October
    print("\n" + "🗓️  OCTOBER 2025 ".center(90, "="))
    oct_normal, oct_hedge, oct_zone = compare_all_strategies(
        'october',
        'data/DOGEUSDT-trades-2025-10.csv',
        model_path
    )
    
    # Overall summary
    print("\n" + "="*90)
    print("OVERALL STRATEGY PERFORMANCE SUMMARY")
    print("="*90)
    
    if all([sept_normal, sept_hedge, sept_zone, oct_normal, oct_hedge, oct_zone]):
        # Calculate combined returns
        combined_returns = {
            'Normal': sept_normal['return_pct'] + oct_normal['return_pct'],
            'Hedging': sept_hedge['return_pct'] + oct_hedge['return_pct'],
            'Zone Recovery': sept_zone['return_pct'] + oct_zone['return_pct']
        }
        
        print(f"\n📊 Combined Returns (Sep + Oct):")
        print(f"{'Strategy':<20} {'Return':<15} {'Rank':<10}")
        print("-"*45)
        
        sorted_strategies = sorted(combined_returns.items(), key=lambda x: x[1], reverse=True)
        for rank, (strategy, ret) in enumerate(sorted_strategies, 1):
            print(f"{strategy:<20} {ret:>13.2f}% {'🏆' if rank == 1 else '  '} #{rank}")
        
        print("\n📈 Monthly Breakdown:")
        print(f"\nSeptember:")
        print(f"  Normal: {sept_normal['return_pct']:>8.2f}% | Hedging: {sept_hedge['return_pct']:>8.2f}% | Zone Recovery: {sept_zone['return_pct']:>8.2f}%")
        
        print(f"\nOctober:")
        print(f"  Normal: {oct_normal['return_pct']:>8.2f}% | Hedging: {oct_hedge['return_pct']:>8.2f}% | Zone Recovery: {oct_zone['return_pct']:>8.2f}%")
        
        print("\n🎯 Strategy Effectiveness:")
        best_overall = max(combined_returns, key=combined_returns.get)
        print(f"  🏆 Best Overall: {best_overall} ({combined_returns[best_overall]:.2f}%)")
        
        # Risk-adjusted performance
        combined_sharpe = {
            'Normal': (sept_normal['sharpe_ratio'] + oct_normal['sharpe_ratio']) / 2,
            'Hedging': (sept_hedge['sharpe_ratio'] + oct_hedge['sharpe_ratio']) / 2,
            'Zone Recovery': (sept_zone['sharpe_ratio'] + oct_zone['sharpe_ratio']) / 2
        }
        
        best_sharpe = max(combined_sharpe, key=combined_sharpe.get)
        print(f"  📊 Best Risk-Adjusted: {best_sharpe} (Sharpe: {combined_sharpe[best_sharpe]:.2f})")
        
        # Drawdown comparison
        avg_drawdown = {
            'Normal': (sept_normal['max_drawdown'] + oct_normal['max_drawdown']) / 2,
            'Hedging': (sept_hedge['max_drawdown'] + oct_hedge['max_drawdown']) / 2,
            'Zone Recovery': (sept_zone['max_drawdown'] + oct_zone['max_drawdown']) / 2
        }
        
        lowest_dd = min(avg_drawdown, key=avg_drawdown.get)
        print(f"  🛡️ Lowest Avg Drawdown: {lowest_dd} ({avg_drawdown[lowest_dd]:.2f}%)")
    
    print("\n" + "="*90)
    print("Comparison complete! Check the generated directories for detailed charts.")
    print("="*90)


if __name__ == "__main__":
    main()
