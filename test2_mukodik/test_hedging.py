"""
Test hedging strategy on September and October 2025 data
Compare results with and without hedging
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
from backtest_with_hedging import HedgingBacktestEngine
import os
from datetime import datetime


def resample_ticks_to_hourly(csv_file):
    """Resample tick/trade data to hourly candlesticks"""
    print(f"Loading tick data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Original tick data shape: {df.shape}")
    
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
    print(f"Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
    
    return ohlcv


def test_month_with_hedging(month_name, csv_file, model_path):
    """Test a specific month with and without hedging"""
    print("\n" + "="*80)
    print(f"TESTING {month_name.upper()} WITH HEDGING")
    print("="*80)
    
    # Resample data
    df_month = resample_ticks_to_hourly(csv_file)
    
    # Load model
    print(f"\nLoading trained model from {model_path}...")
    classifier = EnhancedForexPatternClassifier()
    classifier.load_model(model_path)
    
    # Make predictions
    print(f"\n--- Making Predictions on {month_name} Data ---")
    predictions, probabilities = classifier.predict(df_month)
    
    # Calculate pattern strength scores
    print(f"\n--- Calculating Pattern Strength Scores ---")
    strength_scores = []
    for i in range(len(df_month)):
        if predictions[i] != 'no_pattern':
            score = PatternStrengthScorer.calculate_pattern_strength(
                df_month, predictions[i], i
            )
        else:
            score = 0.0
        strength_scores.append(score)
    
    strength_scores = np.array(strength_scores)
    
    # Test WITHOUT hedging
    print(f"\n{'='*80}")
    print(f"BACKTEST WITHOUT HEDGING - {month_name.upper()}")
    print(f"{'='*80}")
    
    backtester_no_hedge = BacktestingEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0
    )
    
    results_no_hedge = backtester_no_hedge.run_backtest(
        df_month, predictions, probabilities, strength_scores
    )
    
    # Test WITH hedging
    print(f"\n{'='*80}")
    print(f"BACKTEST WITH HEDGING - {month_name.upper()}")
    print(f"{'='*80}")
    
    backtester_hedge = HedgingBacktestEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0,
        enable_hedging=True,
        hedge_threshold=0.15,  # Activate hedge at 15% drawdown
        hedge_ratio=0.5        # Hedge 50% of exposure
    )
    
    results_hedge = backtester_hedge.run_backtest(
        df_month, predictions, probabilities, strength_scores
    )
    
    # Create output directory
    output_dir = f"{month_name}_hedging_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save equity curves
    if results_no_hedge:
        backtester_no_hedge.plot_equity_curve(
            os.path.join(output_dir, f'equity_no_hedge_{month_name}.png')
        )
    
    if results_hedge:
        backtester_hedge.plot_equity_curve(
            os.path.join(output_dir, f'equity_with_hedge_{month_name}.png')
        )
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"HEDGING COMPARISON - {month_name.upper()}")
    print(f"{'='*80}")
    
    if results_no_hedge and results_hedge:
        print(f"\n{'Metric':<30} {'Without Hedge':<20} {'With Hedge':<20} {'Improvement':<15}")
        print("-"*85)
        
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
            no_hedge_val = results_no_hedge[key]
            hedge_val = results_hedge[key]
            
            if unit == '%' and key not in ['return_pct', 'max_drawdown']:
                no_hedge_val *= 100
                hedge_val *= 100
            
            if unit == '$':
                improvement = hedge_val - no_hedge_val
                imp_str = f"+${improvement:,.2f}" if improvement >= 0 else f"${improvement:,.2f}"
            elif unit == '%':
                improvement = hedge_val - no_hedge_val
                imp_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            else:
                improvement = hedge_val - no_hedge_val
                imp_str = f"+{improvement:.2f}" if improvement >= 0 else f"{improvement:.2f}"
            
            if unit == '$':
                print(f"{label:<30} ${no_hedge_val:>17,.2f} ${hedge_val:>17,.2f} {imp_str:>14}")
            elif unit == '%':
                print(f"{label:<30} {no_hedge_val:>18.2f}% {hedge_val:>18.2f}% {imp_str:>14}")
            else:
                print(f"{label:<30} {no_hedge_val:>19.2f} {hedge_val:>19.2f} {imp_str:>14}")
        
        print("-"*85)
        
        # Calculate improvement percentage
        return_improvement = ((results_hedge['return_pct'] - results_no_hedge['return_pct']) / 
                             abs(results_no_hedge['return_pct'])) * 100 if results_no_hedge['return_pct'] != 0 else 0
        
        drawdown_improvement = ((results_no_hedge['max_drawdown'] - results_hedge['max_drawdown']) / 
                               results_no_hedge['max_drawdown']) * 100 if results_no_hedge['max_drawdown'] != 0 else 0
        
        print(f"\n🎯 Key Improvements:")
        print(f"   Return improvement: {return_improvement:+.2f}%")
        print(f"   Drawdown reduction: {drawdown_improvement:.2f}%")
        
        if results_hedge['hedge_pnl'] > 0:
            print(f"   💰 Hedging added: +${results_hedge['hedge_pnl']:,.2f}")
        else:
            print(f"   ⚠️ Hedging cost: ${results_hedge['hedge_pnl']:,.2f}")
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print("="*80)
    
    return results_no_hedge, results_hedge


def main():
    """Main function to test both months"""
    print("="*80)
    print("HEDGING STRATEGY COMPARISON")
    print("Testing September and October 2025")
    print("="*80)
    
    model_path = 'enhanced_forex_pattern_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run enhanced_main.py first to train the model.")
        return
    
    # Test September
    sept_no_hedge, sept_hedge = test_month_with_hedging(
        'september',
        'data/DOGEUSDT-trades-2025-09.csv',
        model_path
    )
    
    # Test October
    oct_no_hedge, oct_hedge = test_month_with_hedging(
        'october',
        'data/DOGEUSDT-trades-2025-10.csv',
        model_path
    )
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL HEDGING IMPACT SUMMARY")
    print("="*80)
    
    if all([sept_no_hedge, sept_hedge, oct_no_hedge, oct_hedge]):
        total_no_hedge_return = sept_no_hedge['return_pct'] + oct_no_hedge['return_pct']
        total_hedge_return = sept_hedge['return_pct'] + oct_hedge['return_pct']
        
        print(f"\nCombined Returns:")
        print(f"  Without Hedging: {total_no_hedge_return:.2f}%")
        print(f"  With Hedging: {total_hedge_return:.2f}%")
        print(f"  Difference: {total_hedge_return - total_no_hedge_return:+.2f}%")
        
        print(f"\nSeptember:")
        print(f"  Without: {sept_no_hedge['return_pct']:+.2f}% | With: {sept_hedge['return_pct']:+.2f}%")
        
        print(f"\nOctober:")
        print(f"  Without: {oct_no_hedge['return_pct']:+.2f}% | With: {oct_hedge['return_pct']:+.2f}%")
        
        print("\n🛡️ Hedging effectiveness:")
        if total_hedge_return > total_no_hedge_return:
            print(f"   ✅ Hedging IMPROVED overall performance")
        else:
            print(f"   ⚠️ Hedging reduced overall performance (may have protected in October)")
    
    print("="*80)


if __name__ == "__main__":
    main()
