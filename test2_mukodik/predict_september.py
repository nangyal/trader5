"""
Predict patterns on September 2025 data using August-trained model
Uses DOGEUSDT-trades-2025-09.csv and the trained model
"""

import pandas as pd
import numpy as np
from forex_pattern_classifier import *
import os
from datetime import datetime

def resample_ticks_to_hourly(csv_file):
    """
    Resample tick/trade data to hourly candlesticks
    Matches the format used in training (DOGEUSDT-1h-2025-08.csv)
    """
    print(f"Loading tick data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Original tick data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert timestamp (milliseconds) to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('datetime')
    
    # Create OHLCV data from ticks
    print("Resampling to 1-hour candlesticks...")
    ohlcv = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum'
    })
    
    # Flatten column names
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Remove NaN rows
    ohlcv = ohlcv.dropna()
    
    print(f"Resampled data shape: {ohlcv.shape}")
    print(f"Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
    
    return ohlcv


def main_prediction():
    """Main function to predict patterns on September data"""
    print("="*70)
    print("PATTERN PREDICTION ON SEPTEMBER 2025 DATA")
    print("Using August-trained model")
    print("="*70)
    
    # Configuration
    september_csv = 'data/DOGEUSDT-trades-2025-09.csv'
    model_path = 'enhanced_forex_pattern_model.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run enhanced_main.py first to train the model.")
        return
    
    # 1. Resample September tick data to hourly
    df_september = resample_ticks_to_hourly(september_csv)
    
    # 2. Load the trained model
    print(f"\nLoading trained model from {model_path}...")
    classifier = EnhancedForexPatternClassifier()
    classifier.load_model(model_path)
    print("Model loaded successfully!")
    
    # 3. Make predictions on September data
    print("\n--- Making Predictions on September Data ---")
    predictions, probabilities = classifier.predict(df_september)
    
    # 4. Calculate pattern strength scores
    print("\n--- Calculating Pattern Strength Scores ---")
    strength_scores = []
    for i in range(len(df_september)):
        if predictions[i] != 'no_pattern':
            score = PatternStrengthScorer.calculate_pattern_strength(
                df_september, predictions[i], i
            )
        else:
            score = 0.0
        strength_scores.append(score)
    
    strength_scores = np.array(strength_scores)
    
    # 5. Display pattern summary
    print("\n" + "="*70)
    print("PATTERN DETECTION RESULTS - SEPTEMBER 2025")
    print("="*70)
    
    pattern_counts = pd.Series(predictions).value_counts()
    print("\nDetected Patterns:")
    for pattern, count in pattern_counts.items():
        percentage = count / len(predictions) * 100
        avg_conf = np.mean([probabilities[i].max() for i in range(len(predictions)) if predictions[i] == pattern])
        avg_strength = np.mean([strength_scores[i] for i in range(len(predictions)) if predictions[i] == pattern])
        
        if pattern != 'no_pattern':
            print(f"  {pattern}: {count} ({percentage:.2f}%)")
            print(f"    Avg Confidence: {avg_conf:.3f}")
            print(f"    Avg Strength: {avg_strength:.3f}")
    
    # 6. Run backtesting to calculate profit
    print("\n--- Running Backtest on September Data ---")
    backtester = BacktestingEngine(
        initial_capital=10000,
        risk_per_trade=0.02,
        take_profit_multiplier=2.0
    )
    
    backtest_results = backtester.run_backtest(
        df_september, predictions, probabilities, strength_scores
    )
    
    # 7. Create visualizations
    if backtest_results:
        print("\n--- Creating Visualizations ---")
        
        # Create output directory
        output_dir = f"september_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot equity curve
        equity_curve_path = os.path.join(output_dir, 'equity_curve_september.png')
        backtester.plot_equity_curve(equity_curve_path)
        
        # Create interactive dashboard
        dashboard = InteractiveDashboard()
        dashboard.html_dir = output_dir  # Set output directory
        
        # Show first 500 candles for visibility
        display_size = min(500, len(df_september))
        dashboard.create_candlestick_chart(
            df_september.head(display_size), 
            predictions[:display_size], 
            probabilities[:display_size],
            strength_scores[:display_size]
        )
        
        dashboard.create_pattern_distribution_chart(predictions)
        dashboard.create_backtest_dashboard(backtest_results)
        
        print(f"\n✓ Visualizations saved to '{output_dir}/'")
        print(f"  - equity_curve_september.png")
        print(f"  - pattern_dashboard_*.html")
        print(f"  - pattern_distribution.html")
        print(f"  - backtest_dashboard_*.html")
    
    # 8. Print detailed summary
    print("\n" + "="*70)
    print("SUMMARY - SEPTEMBER 2025 BACKTEST")
    print("="*70)
    print(f"Training Period: August 2025 (hourly data)")
    print(f"Testing Period: September 2025 (resampled to hourly)")
    print(f"Total Candles Analyzed: {len(df_september)}")
    print(f"Patterns Detected: {len(predictions[predictions != 'no_pattern'])}")
    
    if backtest_results:
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Initial Capital: ${backtest_results['final_capital'] - backtest_results['total_pnl']:,.2f}")
        print(f"  Final Capital: ${backtest_results['final_capital']:,.2f}")
        print(f"  Total P&L: ${backtest_results['total_pnl']:,.2f}")
        print(f"  Return: {backtest_results['return_pct']:.2f}%")
        print(f"  Win Rate: {backtest_results['win_rate']*100:.2f}%")
        print(f"  Profit Factor: {backtest_results['profit_factor']:.2f}")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Total Trades: {backtest_results['total_trades']}")
        
        print("\n  Pattern-Specific Win Rates:")
        for pattern, win_rate in backtest_results['pattern_win_rates'].items():
            print(f"    {pattern}: {win_rate:.1f}%")
    
    print("="*70)
    
    return df_september, predictions, backtest_results


if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")
    required = ['pandas', 'numpy', 'talib', 'xgboost', 'sklearn', 'plotly']
    missing = []
    
    for lib in required:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError:
            missing.append(lib)
            print(f"  ✗ {lib} (missing)")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\nAll dependencies available. Starting prediction...\n")
        main_prediction()
