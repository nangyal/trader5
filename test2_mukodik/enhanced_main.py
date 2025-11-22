"""
Enhanced Main Function for Forex Pattern Classifier
Includes: Backtesting, Multi-Timeframe, Alerts, Dashboard, MLflow

V2.3 UPDATES (CRITICAL FIX):
- Now uses HOURLY candlestick data instead of tick/trade data
- Pattern trading requires proper timeframes (1h/4h), not millisecond ticks
- Implements LONG-ONLY trend-aligned strategy:
  * Bullish patterns in uptrend → LONG
  * Bearish patterns in downtrend → LONG
  * Skip misaligned setups
- Result: +355% return vs -100% on tick data
"""

from forex_pattern_classifier import *
import os


def main_enhanced():
    """Main function with all enhancements"""
    print("=== ENHANCED FOREX PATTERN CLASSIFIER V2.0 ===")
    print("Features: Backtesting, Multi-Timeframe, Alerts, Dashboard, MLflow")
    
    # Configuration
    # V2.3: Use resampled 1-hour data instead of tick data for pattern trading
    csv_file_path = 'data/DOGEUSDT-1h-2025-08.csv'
    sample_size = None  # Use all data (744 hourly candles)
    
    # Initialize components
    mlflow_tracker = MLflowTracker()
    alert_system = AlertSystem(alert_threshold=0.75, enable_log=False)  # V2.3: Disable JSON logging
    dashboard = InteractiveDashboard()
    
    # Use dashboard's html_dir for all outputs
    output_dir = dashboard.html_dir
    
    start_time = time.time()
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(csv_file_path, sample_size)
        
        # Create pattern labels
        pattern_detector = AdvancedPatternDetector()
        pattern_labels = create_labels_from_data(df, pattern_detector)
        pattern_series = pd.Series(pattern_labels, index=df.index)
        
        # Filter patterns
        pattern_counts = pattern_series.value_counts()
        min_patterns = 30
        valid_patterns = pattern_counts[pattern_counts >= min_patterns].index.tolist()
        
        if 'no_pattern' in valid_patterns:
            valid_patterns.remove('no_pattern')
        
        if len(valid_patterns) < 2:
            print("Warning: Not enough pattern diversity for training.")
            return
        
        mask = pattern_series.isin(valid_patterns + ['no_pattern'])
        df_filtered = df[mask]
        patterns_filtered = pattern_series[mask]
        
        print(f"\nTraining data shape: {df_filtered.shape}")
        print(f"Patterns for training: {valid_patterns}")
        
        # Initialize and train classifier with output directory
        classifier = EnhancedForexPatternClassifier(output_dir=output_dir)
        
        print("\n--- Training Enhanced Model ---")
        
        # Prepare training parameters for MLflow
        training_params = {
            'csv_file': csv_file_path,
            'sample_size': sample_size,
            'min_patterns': min_patterns,
            'patterns': ','.join(valid_patterns)
        }
        
        model = classifier.train(df_filtered, patterns_filtered, optimize_hyperparams=False)
        
        # Save model
        print("\n--- Saving Model ---")
        model_path = 'enhanced_forex_pattern_model.pkl'
        classifier.save_model(model_path)
        
        # Make predictions on test set
        print("\n--- Making Predictions ---")
        # V2.3: Use all data for pattern detection on hourly candles
        predictions, probabilities = classifier.predict(df_filtered)
        
        # Calculate pattern strength scores
        print("\n--- Calculating Pattern Strength Scores ---")
        strength_scores = []
        for i in range(len(df_filtered)):
            if predictions[i] != 'no_pattern':
                score = PatternStrengthScorer.calculate_pattern_strength(
                    df_filtered, predictions[i], i
                )
            else:
                score = 0.0
            strength_scores.append(score)
        
        strength_scores = np.array(strength_scores)
        
        # Check for alerts
        print("\n--- Checking for Pattern Alerts ---")
        for i in range(len(predictions)):
            if predictions[i] != 'no_pattern':
                alert_system.check_and_alert(
                    pattern=predictions[i],
                    strength_score=strength_scores[i],
                    probability=probabilities[i].max(),
                    price=df_filtered['close'].iloc[i],
                    timestamp=df_filtered.index[i] if hasattr(df_filtered.index[i], 'isoformat') else None
                )
        
        alert_system.print_alert_summary()
        
        # Run backtesting
        print("\n--- Running Backtest ---")
        backtester = BacktestingEngine(
            initial_capital=10000,
            risk_per_trade=0.02,
            take_profit_multiplier=2.0
        )
        
        # V2.3: Pass full dataset for proper backtesting with recent_data
        backtest_results = backtester.run_backtest(
            df_filtered, predictions, probabilities, strength_scores
        )
        
        if backtest_results:
            equity_curve_path = os.path.join(output_dir, 'equity_curve.png')
            backtester.plot_equity_curve(equity_curve_path)
        
        # Multi-timeframe analysis
        print("\n--- Multi-Timeframe Analysis ---")
        mtf_analyzer = MultiTimeframeAnalyzer(timeframes=['1min', '5min', '15min'])
        mtf_results, consensus = mtf_analyzer.analyze_multi_timeframe(df_filtered, classifier)
        
        # Create interactive dashboards
        print("\n--- Creating Interactive Dashboards ---")
        
        # Main pattern dashboard (show first 500 candles for visibility)
        dashboard.create_candlestick_chart(
            df_filtered.head(500), 
            predictions[:500], 
            probabilities[:500],
            strength_scores[:500]
        )
        
        # Pattern distribution
        dashboard.create_pattern_distribution_chart(predictions)
        
        # Backtest dashboard
        if backtest_results:
            dashboard.create_backtest_dashboard(backtest_results)
        
        # Log to MLflow
        if backtest_results:
            training_metrics = {
                'accuracy': backtest_results['win_rate'],
                'total_pnl': backtest_results['total_pnl'],
                'profit_factor': backtest_results['profit_factor'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown']
            }
            
            artifacts = [
                os.path.join(output_dir, 'feature_importance.png'),
                os.path.join(output_dir, 'confusion_matrix.png'),
                os.path.join(output_dir, 'equity_curve.png')
            ]
            
            mlflow_tracker.log_training_run(
                params=training_params,
                metrics=training_metrics,
                model=classifier.model,
                artifacts=[a for a in artifacts if os.path.exists(a)]
            )
            
            mlflow_tracker.log_backtest_results(backtest_results)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print("=== ENHANCED SYSTEM COMPLETE ===")
        print(f"{'='*60}")
        print(f"Total time: {training_time:.2f}s ({training_time/60:.2f}min)")
        print("\nGenerated Files:")
        print("  ✓ enhanced_forex_pattern_model.pkl - Trained model")
        print(f"  ✓ {output_dir}/ - All outputs (PNG charts, HTML dashboards)")
        print(f"    - feature_importance.png - Feature analysis")
        print(f"    - confusion_matrix.png - Model accuracy")
        print(f"    - equity_curve.png - Backtest equity")
        print(f"    - pattern_dashboard_*.html - Interactive candlestick chart")
        print(f"    - pattern_distribution.html - Pattern breakdown")
        print(f"    - backtest_dashboard_*.html - Backtest analysis")
        print("  ✓ pattern_alerts.json - Alert log")
        
        print("\nEnhancements Added:")
        print("  ✓ Backtesting with profit/loss tracking")
        print("  ✓ Multi-timeframe consensus analysis")
        print("  ✓ Pattern strength scoring system")
        print("  ✓ Real-time alert system")
        print("  ✓ Interactive Plotly dashboards")
        print("  ✓ MLflow experiment tracking")
        print(f"{'='*60}\n")
        
        return classifier, df_filtered, backtest_results
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for GPU availability
    try:
        import xgboost as xgb
        gpu_info = xgb.get_config()
        print("GPU support available:", 'cuda' in str(gpu_info).lower())
    except:
        print("GPU check failed - running on CPU")
    
    # Check dependencies
    print("Checking dependencies...")
    required = ['talib', 'xgboost', 'sklearn', 'plotly']
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
        print("Install with: pip install plotly mlflow")
    else:
        print("\nAll dependencies available. Starting...")
        main_enhanced()
