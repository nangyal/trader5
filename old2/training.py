"""
Training script for the crypto trading framework.
This script trains the classifier model based on historical data.
"""

import config
from classes.classification import Classifier
import pandas as pd
import numpy as np
from pathlib import Path
from classes.training import find_csv_for_coin, load_tick_data, tick_to_ohlcv


def prepare_training_data():
    """
    Load historical data and prepare features for training.
    Returns X (features) and y (labels).
    """
    all_data = []
    
    print("Loading training data from coins...")
    for coin in config.COINS:
        csvs = find_csv_for_coin(coin)
        if not csvs:
            print(f"  No data for {coin}")
            continue
        
        for csv_path in csvs:
            df = load_tick_data(csv_path)
            if df is None or len(df) < 1:
                continue
            
            # Convert to OHLCV for feature extraction
            for tf in config.TIMEFRAMES:
                ohlc = tick_to_ohlcv(df, tf)
                if ohlc is None or len(ohlc) < 50:
                    continue
                
                print(f"  Loaded {coin} {tf}: {len(ohlc)} candles")
                all_data.append({
                    'coin': coin,
                    'timeframe': tf,
                    'data': ohlc
                })
    
    if not all_data:
        print("ERROR: No training data found!")
        return None, None
    
    # Prepare features and labels
    # This is a simplified example - adapt based on your actual feature engineering
    X_list = []
    y_list = []
    
    for item in all_data:
        ohlc = item['data']
        
        # Example feature extraction (customize based on your needs)
        features_df = pd.DataFrame()
        features_df['returns'] = ohlc['close'].pct_change()
        features_df['volatility'] = ohlc['close'].rolling(20).std()
        features_df['range'] = (ohlc['high'] - ohlc['low']) / ohlc['close']
        features_df['volume_ma'] = ohlc['volume'].rolling(20).mean()
        
        # Example labeling: 1 if price goes up by 1% in next 5 candles, 0 otherwise
        future_return = ohlc['close'].shift(-5) / ohlc['close'] - 1
        labels = (future_return > 0.01).astype(int)
        
        # Drop NaN rows
        combined = pd.concat([features_df, labels.rename('label')], axis=1).dropna()
        
        if len(combined) > 0:
            X_list.append(combined.drop('label', axis=1))
            y_list.append(combined['label'])
    
    if not X_list:
        print("ERROR: No features could be extracted!")
        return None, None
    
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    
    print(f"\nTraining data prepared: {len(X)} samples, {X.shape[1]} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_model():
    """
    Train the classifier and save it to disk.
    """
    print("=" * 60)
    print("CRYPTO TRADING FRAMEWORK - MODEL TRAINING")
    print("=" * 60)
    
    # Ensure directories exist
    config.ensure_dirs()
    
    # Load classifier
    clf = Classifier()
    
    # Prepare training data
    X, y = prepare_training_data()
    
    if X is None or y is None or len(X) == 0:
        print("\nERROR: Could not prepare training data. Check your data files.")
        return
    
    # Train the model
    print("\nStarting model training...")
    try:
        clf.train(X, y)
        print("✓ Model training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Save the model
    print(f"\nSaving model to {config.MODEL_PATH}...")
    try:
        clf.save_model(config.MODEL_PATH)
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"✗ Failed to save model: {e}")
        return
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"You can now run 'python start.py' to backtest with this model")


if __name__ == '__main__':
    train_model()
