#!/usr/bin/env python3
"""
Direct test: Load real BTCUSDT data and predict with GPU monitoring
"""
import sys
sys.path.insert(0, '/home/nangyal/Desktop/v4')

from classes.training import find_csv_for_coin, load_tick_data, tick_to_ohlcv
from classes.classification import Classifier
from pathlib import Path
import time

print("="*70)
print("REAL DATA GPU PREDICTION TEST")
print("="*70)

# Load real data
print("\n1. Loading BTCUSDT data...")
csvs = find_csv_for_coin('BTCUSDT')
print(f"   Found {len(csvs)} CSV files")

if csvs:
    df = load_tick_data(csvs[0])
    print(f"   Loaded {len(df)} tick records")
    
    # Resample to 1min
    print("\n2. Resampling to 1min...")
    ohlc = tick_to_ohlcv(df, '1min')
    print(f"   OHLC shape: {ohlc.shape}")
    
    # Load classifier
    print("\n3. Loading classifier with GPU...")
    clf = Classifier()
    clf.load_model_if_exists(Path('models/enhanced_forex_pattern_model.pkl'), force_gpu=True)
    
    # Predict
    print("\n4. Running prediction (watch nvidia-smi in another terminal!)...")
    print("   👀 RUN THIS NOW: watch -n 0.1 nvidia-smi")
    print()
    
    time.sleep(3)  # Give time to start nvidia-smi
    
    for i in range(10):
        print(f"\n   === Iteration {i+1}/10 ===")
        start = time.time()
        predictions, probs = clf.predict(ohlc)
        elapsed = time.time() - start
        
        patterns = set(predictions[predictions != 'no_pattern'])
        print(f"   Time: {elapsed:.2f}s, Patterns found: {len(patterns)}")
        
        time.sleep(0.5)
    
    print("\n✅ Done! Did you see GPU usage spike?")
else:
    print("❌ No BTCUSDT data found!")
