#!/usr/bin/env python3
"""Test GPU usage during actual prediction"""
import pandas as pd
import numpy as np
import time
import threading
import subprocess
from classes.classification import Classifier
from pathlib import Path

def monitor_gpu(stop_event):
    """Monitor GPU usage"""
    max_gpu = 0
    samples = []
    
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=0.5
            )
            gpu = int(result.stdout.strip())
            max_gpu = max(max_gpu, gpu)
            samples.append(gpu)
            
            bar = '█' * (gpu // 5)
            print(f"\rGPU: {gpu:3d}% |{bar:<20}| Max: {max_gpu:3d}%", end='', flush=True)
            
            time.sleep(0.02)  # 20ms - very fast
        except:
            pass
    
    avg_gpu = sum(samples) / len(samples) if samples else 0
    print(f"\n\n📊 Stats: Max={max_gpu}%, Avg={avg_gpu:.1f}%, Samples={len(samples)}")
    return max_gpu, avg_gpu

if __name__ == '__main__':
    print("="*70)
    print("GPU PREDICTION TEST")
    print("="*70)
    
    # Load classifier with GPU
    print("\n1. Loading classifier with GPU...")
    clf = Classifier()
    clf.load_model_if_exists(Path('models/enhanced_forex_pattern_model.pkl'), force_gpu=True)
    
    # Create test data
    print("\n2. Creating test dataset (10,000 samples)...")
    n = 10000
    test_df = pd.DataFrame({
        'open': np.random.randn(n) * 100 + 50000,
        'high': np.random.randn(n) * 100 + 50100,
        'low': np.random.randn(n) * 100 + 49900,
        'close': np.random.randn(n) * 100 + 50000,
        'volume': np.random.randn(n) * 1000 + 10000,
    })
    
    print(f"   Shape: {test_df.shape}")
    
    # Start GPU monitoring
    print("\n3. Starting GPU monitor...")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event,), daemon=True)
    monitor_thread.start()
    time.sleep(0.5)
    
    # Run predictions multiple times
    print("\n4. Running predictions (20 iterations)...\n")
    
    total_time = 0
    for i in range(20):
        start = time.time()
        predictions, probabilities = clf.predict(test_df)
        elapsed = time.time() - start
        total_time += elapsed
        
        if i % 5 == 0:
            print(f"\n   Iteration {i+1}: {elapsed*1000:.1f}ms", flush=True)
        
        time.sleep(0.05)  # Small delay between iterations
    
    # Stop monitoring
    time.sleep(0.5)
    stop_event.set()
    time.sleep(0.5)
    
    print(f"\n\n5. Results:")
    print(f"   Total predictions: {20 * n:,}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Speed: {(20 * n)/total_time:.0f} samples/sec")
    
    print("\n" + "="*70)
    print("💡 If Max GPU > 5%, GPU is being used for prediction")
    print("💡 If Max GPU < 2%, prediction is likely CPU-only")
    print("="*70)
