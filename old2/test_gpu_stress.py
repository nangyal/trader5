#!/usr/bin/env python3
"""
GPU Stress Test - Generate lots of predictions to see GPU usage
Run nvidia-smi in another terminal to watch GPU utilization
"""
import pandas as pd
import numpy as np
from classes.classification import Classifier
from pathlib import Path
import time

print("=" * 70)
print("GPU STRESS TEST - Watch nvidia-smi in another terminal!")
print("=" * 70)

# Load classifier
print("\n1. Loading classifier with GPU...")
classifier = Classifier()
classifier.load_model_if_exists(
    Path('models/enhanced_forex_pattern_model.pkl'), 
    force_gpu=True
)

# Create large test dataset
print("\n2. Creating large test dataset (100,000 samples)...")
n_samples = 100000
test_df = pd.DataFrame({
    'open': np.random.randn(n_samples) * 100 + 50000,
    'high': np.random.randn(n_samples) * 100 + 50100,
    'low': np.random.randn(n_samples) * 100 + 49900,
    'close': np.random.randn(n_samples) * 100 + 50000,
    'volume': np.random.randn(n_samples) * 1000 + 10000,
})

print(f"   Dataset shape: {test_df.shape}")

# Run predictions in a loop
print("\n3. Running predictions (10 iterations)...")
print("   👀 WATCH nvidia-smi NOW to see GPU usage!")
print()

total_time = 0
for i in range(10):
    start = time.time()
    
    # This will use GPU for prediction
    predictions, probabilities = classifier.predict(test_df)
    
    elapsed = time.time() - start
    total_time += elapsed
    
    print(f"   Iteration {i+1}/10: {elapsed*1000:.0f}ms "
          f"({len(predictions)/elapsed:.0f} samples/sec)")
    
    # Small delay to allow monitoring
    time.sleep(0.1)

print(f"\n4. Results:")
print(f"   Total predictions: {10 * n_samples:,}")
print(f"   Total time: {total_time:.2f}s")
print(f"   Average speed: {(10 * n_samples)/total_time:.0f} samples/sec")

# Check GPU config
params = classifier.model.model.get_xgb_params()
print(f"\n5. GPU Configuration:")
print(f"   device: {params.get('device')}")
print(f"   predictor: {params.get('predictor')}")
print(f"   tree_method: {params.get('tree_method')}")

print("\n" + "=" * 70)
print("If GPU utilization was 0%, it means:")
print("  • Predictions are too fast (batch processing helps)")
print("  • GPU is used but monitoring interval missed it")
print("  • Or XGBoost is not actually using GPU for predict")
print("=" * 70)
