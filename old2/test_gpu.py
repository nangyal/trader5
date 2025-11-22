#!/usr/bin/env python3
"""GPU usage test for the classifier"""
import joblib
import pandas as pd
import numpy as np

print("=" * 60)
print("GPU USAGE TEST")
print("=" * 60)

# Load model
print("\n1. Loading model...")
model_data = joblib.load('models/enhanced_forex_pattern_model.pkl')
model = model_data['model']

print("\n2. Model XGBoost parameters:")
params = model.get_xgb_params()
for key in ['tree_method', 'device', 'predictor']:
    if key in params:
        print(f"   {key}: {params[key]}")
    else:
        print(f"   {key}: NOT SET")

# Check if model is actually configured for GPU
print("\n3. Checking GPU configuration:")
try:
    # Get booster
    booster = model.get_booster()
    booster_config = booster.save_config()
    import json
    config = json.loads(booster_config)
    
    print(f"   tree_method: {config.get('learner', {}).get('gradient_booster', {}).get('tree_method', 'N/A')}")
    print(f"   updater: {config.get('learner', {}).get('gradient_booster', {}).get('updater', 'N/A')}")
    
except Exception as e:
    print(f"   Could not read booster config: {e}")

# Create test data
print("\n4. Creating test data...")
test_df = pd.DataFrame({
    'open': np.random.randn(1000) * 100 + 50000,
    'high': np.random.randn(1000) * 100 + 50100,
    'low': np.random.randn(1000) * 100 + 49900,
    'close': np.random.randn(1000) * 100 + 50000,
    'volume': np.random.randn(1000) * 1000 + 10000,
})

# Extract features (like the classifier does)
print("\n5. Testing prediction (this should use GPU if configured)...")
from classes.classification import EnhancedFeatureExtractor

extractor = (EnhancedFeatureExtractor(test_df)
            .add_advanced_price_features()
            .add_professional_technical_indicators()
            .add_pattern_specific_features())

features_df = extractor.get_features_df()
scaler = model_data['scaler']
features_scaled = scaler.transform(features_df)

print(f"   Features shape: {features_scaled.shape}")

# Predict (this is where GPU would be used)
import time
start = time.time()
predictions = model.predict(features_scaled)
elapsed = time.time() - start

print(f"   Prediction time: {elapsed*1000:.2f}ms for {len(features_scaled)} samples")
print(f"   Speed: {len(features_scaled)/elapsed:.0f} samples/sec")

print("\n6. Conclusion:")
if params.get('device') == 'cuda' and params.get('predictor') == 'gpu_predictor':
    print("   ✅ Model IS configured for GPU")
    print("   ⚠️  But 0% GPU utilization means:")
    print("      - Prediction is too fast for monitoring")
    print("      - Or GPU is not actually being used during predict()")
    print("      - Check nvidia-smi during training (not prediction)")
else:
    print("   ❌ Model is NOT configured for GPU")
    print(f"      device: {params.get('device', 'NOT SET')}")
    print(f"      predictor: {params.get('predictor', 'NOT SET')}")

print("=" * 60)
