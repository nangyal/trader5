#!/usr/bin/env python3
"""Direct XGBoost GPU test"""
import xgboost as xgb
import numpy as np
import time

print("Testing XGBoost GPU directly...")

# Create large dataset
n = 100000
X = np.random.rand(n, 50)
y = np.random.randint(0, 3, n)

print(f"Dataset: {X.shape}")

# Train with GPU
print("\nTraining with GPU...")
dtrain = xgb.DMatrix(X, label=y)

params = {
    'device': 'cuda:0',  # XGBoost 2.0+ - no tree_method needed
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 6,
}

print("Parameters:", params)
print("\n👀 Watch nvidia-smi NOW!\n")

start = time.time()
bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=10)
print(f"\nTraining time: {time.time()-start:.2f}s")

# Predict
print("\nPredicting...")
start = time.time()
for i in range(10):
    pred = bst.predict(dtrain)
print(f"Prediction time (10x): {time.time()-start:.2f}s")

print("\n✅ If you saw GPU usage spike during training, GPU works!")
print("❌ If GPU was 0% during training, XGBoost is NOT using GPU")
