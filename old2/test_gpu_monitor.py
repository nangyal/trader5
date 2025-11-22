#!/usr/bin/env python3
"""Monitor GPU while running XGBoost training"""
import subprocess
import time
import threading

def monitor_gpu():
    """Monitor GPU usage every 0.1s"""
    print("\n🔍 GPU Monitoring started (Ctrl+C to stop)...\n")
    try:
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            gpu_util, mem_util, mem_used = result.stdout.strip().split(',')
            
            bar_gpu = '█' * (int(gpu_util) // 5)
            bar_mem = '█' * (int(mem_util) // 5)
            
            print(f"\rGPU: {gpu_util.strip():>3}% |{bar_gpu:<20}| "
                  f"MEM: {mem_util.strip():>3}% |{bar_mem:<20}| "
                  f"{mem_used.strip()} MiB", end='', flush=True)
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")

if __name__ == '__main__':
    # Start monitoring in background
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    # Give it time to start
    time.sleep(0.5)
    
    # Now run the actual training
    print("\n🚀 Starting XGBoost GPU training...\n")
    import xgboost as xgb
    import numpy as np
    
    # Large dataset for GPU stress
    n = 500000  # 500K samples
    print(f"Creating dataset: {n:,} samples x 50 features")
    X = np.random.rand(n, 50)
    y = np.random.randint(0, 3, n)
    
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        'device': 'cuda:0',
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 8,
        'eta': 0.1,
    }
    
    print(f"\n▶️  Training with {params}\n")
    start = time.time()
    
    bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=20)
    
    elapsed = time.time() - start
    print(f"\n\n✅ Training completed in {elapsed:.2f}s")
    
    # Keep monitoring for a bit
    time.sleep(2)
