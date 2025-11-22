#!/usr/bin/env python3
"""Monitor GPU during start.py execution"""
import subprocess
import sys
import time
import threading

def monitor_gpu(stop_event):
    """Monitor GPU usage"""
    max_gpu = 0
    max_mem = 0
    
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            gpu_util, mem_util, mem_used = result.stdout.strip().split(',')
            gpu_util = int(gpu_util.strip())
            mem_util = int(mem_util.strip())
            
            max_gpu = max(max_gpu, gpu_util)
            max_mem = max(max_mem, mem_util)
            
            bar = '█' * (gpu_util // 5)
            print(f"\rGPU: {gpu_util:3d}% |{bar:<20}| MEM: {mem_util:3d}% | Max GPU: {max_gpu}%", 
                  end='', flush=True)
            
            time.sleep(0.05)  # 50ms interval - faster monitoring
        except:
            pass
    
    print(f"\n\n📊 Peak GPU Usage: {max_gpu}% (Memory: {max_mem}%)")
    return max_gpu

if __name__ == '__main__':
    print("🚀 Starting GPU monitor + start.py")
    print("=" * 60)
    
    stop_event = threading.Event()
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event,), daemon=True)
    monitor_thread.start()
    
    time.sleep(0.5)
    
    # Run start.py in subprocess
    print("\n▶️  Running start.py...\n")
    
    try:
        process = subprocess.Popen(
            [sys.executable, 'start.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        process.terminate()
    finally:
        stop_event.set()
        time.sleep(0.5)
