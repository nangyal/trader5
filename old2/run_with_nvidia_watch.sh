#!/bin/bash
# Run start.py with continuous GPU monitoring

echo "Starting GPU monitoring + start.py"
echo "======================================"

# Start nvidia-smi in watch mode in background
watch -n 0.5 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader' &
WATCH_PID=$!

# Give it time to start
sleep 1

# Run start.py
echo "Running start.py..."
timeout 120 python start.py

# Kill watch
kill $WATCH_PID 2>/dev/null

echo ""
echo "Done!"
