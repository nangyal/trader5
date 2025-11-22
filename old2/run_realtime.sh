#!/bin/bash
# Start realtime trading with monitoring

echo "🚀 Starting realtime crypto trading..."
echo "Coins: BTCUSDT, ETHUSDT"
echo "Press Ctrl+C to stop"
echo "=" | tr '=' '=' | head -c 80
echo ""

python start.py 2>&1 | tee /tmp/realtime_$(date +%Y%m%d_%H%M%S).log
