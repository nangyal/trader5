#!/usr/bin/env python
"""Test Binance websocket connection"""
from websocket import create_connection
import json
import time

print("Connecting to Binance BTCUSDT trade stream...")
ws = create_connection('wss://stream.binance.com:9443/ws/btcusdt@trade', timeout=10)
print('✅ Connected!')

for i in range(20):
    msg = ws.recv()
    data = json.loads(msg)
    print(f'Tick {i+1}: Price=${data["p"]}, Qty={data["q"]}, Time={data["T"]}')
    if i < 19:
        time.sleep(0.5)
    
ws.close()
print('\n✅ Test complete - websocket working!')
