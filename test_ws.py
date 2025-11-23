import asyncio
import signal
import sys
from websocket_live_trading import run_live_websocket_trading
import config

running = True

def signal_handler(sig, frame):
    global running
    print('\n\nStopping...')
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def test():
    try:
        await run_live_websocket_trading(
            coins=['BTCUSDT'],
            timeframes=['1min'],
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
            demo_mode=True
        )
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test())
