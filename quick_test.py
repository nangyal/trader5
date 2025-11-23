import asyncio
from websocket_live_trading import LiveWebSocketTrader
import config

async def test():
    trader = LiveWebSocketTrader(
        coins=['BTCUSDT'],
        timeframes=['1min'],
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=True
    )
    await trader.initialize()
    print(f"\n📊 Data loaded:")
    for coin in trader.coins:
        for tf in trader.timeframes:
            df = trader.kline_data[coin][tf]
            print(f"   {coin} {tf}: {len(df)} candles")

asyncio.run(test())
