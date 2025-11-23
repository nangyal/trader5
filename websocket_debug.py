"""
WebSocket Live Trading DEBUG launcher
Enhanced configuration display and interactive controls
"""
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import asyncio
from datetime import datetime
from websocket_live_trading import LiveWebSocketTrader
import config


async def main():
    """
    Debug mode entry point
    """
    print("\n" + "="*80)
    print("🚀 WEBSOCKET LIVE TRADING - DEBUG MODE")
    print("="*80)
    
    # Configuration
    print("\n📋 CONFIGURATION:")
    print(f"   Coins: {config.COINS}")
    print(f"   Timeframes: {config.TIMEFRAMES}")
    print(f"   Initial capital: ${config.BACKTEST_INITIAL_CAPITAL}")
    print(f"   Risk per trade: {config.RISK_PER_TRADE*100}%")
    print(f"   Max concurrent trades: {config.MAX_CONCURRENT_TRADES}")
    print(f"   Max position size: {config.MAX_POSITION_SIZE_PCT*100}%")
    print(f"   Pattern filters:")
    print(f"      Min probability: {config.PATTERN_FILTERS['min_probability']}")
    print(f"      Min strength: {config.PATTERN_FILTERS['min_strength']}")
    print(f"   Partial TP: {config.PARTIAL_TP['enable']}")
    print(f"   Trailing Stop: {config.TRAILING_STOP['enable']}")
    print(f"   Breakeven: {config.BREAKEVEN_STOP['enable']}")
    print(f"   ML Confidence Weighting: {config.ML_CONFIDENCE_WEIGHTING['enable']}")
    print(f"   Demo mode: {config.BINANCE_DEMO_MODE}")
    
    # Initialize trader
    trader = LiveWebSocketTrader(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=config.BINANCE_DEMO_MODE
    )
    
    # Run async initialization (loads historical data and sets capital)
    print("\n🔧 Initializing trader (async)...")
    await trader.initialize()
    
    # Show initial status
    await trader.print_status()
    
    # Confirmation before starting
    print("\n" + "="*80)
    print("⚠️  READY TO START LIVE TRADING")
    print("="*80)
    print(f"   Mode: {'DEMO (Paper Trading)' if config.BINANCE_DEMO_MODE else '🔴 LIVE (Real Money!)'}")
    print(f"   Capital: ${trader.shared_capital:.2f}")
    print(f"   Coins: {', '.join(config.COINS)}")
    print(f"   Timeframes: {', '.join(config.TIMEFRAMES)}")
    print()
    
    # In debug mode, we can add interactive controls
    print("Debug controls:")
    print("   Press Ctrl+C to stop")
    print("   Status updates every 60 seconds")
    print()
    
    input("Press ENTER to start trading...")
    
    # Start WebSocket
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping trading (Ctrl+C pressed)...")
        
        # Final status
        await trader.print_status()
        
        print("\n✅ Trading stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add detailed error logging
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
