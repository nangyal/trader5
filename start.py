"""
Start script - Indítja a kereskedési keretrendszert
A config.py alapján választja ki az adatforrást (backtest vagy websocket)
"""
import sys
import config

# Ensure directories exist
config.ensure_dirs()

print("\n" + "="*80)
print("🚀 CRYPTO TRADING FRAMEWORK")
print("="*80)

print(f"\n📋 Konfiguráció:")
print(f"   Adatforrás: {config.DATA_SOURCE}")
print(f"   Coinok: {', '.join(config.COINS)}")
print(f"   Timeframes: {', '.join(config.TIMEFRAMES)}")
print(f"   Workers: {config.NUM_WORKERS}")

if config.DATA_SOURCE == 'backtest':
    print(f"\n💰 Backtest Beállítások:")
    print(f"   Kezdő tőke: ${config.BACKTEST_INITIAL_CAPITAL}")
    print(f"   Adat könyvtár: {config.DATA_DIR}")
    print(f"   Stat könyvtár: {config.STAT_DIR}")
    print(f"   Model: {config.MODEL_PATH}")
    
    print("\n" + "="*80)
    print("BACKTEST MÓD")
    print("="*80 + "\n")
    
    # Run backtest
    from backtest import run_backtest
    
    results = run_backtest(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        num_workers=config.NUM_WORKERS
    )
    
    # Generate Excel report
    print("\n" + "="*80)
    print("📊 EXCEL STATISZTIKA GENERÁLÁS")
    print("="*80 + "\n")
    
    from excel_stats import generate_excel_report
    
    excel_file = generate_excel_report(results)
    
    print(f"\n✅ Backtest befejezve!")
    print(f"📄 Excel report: {excel_file}")

elif config.DATA_SOURCE == 'websocket':
    print(f"\n🌐 WebSocket Beállítások:")
    print(f"   Binance WS: {config.BINANCE_WS}")
    print(f"   Demo Mode: {config.BINANCE_DEMO_MODE}")
    print(f"   Model: {config.MODEL_PATH}")
    
    print("\n" + "="*80)
    print("WEBSOCKET LIVE TRADING MÓD")
    print("="*80 + "\n")
    
    print("⚠️  FIGYELEM: Live trading mód!")
    if config.BINANCE_DEMO_MODE:
        print("✅ DEMO/TESTNET mód - biztonságos tesztelés")
    else:
        print("⚠️⚠️⚠️  LIVE/MAINNET mód - valódi kereskedés!")
        response = input("\nBiztosan folytatod? (yes/no): ")
        if response.lower() != 'yes':
            print("Leállítva.")
            sys.exit(0)
    
    # Run WebSocket trading
    from websocket_trading import run_websocket_trading
    
    run_websocket_trading(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=config.BINANCE_DEMO_MODE
    )

else:
    print(f"\n❌ Ismeretlen DATA_SOURCE: {config.DATA_SOURCE}")
    print("   Választható: 'backtest' vagy 'websocket'")
    sys.exit(1)
