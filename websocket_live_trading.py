"""
WebSocket Live Trading modul - Production Ready
Real-time OHLCV kline adatok alapján történő kereskedés
1 WebSocket connection, multi-coin/multi-timeframe tracking
"""
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import asyncio
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import traceback
import warnings
warnings.filterwarnings('ignore')

# Binance imports
from binance.client import Client
from binance.exceptions import BinanceAPIException
import websockets

# Local imports
import config
from trading_logic import TradingLogic
from old.forex_pattern_classifier import EnhancedForexPatternClassifier


class LiveWebSocketTrader:
    """
    Live trading WebSocket kezelő több coinhoz és timeframe-hez
    """
    
    def __init__(self, coins, timeframes, api_key, api_secret, demo_mode=True):
        """
        Inicializálás
        
        Args:
            coins: List of coin pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            timeframes: List of timeframes (e.g., ['1m', '5m', '15m'])
            api_key: Binance API key
            api_secret: Binance API secret
            demo_mode: True = Paper trading (no real orders), False = Real trading
        """
        self.coins = coins
        self.timeframes = timeframes
        self.demo_mode = demo_mode
        self.config = config  # Store config module reference
        
        # Binance client
        if demo_mode:
            # DEMO: Use demo API for paper trading
            self.client = Client(api_key, api_secret, demo=True)
            self.ws_url = "wss://stream.binance.com:9443"  # Mainnet WS for real prices
            print("⚠️  DEMO MODE - Paper trading (no real orders)")
            print("   Using DEMO API for balance + MAINNET WebSocket for prices")
        else:
            # LIVE: Use mainnet API for real trading
            self.client = Client(api_key, api_secret)
            self.ws_url = "wss://stream.binance.com:9443"
            print("🔴 LIVE MODE - Real trading with real orders!")
        
        # Load ML model
        print("\n📦 Loading ML model...")
        self.classifier = EnhancedForexPatternClassifier()
        self.classifier.load_model(str(config.MODEL_PATH))
        print(f"✅ Model loaded: {config.MODEL_PATH}")
        
        # Shared capital pool for all traders
        self.shared_capital = 0.0
        self.initial_capital = 0.0
        
        print("\n🔧 DEBUG: Initializing trading logic...")
        # Trading logic per coin
        self.traders = {}
        for coin in coins:
            print(f"   Creating trader for {coin}")
            self.traders[coin] = TradingLogic(config)
        print(f"   ✅ {len(self.traders)} traders initialized")
        
        # OHLCV data storage: {coin: {timeframe: DataFrame}}
        self.kline_data = defaultdict(lambda: defaultdict(lambda: pd.DataFrame()))
        
        # Kline buffer size (keep last N candles)
        self.max_candles = 200
        
        # Timeframe normalization map (config format -> binance format)
        self.timeframe_map = {
            '15s': '15s',
            '30s': '30s', 
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m'
        }
        # Reverse map for incoming klines
        self.reverse_timeframe_map = {v: k for k, v in self.timeframe_map.items()}
        
        # Track processed candles to avoid duplicates
        self.processed_candles = defaultdict(lambda: defaultdict(set))  # {coin: {timeframe: set of timestamps}}
        
        # Stats tracking
        self.last_status_time = time.time()
        self.status_interval = 30  # 30s status update
        
        # BUG #37 FIX: Track last trade open time per coin to prevent duplicates
        self.last_trade_open_time = {}  # coin -> timestamp
        self.trade_open_cooldown = 60  # seconds - prevent same coin trade within 60s
        
        print(f"\n✅ WebSocket Trader initialized")
        print(f"   Coins: {', '.join(coins)}")
        print(f"   Timeframes: {', '.join(timeframes)}")
        print(f"   Max concurrent trades: {config.MAX_CONCURRENT_TRADES}")
        print(f"   Max position size: {config.MAX_POSITION_SIZE_PCT*100}%")
    
    async def initialize(self):
        """
        Async initialization - set initial capital from API and load historical data
        """
        # Initialize shared capital from API
        balance = await self.get_account_balance()
        self.shared_capital = balance
        self.initial_capital = balance
        
        # Set all traders to use shared capital reference
        for trader in self.traders.values():
            trader.capital = self.shared_capital
            trader.initial_capital = self.initial_capital
        
        print(f"\n💰 Shared capital pool: ${self.shared_capital:.2f} USDT")
        
        # Load historical kline data
        print(f"\n📊 Loading historical kline data...")
        await self.load_historical_klines()
    
    async def load_historical_klines(self):
        """
        Betölti az utolsó 1000 candle-t minden coin-timeframe kombinációra
        Note: 15s és 30s timeframe-eket 1s adatokból generáljuk
        """
        # Timeframes supported by Binance get_klines API
        supported_timeframes = ['1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        
        # First, load 1s data if needed for generating 15s/30s
        needs_1s_data = any(tf in ['15s', '30s'] for tf in self.timeframes)
        base_1s_data = {}
        
        if needs_1s_data:
            for coin in self.coins:
                try:
                    print(f"   Loading {coin} 1s for resampling...", end=' ')
                    klines = self.client.get_klines(
                        symbol=coin,
                        interval='1s',
                        limit=1000  # 1000 seconds = ~16 minutes
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    base_1s_data[coin] = df
                    print(f"✅ {len(df)} candles")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # Now load/generate all timeframes
        for coin in self.coins:
            for timeframe in self.timeframes:
                try:
                    binance_tf = self.timeframe_map.get(timeframe, timeframe)
                    
                    # Generate 15s or 30s from 1s data
                    if timeframe in ['15s', '30s']:
                        if coin not in base_1s_data:
                            print(f"   Skipping {coin} {timeframe} (no 1s data available)")
                            continue
                        
                        print(f"   Generating {coin} {timeframe} from 1s...", end=' ')
                        
                        # Resample 1s to 15s or 30s
                        resample_rule = timeframe  # '15s' or '30s'
                        df_resampled = base_1s_data[coin].resample(resample_rule).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        
                        self.kline_data[coin][timeframe] = df_resampled
                        print(f"✅ {len(df_resampled)} candles")
                        continue
                    
                    # Skip unsupported timeframes
                    if binance_tf not in supported_timeframes:
                        print(f"   Skipping {coin} {timeframe} (not supported)")
                        continue
                    
                    # Fetch klines from API for supported timeframes
                    print(f"   Loading {coin} {timeframe}...", end=' ')
                    klines = self.client.get_klines(
                        symbol=coin,
                        interval=binance_tf,
                        limit=1000
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    
                    # Keep only OHLCV columns
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Store in kline_data
                    self.kline_data[coin][timeframe] = df
                    
                    print(f"✅ {len(df)} candles")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        print(f"✅ Historical data loaded for all pairs")
    
    async def get_account_balance(self):
        """
        Lekéri az aktuális USDT balance-t az API-n keresztül
        Works for both DEMO and LIVE mode
        
        Returns:
            float: USDT balance
        """
        try:
            account = self.client.get_account()
            
            # Collect all relevant balances
            balances = {}
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                # Only show assets with non-zero balance
                if total > 0:
                    balances[asset] = {'free': free, 'locked': locked, 'total': total}
            
            # Print account balances
            print(f"\n💼 Account balances:")
            for asset, bal in balances.items():
                if asset in ['USDT', 'BTC', 'ETH', 'BNB']:  # Show main assets
                    print(f"   {asset}: Free: {bal['free']:.8f} | Locked: {bal['locked']:.8f} | Total: {bal['total']:.8f}")
            
            # Return USDT balance
            return balances.get('USDT', {}).get('free', 0.0)
            
        except BinanceAPIException as e:
            print(f"⚠️  Balance lekérés hiba: {e}")
            # Fallback to config capital if API fails
            if self.demo_mode:
                print(f"   Using config capital: ${config.BACKTEST_INITIAL_CAPITAL}")
                return config.BACKTEST_INITIAL_CAPITAL
            return 0.0
    
    def sync_trader_capital(self):
        """
        Sync all traders' capital with the shared pool
        """
        for trader in self.traders.values():
            trader.capital = self.shared_capital
    
    async def print_status(self):
        """
        30s-enként kiírja a státuszt
        """
        # Aggregate stats from all traders
        total_active_trades = sum(len(t.active_trades) for t in self.traders.values())
        total_pnl = sum(t.total_pnl for t in self.traders.values())
        total_closed_trades = sum(len(t.closed_trades) for t in self.traders.values())
        
        # BUG #43 FIX: Calculate total invested capital using stored position_value
        # Don't recalculate with current price - use original entry value
        total_invested = 0.0
        for trader in self.traders.values():
            for trade in trader.active_trades:
                # Use stored position_value from trade open
                total_invested += trade['position_value']
        
        # Use shared capital pool
        pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0.0
        
        print("\n" + "="*80)
        print(f"📊 LIVE TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"💰 Shared capital: ${self.shared_capital:.2f} USDT")
        print(f"💵 Befektetett tőke: ${total_invested:.2f} USDT")
        print(f"📈 Összes P&L: ${total_pnl:.2f} USDT ({pnl_pct:+.2f}%)")
        print(f"🔄 Aktív kereskedések: {total_active_trades}")
        print(f"✅ Lezárt kereskedések: {total_closed_trades}")
        
        if total_active_trades > 0:
            print(f"\n📋 Nyitott pozíciók:")
            for coin, trader in self.traders.items():
                for trade in trader.active_trades:
                    # Use stored position_value for consistency
                    position_value = trade['position_value']
                    print(f"   • {trade['coin']} | "
                          f"{trade['position_size']:.4f} | "
                          f"${position_value:.2f} USDT | "
                          f"Pattern: {trade['pattern']} | "
                          f"Entry: ${trade['entry_price']:.2f}")
        
        print("="*80 + "\n")
    
    async def process_kline(self, coin, timeframe, kline):
        """
        Feldolgoz egy új kline-t és frissíti az OHLCV dataframe-et
        
        Args:
            coin: Coin pair
            timeframe: Timeframe
            kline: Kline adat (dict)
        """
        # Parse kline
        timestamp = pd.to_datetime(kline['t'], unit='ms')
        is_closed = kline['x']
        
        # Debug: log every closed candle
        if is_closed:
            print(f"📌 RAW CANDLE CLOSE: {coin} {timeframe} | x={kline['x']} | timestamp={timestamp}")
        
        new_row = pd.DataFrame({
            'open': [float(kline['o'])],
            'high': [float(kline['h'])],
            'low': [float(kline['l'])],
            'close': [float(kline['c'])],
            'volume': [float(kline['v'])]
        }, index=[timestamp])
        
        # Debug log every 30 seconds
        current_time = time.time()
        if not hasattr(self, '_last_kline_log'):
            self._last_kline_log = {}
        
        if coin not in self._last_kline_log or (current_time - self._last_kline_log.get(coin, 0)) > 30:
            candle_count = len(self.kline_data[coin].get(timeframe, pd.DataFrame()))
            print(f"📊 Kline update: {coin} {timeframe} | Candles: {candle_count} | Closed: {is_closed}")
            self._last_kline_log[coin] = current_time
        
        # Update or append
        if coin not in self.kline_data or timeframe not in self.kline_data[coin]:
            self.kline_data[coin][timeframe] = new_row
        else:
            df = self.kline_data[coin][timeframe]
            
            # Check if this timestamp already exists (update) or new (append)
            if timestamp in df.index:
                # BUG FIX #36: Check if already processed BEFORE updating
                already_processed = timestamp in self.processed_candles[coin][timeframe]
                
                # Update existing candle (direct dict access)
                self.kline_data[coin][timeframe].loc[timestamp] = new_row.iloc[0]
                
                # Check if candle is now closed and needs processing
                # ONLY process if NOT already processed (prevent duplicate trades)
                if is_closed and not already_processed:
                    self.processed_candles[coin][timeframe].add(timestamp)
                    # Keep only last 100 processed timestamps to save memory
                    if len(self.processed_candles[coin][timeframe]) > 100:
                        oldest = min(self.processed_candles[coin][timeframe])
                        self.processed_candles[coin][timeframe].discard(oldest)
                    
                    print(f"🔔 CANDLE CLOSED: {coin} {timeframe} | Processing trading logic...")
                    await self.process_closed_candle(coin, timeframe, self.kline_data[coin][timeframe])
            else:
                # Append new candle
                df = pd.concat([df, new_row])
                
                # Keep only last N candles
                if len(df) > self.max_candles:
                    df = df.iloc[-self.max_candles:]
                
                # Update dict with new dataframe
                self.kline_data[coin][timeframe] = df
                
                # NEW CANDLE CLOSED - run trading logic (only once per candle)
                if kline['x']:  # Candle is closed
                    # Check if already processed this candle
                    if timestamp not in self.processed_candles[coin][timeframe]:
                        self.processed_candles[coin][timeframe].add(timestamp)
                        # Keep only last 100 processed timestamps to save memory
                        if len(self.processed_candles[coin][timeframe]) > 100:
                            oldest = min(self.processed_candles[coin][timeframe])
                            self.processed_candles[coin][timeframe].discard(oldest)
                        
                        print(f"🔔 CANDLE CLOSED: {coin} {timeframe} | Processing trading logic...")
                        await self.process_closed_candle(coin, timeframe, df)
    
    async def process_closed_candle(self, coin, timeframe, df_ohlcv):
        """
        Új candle bezárult - futtatja a trading logikát
        
        Args:
            coin: Coin pair
            timeframe: Timeframe
            df_ohlcv: OHLCV DataFrame
        """
        print(f"🔍 process_closed_candle called: {coin} {timeframe} | Candles: {len(df_ohlcv)}")
        
        if len(df_ohlcv) < 60:
            # Nincs elég adat még
            print(f"   ⚠️ Not enough data: {len(df_ohlcv)}/60 candles")
            return
        
        # Check trading hours (csak nappal kereskedik)
        if self.config.TRADING_HOURS['enable']:
            current_hour = datetime.utcnow().hour
            start_hour = self.config.TRADING_HOURS['start_hour']
            end_hour = self.config.TRADING_HOURS['end_hour']
            
            if not (start_hour <= current_hour < end_hour):
                # Kereskedési időn kívül - NEM nyit új trade-eket
                # De zárja a meglévőket!
                trader = self.traders[coin]
                current_candle = df_ohlcv.iloc[-1]
                
                # Csak aktív trade-ek ellenőrzése (close only mode)
                for trade in list(trader.active_trades):
                    should_close, exit_price, exit_reason, partial_ratio = trader.check_trade_exit(
                        trade, current_candle
                    )
                    
                    if should_close:
                        if self.demo_mode:
                            pnl = trader.close_trade(trade, exit_price, exit_reason, 
                                                    datetime.now(), partial_ratio)
                            self.shared_capital = trader.capital
                            self.sync_trader_capital()
                            print(f"🔴 DEMO CLOSE (off-hours): {coin} {timeframe} | {exit_reason} | "
                                  f"P&L: ${pnl:.2f} | Shared Capital: ${self.shared_capital:.2f}")
                return  # Nem nyit új kereskedést
        
        trader = self.traders[coin]
        
        # Get last candle
        current_candle = df_ohlcv.iloc[-1]
        current_price = current_candle['close']
        
        # Check ALL active trades for exit (not just matching timeframe)
        # Use current price for exit checks on all timeframes
        for trade in list(trader.active_trades):
            should_close, exit_price, exit_reason, partial_ratio = trader.check_trade_exit(
                trade, current_candle
            )
            
            if should_close:
                # DEMO vs LIVE mode handling
                if self.demo_mode:
                    # DEMO: Just close internally
                    pnl = trader.close_trade(trade, exit_price, exit_reason, 
                                            datetime.now(), partial_ratio)
                    
                    # Update shared capital (trader.capital already increased by close_trade)
                    self.shared_capital = trader.capital
                    
                    # Sync to other traders (they need the updated shared capital)
                    self.sync_trader_capital()
                    
                    print(f"🔴 DEMO CLOSE: {coin} {timeframe} | {exit_reason} | "
                          f"P&L: ${pnl:.2f} | Shared Capital: ${self.shared_capital:.2f}")
                else:
                    # LIVE: Execute real close order
                    try:
                        # TODO: Implement real order execution
                        # order = self.client.create_order(
                        #     symbol=coin,
                        #     side='SELL',
                        #     type='MARKET',
                        #     quantity=trade['position_size'] * partial_ratio
                        # )
                        
                        # For now, close internally like demo
                        pnl = trader.close_trade(trade, exit_price, exit_reason, 
                                                datetime.now(), partial_ratio)
                        
                        # Update shared capital (trader.capital already increased by close_trade)
                        self.shared_capital = trader.capital
                        
                        # Sync to other traders (they need the updated shared capital)
                        self.sync_trader_capital()
                        
                        print(f"🔴 LIVE CLOSE: {coin} {timeframe} | {exit_reason} | "
                              f"P&L: ${pnl:.2f} | Shared Capital: ${self.shared_capital:.2f}")
                        print(f"   ⚠️  Real order execution not yet implemented")
                    except BinanceAPIException as e:
                        print(f"❌ Close order execution failed: {e}")
        
        # Run ML prediction on last 60 candles (use .copy() to avoid warnings)
        try:
            predictions, probabilities = self.classifier.predict(df_ohlcv.iloc[-60:].copy())
        except Exception as e:
            print(f"❌ Prediction error {coin} {timeframe}: {e}")
            return        # Get last prediction
        pattern = predictions[-1]
        pattern_prob = np.max(probabilities[-1])
        pattern_strength = pattern_prob  # Use prob as strength proxy
        
        # Always log pattern detection (if not 'no_pattern')
        if pattern != 'no_pattern':
            print(f"\n🔍 PATTERN DETECTED: {coin} {timeframe}")
            print(f"   Pattern: {pattern}")
            print(f"   Probability: {pattern_prob:.3f}")
            print(f"   Strength: {pattern_strength:.3f}")
        
        # Decrement cooldown counter on every candle
        trader.decrement_cooldown()
        
        # Check global concurrent trades limit (shared pool level)
        total_active_trades = sum(len(t.active_trades) for t in self.traders.values())
        if total_active_trades >= config.MAX_CONCURRENT_TRADES:
            if pattern != 'no_pattern':
                print(f"   ⛔ SKIP: Max concurrent trades reached (GLOBAL: {total_active_trades}/{config.MAX_CONCURRENT_TRADES})")
            return
        
        # BUG #37 FIX: Check per-coin cooldown to prevent duplicate trades from multiple timeframes
        current_time = time.time()
        last_open_time = self.last_trade_open_time.get(coin, 0)
        time_since_last_open = current_time - last_open_time
        
        if time_since_last_open < self.trade_open_cooldown:
            if pattern != 'no_pattern':
                print(f"   ⛔ SKIP: Trade cooldown active for {coin} ({time_since_last_open:.1f}s / {self.trade_open_cooldown}s)")
            return
        
        # Check if we should open trade (pattern/prob/strength filters)
        if not trader.should_open_trade(pattern, pattern_prob, pattern_strength):
            if pattern != 'no_pattern':
                # Explain why trade was skipped
                if pattern == 'no_pattern':
                    print(f"   ⛔ SKIP: No pattern detected")
                elif pattern_prob < trader.pattern_filters['min_probability']:
                    print(f"   ⛔ SKIP: Probability too low ({pattern_prob:.3f} < {trader.pattern_filters['min_probability']})")
                elif pattern_strength < trader.pattern_filters['min_strength']:
                    print(f"   ⛔ SKIP: Strength too low ({pattern_strength:.3f} < {trader.pattern_filters['min_strength']})")
                elif len(trader.active_trades) >= trader.config.MAX_CONCURRENT_TRADES:
                    print(f"   ⛔ SKIP: Max concurrent trades reached ({len(trader.active_trades)}/{trader.config.MAX_CONCURRENT_TRADES})")
                else:
                    print(f"   ⛔ SKIP: Pattern filter criteria not met")
            return
        
        # Get recent data for trend calc
        recent_data = df_ohlcv.iloc[-30:].copy()
        
        # Calculate targets
        entry_price = current_candle['close']
        sl, tp, direction, params = trader.calculate_pattern_targets(
            pattern, entry_price, current_candle, recent_data
        )
        
        if direction == 'skip':
            print(f"   ⛔ SKIP: Trend/direction check failed")
            print(f"      - Price below EMA50 or bearish pattern or wrong trend alignment")
            return
        
        # Use SHARED capital for position sizing (thread-safe allocation)
        position_size = trader.calculate_position_size(
            entry_price, sl, self.shared_capital, ml_probability=pattern_prob
        )
        
        if position_size <= 0:
            print(f"   ⛔ SKIP: Position size invalid ({position_size:.4f})")
            print(f"      - Shared Capital: ${self.shared_capital:.2f}")
            print(f"      - Entry: ${entry_price:.2f}, SL: ${sl:.2f}")
            return
        
        # Sync capital to trader BEFORE opening trade
        self.sync_trader_capital()
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # DEMO vs LIVE mode handling
        if self.demo_mode:
            # DEMO: Just track the trade internally (paper trading)
            # Note: trader.open_trade() will deduct from trader.capital
            trade = trader.open_trade(
                coin=coin,
                pattern=pattern,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                position_size=position_size,
                probability=pattern_prob,
                strength=pattern_strength,
                timeframe=timeframe,
                entry_time=datetime.now()
            )
            
            # Update shared capital (trader.capital was reduced by open_trade)
            self.shared_capital = trader.capital
            
            # Sync to other traders
            self.sync_trader_capital()
            
            # BUG #37 FIX: Update last trade open time for this coin
            self.last_trade_open_time[coin] = time.time()
            
            print(f"🟢 DEMO OPEN: {coin} {timeframe} | Pattern: {pattern} | "
                  f"Size: {position_size:.4f} (${position_value:.2f}) | "
                  f"Entry: ${entry_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | "
                  f"Shared Capital: ${self.shared_capital:.2f}")
        else:
            # LIVE: Execute real order via Binance API
            try:
                # TODO: Implement real order execution
                # order = self.client.create_order(
                #     symbol=coin,
                #     side='BUY',
                #     type='MARKET',
                #     quantity=position_size
                # )
                
                # For now, track internally like demo
                trade = trader.open_trade(
                    coin=coin,
                    pattern=pattern,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size=position_size,
                    probability=pattern_prob,
                    strength=pattern_strength,
                    timeframe=timeframe,
                    entry_time=datetime.now()
                )
                
                # Update shared capital (trader.capital was reduced by open_trade)
                self.shared_capital = trader.capital
                
                # Sync to other traders
                self.sync_trader_capital()
                
                # BUG #37 FIX: Update last trade open time for this coin
                self.last_trade_open_time[coin] = time.time()
                
                print(f"🟢 LIVE OPEN: {coin} {timeframe} | Pattern: {pattern} | "
                      f"Size: {position_size:.4f} (${position_value:.2f}) | "
                      f"Entry: ${entry_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | "
                      f"Shared Capital: ${self.shared_capital:.2f}")
                print(f"   ⚠️  Real order execution not yet implemented")
            except BinanceAPIException as e:
                print(f"❌ Order execution failed: {e}")
    
    async def handle_ws_message(self, message):
        """
        WebSocket üzenet feldolgozása
        
        Args:
            message: JSON üzenet
        """
        try:
            data = json.loads(message)
            
            # Debug: log first message
            if not hasattr(self, '_first_message_logged'):
                print(f"📥 First WS message received: {str(data)[:200]}...")
                self._first_message_logged = True
            
            # Multi-stream wrapper format
            if 'stream' in data and 'data' in data:
                data = data['data']
            
            # Kline event
            if 'e' in data and data['e'] == 'kline':
                kline = data['k']
                symbol = kline['s']
                interval = kline['i']  # Binance format (1m, 5m, etc.)
                
                # Normalize timeframe to config format
                normalized_tf = self.reverse_timeframe_map.get(interval, interval)
                
                await self.process_kline(symbol, normalized_tf, kline)
            else:
                # Log non-kline messages
                if not hasattr(self, '_unknown_message_logged'):
                    print(f"⚠️  Non-kline message: {data.get('e', 'unknown type')}")
                    self._unknown_message_logged = True
            
        except Exception as e:
            print(f"❌ Message processing error: {e}")
            traceback.print_exc()
    
    async def run(self):
        """
        Fő loop - WebSocket connection és message handling
        """
        # Build stream names for multi-stream
        streams = []
        for coin in self.coins:
            for tf in self.timeframes:
                # Convert to Binance format using map
                binance_tf = self.timeframe_map.get(tf, tf)
                stream = f"{coin.lower()}@kline_{binance_tf}"
                streams.append(stream)
        
        # Multi-stream WebSocket URL
        # Format: wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"
        
        print(f"\n🌐 Connecting to WebSocket...")
        print(f"   URL: {stream_url[:80]}...")
        print(f"   Streams: {len(streams)}")
        
        # Main loop
        while True:
            try:
                async with websockets.connect(stream_url) as websocket:
                    print(f"✅ WebSocket connected!")
                    
                    # Message loop
                    while True:
                        try:
                            message = await websocket.recv()
                            await self.handle_ws_message(message)
                            
                            # Print status every 30s
                            if time.time() - self.last_status_time > self.status_interval:
                                await self.print_status()
                                self.last_status_time = time.time()
                        
                        except websockets.exceptions.ConnectionClosed:
                            print("⚠️  WebSocket connection closed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"❌ Message error: {e}")
                            traceback.print_exc()
            
            except Exception as e:
                print(f"❌ WebSocket connection error: {e}")
                print("⏳ Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


async def run_live_websocket_trading(coins, timeframes, api_key, api_secret, demo_mode=True):
    """
    Fő entry point a live trading-hez
    
    Args:
        coins: List of coin pairs
        timeframes: List of timeframes
        api_key: Binance API key
        api_secret: Binance API secret
        demo_mode: True = Testnet, False = Mainnet
    """
    print("\n" + "="*80)
    print("🚀 LIVE WEBSOCKET TRADING INDÍTÁSA")
    print("="*80)
    
    trader = LiveWebSocketTrader(coins, timeframes, api_key, api_secret, demo_mode)
    
    # Initialize capital from API
    await trader.initialize()
    
    await trader.run()


if __name__ == '__main__':
    # Test
    import config
    
    asyncio.run(run_live_websocket_trading(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=config.BINANCE_DEMO_MODE
    ))
