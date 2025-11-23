"""
WebSocket Live Trading modul HEDGING-gel - Production Ready
Real-time OHLCV kline adatok + Dynamic Hedging protection
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
from hedge_manager import HedgeManager
from old.forex_pattern_classifier import EnhancedForexPatternClassifier


class LiveWebSocketHedgingTrader:
    """
    Live trading WebSocket kezelő HEDGING-gel
    Kombinálja a WebSocket real-time trading-et a dynamic hedging-gel
    """
    
    def __init__(self, coins, timeframes, api_key, api_secret, demo_mode=True):
        """
        Inicializálás
        
        Args:
            coins: List of coin pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            timeframes: List of timeframes (e.g., ['1m', '5m', '15m'])
            api_key: Binance API key
            api_secret: Binance API secret
            demo_mode: True = Paper trading, False = Real trading
        """
        self.coins = coins
        self.timeframes = timeframes
        self.demo_mode = demo_mode
        self.config = config
        
        # Binance client
        if demo_mode:
            self.client = Client(api_key, api_secret, demo=True)
            self.ws_url = "wss://stream.binance.com:9443"
            print("⚠️  DEMO MODE - Paper trading (no real orders)")
            print("   Using DEMO API for balance + MAINNET WebSocket for prices")
        else:
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
        self.peak_capital = 0.0
        self.peak_equity = 0.0
        
        print("\n🔧 DEBUG: Initializing trading logic + hedging...")
        # Trading logic per coin
        self.traders = {}
        for coin in coins:
            print(f"   Creating trader for {coin}")
            self.traders[coin] = TradingLogic(config)
        print(f"   ✅ {len(self.traders)} traders initialized")
        
        # Hedge Manager (közös minden coin-ra)
        self.hedge_manager = HedgeManager(config)
        self.active_hedges = []  # Aktív hedge trade-ek
        print(f"   ✅ Hedge manager initialized")
        print(f"      - Hedge threshold: {config.HEDGING['hedge_threshold']*100:.1f}%")
        print(f"      - Recovery threshold: {config.HEDGING['hedge_recovery_threshold']*100:.1f}%")
        print(f"      - Hedge ratio: {config.HEDGING['hedge_ratio']*100:.1f}%")
        print(f"      - Dynamic hedge: {config.HEDGING['dynamic_hedge']}")
        
        # OHLCV data storage
        self.kline_data = defaultdict(lambda: defaultdict(lambda: pd.DataFrame()))
        self.max_candles = 200
        
        # Timeframe normalization map
        self.timeframe_map = {
            '15s': '15s',
            '30s': '30s',
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m'
        }
        self.reverse_timeframe_map = {v: k for k, v in self.timeframe_map.items()}
        
        # Track processed candles
        self.processed_candles = defaultdict(lambda: defaultdict(set))
        
        # Stats tracking
        self.last_status_time = time.time()
        self.status_interval = 30  # 30s status update
        
        # BUG #37 FIX: Track last trade open time per coin to prevent duplicates
        self.last_trade_open_time = {}  # coin -> timestamp
        self.trade_open_cooldown = 60  # seconds - prevent same coin trade within 60s
        
        print(f"\n✅ Hedging Trader initialized")
        print(f"   Coins: {', '.join(coins)}")
        print(f"   Timeframes: {', '.join(timeframes)}")
        print(f"   Max concurrent trades: {config.MAX_CONCURRENT_TRADES}")
        print(f"   Max position size: {config.MAX_POSITION_SIZE_PCT*100}%")
    
    async def initialize(self):
        """Async initialization - set initial capital and load data"""
        balance = await self.get_account_balance()
        self.shared_capital = balance
        self.initial_capital = balance
        self.peak_capital = balance
        self.peak_equity = balance
        
        # Set all traders to use shared capital
        for trader in self.traders.values():
            trader.capital = self.shared_capital
            trader.initial_capital = self.initial_capital
        
        print(f"\n💰 Shared capital pool: ${self.shared_capital:.2f} USDT")
        
        # Load historical kline data
        print(f"\n📊 Loading historical kline data...")
        await self.load_historical_klines()
    
    async def load_historical_klines(self):
        """Betölti az utolsó 1000 candle-t minden coin-timeframe kombinációra"""
        # Same implementation as websocket_live_trading.py
        supported_timeframes = ['1s', '1m', '3m', '5m', '15m', '30m', '1h']
        
        needs_1s_data = any(tf in ['15s', '30s'] for tf in self.timeframes)
        base_1s_data = {}
        
        if needs_1s_data:
            for coin in self.coins:
                try:
                    print(f"   Loading {coin} 1s for resampling...", end=' ')
                    klines = self.client.get_klines(symbol=coin, interval='1s', limit=1000)
                    
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    base_1s_data[coin] = df
                    print(f"✅ {len(df)} candles")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        for coin in self.coins:
            for timeframe in self.timeframes:
                try:
                    binance_tf = self.timeframe_map.get(timeframe, timeframe)
                    
                    if timeframe in ['15s', '30s']:
                        if coin not in base_1s_data:
                            continue
                        
                        print(f"   Generating {coin} {timeframe} from 1s...", end=' ')
                        resample_rule = timeframe
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
                    
                    if binance_tf not in supported_timeframes:
                        continue
                    
                    print(f"   Loading {coin} {timeframe}...", end=' ')
                    klines = self.client.get_klines(symbol=coin, interval=binance_tf, limit=1000)
                    
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    self.kline_data[coin][timeframe] = df
                    print(f"✅ {len(df)} candles")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        print(f"✅ Historical data loaded for all pairs")
    
    async def get_account_balance(self):
        """Lekéri az aktuális USDT balance-t"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    balances[asset] = {'free': free, 'locked': locked, 'total': total}
            
            print(f"\n💼 Account balances:")
            for asset, bal in balances.items():
                if asset in ['USDT', 'BTC', 'ETH', 'BNB']:
                    print(f"   {asset}: Free: {bal['free']:.8f} | Locked: {bal['locked']:.8f} | Total: {bal['total']:.8f}")
            
            return balances.get('USDT', {}).get('free', 0.0)
        except BinanceAPIException as e:
            print(f"⚠️  Balance error: {e}")
            if self.demo_mode:
                print(f"   Using config capital: ${config.BACKTEST_INITIAL_CAPITAL}")
                return config.BACKTEST_INITIAL_CAPITAL
            return 0.0
    
    def sync_trader_capital(self):
        """Sync all traders' capital with shared pool"""
        for trader in self.traders.values():
            trader.capital = self.shared_capital
    
    def calculate_total_equity(self):
        """
        Számítja a total equity-t (capital + unrealized P&L)
        
        Returns:
            float: Total equity
        """
        unrealized_pnl = 0.0
        
        # All active LONG trades unrealized P&L
        for trader in self.traders.values():
            for trade in trader.active_trades:
                if not trade.get('is_hedge', False):
                    # BUG #56 FIX: Use SAME timeframe as trade was opened on
                    # Don't pick random timeframe - ensures consistent pricing
                    coin = trade['coin']
                    trade_tf = trade.get('timeframe')  # Timeframe trade was opened on
                    
                    if coin in self.kline_data and self.kline_data[coin]:
                        # Prefer trade's timeframe if available
                        if trade_tf and trade_tf in self.kline_data[coin]:
                            tf_data = self.kline_data[coin][trade_tf]
                            if len(tf_data) > 0:
                                current_price = tf_data.iloc[-1]['close']
                                pnl = (current_price - trade['entry_price']) * trade['position_size']
                                unrealized_pnl += pnl
                        else:
                            # Fallback: any available timeframe (for backwards compatibility)
                            for tf_data in self.kline_data[coin].values():
                                if len(tf_data) > 0:
                                    current_price = tf_data.iloc[-1]['close']
                                    pnl = (current_price - trade['entry_price']) * trade['position_size']
                                    unrealized_pnl += pnl
                                    break
        
        # All active HEDGE trades unrealized P&L
        for hedge in self.active_hedges:
            # BUG #56 FIX: Hedges don't have timeframe (created at hedge activation)
            # Use any available timeframe - all should be very close in price
            coin = hedge['coin']
            if coin in self.kline_data and self.kline_data[coin]:
                # Prefer shorter timeframe for more recent price (1m > 5m > 15m)
                for tf in ['1m', '5m', '15m', '30m', '1h']:
                    if tf in self.kline_data[coin]:
                        tf_data = self.kline_data[coin][tf]
                        if len(tf_data) > 0:
                            current_price = tf_data.iloc[-1]['close']
                            # SHORT: entry - current
                            pnl = (hedge['entry_price'] - current_price) * hedge['position_size']
                            unrealized_pnl += pnl
                            break
                else:
                    # Fallback: any available timeframe
                    for tf_data in self.kline_data[coin].values():
                        if len(tf_data) > 0:
                            current_price = tf_data.iloc[-1]['close']
                            pnl = (hedge['entry_price'] - current_price) * hedge['position_size']
                            unrealized_pnl += pnl
                            break
        
        return self.shared_capital + unrealized_pnl
    
    async def print_status(self):
        """30s-enként kiírja a státuszt + hedging info"""
        total_active_trades = sum(len(t.active_trades) for t in self.traders.values())
        total_pnl = sum(t.total_pnl for t in self.traders.values())
        total_closed_trades = sum(len(t.closed_trades) for t in self.traders.values())
        
        # BUG #43 FIX: Calculate invested capital using stored position_value
        # Don't recalculate with current price - use original entry value
        total_invested = 0.0
        for trader in self.traders.values():
            for trade in trader.active_trades:
                if not trade.get('is_hedge', False):
                    # Use stored position_value from trade open
                    total_invested += trade['position_value']
        
        # Hedge invested capital (locked for SHORT positions)
        hedge_invested = sum(h['position_value'] for h in self.active_hedges)
        
        # Calculate equity and drawdown
        equity = self.calculate_total_equity()
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Update peaks
        self.peak_capital = max(self.peak_capital, self.shared_capital)
        self.peak_equity = max(self.peak_equity, equity)
        
        # Update hedge manager equity curve
        self.hedge_manager.update_equity_curve(equity)
        
        pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0.0
        
        print("\n" + "="*80)
        print(f"🛡️  LIVE HEDGING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"💰 Shared capital: ${self.shared_capital:.2f} USDT")
        print(f"💵 Befektetett tőke: ${total_invested:.2f} USDT")
        print(f"🛡️  Hedge tőke: ${hedge_invested:.2f} USDT ({len(self.active_hedges)} hedge)")
        print(f"📊 Total Equity: ${equity:.2f} USDT")
        print(f"📉 Drawdown: {drawdown*100:.2f}% (Peak: ${self.peak_equity:.2f})")
        print(f"📈 Összes P&L: ${total_pnl:.2f} USDT ({pnl_pct:+.2f}%)")
        print(f"🔄 Aktív kereskedések: {total_active_trades}")
        print(f"✅ Lezárt kereskedések: {total_closed_trades}")
        print(f"🔧 Hedge activations: {self.hedge_manager.hedge_activations}")
        
        if total_active_trades > 0 or len(self.active_hedges) > 0:
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
            
            if self.active_hedges:
                print(f"\n🛡️  Aktív hedge-ek:")
                for hedge in self.active_hedges:
                    print(f"   • {hedge['coin']} | "
                          f"SHORT {hedge['position_size']:.4f} | "
                          f"${hedge['position_value']:.2f} USDT | "
                          f"Entry: ${hedge['entry_price']:.2f}")
        
        print("="*80 + "\n")
    
    async def process_kline(self, coin, timeframe, kline):
        """Feldolgoz egy új kline-t és frissíti az OHLCV dataframe-et"""
        timestamp = pd.to_datetime(kline['t'], unit='ms')
        is_closed = kline['x']
        
        if is_closed:
            print(f"📌 RAW CANDLE CLOSE: {coin} {timeframe} | x={kline['x']} | timestamp={timestamp}")
        
        new_row = pd.DataFrame({
            'open': [float(kline['o'])],
            'high': [float(kline['h'])],
            'low': [float(kline['l'])],
            'close': [float(kline['c'])],
            'volume': [float(kline['v'])]
        }, index=[timestamp])
        
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
            
            if timestamp in df.index:
                # BUG FIX #36: Check if already processed BEFORE updating
                already_processed = timestamp in self.processed_candles[coin][timeframe]
                
                # Update existing candle
                self.kline_data[coin][timeframe].loc[timestamp] = new_row.iloc[0]
                
                # ONLY process if NOT already processed
                if is_closed and not already_processed:
                    self.processed_candles[coin][timeframe].add(timestamp)
                    if len(self.processed_candles[coin][timeframe]) > 100:
                        oldest = min(self.processed_candles[coin][timeframe])
                        self.processed_candles[coin][timeframe].discard(oldest)
                    
                    print(f"🔔 CANDLE CLOSED: {coin} {timeframe} | Processing trading logic...")
                    await self.process_closed_candle(coin, timeframe, self.kline_data[coin][timeframe])
            else:
                # Append new candle
                df = pd.concat([df, new_row])
                
                if len(df) > self.max_candles:
                    df = df.iloc[-self.max_candles:]
                
                self.kline_data[coin][timeframe] = df
                
                if kline['x']:
                    if timestamp not in self.processed_candles[coin][timeframe]:
                        self.processed_candles[coin][timeframe].add(timestamp)
                        if len(self.processed_candles[coin][timeframe]) > 100:
                            oldest = min(self.processed_candles[coin][timeframe])
                            self.processed_candles[coin][timeframe].discard(oldest)
                        
                        print(f"🔔 CANDLE CLOSED: {coin} {timeframe} | Processing trading logic...")
                        await self.process_closed_candle(coin, timeframe, df)
    
    async def process_closed_candle(self, coin, timeframe, df_ohlcv):
        """
        Új candle bezárult - trading logic + hedging logic
        """
        print(f"🔍 process_closed_candle called: {coin} {timeframe} | Candles: {len(df_ohlcv)}")
        
        if len(df_ohlcv) < 60:
            print(f"   ⚠️ Not enough data: {len(df_ohlcv)}/60 candles")
            return
        
        # Get current candle and price
        trader = self.traders[coin]
        current_candle = df_ohlcv.iloc[-1]
        current_price = current_candle['close']
        
        # ========================================
        # 1. CHECK HEDGE EXITS FIRST
        # ========================================
        for hedge in list(self.active_hedges):
            if hedge['coin'] == coin:
                should_close, exit_price, exit_reason = self.hedge_manager.check_hedge_exit(hedge, current_candle)
                
                if should_close:
                    pnl = self.hedge_manager.calculate_hedge_pnl(hedge, exit_price)
                    
                    # Return capital: original position_value + P&L
                    # SHORT P&L = (entry - exit) * size
                    # If profitable (price dropped): PnL positive
                    # Total return = position_value (locked capital) + PnL (profit/loss)
                    self.shared_capital += hedge['position_value'] + pnl
                    self.sync_trader_capital()
                    
                    # Remove from active hedges
                    self.active_hedges.remove(hedge)
                    
                    print(f"🛡️  HEDGE CLOSE: {coin} | {exit_reason} | "
                          f"P&L: ${pnl:.2f} | Capital: ${self.shared_capital:.2f}")
        
        # ========================================
        # 2. CHECK MAIN TRADE EXITS
        # ========================================
        for trade in list(trader.active_trades):
            should_close, exit_price, exit_reason, partial_ratio = trader.check_trade_exit(trade, current_candle)
            
            if should_close:
                if self.demo_mode:
                    pnl = trader.close_trade(trade, exit_price, exit_reason, datetime.now(), partial_ratio)
                    self.shared_capital = trader.capital
                    self.sync_trader_capital()
                    
                    print(f"🔴 DEMO CLOSE: {coin} {timeframe} | {exit_reason} | "
                          f"P&L: ${pnl:.2f} | Shared Capital: ${self.shared_capital:.2f}")
                    
                    # BUG #53 FIX: Check if hedge became orphaned (no more LONG trades to protect)
                    if self.active_hedges:
                        # Count remaining LONG trades (non-hedge)
                        remaining_long_trades = sum(
                            1 for t in self.traders.values()
                            for trade_item in t.active_trades
                            if trade_item['direction'] == 'long' and not trade_item.get('is_hedge', False)
                        )
                        
                        if remaining_long_trades == 0:
                            # No more LONG exposure → force close all hedges
                            print(f"⚠️  ORPHANED HEDGE DETECTED - All LONG trades closed, forcing hedge close")
                            
                            total_recovered = 0.0
                            for hedge in list(self.active_hedges):
                                hedge_coin = hedge['coin']
                                hedge_current_price = None
                                
                                if hedge_coin in self.kline_data and self.kline_data[hedge_coin]:
                                    for tf_data in self.kline_data[hedge_coin].values():
                                        if len(tf_data) > 0:
                                            hedge_current_price = tf_data.iloc[-1]['close']
                                            break
                                
                                if hedge_current_price:
                                    pnl_hedge = self.hedge_manager.calculate_hedge_pnl(hedge, hedge_current_price)
                                    recovered = hedge['position_value'] + pnl_hedge
                                    self.shared_capital += recovered
                                    total_recovered += recovered
                                    
                                    print(f"🛡️  FORCE CLOSE ORPHANED HEDGE: {hedge['coin']} | "
                                          f"P&L: ${pnl_hedge:.2f} | Recovered: ${recovered:.2f}")
                            
                            self.sync_trader_capital()
                            self.active_hedges = []
                            print(f"   Total recovered from orphaned hedges: ${total_recovered:.2f}")
        
        # BUG #51 FIX: Calculate equity and update peaks AFTER trade/hedge exits
        # This ensures peak tracking and drawdown calculation use consistent state
        equity = self.calculate_total_equity()
        self.peak_capital = max(self.peak_capital, self.shared_capital)
        self.peak_equity = max(self.peak_equity, equity)
        
        # Update hedge manager equity curve for dynamic threshold
        self.hedge_manager.update_equity_curve(equity)
        
        # ========================================
        # 3. HEDGE MANAGEMENT - Check if we need hedge activation/deactivation
        # ========================================
        
        # BUG #44-47 FIX: Use None for coin - hedge is GLOBAL, not coin-specific!
        # Drawdown is calculated from TOTAL equity, not per-coin
        # Using coin-specific config here would cause inconsistent thresholds
        should_activate_hedge, current_drawdown = self.hedge_manager.should_hedge(
            self.shared_capital,
            self.peak_capital,
            equity,
            self.peak_equity,
            None  # Global config, not coin-specific
        )
        
        if should_activate_hedge and not self.active_hedges:
            # BUG #46 FIX: Create separate hedge for EACH coin with exposure
            # Don't mix coins - hedge BTC with BTC SHORT, ETH with ETH SHORT
            
            # Group active trades by coin
            trades_by_coin = {}
            for t in self.traders.values():
                for trade in t.active_trades:
                    if trade['direction'] == 'long' and not trade.get('is_hedge', False):
                        coin_name = trade['coin']
                        if coin_name not in trades_by_coin:
                            trades_by_coin[coin_name] = []
                        trades_by_coin[coin_name].append(trade)
            
            if trades_by_coin:
                total_hedge_value = 0.0
                hedges_created = []
                
                # Create hedge for each coin separately
                for coin_name, coin_trades in trades_by_coin.items():
                    # BUG #57 FIX: Get current price - prefer shorter timeframe (fresher price)
                    coin_price = None
                    if coin_name in self.kline_data and self.kline_data[coin_name]:
                        # Prefer 1m > 5m > 15m > 30m > 1h for most recent price
                        for tf in ['1m', '5m', '15m', '30m', '1h']:
                            if tf in self.kline_data[coin_name]:
                                tf_data = self.kline_data[coin_name][tf]
                                if len(tf_data) > 0:
                                    coin_price = tf_data.iloc[-1]['close']
                                    break
                        
                        # Fallback: any available timeframe
                        if coin_price is None:
                            for tf_data in self.kline_data[coin_name].values():
                                if len(tf_data) > 0:
                                    coin_price = tf_data.iloc[-1]['close']
                                    break
                    
                    if coin_price:
                        hedge_trade = self.hedge_manager.create_hedge_trade(
                            coin_trades,
                            coin_price,
                            coin_name,
                            datetime.now()
                        )
                        
                        if hedge_trade:
                            hedges_created.append(hedge_trade)
                            total_hedge_value += hedge_trade['position_value']
                
                if hedges_created:
                    # Deduct total hedge capital
                    self.shared_capital -= total_hedge_value
                    self.sync_trader_capital()
                    
                    # Add all hedges
                    self.active_hedges.extend(hedges_created)
                    
                    print(f"\n🛡️  HEDGE ACTIVATED!")
                    print(f"   Drawdown: {current_drawdown*100:.2f}%")
                    print(f"   Total hedge count: {len(hedges_created)}")
                    print(f"   Total hedge value: ${total_hedge_value:.2f}")
                    print(f"   Hedge ratio: {self.config.HEDGING['hedge_ratio']*100:.0f}%")
                    for h in hedges_created:
                        print(f"      • {h['coin']}: {h['position_size']:.6f} @ ${h['entry_price']:.2f} = ${h['position_value']:.2f}")
                    print(f"   Capital after hedge: ${self.shared_capital:.2f}\n")
        
        # Check if we should close existing hedges (recovery)
        elif self.active_hedges:
            # BUG #47 FIX: Use None for coin - recovery is global
            should_deactivate = self.hedge_manager.should_close_hedge(
                self.shared_capital,
                self.peak_capital,
                equity,
                self.peak_equity,
                None  # Global config, not coin-specific
            )
            
            if should_deactivate:
                # BUG #46 FIX: Close all hedges with correct prices per coin
                total_recovered = 0.0
                for hedge in list(self.active_hedges):
                    # BUG #57 FIX: Get current price - prefer shorter timeframe
                    hedge_coin = hedge['coin']
                    hedge_current_price = None
                    
                    if hedge_coin in self.kline_data and self.kline_data[hedge_coin]:
                        # Prefer 1m > 5m > 15m > 30m > 1h for most recent price
                        for tf in ['1m', '5m', '15m', '30m', '1h']:
                            if tf in self.kline_data[hedge_coin]:
                                tf_data = self.kline_data[hedge_coin][tf]
                                if len(tf_data) > 0:
                                    hedge_current_price = tf_data.iloc[-1]['close']
                                    break
                        
                        # Fallback: any available timeframe
                        if hedge_current_price is None:
                            for tf_data in self.kline_data[hedge_coin].values():
                                if len(tf_data) > 0:
                                    hedge_current_price = tf_data.iloc[-1]['close']
                                    break
                    
                    if hedge_current_price:
                        pnl = self.hedge_manager.calculate_hedge_pnl(hedge, hedge_current_price)
                        
                        # Return locked capital + P&L
                        recovered = hedge['position_value'] + pnl
                        self.shared_capital += recovered
                        total_recovered += recovered
                        
                        print(f"🛡️  HEDGE RECOVERY CLOSE: {hedge['coin']} | "
                              f"P&L: ${pnl:.2f} | Recovered: ${recovered:.2f}")
                
                self.sync_trader_capital()
                self.active_hedges = []
                
                print(f"   Total recovered: ${total_recovered:.2f} | Drawdown: {current_drawdown*100:.2f}%")
        
        # ========================================
        # 4. PATTERN DETECTION & TRADE OPENING
        # ========================================
        
        try:
            predictions, probabilities = self.classifier.predict(df_ohlcv.iloc[-60:].copy())
        except Exception as e:
            print(f"❌ Prediction error {coin} {timeframe}: {e}")
            return
        
        pattern = predictions[-1]
        pattern_prob = np.max(probabilities[-1])
        pattern_strength = pattern_prob
        
        if pattern != 'no_pattern':
            print(f"\n🔍 PATTERN DETECTED: {coin} {timeframe}")
            print(f"   Pattern: {pattern}")
            print(f"   Probability: {pattern_prob:.3f}")
            print(f"   Strength: {pattern_strength:.3f}")
        
        trader.decrement_cooldown()
        
        # Check global concurrent trades limit
        total_active_trades = sum(len(t.active_trades) for t in self.traders.values())
        if total_active_trades >= config.MAX_CONCURRENT_TRADES:
            if pattern != 'no_pattern':
                print(f"   ⛔ SKIP: Max concurrent trades (GLOBAL: {total_active_trades}/{config.MAX_CONCURRENT_TRADES})")
            return
        
        # BUG #37 FIX: Check per-coin cooldown to prevent duplicate trades from multiple timeframes
        current_time = time.time()
        last_open_time = self.last_trade_open_time.get(coin, 0)
        time_since_last_open = current_time - last_open_time
        
        if time_since_last_open < self.trade_open_cooldown:
            if pattern != 'no_pattern':
                print(f"   ⛔ SKIP: Trade cooldown active for {coin} ({time_since_last_open:.1f}s / {self.trade_open_cooldown}s)")
            return
        
        if not trader.should_open_trade(pattern, pattern_prob, pattern_strength):
            return
        
        # Calculate targets
        recent_data = df_ohlcv.iloc[-30:].copy()
        entry_price = current_candle['close']
        sl, tp, direction, params = trader.calculate_pattern_targets(
            pattern, entry_price, current_candle, recent_data
        )
        
        if direction == 'skip':
            print(f"   ⛔ SKIP: Trend/direction check failed")
            return
        
        # Position sizing
        position_size = trader.calculate_position_size(
            entry_price, sl, self.shared_capital, ml_probability=pattern_prob
        )
        
        if position_size <= 0:
            print(f"   ⛔ SKIP: Position size invalid ({position_size:.4f})")
            return
        
        self.sync_trader_capital()
        position_value = position_size * entry_price
        
        # DEMO mode trade opening
        if self.demo_mode:
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
            
            # Mark that this trade was opened in hedging mode
            trade['hedging_used'] = 'yes'
            
            self.shared_capital = trader.capital
            self.sync_trader_capital()
            
            # BUG #37 FIX: Update last trade open time for this coin
            self.last_trade_open_time[coin] = time.time()
            
            print(f"🟢 DEMO OPEN: {coin} {timeframe} | Pattern: {pattern} | "
                  f"Size: {position_size:.4f} (${position_value:.2f}) | "
                  f"Entry: ${entry_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | "
                  f"Shared Capital: ${self.shared_capital:.2f}")
    
    async def handle_ws_message(self, message):
        """WebSocket üzenet feldolgozása"""
        try:
            data = json.loads(message)
            
            if not hasattr(self, '_first_message_logged'):
                print(f"📥 First WS message received: {str(data)[:200]}...")
                self._first_message_logged = True
            
            if 'stream' in data and 'data' in data:
                data = data['data']
            
            if 'e' in data and data['e'] == 'kline':
                kline = data['k']
                symbol = kline['s']
                interval = kline['i']
                
                normalized_tf = self.reverse_timeframe_map.get(interval, interval)
                await self.process_kline(symbol, normalized_tf, kline)
        except Exception as e:
            print(f"❌ Message processing error: {e}")
            traceback.print_exc()
    
    async def run(self):
        """Fő loop - WebSocket connection és message handling"""
        streams = []
        for coin in self.coins:
            for tf in self.timeframes:
                binance_tf = self.timeframe_map.get(tf, tf)
                stream = f"{coin.lower()}@kline_{binance_tf}"
                streams.append(stream)
        
        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"
        
        print(f"\n🌐 Connecting to WebSocket...")
        print(f"   URL: {stream_url[:80]}...")
        print(f"   Streams: {len(streams)}")
        
        while True:
            try:
                async with websockets.connect(stream_url) as websocket:
                    print(f"✅ WebSocket connected!")
                    
                    while True:
                        try:
                            message = await websocket.recv()
                            await self.handle_ws_message(message)
                            
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


async def run_live_websocket_hedging_trading(coins, timeframes, api_key, api_secret, demo_mode=True):
    """
    Fő entry point a live hedging trading-hez
    """
    print("\n" + "="*80)
    print("🛡️  LIVE WEBSOCKET HEDGING TRADING INDÍTÁSA")
    print("="*80)
    
    trader = LiveWebSocketHedgingTrader(coins, timeframes, api_key, api_secret, demo_mode)
    
    await trader.initialize()
    await trader.run()


if __name__ == '__main__':
    import config
    
    asyncio.run(run_live_websocket_hedging_trading(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=config.BINANCE_DEMO_MODE
    ))
