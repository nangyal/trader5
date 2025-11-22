"""
Multi-Symbol Live Trading System with Binance WebSocket TICK Data
- WebSocket TRADE streams (real-time tick data) for multiple symbols
- Converts tick data to 5s, 15s, 1min timeframes in real-time
- Pattern detection with trained model on all timeframes
- Automatic order execution (2% risk per trade)
- Real-time balance updates
- Multi-processing for parallel symbol handling

V5.0: MULTI-SYMBOL MULTI-PROCESSOR
"""

import os
# Suppress all XGBoost warnings at C++ level
os.environ['XGBOOST_VERBOSITY'] = '0'

import sys
import pandas as pd
import numpy as np
import joblib
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from queue import Queue, Empty, Full
from threading import Thread, Lock
from multiprocessing import Process, Manager, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Suppress XGBoost parameter warnings
logging.getLogger('xgboost').setLevel(logging.ERROR)

# Suppress WebSocket library verbose messages
# This prevents "Error receiving message" spam during reconnections
logging.getLogger('binance.threaded_stream').setLevel(logging.CRITICAL)
logging.getLogger('binance.websockets').setLevel(logging.CRITICAL)
logging.getLogger('websocket').setLevel(logging.CRITICAL)

# Redirect websocket-client library's direct print statements
class WebSocketErrorFilter:
    """Filter out websocket error spam"""
    def __init__(self, stream):
        self.stream = stream
        self.last_error_time = 0
        
    def write(self, message):
        # Rate limit "Error receiving message" spam (max 1 per second)
        if 'Error receiving message' in message or 'Read loop has been closed' in message:
            current_time = time.time()
            if current_time - self.last_error_time >= 1.0:
                self.last_error_time = current_time
                self.stream.write(message)
        else:
            self.stream.write(message)
    
    def flush(self):
        self.stream.flush()

# Apply stderr filter
sys.stderr = WebSocketErrorFilter(sys.stderr)

from binance.client import Client
from binance import ThreadedWebsocketManager

from forex_pattern_classifier import (
    AdvancedPatternDetector,
    PatternStrengthScorer,
    EnhancedForexPatternClassifier
)

# Import configuration
from config import api_config, trading_config, pattern_config


# TRADING SYMBOLS - monitoring multiple cryptos
TRADING_SYMBOLS = [
    'ZECUSDT',
    'NEOUSDT',
    'SNXUSDT',
    'DASHUSDT',
    'XRPUSDT',
    'APTUSDT',
    'CAKEUSDT',
    'DOTUSDT',
    'ICPUSDT',
    'ETHUSDT',
    'ARUSDT',
    'NEARUSDT',
    'TONUSDT',
    'PENDLEUSDT',
    'ATOMUSDT'
]

# TIMEFRAMES - 5s, 15s, 1min
TIMEFRAMES = {
    '5s': 5,
    '15s': 15,
    '1min': 60
}


class MultiSymbolTrader:
    """
    Live trading system for a single symbol (runs in separate process)
    """
    
    def __init__(self, api_key, api_secret, symbol, shared_state, risk_per_trade=None):
        """
        Initialize live trader for one symbol
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbol: Trading pair (e.g., 'ETHUSDT')
            shared_state: Multiprocessing Manager dict for shared data
            risk_per_trade: Base risk percentage
        """
        self.symbol = symbol
        self.shared_state = shared_state
        
        if risk_per_trade is None:
            risk_per_trade = trading_config.RISK_PER_TRADE
        
        # Binance DEMO (demo.binance.vision) - demo=True automatically sets correct URL
        self.client = Client(api_key, api_secret, demo=True)
        
        # Increase timeout for demo API (slower than production)
        self.client.REQUEST_TIMEOUT = api_config.CONNECTION_CONFIG['timeout']
        
        # Configure session for retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,  # 3 retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client.session.mount("https://", adapter)
        self.client.session.mount("http://", adapter)
        
        self.risk_per_trade = risk_per_trade
        self.use_tiered_risk = trading_config.USE_TIERED_RISK
        self.risk_tiers = trading_config.RISK_TIERS
        
        # WebSocket manager
        self.twm = None
        self.ws_running = False
        self.reconnect_lock = Lock()
        
        # TICK DATA STORAGE
        self.tick_buffer = deque(maxlen=10000)
        self.tick_lock = Lock()
        self.tick_queue = Queue(maxsize=5000)
        self.tick_processor_thread = None
        self.tick_processor_stop = False
        self.last_queue_warning = 0.0
        self._tick_processor_started = False
        
        # TIMEFRAME CANDLE BUFFERS - dynamic based on TIMEFRAMES
        self.candles = {tf: deque(maxlen=2000) for tf in TIMEFRAMES.keys()}
        self.current_candle = {tf: None for tf in TIMEFRAMES.keys()}
        self.candle_locks = {tf: Lock() for tf in TIMEFRAMES.keys()}
        
        # Trading state
        self.active_trades = []
        
        # Symbol trading rules
        self.symbol_info = None
        self.lot_size_filter = None
        self.min_notional = None
        
        # Pattern detection
        self.model = None
        self.classifier = None
        
        # Performance tracking
        self.trades_history = []
        
        print(f"[{self.symbol}] 🚀 Trader initialized")
    
    def load_model(self, model_path=None):
        """Load trained pattern detection model"""
        if model_path is None:
            from config import model_config
            model_path = model_config.MODEL_SAVE_PATH
        
        try:
            self.classifier = EnhancedForexPatternClassifier()
            self.classifier.load_model(model_path)
            self.model = self.classifier.model
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return False
    
    def get_tiered_risk_percentage(self, current_capital):
        """
        Calculate risk percentage based on tiered compounding strategy
        
        Args:
            current_capital: Current account balance
            
        Returns:
            float: Risk percentage to use (0.0075 to 0.02)
        """
        if not self.use_tiered_risk:
            return self.risk_per_trade
        
        capital_ratio = current_capital / self.initial_capital
        
        for tier in self.risk_tiers:
            if capital_ratio < tier['max_capital_ratio']:
                return tier['risk']
        
        return self.risk_tiers[-1]['risk']  # Default to most conservative
    
    def get_account_balance(self, max_retries=3):
        """Get current USDT balance from demo account with retry logic"""
        from requests.exceptions import ReadTimeout, ConnectionError
        
        with self.balance_lock:
            for attempt in range(max_retries):
                try:
                    account = self.client.get_account()
                    
                    # Find USDT balance
                    for asset in account['balances']:
                        if asset['asset'] == 'USDT':
                            self.balance = float(asset['free'])
                            print(f"💰 Balance: ${self.balance:.2f} USDT")
                            return self.balance
                    
                    print("⚠️  No USDT balance found")
                    return 0.0
                    
                except (ReadTimeout, ConnectionError) as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                        print(f"⚠️  API timeout (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Error getting balance after {max_retries} attempts: {e}")
                        print("   💡 Demo API may be slow - check https://testnet.binance.vision/")
                        return 0.0
                except Exception as e:
                    print(f"❌ Error getting balance: {e}")
                    return 0.0
    
    def get_symbol_info(self):
        """Get trading rules for the symbol (LOT_SIZE, MIN_NOTIONAL, etc.)"""
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] == self.symbol:
                    self.symbol_info = symbol_data
                    
                    # Extract LOT_SIZE filter
                    for f in symbol_data['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            self.lot_size_filter = f
                            print(f"📊 LOT_SIZE: min={f['minQty']}, max={f['maxQty']}, step={f['stepSize']}")
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.min_notional = float(f['minNotional'])
                            print(f"📊 MIN_NOTIONAL: {f['minNotional']} USDT")
                    
                    return True
            
            print(f"⚠️  Symbol {self.symbol} not found in exchange info")
            return False
            
        except Exception as e:
            print(f"❌ Error getting symbol info: {e}")
            return False
    
    def load_historical_candles(self):
        """Load historical 1min candles and build initial 15s/30s buffers"""
        print("\n📥 Loading historical 1min candles (will resample to 15s/30s)...")
        
        from requests.exceptions import ReadTimeout, ConnectionError
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load 1min klines (limit: 500 = ~8 hours)
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval='1m',
                    limit=500
                )
                
                # Build 1min candles
                for k in klines[:-1]:  # Exclude current incomplete candle
                    candle = {
                        'time': datetime.fromtimestamp(k[0] / 1000),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    }
                    self.candles_1min.append(candle)
                
                print(f"   ✅ 1min: {len(self.candles_1min)} candles loaded")
                
                # Build 15s candles from 1min data (1min = 4x 15s)
                # NOTE: This is approximate - we split 1min candles into 4x 15s
                # Real tick data will override these once WebSocket connects
                for i, candle_1m in enumerate(list(self.candles_1min)):
                    # Split each 1min into 4x 15s candles (equal OHLC approximation)
                    for j in range(4):
                        candle_15s = {
                            'time': candle_1m['time'] + timedelta(seconds=j*15),
                            'open': candle_1m['open'],
                            'high': candle_1m['high'],
                            'low': candle_1m['low'],
                            'close': candle_1m['close'],
                            'volume': candle_1m['volume'] / 4  # Distribute volume evenly
                        }
                        self.candles_15s.append(candle_15s)
                
                print(f"   ✅ 15s: {len(self.candles_15s)} candles generated (approximated)")
                
                # Build 30s candles from 1min data (1min = 2x 30s)
                for candle_1m in list(self.candles_1min):
                    for j in range(2):
                        candle_30s = {
                            'time': candle_1m['time'] + timedelta(seconds=j*30),
                            'open': candle_1m['open'],
                            'high': candle_1m['high'],
                            'low': candle_1m['low'],
                            'close': candle_1m['close'],
                            'volume': candle_1m['volume'] / 2
                        }
                        self.candles_30s.append(candle_30s)
                
                print(f"   ✅ 30s: {len(self.candles_30s)} candles generated (approximated)")
                print(f"\n   💡 Note: 15s/30s are approximated from 1min data")
                print(f"   💡 Once WebSocket connects, real tick data will be used")
                break  # Success
                
            except (ReadTimeout, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"   ⚠️  Timeout (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed after {max_retries} attempts: {e}")
            except Exception as e:
                print(f"   ❌ Error loading candles: {e}")
                break
        
        print(f"✅ Historical data loaded\n")
    
    def start_websocket(self):
        """Start WebSocket for real-time TICK (trade) data"""
        print(f"🔌 Starting WebSocket TICK stream for {self.symbol}...")
        print(f"   📡 Using MAINNET with 443 port (wss://stream.binance.com:443)")
        print(f"   💡 Note: Trade execution on TESTNET, market data from MAINNET")
        
        # Create new WebSocket manager
        # Use testnet=False for MAINNET, library will use wss://stream.binance.com:443
        self.twm = ThreadedWebsocketManager(
            testnet=False,
            max_queue_size=5000,
        )
        self.twm.start()
        self.ws_running = True
        
        # Subscribe to TRADE stream (real-time tick data)
        print(f"   📊 Subscribing to TRADE stream (ticks)...")
        print(f"   📊 Building timeframes: 1min, 15s, 30s")
        
        self.twm.start_trade_socket(
            callback=self._handle_tick,
            symbol=self.symbol
        )
        
        print(f"✅ WebSocket connected and streaming TICKS")
        print(f"   Symbol: {self.symbol}")
        print(f"   Waiting for tick data...")
    
    def stop_websocket(self):
        """Stop WebSocket streams"""
        if self.twm and self.ws_running:
            try:
                self.ws_running = False  # Set to False BEFORE stopping
                self.twm.stop()
            except Exception as e:
                # Ignore errors during stop (connection already closed)
                pass
    
    def reconnect_websocket(self):
        """Reconnect WebSocket after error"""
        # Use try to acquire lock - if already locked, another thread is reconnecting
        acquired = self.reconnect_lock.acquire(blocking=False)
        if not acquired:
            # Silently skip if reconnection already in progress
            return
        
        try:
            print("\n🔄 Reconnecting WebSocket...")
            
            # Mark as not running
            old_ws_running = self.ws_running
            self.ws_running = False
            
            # Abandon old connection (don't call stop - it hangs)
            # Just let the old TWM die and create a new one
            old_twm = self.twm
            self.twm = None
            
            if old_twm:
                print("   🗑️  Abandoning old WebSocket connection (will be garbage collected)")
            
            # Wait before reconnecting
            print("   ⏳ Waiting 2 seconds before reconnecting...")
            time.sleep(2)
            
            # Retry loop
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"   🔌 Reconnection attempt {attempt}/{max_retries}...")
                    self.start_websocket()
                    print("✅ WebSocket reconnected successfully!\n")
                    return True  # Success!
                except Exception as e:
                    if attempt < max_retries:
                        wait_time = attempt * 3  # 3, 6, 9 seconds
                        print(f"   ❌ Attempt {attempt} failed: {e}")
                        print(f"   ⏳ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ All {max_retries} reconnection attempts failed!")
                        print(f"   💡 Manual restart may be required.")
                        self.ws_running = False
                        return False
        except Exception as e:
            print(f"❌ Critical error in reconnection handler: {e}")
            self.ws_running = False
            return False
        finally:
            self.reconnect_lock.release()
    
    def _reconnect_wrapper(self):
        """Wrapper to reset reconnecting flag after attempt"""
        try:
            success = self.reconnect_websocket()
            if success:
                # Reset flags on success
                self._reconnecting = False
                if hasattr(self, '_reconnecting_since'):
                    delattr(self, '_reconnecting_since')
            else:
                # On failure, flag stays set (will timeout after 60s)
                print(f"   ⚠️  Reconnection failed, flag will timeout in 60s")
        except Exception as e:
            print(f"❌ Reconnect wrapper error: {e}")
            # Reset flags on exception
            self._reconnecting = False
            if hasattr(self, '_reconnecting_since'):
                delattr(self, '_reconnecting_since')
    
    def _start_tick_processor(self):
        """Ensure background tick processor thread is running"""
        if self.tick_processor_thread and self.tick_processor_thread.is_alive() and not self.tick_processor_stop:
            return
        self.tick_processor_stop = False
        self.tick_processor_thread = Thread(target=self._tick_processor_loop, daemon=True)
        self.tick_processor_thread.start()
        if not self._tick_processor_started:
            print("   🧵 Tick processor thread started")
            self._tick_processor_started = True

    def _tick_processor_loop(self):
        """Continuously process ticks off the queue to keep websocket callback light"""
        while not self.tick_processor_stop:
            try:
                tick = self.tick_queue.get(timeout=1)
            except Empty:
                continue

            if tick is None:
                self.tick_queue.task_done()
                break

            try:
                self._process_tick(tick)
            except Exception as e:
                print(f"⚠️  Tick processing error: {e}")
            finally:
                self.tick_queue.task_done()

    def _process_tick(self, tick):
        """Handle trade tick outside of websocket thread"""
        trade_time_ms, price_raw, qty_raw = tick
        tick_data = {
            'time': datetime.fromtimestamp(trade_time_ms / 1000),
            'price': float(price_raw),
            'quantity': float(qty_raw)
        }

        with self.tick_lock:
            self.tick_buffer.append(tick_data)

        self._update_candle(tick_data, '1min', 60)
        self._update_candle(tick_data, '15s', 15)
        self._update_candle(tick_data, '30s', 30)

    def _handle_tick(self, msg):
        """Handle real-time TICK (trade) data and aggregate into timeframes"""
        # Handle WebSocket errors
        if isinstance(msg, dict):
            if msg.get('e') == 'error':
                # Rate limit error messages (max 1 per second per error type)
                error_msg = str(msg)
                error_key = 'websocket_error'
                current_time = time.time()
                
                # Check if we should print this error (1 second cooldown)
                if error_key not in self.last_error_time or (current_time - self.last_error_time[error_key]) >= 1.0:
                    self.last_error_time[error_key] = current_time
                    print(f"\n⚠️  WebSocket error: {msg}")
                    
                    # Only trigger reconnection ONCE (until successful or timeout)
                    if 'Read loop has been closed' in error_msg or msg.get('m') == 'Read loop has been closed':
                        # Check if reconnection already triggered
                        now = time.time()
                        
                        # Reset reconnecting flag if stuck for > 60 seconds
                        if hasattr(self, '_reconnecting_since'):
                            if now - self._reconnecting_since > 60:
                                print(f"   ⏰ Reconnection timeout (60s), resetting flag...")
                                self._reconnecting = False
                                delattr(self, '_reconnecting_since')
                        
                        if not hasattr(self, '_reconnecting') or not self._reconnecting:
                            self._reconnecting = True  # Mark as reconnecting
                            self._reconnecting_since = now  # Track start time
                            print(f"   🔄 Triggering reconnection...")
                            # Start reconnection in NON-daemon thread (so it can complete)
                            reconnect_thread = Thread(target=self._reconnect_wrapper, daemon=False)
                            reconnect_thread.start()
                return
        
        # Handle trade messages
        if msg.get('e') == 'trade':
            tick = (msg['T'], msg['p'], msg['q'])

            if not self.tick_processor_thread or not self.tick_processor_thread.is_alive():
                self._start_tick_processor()

            try:
                self.tick_queue.put_nowait(tick)
            except Full:
                # Drop the oldest tick to keep stream real-time
                try:
                    self.tick_queue.get_nowait()
                    self.tick_queue.task_done()
                except Empty:
                    pass
                try:
                    self.tick_queue.put_nowait(tick)
                except Full:
                    pass

                current_time = time.time()
                if current_time - self.last_queue_warning > 5:
                    self.last_queue_warning = current_time
                    print("⚠️  Tick queue full, dropping oldest tick to stay real-time")
    
    def _update_candle(self, tick, timeframe, seconds):
        """Update or complete candle for a specific timeframe"""
        with self.candle_locks[timeframe]:
            # Get current candle
            if timeframe == '1min':
                current_candle = self.current_candle_1min
                candle_deque = self.candles_1min
            elif timeframe == '15s':
                current_candle = self.current_candle_15s
                candle_deque = self.candles_15s
            else:  # 30s
                current_candle = self.current_candle_30s
                candle_deque = self.candles_30s
            
            # Calculate candle start time (round down to timeframe boundary)
            tick_time = tick['time']
            candle_start = datetime(
                tick_time.year, tick_time.month, tick_time.day,
                tick_time.hour, tick_time.minute,
                (tick_time.second // seconds) * seconds
            )
            
            # Check if we need to start a new candle
            if current_candle is None or current_candle['time'] != candle_start:
                # Close previous candle if exists
                if current_candle is not None:
                    candle_deque.append(current_candle)
                    
                    # Run pattern detection on closed candle
                    if len(candle_deque) >= 100:
                        self._check_pattern_on_timeframe(timeframe, candle_deque)
                
                # Start new candle
                current_candle = {
                    'time': candle_start,
                    'open': tick['price'],
                    'high': tick['price'],
                    'low': tick['price'],
                    'close': tick['price'],
                    'volume': tick['quantity']
                }
            else:
                # Update existing candle
                current_candle['high'] = max(current_candle['high'], tick['price'])
                current_candle['low'] = min(current_candle['low'], tick['price'])
                current_candle['close'] = tick['price']
                current_candle['volume'] += tick['quantity']
            
            # Store updated candle
            if timeframe == '1min':
                self.current_candle_1min = current_candle
            elif timeframe == '15s':
                self.current_candle_15s = current_candle
            else:  # 30s
                self.current_candle_30s = current_candle
    
    def _check_pattern_on_timeframe(self, timeframe, candles):
        """Check for patterns on a specific timeframe"""
        if self.model is None or self.classifier is None:
            return
        
        # Need at least 100 candles for reliable pattern detection
        if len(candles) < 100:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(list(candles))
            df.set_index('time', inplace=True)
            
            # Use last 100 candles for prediction
            df_subset = df.tail(100)
            
            # Predict patterns
            predictions, probabilities = self.classifier.predict(df_subset)
            
            # Check last prediction (index is -1 for the subset)
            last_pattern = predictions[-1]
            last_prob = probabilities[-1].max()
            
            # ✅ PRODUCTION: Standard thresholds (70% probability, 70% strength)
            if last_pattern != 'no_pattern' and last_prob >= 0.7:  # 70% probability
                # Calculate pattern strength (use index relative to subset)
                scorer = PatternStrengthScorer()
                strength = scorer.calculate_pattern_strength(
                    df_subset, 
                    last_pattern, 
                    len(df_subset) - 1,  # Last index in subset
                    window=50
                )
                
                # ✅ PRODUCTION: Standard strength threshold (70%)
                if strength >= 0.7:  # 70% strength
                    self._process_pattern_signal(last_pattern, last_prob, strength, timeframe)
        
        except Exception as e:
            print(f"⚠️  Pattern detection error on {timeframe}: {e}")
    
    def _process_pattern_signal(self, pattern, probability, strength, timeframe):
        """Process detected pattern signal"""
        print(f"\n🎯 PATTERN DETECTED on {timeframe}:")
        print(f"   Pattern: {pattern}")
        print(f"   Probability: {probability:.1%}")
        print(f"   Strength: {strength:.1%}")
        
        # Execute trade based on pattern
        self._execute_pattern_trade(pattern, probability, strength, timeframe)
    
    def _execute_pattern_trade(self, pattern, probability, strength, timeframe):
        """Execute trade based on detected pattern"""
        try:
            # Get current price from latest candle
            candles_map = {
                '1min': self.candles_1min,
                '15s': self.candles_15s,
                '30s': self.candles_30s
            }
            
            candles = candles_map.get(timeframe)
            if not candles or len(candles) == 0:
                print(f"   ❌ No candles available for {timeframe}")
                return
            
            current_price = candles[-1]['close']
            
            # Calculate stop loss and take profit
            stop_loss, take_profit, direction = self._calculate_targets(
                pattern, current_price, candles
            )
            
            if direction == 'skip':
                print(f"   ⏭️  Skipping misaligned pattern")
                return
            
            # Check if we already have an active trade (max 4)
            if len(self.active_trades) >= 4:
                print(f"   ⏸️  Max concurrent trades reached (4)")
                return
            
            # Calculate position size based on tiered risk management
            with self.balance_lock:
                current_balance = self.balance
            
            if current_balance == 0:
                current_balance = self.get_account_balance()
            
            # Calculate total capital (free balance + active positions)
            active_position_value = 0
            for trade in self.active_trades:
                # Assume current price for active positions
                active_position_value += trade['quantity'] * current_price
            
            total_capital = current_balance + active_position_value
            
            print(f"   💰 Free balance: ${current_balance:.2f}")
            print(f"   📊 Active positions: ${active_position_value:.2f}")
            print(f"   💼 Total capital: ${total_capital:.2f}")
            
            # Check minimum balance requirement (MIN_NOTIONAL / 20% = minimum balance)
            min_balance_required = (self.min_notional or 10.0) / 0.20  # Need at least 5x MIN_NOTIONAL
            if current_balance < min_balance_required:
                print(f"   ⚠️  Insufficient free balance: ${current_balance:.2f} < ${min_balance_required:.2f} minimum")
                print(f"   💡 Minimum free balance needed: ${min_balance_required:.2f} (to meet MIN_NOTIONAL with 20% position size)")
                return
            
            # Get tiered risk percentage based on TOTAL capital (not just free balance)
            current_risk_pct = self.get_tiered_risk_percentage(total_capital)
            risk_amount = total_capital * current_risk_pct
            
            if direction == 'long':
                risk_per_unit = current_price - stop_loss
            else:
                risk_per_unit = stop_loss - current_price
            
            if risk_per_unit <= 0:
                print(f"   ❌ Invalid risk calculation")
                return
            
            # Calculate quantity based on risk
            quantity = risk_amount / risk_per_unit
            
            # 🔒 SAFETY: Cap total exposure to 80% of capital (4 trades × 20% each)
            # Check total active position value
            total_active_value = sum(t['quantity'] * current_price for t in self.active_trades)
            max_total_exposure = total_capital * 0.80  # Max 80% total (4 × 20%)
            
            if total_active_value >= max_total_exposure:
                print(f"   ⚠️  Total exposure ${total_active_value:.2f} >= max ${max_total_exposure:.2f} (80% of capital)")
                print(f"   ⏸️  Skipping trade to avoid over-exposure")
                return
            
            # 🔒 SAFETY: Cap position size to 20% of TOTAL capital per trade
            # This prevents overleveraging when stop-loss is very tight
            max_position_value = total_capital * 0.20  # Max 20% of TOTAL capital per position
            position_value = quantity * current_price
            
            if position_value > max_position_value:
                print(f"   ⚠️  Position value ${position_value:.2f} exceeds max ${max_position_value:.2f} (20% of total capital)")
                print(f"   ⚠️  Reducing quantity from {quantity} to fit 20% position size limit...")
                quantity = max_position_value / current_price
                position_value = quantity * current_price
                print(f"   ✅ Adjusted quantity: {quantity} ETH (${position_value:.2f})")
            
            # Round quantity to LOT_SIZE step (e.g., 0.001 for ETHUSDT on testnet)
            if self.lot_size_filter:
                step_size = float(self.lot_size_filter['stepSize'])
                min_qty = float(self.lot_size_filter['minQty'])
                max_qty = float(self.lot_size_filter['maxQty'])
                
                # Round to step_size precision
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = round(quantity - (quantity % step_size), precision)
                
                # Validate quantity
                if quantity < min_qty:
                    print(f"   ❌ Quantity {quantity} < minimum {min_qty}")
                    return
                if quantity > max_qty:
                    print(f"   ⚠️  Quantity {quantity} > maximum {max_qty}, capping to {max_qty}")
                    quantity = max_qty
            else:
                # Fallback if no LOT_SIZE info (use 5 decimals for ETH)
                quantity = round(quantity, 5)
            
            # Check minimum notional value (quantity * price >= min_notional)
            if self.min_notional:
                notional_value = quantity * current_price
                if notional_value < self.min_notional:
                    print(f"   ❌ Notional value ${notional_value:.2f} < minimum ${self.min_notional:.2f}")
                    return
            
            if quantity <= 0:
                print(f"   ❌ Invalid quantity: {quantity}")
                return
            
            # Calculate final position value
            final_position_value = quantity * current_price
            
            print(f"\n📊 TRADE SETUP:")
            print(f"   Direction: {direction.upper()}")
            print(f"   Entry: {current_price:.2f} USDT")
            print(f"   Stop Loss: {stop_loss:.2f} USDT")
            print(f"   Take Profit: {take_profit:.2f} USDT")
            print(f"   Quantity: {quantity} ETH")
            print(f"   Position Value: ${final_position_value:.2f} ({final_position_value/total_capital*100:.1f}% of total capital)")
            print(f"   Risk: {risk_amount:.2f} USDT ({current_risk_pct*100:.2f}% of total capital)")
            print(f"   Free Balance: ${current_balance:.2f}")
            
            # Execute order
            if direction == 'long':
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                print(f"   ✅ BUY order executed: {order['orderId']}")
            else:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                print(f"   ✅ SELL order executed: {order['orderId']}")
            
            # Store trade info
            trade = {
                'order_id': order['orderId'],
                'pattern': pattern,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': quantity,
                'risk_amount': risk_amount,
                'entry_time': datetime.now(),
                'timeframe': timeframe,
                'strength': strength,
                'probability': probability,
                'status': 'open'
            }
            
            self.active_trades.append(trade)
            
            # Update balance
            self.get_account_balance()
            
        except Exception as e:
            print(f"   ❌ Error executing trade: {e}")
    
    def _calculate_targets(self, pattern, entry_price, candles):
        """Calculate stop loss and take profit targets (from config)"""
        base_pattern = pattern.split('_')[0] if '_' in pattern else pattern
        
        # Get targets from config
        targets = trading_config.PATTERN_TARGETS.get(
            pattern, 
            {'sl_pct': 0.015, 'tp_pct': 0.03}  # Default fallback
        )
        
        # Determine trend from candles (using config settings)
        trend_config = trading_config.TREND_ALIGNMENT
        if len(candles) >= trend_config['lookback_period']:
            closes = np.array([c['close'] for c in list(candles)[-trend_config['lookback_period']:]])
            
            # ✅ PRODUCTION: EMA filter ENABLED for trend confirmation
            if trend_config['use_ema_filter']:
                ema_period = trend_config['ema_period']
                ema50 = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().iloc[-1]
                current_price = closes[-1]
                
                # Skip if price below EMA(50) - trend filter
                if current_price < ema50:
                    print(f"   ⏭️  Price ${current_price:.2f} < EMA50 ${ema50:.2f} - skipping (trend filter)")
                    return 0, 0, 'skip'
                else:
                    print(f"   ✅ Price ${current_price:.2f} > EMA50 ${ema50:.2f} - trend OK")
            
            # ✅ PRODUCTION: ATR volatility filter ENABLED
            volatility_config = trading_config.VOLATILITY_FILTER
            if volatility_config['enable']:
                atr_period = volatility_config['atr_period']
                highs = np.array([c['high'] for c in list(candles)[-atr_period:]])
                lows = np.array([c['low'] for c in list(candles)[-atr_period:]])
                closes_14 = closes[-atr_period:]
                
                tr1 = highs - lows
                tr2 = np.abs(highs - np.roll(closes_14, 1))
                tr3 = np.abs(lows - np.roll(closes_14, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr)
                atr_pct = (atr / current_price) * 100
                
                print(f"   📊 ATR: {atr:.2f}, ATR%: {atr_pct:.2f}%, Min required: {volatility_config['min_atr_pct']:.2f}%")
                
                # Skip if ATR% outside configured range
                if atr_pct < volatility_config['min_atr_pct']:
                    print(f"   ⏭️  ATR {atr_pct:.2f}% < {volatility_config['min_atr_pct']:.2f}% - too low volatility, skipping")
                    return 0, 0, 'skip'
                else:
                    print(f"   ✅ ATR {atr_pct:.2f}% >= {volatility_config['min_atr_pct']:.2f}% - volatility OK")
            
            # LONG-ONLY: Use pattern classification from config
            bullish_patterns = [p.split('_')[0] for p in trend_config['bullish_patterns']]
            bearish_patterns = [p.split('_')[0] for p in trend_config['bearish_patterns']]
            
            is_bullish = any(bp in base_pattern for bp in bullish_patterns)
            is_bearish = any(bp in base_pattern for bp in bearish_patterns)
            
            print(f"   📊 Pattern: {pattern} (base: {base_pattern})")
            print(f"   📊 Bullish patterns: {bullish_patterns}")
            print(f"   📊 Bearish patterns: {bearish_patterns}")
            print(f"   📊 Is bullish: {is_bullish}, Is bearish: {is_bearish}")
            
            # Skip descending triangles if in blacklist
            blacklist = trading_config.PATTERN_FILTERS.get('blacklist_patterns', [])
            if pattern in blacklist:
                print(f"   ⏭️  Pattern in blacklist: {blacklist}")
                return 0, 0, 'skip'
            
            direction = 'long'
        else:
            return 0, 0, 'skip'  # Not enough data
        
        stop_loss = entry_price * (1 - targets['sl_pct'])
        take_profit = entry_price * (1 + targets['tp_pct'])
        
        return stop_loss, take_profit, direction
    
    def monitor_trades(self):
        """Monitor active trades and close when targets are hit"""
        print("\n👁️  Starting trade monitoring...")
        
        while True:
            try:
                if not self.active_trades:
                    time.sleep(5)
                    continue
                
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker['price'])
                
                trades_to_close = []
                
                for i, trade in enumerate(self.active_trades):
                    if trade['status'] != 'open':
                        continue
                    
                    # Check stop loss and take profit
                    if trade['direction'] == 'long':
                        if current_price <= trade['stop_loss']:
                            # Stop loss hit
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'stop_loss'
                            # pnl will be computed precisely when closing (avoid fixed multipliers)
                            trades_to_close.append(i)
                            
                        elif current_price >= trade['take_profit']:
                            # Take profit hit
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'take_profit'
                            # pnl will be computed precisely when closing
                            trades_to_close.append(i)
                    
                    else:  # short
                        if current_price >= trade['stop_loss']:
                            # Stop loss hit
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'stop_loss'
                            # pnl will be computed precisely when closing
                            trades_to_close.append(i)
                            
                        elif current_price <= trade['take_profit']:
                            # Take profit hit
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'take_profit'
                            # pnl will be computed precisely when closing
                            trades_to_close.append(i)
                
                # Close trades
                for idx in reversed(trades_to_close):
                    trade = self.active_trades[idx]
                    self._close_trade(trade)
                    self.active_trades.pop(idx)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"❌ Error monitoring trades: {e}")
                time.sleep(5)
    
    def _close_trade(self, trade):
        """Close a trade and update balance"""
        try:
            # Execute closing order
            if trade['direction'] == 'long':
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=trade['quantity']
                )
            else:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=trade['quantity']
                )
            
            trade['exit_time'] = datetime.now()
            trade['exit_order_id'] = order['orderId']
            trade['status'] = 'closed'
            # Compute precise P&L based on entry/exit prices and quantity
            try:
                entry = float(trade.get('entry_price', 0.0))
                exit_price = float(trade.get('exit_price', entry))
                qty = float(trade.get('quantity', 0.0))

                if trade['direction'] == 'long':
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty

                # If quantity or prices are missing, fall back to risk_amount sign
                if qty <= 0 or entry == 0.0:
                    pnl = trade.get('risk_amount', 0.0) * (1 if trade.get('exit_reason') == 'take_profit' else -1)

                trade['pnl'] = round(pnl, 2)
            except Exception:
                trade['pnl'] = trade.get('risk_amount', 0.0)

            # Update statistics
            self.total_pnl += trade['pnl']
            self.trades_history.append(trade)
            
            # Print trade result
            print(f"\n{'='*60}")
            print(f"🔔 TRADE CLOSED - {trade['exit_reason'].upper()}")
            print(f"{'='*60}")
            print(f"Pattern: {trade['pattern']}")
            print(f"Direction: {trade['direction'].upper()}")
            print(f"Entry: {trade['entry_price']:.6f} USDT")
            print(f"Exit: {trade['exit_price']:.6f} USDT")
            print(f"P&L: {trade['pnl']:+.2f} USDT")
            print(f"Duration: {trade['exit_time'] - trade['entry_time']}")
            print(f"Total P&L: {self.total_pnl:+.2f} USDT")
            print(f"{'='*60}\n")
            
            # Update balance
            self.get_account_balance()
            
        except Exception as e:
            print(f"❌ Error closing trade: {e}")
    
    def print_status(self):
        """Print current system status"""
        while True:
            try:
                print(f"\n{'='*60}")
                print(f"📊 LIVE TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                print(f"Symbol: {self.symbol}")
                print(f"Balance: {self.balance:.2f} USDT")
                print(f"Active Trades: {len(self.active_trades)}")
                print(f"Total Trades: {len(self.trades_history)}")
                print(f"Total P&L: {self.total_pnl:+.2f} USDT")
                
                if self.trades_history:
                    wins = sum(1 for t in self.trades_history if t['pnl'] > 0)
                    win_rate = wins / len(self.trades_history) * 100
                    print(f"Win Rate: {win_rate:.1f}%")
                
                print(f"\nData Points:")
                print(f"  Ticks buffered: {len(self.tick_buffer)}")
                print(f"  1min candles: {len(self.candles_1min)}")
                print(f"  15s candles: {len(self.candles_15s)}")
                print(f"  30s candles: {len(self.candles_30s)}")
                
                # Show current building candle from each timeframe
                if self.current_candle_1min:
                    c = self.current_candle_1min
                    print(f"    └─ Current 1min: {c['time'].strftime('%H:%M:%S')} - O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f}")
                
                if self.current_candle_15s:
                    c = self.current_candle_15s
                    print(f"    └─ Current 15s: {c['time'].strftime('%H:%M:%S')} - O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f}")
                
                if self.current_candle_30s:
                    c = self.current_candle_30s
                    print(f"    └─ Current 30s: {c['time'].strftime('%H:%M:%S')} - O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f}")
                
                if self.active_trades:
                    print(f"\nActive Trades:")
                    for trade in self.active_trades:
                        print(f"  • {trade['pattern']} ({trade['direction']}) @ {trade['entry_price']:.6f}")
                
                print(f"{'='*60}\n")
                
                time.sleep(30)  # Print status every 30 seconds
                
            except Exception as e:
                print(f"❌ Error printing status: {e}")
                time.sleep(30)
    
    def run(self):
        """Start the live trading system"""
        print("\n" + "="*60)
        print("🚀 STARTING LIVE TRADING SYSTEM")
        print("="*60)
        
        # Get symbol trading rules first
        print("\n📊 Getting symbol trading rules...")
        self.get_symbol_info()
        
        # Get initial balance
        self.get_account_balance()
        
        # Load historical candles
        self.load_historical_candles()
        
        # Start WebSocket
        self.start_websocket()
        
        # Start monitoring thread
        monitor_thread = Thread(target=self.monitor_trades, daemon=True)
        monitor_thread.start()
        
        # Start status thread
        status_thread = Thread(target=self.print_status, daemon=True)
        status_thread.start()
        
        print("\n✅ System is running. Press Ctrl+C to stop.")
        print("💡 Note: WebSocket will auto-reconnect if connection drops.\n")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping live trading system...")
            self.stop_websocket()
            print("✅ System stopped.")
            
            # Print final summary
            if self.trades_history:
                print(f"\n{'='*60}")
                print("📈 FINAL SUMMARY")
                print(f"{'='*60}")
                print(f"Total Trades: {len(self.trades_history)}")
                print(f"Total P&L: {self.total_pnl:+.2f} USDT")
                
                wins = sum(1 for t in self.trades_history if t['pnl'] > 0)
                win_rate = wins / len(self.trades_history) * 100
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Final Balance: {self.balance:.2f} USDT")
                print(f"{'='*60}\n")


def main():
    """Main function to run live trading"""
    
    # Get API credentials from config
    api_key, api_secret = api_config.get_api_credentials()
    
    print(f"🔑 Using API environment: {api_config.ENVIRONMENT}")
    
    # Initialize trader with ETHUSDT (tick-based)
    trader = MultiSymbolTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbol='ETHUSDT',  # Changed to ETHUSDT
        shared_state=None,  # No shared state for single-symbol mode
        risk_per_trade=trading_config.RISK_PER_TRADE
    )
    
    print(f"\n💡 TICK-BASED TRADING:")
    print(f"   • WebSocket receives real-time trade ticks")
    print(f"   • Ticks are aggregated into 1min, 15s, 30s candles")
    print(f"   • Pattern detection runs on all timeframes")
    print(f"   • More accurate than pre-aggregated klines\n")
    
    # Load trained model (path from config)
    if not trader.load_model():
        print("⚠️  No trained model found. Please train the model first.")
        print("   Run: python enhanced_main.py")
        return
    
    # Start trading
    trader.run()


if __name__ == "__main__":
    main()
