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
import json
import csv
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from collections import deque
from queue import Queue, Empty, Full
from threading import Thread, Lock, Event
from multiprocessing import Process, Manager, cpu_count
import websocket
import warnings
warnings.filterwarnings('ignore')

# Suppress XGBoost parameter warnings
logging.getLogger('xgboost').setLevel(logging.ERROR)

# Suppress WebSocket library verbose messages
logging.getLogger('binance.threaded_stream').setLevel(logging.CRITICAL)
logging.getLogger('binance.websockets').setLevel(logging.CRITICAL)
logging.getLogger('websocket').setLevel(logging.CRITICAL)

from binance.client import Client

from forex_pattern_classifier import (
    PatternStrengthScorer,
    EnhancedForexPatternClassifier
)

# Import configuration
from config import api_config, trading_config, pattern_config


# TRADING SYMBOLS - unique list
TRADING_SYMBOLS = list(set([
    'ZECUSDT', 'NEOUSDT', 'SNXUSDT', 'DASHUSDT', 'XRPUSDT',
    'APTUSDT', 'CAKEUSDT', 'DOTUSDT', 'ICPUSDT', 'ETHUSDT',
    'ARUSDT', 'NEARUSDT', 'TONUSDT', 'PENDLEUSDT', 'ATOMUSDT'
]))

# TIMEFRAMES - 5s, 15s, 1min
TIMEFRAMES = {
    '5s': 5,
    '15s': 15,
    '1min': 60
}


class MultiSymbolTrader:
    """Live trading system for a single symbol (runs in separate process)"""
    
    def __init__(self, api_key, api_secret, symbol, shared_state):
        self.symbol = symbol
        self.shared_state = shared_state
        
        # Binance DEMO client
        self.client = Client(api_key, api_secret, demo=True)
        self.client.REQUEST_TIMEOUT = api_config.CONNECTION_CONFIG['timeout']
        
        # Configure session for retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client.session.mount("https://", adapter)
        self.client.session.mount("http://", adapter)
        
        self.risk_per_trade = trading_config.RISK_PER_TRADE
        self.use_tiered_risk = trading_config.USE_TIERED_RISK
        self.risk_tiers = trading_config.RISK_TIERS
        
        # WebSocket
        self.twm = None
        self.ws_running = False
        
        # TICK DATA
        self.tick_buffer = deque(maxlen=10000)
        self.tick_lock = Lock()
        self.tick_queue = Queue(maxsize=5000)
        self.tick_processor_thread = None
        self.tick_processor_stop = False
        
        # TIMEFRAME CANDLES
        self.candles = {tf: deque(maxlen=2000) for tf in TIMEFRAMES.keys()}
        self.current_candle = {tf: None for tf in TIMEFRAMES.keys()}
        self.candle_locks = {tf: Lock() for tf in TIMEFRAMES.keys()}
        
        # Trading state
        self.active_trades = []
        self.symbol_info = None
        self.lot_size_filter = None
        self.min_notional = None
        
        # Pattern detection
        self.model = None
        self.classifier = None
        self.trades_history = []
        
        print(f"[{self.symbol}] 🚀 Trader initialized")
    
    @staticmethod
    def _ensure_trade_log():
        """Ensure shared trade log file exists with headers"""
        log_file = 'trades_log.csv'
        try:
            if not os.path.exists(log_file):
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'symbol', 'action', 'direction', 'price', 'quantity', 
                                   'value', 'pattern', 'timeframe', 'probability', 'strength', 
                                   'stop_loss', 'take_profit', 'order_id', 'reason', 'pnl_usdt'])
        except Exception as e:
            logging.error(f"Error initializing trade log: {e}")
    
    def _log_trade(self, action, direction, price, quantity, value, pattern='', timeframe='', 
                   probability=0.0, strength=0.0, stop_loss=0.0, take_profit=0.0, order_id='', reason='', pnl_usdt=0.0):
        """Log trade to shared CSV file"""
        try:
            self._ensure_trade_log()
            with open('trades_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    self.symbol,
                    action,
                    direction,
                    price,
                    quantity,
                    value,
                    pattern,
                    timeframe,
                    probability,
                    strength,
                    stop_loss,
                    take_profit,
                    order_id,
                    reason,
                    pnl_usdt
                ])
        except Exception as e:
            logging.error(f"Error logging trade: {e}")

    @staticmethod
    def _ensure_negative_balance_log():
        """Ensure negative-balance log file exists with headers"""
        log_file = 'negative_balance_trades.csv'
        try:
            if not os.path.exists(log_file):
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'symbol', 'order_id', 'action', 'direction',
                        'entry_time', 'entry_price', 'exit_time', 'exit_price', 'quantity',
                        'pnl_usdt', 'balance_before', 'balance_after', 'total_pnl', 'reason',
                        'pattern', 'timeframe'
                    ])
        except Exception as e:
            logging.error(f"Error initializing negative balance log: {e}")

    def _log_negative_balance_event(self, trade, balance_before, balance_after):
        """Log an event when a trade causes a negative account balance"""
        try:
            self._ensure_negative_balance_log()
            with open('negative_balance_trades.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    self.symbol,
                    trade.get('order_id', ''),
                    'CLOSE',
                    trade.get('direction', ''),
                    trade.get('entry_time').strftime('%Y-%m-%d %H:%M:%S') if trade.get('entry_time') else '',
                    trade.get('entry_price', ''),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    trade.get('exit_price', ''),
                    trade.get('quantity', 0),
                    round(trade.get('pnl', 0.0), 2),
                    round(balance_before, 2),
                    round(balance_after, 2),
                    round(self.shared_state.get('total_pnl', 0.0), 2),
                    trade.get('exit_reason', ''),
                    trade.get('pattern', ''),
                    trade.get('timeframe', '')
                ])
        except Exception as e:
            logging.error(f"Error logging negative balance event: {e}")

    
    def adjust_stop_loss(self, new_stop_price):
        """Dynamically adjust stop loss for current position"""
        if self.position is None:
            return False
        
        try:
            # Cancel existing stop loss order if it exists
            if self.position.get('stop_order_id'):
                try:
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=self.position['stop_order_id']
                    )
                except Exception as e:
                    print(f"[{self.symbol}] ⚠️  Could not cancel old stop loss: {e}")
            
            # Place new stop loss order
            side = 'SELL' if self.position['side'] == 'LONG' else 'BUY'
            stop_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='STOP_MARKET',
                stopPrice=new_stop_price,
                closePosition=True
            )
            
            self.position['stop_loss'] = new_stop_price
            self.position['stop_order_id'] = stop_order['orderId']
            print(f"[{self.symbol}] 🎯 Stop loss adjusted to {new_stop_price}")
            return True
            
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error adjusting stop loss: {e}")
            return False
    
    def adjust_take_profit(self, new_tp_price):
        """Dynamically adjust take profit for current position"""
        if self.position is None:
            return False
        
        try:
            # Cancel existing take profit order if it exists
            if self.position.get('tp_order_id'):
                try:
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=self.position['tp_order_id']
                    )
                except Exception as e:
                    print(f"[{self.symbol}] ⚠️  Could not cancel old take profit: {e}")
            
            # Place new take profit order
            side = 'SELL' if self.position['side'] == 'LONG' else 'BUY'
            tp_order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=new_tp_price,
                closePosition=True
            )
            
            self.position['take_profit'] = new_tp_price
            self.position['tp_order_id'] = tp_order['orderId']
            print(f"[{self.symbol}] 🎯 Take profit adjusted to {new_tp_price}")
            return True
            
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error adjusting take profit: {e}")
            return False
    
    def enable_trailing_stop(self, callback_rate=1.0, activation_price=None):
        """Enable trailing stop for current position
        
        Args:
            callback_rate: Percentage distance from peak (e.g., 1.0 = 1%)
            activation_price: Price at which trailing starts (optional)
        """
        if self.position is None:
            return False
        
        try:
            # Cancel existing stop loss
            if self.position.get('stop_order_id'):
                try:
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=self.position['stop_order_id']
                    )
                except Exception as e:
                    print(f"[{self.symbol}] ⚠️  Could not cancel old stop: {e}")
            
            # Place trailing stop order
            side = 'SELL' if self.position['side'] == 'LONG' else 'BUY'
            order_params = {
                'symbol': self.symbol,
                'side': side,
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'closePosition': True
            }
            
            if activation_price:
                order_params['activationPrice'] = activation_price
            
            trailing_order = self.client.futures_create_order(**order_params)
            
            self.position['trailing_stop'] = True
            self.position['callback_rate'] = callback_rate
            self.position['stop_order_id'] = trailing_order['orderId']
            print(f"[{self.symbol}] 📈 Trailing stop enabled ({callback_rate}%)")
            return True
            
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error enabling trailing stop: {e}")
            return False
    
    def load_model(self, model_path=None):
        """Load trained pattern detection model"""
        if model_path is None:
            from config import model_config
            model_path = model_config.MODEL_SAVE_PATH
        
        try:
            self.classifier = EnhancedForexPatternClassifier()
            self.classifier.load_model(model_path)
            self.model = self.classifier.model
            print(f"[{self.symbol}] ✅ Model loaded")
            return True
        except Exception as e:
            print(f"[{self.symbol}] ❌ Failed to load model: {e}")
            return False
    
    def get_tiered_risk_percentage(self, current_capital):
        """Calculate risk percentage based on tiered compounding strategy"""
        if not self.use_tiered_risk:
            return self.risk_per_trade
        
        initial_capital = self.shared_state.get('initial_capital', 10000.0)
        capital_ratio = current_capital / initial_capital
        
        for tier in self.risk_tiers:
            if capital_ratio < tier['max_capital_ratio']:
                return tier['risk']
        
        return self.risk_tiers[-1]['risk']
    
    def get_symbol_info(self):
        """Get trading rules for the symbol"""
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] == self.symbol:
                    self.symbol_info = symbol_data
                    
                    for f in symbol_data['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            self.lot_size_filter = f
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.min_notional = float(f['minNotional'])
                    
                    return True
            return False
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error getting symbol info: {e}")
            return False
    
    def load_historical_candles(self):
        """Load historical 1min candles and build initial buffers"""
        print(f"[{self.symbol}] 📥 Loading historical candles...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                klines = self.client.get_klines(symbol=self.symbol, interval='1m', limit=500)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"[{self.symbol}] ❌ Failed to load candles: {e}")
                    return
        
        try:
            
            # Build 1min candles
            for k in klines[:-1]:
                candle = {
                    'time': datetime.fromtimestamp(k[0] / 1000),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                }
                self.candles['1min'].append(candle)
            
            # Build 5s and 15s from 1min (approximated)
            for candle_1m in list(self.candles['1min']):
                # 1min = 12x 5s
                for j in range(12):
                    candle_5s = {
                        'time': candle_1m['time'] + timedelta(seconds=j*5),
                        'open': candle_1m['open'],
                        'high': candle_1m['high'],
                        'low': candle_1m['low'],
                        'close': candle_1m['close'],
                        'volume': candle_1m['volume'] / 12
                    }
                    self.candles['5s'].append(candle_5s)
                
                # 1min = 4x 15s
                for j in range(4):
                    candle_15s = {
                        'time': candle_1m['time'] + timedelta(seconds=j*15),
                        'open': candle_1m['open'],
                        'high': candle_1m['high'],
                        'low': candle_1m['low'],
                        'close': candle_1m['close'],
                        'volume': candle_1m['volume'] / 4
                    }
                    self.candles['15s'].append(candle_15s)
            
            print(f"[{self.symbol}] ✅ Loaded {len(self.candles['1min'])} candles")
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error processing candles: {e}")
    
    def start_websocket(self):
        """Start WebSocket for real-time TICK data"""
        # WebSocket will be started globally, not per symbol
        pass
    
    def stop_websocket(self):
        """Stop WebSocket streams - not used in multi-stream mode"""
        pass
    
    def _start_tick_processor(self):
        """Start background tick processor thread"""
        if self.tick_processor_thread and self.tick_processor_thread.is_alive():
            return
        self.tick_processor_stop = False
        self.tick_processor_thread = Thread(target=self._tick_processor_loop, daemon=True)
        self.tick_processor_thread.start()

    def _tick_processor_loop(self):
        """Process ticks from queue"""
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
                print(f"[{self.symbol}] ⚠️  Tick processing error: {e}")
            finally:
                self.tick_queue.task_done()

    def _process_tick(self, tick):
        """Process a single tick"""
        trade_time_ms, price_raw, qty_raw = tick
        tick_data = {
            'time': datetime.fromtimestamp(trade_time_ms / 1000),
            'price': float(price_raw),
            'quantity': float(qty_raw)
        }

        with self.tick_lock:
            self.tick_buffer.append(tick_data)

        # Update all timeframes
        for tf_name, tf_seconds in TIMEFRAMES.items():
            self._update_candle(tick_data, tf_name, tf_seconds)

    def _handle_tick(self, msg):
        """Handle WebSocket tick messages"""
        if isinstance(msg, dict) and msg.get('e') == 'error':
            return
        
        if msg.get('e') == 'trade':
            tick = (msg['T'], msg['p'], msg['q'])

            if not self.tick_processor_thread or not self.tick_processor_thread.is_alive():
                self._start_tick_processor()

            try:
                self.tick_queue.put_nowait(tick)
            except Full:
                try:
                    self.tick_queue.get_nowait()
                    self.tick_queue.task_done()
                except Empty:
                    pass
                try:
                    self.tick_queue.put_nowait(tick)
                except Full:
                    pass
    
    def _update_candle(self, tick, timeframe, seconds):
        """Update or complete candle for a specific timeframe"""
        with self.candle_locks[timeframe]:
            current_candle = self.current_candle[timeframe]
            candle_deque = self.candles[timeframe]
            
            tick_time = tick['time']
            candle_start = datetime(
                tick_time.year, tick_time.month, tick_time.day,
                tick_time.hour, tick_time.minute,
                (tick_time.second // seconds) * seconds
            )
            
            if current_candle is None or current_candle['time'] != candle_start:
                if current_candle is not None:
                    candle_deque.append(current_candle)
                    
                    if len(candle_deque) >= 100:
                        self._check_pattern_on_timeframe(timeframe, candle_deque)
                
                current_candle = {
                    'time': candle_start,
                    'open': tick['price'],
                    'high': tick['price'],
                    'low': tick['price'],
                    'close': tick['price'],
                    'volume': tick['quantity']
                }
            else:
                current_candle['high'] = max(current_candle['high'], tick['price'])
                current_candle['low'] = min(current_candle['low'], tick['price'])
                current_candle['close'] = tick['price']
                current_candle['volume'] += tick['quantity']
            
            self.current_candle[timeframe] = current_candle
    
    def _check_pattern_on_timeframe(self, timeframe, candles):
        """Check for patterns on a specific timeframe"""
        if self.model is None or self.classifier is None:
            return
        
        if len(candles) < 100:
            return
        
        try:
            df = pd.DataFrame(list(candles))
            df.set_index('time', inplace=True)
            df_subset = df.tail(100)
            
            predictions, probabilities = self.classifier.predict(df_subset)
            
            last_pattern = predictions[-1]
            last_prob = probabilities[-1].max()
            
            if last_pattern != 'no_pattern' and last_prob >= 0.7:
                scorer = PatternStrengthScorer()
                strength = scorer.calculate_pattern_strength(
                    df_subset, 
                    last_pattern, 
                    len(df_subset) - 1,
                    window=50
                )
                
                if strength >= 0.7:
                    self._process_pattern_signal(last_pattern, last_prob, strength, timeframe)
        
        except Exception as e:
            pass
    
    def _process_pattern_signal(self, pattern, probability, strength, timeframe):
        """Process detected pattern signal"""
        print(f"\n[{self.symbol}] 🎯 PATTERN on {timeframe}: {pattern} ({probability:.1%}, {strength:.1%})")
        self._execute_pattern_trade(pattern, probability, strength, timeframe)
    
    def _execute_pattern_trade(self, pattern, probability, strength, timeframe):
        """Execute trade based on pattern"""
        try:
            candles = self.candles[timeframe]
            if not candles or len(candles) == 0:
                return
            
            current_price = candles[-1]['close']
            
            stop_loss, take_profit, direction = self._calculate_targets(pattern, current_price, candles)
            
            if direction == 'skip':
                return
            
            # Check max concurrent trades (shared across all symbols)
            total_active = self.shared_state.get('total_active_trades', 0)
            if total_active >= trading_config.MAX_CONCURRENT_TRADES:
                print(f"[{self.symbol}] ⏸️  Max total trades reached ({trading_config.MAX_CONCURRENT_TRADES})")
                return
            
            # Check if this symbol has reached its max trades
            symbol_active = len(self.active_trades)
            max_per_symbol = getattr(trading_config, 'MAX_TRADES_PER_SYMBOL', 1)
            if max_per_symbol and symbol_active >= max_per_symbol:
                print(f"[{self.symbol}] ⏸️  Max trades per symbol reached ({max_per_symbol})")
                return
            
            # Get shared balance
            current_balance = self.shared_state.get('balance', 0.0)
            if current_balance == 0:
                return
            
            # Calculate position
            active_position_value = sum(t['quantity'] * current_price for t in self.active_trades)
            total_capital = current_balance + active_position_value
            
            current_risk_pct = self.get_tiered_risk_percentage(total_capital)
            risk_amount = total_capital * current_risk_pct
            
            if direction == 'long':
                risk_per_unit = current_price - stop_loss
            else:
                risk_per_unit = stop_loss - current_price
            
            if risk_per_unit <= 0:
                return
            
            quantity = risk_amount / risk_per_unit
            
            # Cap at 5% of total capital
            max_position_value = total_capital * 0.05
            position_value = quantity * current_price
            
            if position_value > max_position_value:
                quantity = max_position_value / current_price
                position_value = quantity * current_price
            
            # Round quantity to LOT_SIZE
            if self.lot_size_filter:
                step_size = float(self.lot_size_filter['stepSize'])
                min_qty = float(self.lot_size_filter['minQty'])
                max_qty = float(self.lot_size_filter['maxQty'])
                
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = round(quantity - (quantity % step_size), precision)
                
                if quantity < min_qty or quantity > max_qty:
                    return
            else:
                quantity = round(quantity, 5)
            
            # Check MIN_NOTIONAL (add safety margin)
            if self.min_notional:
                notional_value = quantity * current_price
                # Add 10% safety margin to MIN_NOTIONAL
                min_required = self.min_notional * 1.1
                if notional_value < min_required:
                    print(f"[{self.symbol}] ⚠️  Trade too small: ${notional_value:.2f} < ${min_required:.2f} (MIN_NOTIONAL)")
                    return
            else:
                # Fallback: ensure at least $10 USDT trade value
                notional_value = quantity * current_price
                if notional_value < 10.0:
                    print(f"[{self.symbol}] ⚠️  Trade too small: ${notional_value:.2f} < $10.00")
                    return
            
            if quantity <= 0:
                return
            
            final_position_value = quantity * current_price
            
            print(f"\n[{self.symbol}] 📊 TRADE SETUP:")
            print(f"   {direction.upper()} @ {current_price:.2f} USDT")
            print(f"   SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            print(f"   Qty: {quantity}, Value: ${final_position_value:.2f}")
            
            # Execute order
            if direction == 'long':
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                print(f"[{self.symbol}] ✅ BUY executed: {order['orderId']}")
                
                # Log BUY trade
                self._log_trade(
                    action='BUY',
                    direction=direction,
                    price=current_price,
                    quantity=quantity,
                    value=current_price * quantity,
                    pattern=pattern,
                    timeframe=timeframe,
                    probability=probability,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=str(order['orderId']),
                    reason='Pattern entry'
                )
            else:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                print(f"[{self.symbol}] ✅ SELL executed: {order['orderId']}")
                
                # Log SELL trade
                self._log_trade(
                    action='SELL',
                    direction=direction,
                    price=current_price,
                    quantity=quantity,
                    value=current_price * quantity,
                    pattern=pattern,
                    timeframe=timeframe,
                    probability=probability,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_id=str(order['orderId']),
                    reason='Pattern entry'
                )
            
            # Store trade
            trade = {
                'order_id': order['orderId'],
                'symbol': self.symbol,
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
            
            # Update shared state
            self.shared_state['total_active_trades'] = total_active + 1
            self.shared_state['balance'] = current_balance - final_position_value
            
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error executing trade: {e}")
    
    def _calculate_targets(self, pattern, entry_price, candles):
        """Calculate stop loss and take profit targets"""
        base_pattern = pattern.split('_')[0] if '_' in pattern else pattern
        
        targets = trading_config.PATTERN_TARGETS.get(
            pattern, 
            {'sl_pct': 0.015, 'tp_pct': 0.03}
        )
        
        trend_config = trading_config.TREND_ALIGNMENT
        if len(candles) >= trend_config['lookback_period']:
            closes = np.array([c['close'] for c in list(candles)[-trend_config['lookback_period']:]])
            
            # EMA filter
            if trend_config['use_ema_filter']:
                ema_period = trend_config['ema_period']
                ema50 = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().iloc[-1]
                current_price = closes[-1]
                
                if current_price < ema50:
                    return 0, 0, 'skip'
            
            # ATR filter
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
                
                if atr_pct < volatility_config['min_atr_pct']:
                    return 0, 0, 'skip'
            
            # Pattern classification
            bullish_patterns = [p.split('_')[0] for p in trend_config['bullish_patterns']]
            
            # Blacklist check
            blacklist = trading_config.PATTERN_FILTERS.get('blacklist_patterns', [])
            if pattern in blacklist:
                return 0, 0, 'skip'
            
            direction = 'long'
        else:
            return 0, 0, 'skip'
        
        stop_loss = entry_price * (1 - targets['sl_pct'])
        take_profit = entry_price * (1 + targets['tp_pct'])
        
        return stop_loss, take_profit, direction
    
    def monitor_trades(self):
        """Monitor active trades and dynamically adjust stops"""
        while True:
            try:
                if not self.active_trades:
                    time.sleep(5)
                    continue
                
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker['price'])
                
                trades_to_close = []
                
                for i, trade in enumerate(self.active_trades):
                    if trade['status'] != 'open':
                        continue
                    
                    entry_price = trade['entry_price']
                    
                    # Calculate profit percentage
                    if trade['direction'] == 'long':
                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                        
                        # Dynamic stop loss adjustment
                        if not trade.get('trailing_enabled', False):
                            # Move to breakeven after 2% profit
                            if profit_pct > 2.0 and trade['stop_loss'] < entry_price:
                                new_stop = entry_price * 1.001
                                if self.adjust_stop_loss(new_stop):
                                    trade['stop_loss'] = new_stop
                                    print(f"[{self.symbol}] 🛡️  Stop moved to breakeven (BE+0.1%)")
                            
                            # Enable trailing stop after 5% profit
                            elif profit_pct > 5.0:
                                if self.enable_trailing_stop(callback_rate=1.5):
                                    trade['trailing_enabled'] = True
                                    print(f"[{self.symbol}] 📈 Trailing stop activated (1.5%)")
                        
                        # Check exit conditions
                        if current_price <= trade['stop_loss']:
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'stop_loss'
                            trades_to_close.append(i)
                        elif current_price >= trade['take_profit']:
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'take_profit'
                            trades_to_close.append(i)
                    
                    else:  # SHORT
                        profit_pct = ((entry_price - current_price) / entry_price) * 100
                        
                        # Dynamic stop loss adjustment
                        if not trade.get('trailing_enabled', False):
                            # Move to breakeven after 2% profit
                            if profit_pct > 2.0 and trade['stop_loss'] > entry_price:
                                new_stop = entry_price * 0.999
                                if self.adjust_stop_loss(new_stop):
                                    trade['stop_loss'] = new_stop
                                    print(f"[{self.symbol}] 🛡️  Stop moved to breakeven (BE-0.1%)")
                            
                            # Enable trailing stop after 5% profit
                            elif profit_pct > 5.0:
                                if self.enable_trailing_stop(callback_rate=1.5):
                                    trade['trailing_enabled'] = True
                                    print(f"[{self.symbol}] 📈 Trailing stop activated (1.5%)")
                        
                        # Check exit conditions
                        if current_price >= trade['stop_loss']:
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'stop_loss'
                            trades_to_close.append(i)
                        elif current_price <= trade['take_profit']:
                            trade['exit_price'] = current_price
                            trade['exit_reason'] = 'take_profit'
                            trades_to_close.append(i)
                
                for idx in reversed(trades_to_close):
                    trade = self.active_trades[idx]
                    self._close_trade(trade)
                    self.active_trades.pop(idx)
                
                time.sleep(1)
                
            except Exception as e:
                time.sleep(5)
    
    def _close_trade(self, trade):
        """Close a trade"""
        try:
            if trade['direction'] == 'long':
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=trade['quantity']
                )
                
                # Calculate PnL
                entry_price = float(trade.get('entry_price', 0.0))
                exit_price = float(trade.get('exit_price', entry_price))
                qty = float(trade.get('quantity', 0.0))
                pnl = (exit_price - entry_price) * qty
                
                # Log SELL (close long)
                self._log_trade(
                    action='SELL',
                    direction='CLOSE_LONG',
                    price=trade.get('exit_price', 0),
                    quantity=trade['quantity'],
                    value=trade.get('exit_price', 0) * trade['quantity'],
                    pattern=trade.get('pattern', ''),
                    timeframe=trade.get('timeframe', ''),
                    probability=trade.get('probability', 0),
                    strength=trade.get('strength', 0),
                    stop_loss=trade.get('stop_loss', 0),
                    take_profit=trade.get('take_profit', 0),
                    order_id=str(order['orderId']),
                    reason=trade.get('exit_reason', 'CLOSE'),
                    pnl_usdt=round(pnl, 2)
                )
            else:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=trade['quantity']
                )
                
                # Calculate PnL
                entry_price = float(trade.get('entry_price', 0.0))
                exit_price = float(trade.get('exit_price', entry_price))
                qty = float(trade.get('quantity', 0.0))
                pnl = (entry_price - exit_price) * qty
                
                # Log BUY (close short)
                self._log_trade(
                    action='BUY',
                    direction='CLOSE_SHORT',
                    price=trade.get('exit_price', 0),
                    quantity=trade['quantity'],
                    value=trade.get('exit_price', 0) * trade['quantity'],
                    pattern=trade.get('pattern', ''),
                    timeframe=trade.get('timeframe', ''),
                    probability=trade.get('probability', 0),
                    strength=trade.get('strength', 0),
                    stop_loss=trade.get('stop_loss', 0),
                    take_profit=trade.get('take_profit', 0),
                    order_id=str(order['orderId']),
                    reason=trade.get('exit_reason', 'CLOSE'),
                    pnl_usdt=round(pnl, 2)
                )
            
            trade['exit_time'] = datetime.now()
            trade['exit_order_id'] = order['orderId']
            trade['status'] = 'closed'
            
            entry = float(trade.get('entry_price', 0.0))
            exit_price = float(trade.get('exit_price', entry))
            qty = float(trade.get('quantity', 0.0))

            if trade['direction'] == 'long':
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

            trade['pnl'] = round(pnl, 2)
            
            self.trades_history.append(trade)
            
            print(f"\n[{self.symbol}] 🔔 CLOSED - {trade['exit_reason'].upper()}")
            print(f"   P&L: {trade['pnl']:+.2f} USDT")
            
            # Update shared state
            current_total = self.shared_state.get('total_active_trades', 0)
            self.shared_state['total_active_trades'] = max(0, current_total - 1)
            
            current_balance = self.shared_state.get('balance', 0.0)
            position_value = qty * exit_price
            balance_before = current_balance
            balance_after = current_balance + position_value + pnl
            # Update shared balance
            self.shared_state['balance'] = balance_after
            
            total_pnl = self.shared_state.get('total_pnl', 0.0)
            self.shared_state['total_pnl'] = total_pnl + pnl

            # If the account balance becomes negative after this close, log details for analysis
            if balance_after < 0:
                print(f"[{self.symbol}] ⚠️ ACCOUNT NEGATIVE AFTER TRADE: {balance_after:.2f} USDT - logging event")
                try:
                    self._log_negative_balance_event(trade, balance_before, balance_after)
                except Exception as e:
                    logging.error(f"Failed to write negative-balance event: {e}")
            
        except Exception as e:
            print(f"[{self.symbol}] ❌ Error closing trade: {e}")
    
    def run(self):
        """Run the trader for this symbol"""
        print(f"[{self.symbol}] 🚀 Starting trader process...")
        
        self.get_symbol_info()
        self.load_historical_candles()
        
        # Start tick processor thread
        self._start_tick_processor()
        
        # Start trade monitoring thread
        monitor_thread = Thread(target=self.monitor_trades, daemon=True)
        monitor_thread.start()
        
        print(f"[{self.symbol}] ✅ Ready, waiting for ticks from WebSocket...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.tick_processor_stop = True


def run_symbol_trader(symbol, api_key, api_secret, shared_state, model_path, tick_queues):
    """Run trader for a single symbol (called in separate process)"""
    try:
        trader = MultiSymbolTrader(api_key, api_secret, symbol, shared_state)
        
        # Set the tick queue for this symbol
        trader.tick_queue = tick_queues[symbol]
        
        if not trader.load_model(model_path):
            print(f"[{symbol}] ❌ Model load failed")
            return
        
        trader.run()
        
    except Exception as e:
        print(f"[{symbol}] ❌ Process error: {e}")


def print_global_status(shared_state):
    """Print global trading status across all symbols"""
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"📊 MULTI-SYMBOL STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Active Symbols: {len(TRADING_SYMBOLS)}")
            print(f"Timeframes: {list(TIMEFRAMES.keys())}")
            print(f"Total Balance: ${shared_state.get('balance', 0.0):.2f} USDT")
            print(f"Active Trades: {shared_state.get('total_active_trades', 0)}")
            print(f"Total P&L: {shared_state.get('total_pnl', 0.0):+.2f} USDT")
            print(f"{'='*60}\n")
            
            time.sleep(30)
            
        except Exception as e:
            print(f"❌ Status error: {e}")
            time.sleep(30)


def websocket_distributor(symbols, tick_queues):
    """Single WebSocket connection distributing ticks with manual reconnect logic"""
    streams = [f"{symbol.lower()}@trade" for symbol in symbols]
    stream_path = '/'.join(streams)
    ws_url = f"wss://stream.binance.com:9443/stream?streams={stream_path}"

    reconnect_event = Event()
    stop_event = Event()
    max_backoff = 60
    current_ws = {'app': None}

    def _dispatch_tick(payload):
        try:
            stream_name = payload.get('stream')
            data = payload.get('data', {})
            if not stream_name or data.get('e') != 'trade':
                return

            symbol = stream_name.split('@')[0].upper()
            if symbol not in tick_queues:
                return

            tick = (data['T'], data['p'], data['q'])

            try:
                tick_queues[symbol].put_nowait(tick)
            except Full:
                try:
                    tick_queues[symbol].get_nowait()
                    tick_queues[symbol].task_done()
                    tick_queues[symbol].put_nowait(tick)
                except Empty:
                    pass
        except Exception as exc:
            print(f"⚠️  Tick dispatch error: {exc}")

    def on_message(ws, message):
        try:
            payload = json.loads(message)
            _dispatch_tick(payload)
        except json.JSONDecodeError as exc:
            print(f"⚠️  Invalid JSON from WebSocket: {exc}")

    def on_error(ws, error):
        print(f"⚠️  WebSocket error: {error}")
        reconnect_event.set()

    def on_close(ws, close_status_code, close_msg):
        print(f"⚠️  WebSocket closed (code={close_status_code}, msg={close_msg})")
        reconnect_event.set()

    def on_open(ws):
        reconnect_event.clear()
        print(f"✅ Multi-stream WebSocket connected ({len(symbols)} symbols) -> {ws_url}")

    def run_ws_loop():
        backoff = 1
        while not stop_event.is_set():
            if reconnect_event.is_set():
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)
            else:
                backoff = 1

            reconnect_event.clear()

            try:
                ws_app = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                current_ws['app'] = ws_app
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
                current_ws['app'] = None
                # If run_forever exits without stop_event, trigger reconnect
                if not stop_event.is_set():
                    reconnect_event.set()
            except Exception as exc:
                print(f"❌ WebSocket run loop error: {exc}")
                reconnect_event.set()

        # ensure connection is closed when stopping
        if current_ws['app'] is not None:
            try:
                current_ws['app'].close()
            except Exception:
                pass

    thread = Thread(target=run_ws_loop, daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        reconnect_event.set()
        if current_ws['app'] is not None:
            try:
                current_ws['app'].close()
            except Exception:
                pass

    return {
        'stop': stop,
        'thread': thread,
        'url': ws_url
    }


def main():
    """Main function to run multi-symbol trading"""
    
    api_key, api_secret = api_config.get_api_credentials()
    from config import model_config
    model_path = model_config.MODEL_SAVE_PATH
    
    print(f"\n{'='*60}")
    print("🚀 MULTI-SYMBOL MULTI-PROCESSOR LIVE TRADING")
    print(f"{'='*60}")
    
    # Get actual balance from Binance
    print("📥 Fetching balance from Binance...")
    try:
        client = Client(api_key, api_secret, demo=True)
        account = client.get_account()
        
        # Find USDT balance
        usdt_balance = 0.0
        for asset in account['balances']:
            if asset['asset'] == 'USDT':
                usdt_balance = float(asset['free'])
                break
        
        if usdt_balance > 0:
            starting_balance = usdt_balance
            print(f"✅ Binance USDT Balance: ${starting_balance:.2f}")
        else:
            starting_balance = 10000.0
            print(f"⚠️  No USDT found, using default: ${starting_balance:.2f}")
    except Exception as e:
        starting_balance = 10000.0
        print(f"⚠️  Could not fetch balance: {e}")
        print(f"   Using default: ${starting_balance:.2f}")
    
    print(f"\nSymbols: {len(TRADING_SYMBOLS)}")
    for symbol in TRADING_SYMBOLS:
        print(f"  • {symbol}")
    print(f"Timeframes: {list(TIMEFRAMES.keys())}")
    print(f"Max concurrent trades: {trading_config.MAX_CONCURRENT_TRADES}")
    print(f"CPUs available: {cpu_count()}")
    print(f"{'='*60}\n")
    
    # Create shared state using Manager
    manager = Manager()
    shared_state = manager.dict()
    
    # Initialize shared state with actual Binance balance
    shared_state['balance'] = starting_balance
    shared_state['initial_capital'] = starting_balance
    shared_state['total_active_trades'] = 0
    shared_state['total_pnl'] = 0.0
    
    # Create shared tick queues for all symbols
    tick_queues = {symbol: manager.Queue(maxsize=5000) for symbol in TRADING_SYMBOLS}
    
    # Start single WebSocket connection for all symbols
    ws_controller = websocket_distributor(TRADING_SYMBOLS, tick_queues)
    
    # Start status thread
    status_thread = Thread(target=print_global_status, args=(shared_state,), daemon=True)
    status_thread.start()
    
    # Create process for each symbol
    processes = []
    
    print(f"🚀 Starting {len(TRADING_SYMBOLS)} parallel processes...")
    
    # Start all processes in parallel (no delay)
    for symbol in TRADING_SYMBOLS:
        p = Process(
            target=run_symbol_trader,
            args=(symbol, api_key, api_secret, shared_state, model_path, tick_queues)
        )
        p.start()
        processes.append(p)
        print(f"   ✅ {symbol} process started")
    
    print(f"\n✅ All {len(processes)} processes started in parallel\n")
    print(f"💡 Each symbol runs on separate CPU core")
    print(f"💡 Single WebSocket serves all {len(TRADING_SYMBOLS)} symbols\n")
    
    try:
        # Keep main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping all traders...")
        
        # Stop WebSocket
        if ws_controller:
            ws_controller['stop']()
            print("✅ WebSocket stopped")
        
        for p in processes:
            p.terminate()
            p.join(timeout=5)
        
        print("✅ All processes stopped")
        
        print(f"\n{'='*60}")
        print("📈 FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Final Balance: ${shared_state.get('balance', 0.0):.2f} USDT")
        print(f"Total P&L: {shared_state.get('total_pnl', 0.0):+.2f} USDT")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
