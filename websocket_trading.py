"""
WebSocket Live Trading modul
Real-time tick adatok alapján történő kereskedés Binance WebSocket-en keresztül
Ugyanazt a trading logikát használja mint a backtest!
"""
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from threading import Thread, Lock
from queue import Queue, Empty, Full
import websocket
import warnings
warnings.filterwarnings('ignore')

# Binance client
from binance.client import Client

# Imports
import config
from trading_logic import TradingLogic


class WebSocketTrader:
    """
    WebSocket-alapú élő kereskedő
    """
    
    def __init__(self, coin, timeframes, api_key, api_secret, demo_mode=True):
        """
        Inicializálás
        
        Args:
            coin: str, pl. 'BTCUSDT'
            timeframes: list of str, pl. ['15s', '30s', '1min']
            api_key: Binance API kulcs
            api_secret: Binance API secret
            demo_mode: bool, testnet használata
        """
        self.coin = coin
        self.timeframes = timeframes
        
        # Binance client
        self.client = Client(api_key, api_secret, testnet=demo_mode)
        
        # WebSocket
        self.ws_url = config.BINANCE_WS
        self.ws = None
        self.ws_running = False
        
        # Tick buffer
        self.tick_buffer = deque(maxlen=10000)
        self.tick_lock = Lock()
        self.tick_queue = Queue(maxsize=5000)
        
        # Candle buffers per timeframe
        self.candles = {tf: deque(maxlen=2000) for tf in timeframes}
        self.current_candle = {tf: None for tf in timeframes}
        self.candle_locks = {tf: Lock() for tf in timeframes}
        
        # Trading logic
        self.trading = None
        
        # ML Model
        self.classifier = None
        self.scorer = None
        
        print(f"[{self.coin}] ✅ WebSocket Trader inicializálva")
    
    def load_model(self, model_path):
        """
        Betölti az ML modelt
        """
        try:
            from old.forex_pattern_classifier import EnhancedForexPatternClassifier, PatternStrengthScorer
            
            self.classifier = EnhancedForexPatternClassifier()
            self.classifier.load_model(str(model_path))
            
            self.scorer = PatternStrengthScorer()
            
            print(f"[{self.coin}] ✅ Model betöltve: {model_path}")
            return True
        except Exception as e:
            print(f"[{self.coin}] ❌ Model betöltési hiba: {e}")
            return False
    
    def initialize_trading_logic(self):
        """
        Inicializálja a trading logikát
        """
        # Get balance from Binance
        try:
            account = self.client.get_account()
            usdt_balance = 0.0
            
            for asset in account['balances']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['free'])
                    break
            
            if usdt_balance > 0:
                initial_capital = usdt_balance
                print(f"[{self.coin}] 💰 Binance USDT egyenleg: ${initial_capital:.2f}")
            else:
                initial_capital = config.BACKTEST_INITIAL_CAPITAL
                print(f"[{self.coin}] ⚠️  Nincs USDT, alapértelmezett: ${initial_capital:.2f}")
        
        except Exception as e:
            initial_capital = config.BACKTEST_INITIAL_CAPITAL
            print(f"[{self.coin}] ⚠️  Balance lekérési hiba: {e}")
            print(f"[{self.coin}]    Alapértelmezett: ${initial_capital:.2f}")
        
        self.trading = TradingLogic(config, initial_capital=initial_capital)
        print(f"[{self.coin}] ✅ Trading logic inicializálva (${initial_capital:.2f})")
    
    def load_historical_candles(self):
        """
        Betölti a kezdő historikus 1min candle-eket
        """
        try:
            print(f"[{self.coin}] 📥 Historikus candle-ek betöltése...")
            
            klines = self.client.get_klines(symbol=self.coin, interval='1m', limit=500)
            
            # Build 1min candles
            for k in klines[:-1]:  # Skip last (incomplete)
                candle = {
                    'time': datetime.fromtimestamp(k[0] / 1000),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                }
                self.candles['1min'].append(candle)
            
            # Approximate 15s and 30s from 1min
            for candle_1m in list(self.candles['1min']):
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
                
                # 1min = 2x 30s
                for j in range(2):
                    candle_30s = {
                        'time': candle_1m['time'] + timedelta(seconds=j*30),
                        'open': candle_1m['open'],
                        'high': candle_1m['high'],
                        'low': candle_1m['low'],
                        'close': candle_1m['close'],
                        'volume': candle_1m['volume'] / 2
                    }
                    self.candles['30s'].append(candle_30s)
            
            print(f"[{self.coin}] ✅ {len(self.candles['1min'])} candle betöltve")
            
        except Exception as e:
            print(f"[{self.coin}] ❌ Historikus candle hiba: {e}")
    
    def start_websocket(self):
        """
        Elindítja a WebSocket connection-t
        """
        stream = f"{self.coin.lower()}@trade"
        ws_url = f"{self.ws_url}/{stream}"
        
        def on_message(ws, message):
            self._handle_tick(json.loads(message))
        
        def on_error(ws, error):
            print(f"[{self.coin}] ⚠️  WebSocket hiba: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"[{self.coin}] 🔌 WebSocket bezárva")
            self.ws_running = False
        
        def on_open(ws):
            print(f"[{self.coin}] ✅ WebSocket csatlakozva")
            self.ws_running = True
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket in separate thread
        ws_thread = Thread(target=self.ws.run_forever, daemon=True)
        ws_thread.start()
        
        # Start tick processor thread
        tick_thread = Thread(target=self._tick_processor_loop, daemon=True)
        tick_thread.start()
        
        print(f"[{self.coin}] 🚀 WebSocket elindítva")
    
    def _handle_tick(self, msg):
        """
        WebSocket tick üzenet feldolgozása
        """
        if msg.get('e') != 'trade':
            return
        
        tick = (msg['T'], msg['p'], msg['q'])  # (timestamp_ms, price, quantity)
        
        try:
            self.tick_queue.put_nowait(tick)
        except Full:
            # Queue full, drop oldest
            try:
                self.tick_queue.get_nowait()
                self.tick_queue.task_done()
                self.tick_queue.put_nowait(tick)
            except:
                pass
    
    def _tick_processor_loop(self):
        """
        Tick processor thread
        """
        while True:
            try:
                tick = self.tick_queue.get(timeout=1)
            except Empty:
                continue
            
            try:
                self._process_tick(tick)
            except Exception as e:
                print(f"[{self.coin}] ⚠️  Tick feldolgozási hiba: {e}")
            finally:
                self.tick_queue.task_done()
    
    def _process_tick(self, tick):
        """
        Feldolgoz egy tick-et és frissíti a candle-eket
        """
        trade_time_ms, price_raw, qty_raw = tick
        
        tick_data = {
            'time': datetime.fromtimestamp(trade_time_ms / 1000),
            'price': float(price_raw),
            'quantity': float(qty_raw)
        }
        
        with self.tick_lock:
            self.tick_buffer.append(tick_data)
        
        # Update all timeframes
        for tf_name in self.timeframes:
            # Get timeframe in seconds
            tf_seconds = {
                '15s': 15,
                '30s': 30,
                '1min': 60,
                '5min': 300
            }.get(tf_name, 60)
            
            self._update_candle(tick_data, tf_name, tf_seconds)
    
    def _update_candle(self, tick, timeframe, seconds):
        """
        Frissíti vagy lezárja a candle-t adott timeframe-en
        """
        with self.candle_locks[timeframe]:
            current = self.current_candle[timeframe]
            candle_deque = self.candles[timeframe]
            
            tick_time = tick['time']
            candle_start = datetime(
                tick_time.year, tick_time.month, tick_time.day,
                tick_time.hour, tick_time.minute,
                (tick_time.second // seconds) * seconds
            )
            
            if current is None or current['time'] != candle_start:
                # Új candle
                if current is not None:
                    # Előző candle lezárása
                    candle_deque.append(current)
                    
                    # Pattern check ha van elég adat
                    if len(candle_deque) >= 100:
                        self._check_pattern_on_timeframe(timeframe, candle_deque)
                
                # Új candle létrehozása
                current = {
                    'time': candle_start,
                    'open': tick['price'],
                    'high': tick['price'],
                    'low': tick['price'],
                    'close': tick['price'],
                    'volume': tick['quantity']
                }
            else:
                # Frissítsd a current candle-t
                current['high'] = max(current['high'], tick['price'])
                current['low'] = min(current['low'], tick['price'])
                current['close'] = tick['price']
                current['volume'] += tick['quantity']
            
            self.current_candle[timeframe] = current
    
    def _check_pattern_on_timeframe(self, timeframe, candles):
        """
        Ellenőrzi a pattern-eket adott timeframe-en
        """
        if self.classifier is None or self.scorer is None:
            return
        
        if len(candles) < 100:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(list(candles))
            df = df.set_index('time')
            df_subset = df.tail(100)
            
            # Predict
            predictions, probabilities = self.classifier.predict(df_subset)
            
            last_pattern = predictions[-1]
            last_prob = probabilities[-1].max()
            
            if last_pattern == 'no_pattern':
                return
            
            # Calculate strength
            strength = self.scorer.calculate_pattern_strength(
                df_subset,
                last_pattern,
                len(df_subset) - 1,
                window=50
            )
            
            # Check if should open trade
            if self.trading.should_open_trade(last_pattern, last_prob, strength):
                self._process_pattern_signal(last_pattern, last_prob, strength, timeframe, df_subset)
        
        except Exception as e:
            print(f"[{self.coin}] ⚠️  Pattern check hiba ({timeframe}): {e}")
    
    def _process_pattern_signal(self, pattern, probability, strength, timeframe, recent_data):
        """
        Feldolgozza a pattern signalt és megnyitja a trade-et
        """
        print(f"\n[{self.coin}] 🎯 PATTERN ({timeframe}): {pattern} (P={probability:.1%}, S={strength:.1%})")
        
        try:
            # Get current price
            candles = self.candles[timeframe]
            if not candles or len(candles) == 0:
                return
            
            current_price = candles[-1]['close']
            current_candle = candles[-1]
            
            # Calculate targets
            sl, tp, direction, params = self.trading.calculate_pattern_targets(
                pattern, current_price, current_candle, recent_data
            )
            
            if direction == 'skip':
                print(f"[{self.coin}]   ⏭️  Skipped (trend misalignment)")
                return
            
            # Calculate position size
            position_size = self.trading.calculate_position_size(
                current_price, sl, self.trading.capital
            )
            
            if position_size <= 0:
                print(f"[{self.coin}]   ⚠️  Pozíció méret = 0")
                return
            
            # Check minimum notional (Binance limit)
            position_value = position_size * current_price
            if position_value < 10.0:
                print(f"[{self.coin}]   ⚠️  Túl kicsi (${position_value:.2f} < $10)")
                return
            
            print(f"[{self.coin}]   📊 TRADE SETUP:")
            print(f"[{self.coin}]      {direction.upper()} @ ${current_price:.2f}")
            print(f"[{self.coin}]      SL: ${sl:.2f}, TP: ${tp:.2f}")
            print(f"[{self.coin}]      Qty: {position_size:.4f}, Value: ${position_value:.2f}")
            
            # Execute order (DEMO MODE - nincs valódi order!)
            if config.BINANCE_DEMO_MODE:
                print(f"[{self.coin}]   🎮 DEMO MODE - order nem kerül végrehajtásra")
                
                # Szimulált order nyitás
                trade = self.trading.open_trade(
                    coin=self.coin,
                    pattern=pattern,
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    position_size=position_size,
                    probability=probability,
                    strength=strength,
                    timeframe=timeframe
                )
                
                print(f"[{self.coin}]   ✅ DEMO trade megnyitva")
            else:
                # Real order execution (implementáld később!)
                print(f"[{self.coin}]   ⚠️  LIVE MODE - order execution not implemented yet!")
        
        except Exception as e:
            print(f"[{self.coin}] ❌ Trade execution hiba: {e}")
    
    def monitor_trades(self):
        """
        Monitor thread - ellenőrzi az aktív trade-eket
        """
        print(f"[{self.coin}] 👀 Trade monitoring elindítva")
        
        while True:
            try:
                if not self.trading.active_trades:
                    time.sleep(5)
                    continue
                
                # Get current price (from latest candle)
                current_candle = None
                for tf in self.timeframes:
                    candles = self.candles[tf]
                    if candles and len(candles) > 0:
                        current_candle = candles[-1]
                        break
                
                if current_candle is None:
                    time.sleep(1)
                    continue
                
                # Check all active trades
                for trade in list(self.trading.active_trades):
                    should_close, exit_price, exit_reason = self.trading.check_trade_exit(
                        trade, current_candle
                    )
                    
                    if should_close:
                        pnl = self.trading.close_trade(trade, exit_price, exit_reason)
                        
                        print(f"\n[{self.coin}] 🔔 TRADE CLOSED - {exit_reason.upper()}")
                        print(f"[{self.coin}]    P&L: {pnl:+.2f} USDT")
                        print(f"[{self.coin}]    Capital: ${self.trading.capital:.2f}")
                
                time.sleep(1)
            
            except Exception as e:
                print(f"[{self.coin}] ⚠️  Monitor hiba: {e}")
                time.sleep(5)
    
    def run(self):
        """
        Elindítja a WebSocket trader-t
        """
        print(f"\n[{self.coin}] 🚀 WebSocket Trading indítása...")
        
        # Initialize
        self.initialize_trading_logic()
        self.load_historical_candles()
        
        # Start WebSocket
        self.start_websocket()
        
        # Wait for WebSocket to connect
        time.sleep(2)
        
        if not self.ws_running:
            print(f"[{self.coin}] ❌ WebSocket nem csatlakozott!")
            return
        
        # Start monitoring thread
        monitor_thread = Thread(target=self.monitor_trades, daemon=True)
        monitor_thread.start()
        
        print(f"[{self.coin}] ✅ WebSocket Trading fut! Várakozás tick-ekre...")
        
        # Keep alive
        try:
            while True:
                time.sleep(10)
                
                # Print status
                stats = self.trading.get_statistics()
                print(f"\n[{self.coin}] 📊 Status:")
                print(f"   Capital: ${stats['final_capital']:.2f}")
                print(f"   Active trades: {len(self.trading.active_trades)}")
                print(f"   Total trades: {stats['total_trades']}")
                print(f"   Win rate: {stats['win_rate']*100:.1f}%")
                print(f"   P&L: {stats['total_pnl']:+.2f} USDT")
        
        except KeyboardInterrupt:
            print(f"\n[{self.coin}] ⏹️  Leállítás...")
            if self.ws:
                self.ws.close()


def run_websocket_trading(coins, timeframes, api_key, api_secret, demo_mode=True):
    """
    Több coin WebSocket trading (multithread)
    """
    print("\n" + "="*80)
    print("🚀 WEBSOCKET LIVE TRADING")
    print("="*80)
    
    print(f"\nCoinok: {len(coins)}")
    for coin in coins:
        print(f"  • {coin}")
    
    print(f"\nTimeframes: {timeframes}")
    print(f"Demo mode: {demo_mode}")
    
    # Create traders
    traders = []
    for coin in coins:
        trader = WebSocketTrader(coin, timeframes, api_key, api_secret, demo_mode)
        
        # Load model
        if not trader.load_model(config.MODEL_PATH):
            print(f"⚠️  {coin}: Model betöltési hiba, kihagyva")
            continue
        
        traders.append(trader)
    
    if not traders:
        print("❌ Nincs trader inicializálva!")
        return
    
    # Start all traders in separate threads
    threads = []
    for trader in traders:
        thread = Thread(target=trader.run, daemon=True)
        thread.start()
        threads.append(thread)
    
    print(f"\n✅ {len(threads)} trader elindítva!")
    
    # Keep alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n⏹️  Leállítás...")


if __name__ == '__main__':
    config.ensure_dirs()
    
    run_websocket_trading(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        demo_mode=config.BINANCE_DEMO_MODE
    )
