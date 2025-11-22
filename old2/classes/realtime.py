import threading
import time
import json
from websocket import create_connection
from multiprocessing import Process, Queue, Manager
import importlib
from classes.classification import Classifier
from pathlib import Path
import pandas as pd
import config


# Global Binance client and shared capital (initialized once for all workers)
_binance_client = None
_shared_portfolio = None

def _init_global_binance_client():
    """Initialize Binance client once and fetch initial capital"""
    global _binance_client, _shared_portfolio
    
    try:
        import sys
        if 'requests' in sys.modules:
            del sys.modules['requests']
        
        from binance.client import Client as BinanceClient
        
        _binance_client = BinanceClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, demo=config.BINANCE_DEMO_MODE)
        print(f'🔗 Binance Demo API connected', flush=True)
        
        # Get initial capital
        try:
            account = _binance_client.get_account()
            for asset in account['balances']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['free'])
                    if usdt_balance > 0:
                        initial_capital = usdt_balance
                        print(f'💰 Binance USDT Balance: ${initial_capital:.2f}', flush=True)
                        return initial_capital
        except Exception as e:
            print(f'⚠️  Could not fetch balance: {e}', flush=True)
    except Exception as e:
        print(f'⚠️  Failed to initialize Binance client: {e}', flush=True)
    
    return config.BACKTEST_INITIAL_CAPITAL


class RealtimeRunner:
    def __init__(self, config_obj, TradingLogicClass, logger):
        self.config = config_obj
        self.TradingLogicClass = TradingLogicClass
        self.logger = logger
        self.queues = {coin: Queue() for coin in config_obj.COINS}

    def _ws_worker(self, symbol, queue):
        # Binance trade stream
        ws_url = f"{config.BINANCE_WS}/{symbol.lower()}@trade"
        print(f'🌐 Opening websocket for {symbol}: {ws_url}', flush=True)
        
        while True:  # Auto-reconnect loop
            try:
                ws = create_connection(ws_url, timeout=10)
                print(f'✅ Websocket connected for {symbol}', flush=True)
                
                while True:
                    msg = ws.recv()
                    data = json.loads(msg)
                    # Extract trade data from Binance format
                    tick = {
                        'price': float(data.get('p', data.get('price', 0))),
                        'qty': float(data.get('q', data.get('qty', 0))),
                        'time': pd.to_datetime(data.get('T', time.time()*1000), unit='ms')
                    }
                    queue.put(tick)
            except KeyboardInterrupt:
                print(f'⏹️  Websocket stopped for {symbol}', flush=True)
                break
            except Exception as e:
                print(f'⚠️  Websocket error for {symbol}: {e}, reconnecting in 5s...', flush=True)
                time.sleep(5)

    def _worker_process(self, symbol, queue):
        # Instantiates trading logic for symbol
        # Keep this method for single-threaded debug, but in general spawn processes via top-level function
        logic = self.TradingLogicClass(None)
        # Simple aggregator to timeframe
        buffer = []
        while True:
            tick = queue.get()
            buffer.append(tick)
            # produce 15s
            now = pd.Timestamp.now()
            if len(buffer) >= 5:
                df = pd.DataFrame(buffer)
                df = df.set_index('time').resample('15S').agg({'price': ['first', 'max', 'min', 'last'], 'qty': 'sum'})
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                # send to logic
                signals = logic.on_tick(df.tail(50))
                # log decisions
                if signals:
                    for s in signals:
                        self.logger.log_trade(symbol, 'realtime', s)
                buffer = []

    def run(self):
        # Initialize Binance client and get shared capital ONCE before starting workers
        print('🔗 Initializing Binance Demo API...', flush=True)
        initial_capital = _init_global_binance_client()
        
        # Create shared portfolio state using Manager
        manager = Manager()
        shared_portfolio = manager.dict({
            'capital': initial_capital,
            'initial_capital': initial_capital,
            'active_trades': 0,
            'total_invested': 0.0,
            'total_pnl': 0.0,
            'closed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        })
        
        # Spawn websocket thread + processing for each coin
        threads = []
        processes = []
        for coin, q in self.queues.items():
            t = threading.Thread(target=self._ws_worker, args=(coin, q), daemon=True)
            # Pass shared portfolio to worker
            p = Process(target=_realtime_worker_process, args=(coin, q, self.config.TRADING_LOGIC, str(self.config.MODEL_PATH), shared_portfolio), daemon=True)
            t.start()
            p.start()
            threads.append(t)
            processes.append(p)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('Realtime stopped by user')


def _realtime_worker_process(symbol, queue, trading_logic_module_name, model_path, shared_portfolio):
    """Top-level worker for real-time processing to avoid pickling bound methods.

    It creates a `TradingLogic` instance inside the process to avoid passing module
    objects through multiprocessing queues.
    """
    import importlib
    from classes.classification import Classifier
    from pathlib import Path
    import pandas as pd
    import config

    print(f'🚀 Realtime worker started for {symbol}', flush=True)
    
    # Use shared portfolio state instead of local
    portfolio = shared_portfolio
    
    # Initialize Binance Demo client for API trading (in background to avoid blocking)
    client = None
    
    def init_binance_client():
        nonlocal client
        try:
            import sys
            # Temporarily remove requests from sys.modules to avoid circular import
            if 'requests' in sys.modules:
                del sys.modules['requests']
            
            from binance.client import Client as BinanceClient
            
            client = BinanceClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, demo=config.BINANCE_DEMO_MODE)
            print(f'🔗 [{symbol}] Binance Demo API connected', flush=True)
        except Exception as e:
            print(f'⚠️  [{symbol}] Failed to initialize Binance client: {e}', flush=True)
    
    # Start Binance client initialization in background
    from threading import Thread
    Thread(target=init_binance_client, daemon=True).start()
    
    clf = Classifier()
    try:
        print(f'🔧 Loading model for {symbol}...', flush=True)
        clf.load_model_if_exists(Path(model_path), force_gpu=False)  # CPU mode for workers
        print(f'✅ Model loaded for {symbol}', flush=True)
    except Exception as e:
        print(f'⚠️  Model load error for {symbol}: {e}', flush=True)

    try:
        module = importlib.import_module('trading_logics.' + trading_logic_module_name)
        TradingLogicClass = module.TradingLogic
    except Exception as e:
        print('Realtime worker import error for trading logic:', e)
        return

    logic = TradingLogicClass(clf)

    buffer = []
    tick_count = 0
    last_signal_time = pd.Timestamp.now()
    last_status_time = time.time()
    status_interval = 30  # seconds
    
    print(f'📊 Starting live tick aggregation for {symbol}...', flush=True)
    
    # Load historical data in background (non-blocking)
    def load_historical_async():
        try:
            import requests
            print(f'📥 [{symbol}] Loading historical data in background...', flush=True)
            
            url = 'https://api.binance.com/api/v3/klines'
            params = {'symbol': symbol, 'interval': '1m', 'limit': 500}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                
                for k in klines[:-1]:
                    tick_time = pd.to_datetime(k[0], unit='ms')
                    for price in [float(k[1]), float(k[2]), float(k[3]), float(k[4])]:
                        buffer.append({
                            'time': tick_time,
                            'price': price,
                            'qty': float(k[5]) / 4
                        })
                
                print(f'✅ [{symbol}] Loaded {len(klines)} candles -> {len(buffer)} ticks', flush=True)
        except Exception as e:
            print(f'⚠️  [{symbol}] Historical load error: {e}', flush=True)
    
    from threading import Thread
    Thread(target=load_historical_async, daemon=True).start()
    
    while True:
        tick = queue.get()
        buffer.append(tick)
        tick_count += 1
        
        # Periodic portfolio status update (every 30 seconds) - check on every tick
        current_time = time.time()
        if current_time - last_status_time >= status_interval:
            # Update capital from API
            if client:
                try:
                    account = client.get_account()
                    for asset in account['balances']:
                        if asset['asset'] == 'USDT':
                            portfolio['capital'] = float(asset['free'])
                            break
                except:
                    pass
            
            win_rate = (portfolio['winning_trades'] / portfolio['closed_trades'] * 100) if portfolio['closed_trades'] > 0 else 0
            total_return_pct = (portfolio['total_pnl'] / portfolio['initial_capital'] * 100) if portfolio['initial_capital'] > 0 else 0
            print(f'\n📊 [{symbol}] Portfolio Update:', flush=True)
            print(f'   Capital: ${portfolio["capital"]:.2f}', flush=True)
            print(f'   Active Trades: {portfolio["active_trades"]}', flush=True)
            print(f'   Invested: ${portfolio["total_invested"]:.2f}', flush=True)
            print(f'   Total P&L: ${portfolio["total_pnl"]:.2f} ({total_return_pct:+.2f}%)', flush=True)
            print(f'   Closed Trades: {portfolio["closed_trades"]}', flush=True)
            print(f'   Win Rate: {win_rate:.1f}%\n', flush=True)
            last_status_time = current_time
        
        # Aggregate every 30 ticks or when buffer is large enough
        if len(buffer) >= 30:
            try:
                df = pd.DataFrame(buffer)
                if df.empty:
                    buffer = []
                    continue
                    
                # Set time index and resample to 15s OHLCV
                df = df.set_index('time').sort_index()
                ohlc = df['price'].resample('15S').ohlc()
                vol = df['qty'].resample('15S').sum().rename('volume')
                df_agg = pd.concat([ohlc, vol], axis=1).dropna()
                
                if len(df_agg) < 50:  # Need minimum bars for pattern detection
                    buffer = buffer[-100:]  # Keep recent ticks
                    continue
                
                # Detect patterns in live data (NO backtest simulation)
                try:
                    recent_window = df_agg.tail(200)  # Use recent 200 bars
                    signals = logic.detect_live_patterns(symbol, '15s', recent_window)
                    
                    if signals:
                        # Filter for new signals only
                        new_trades = [s for s in signals if pd.to_datetime(s.get('entry_time', 0)) > last_signal_time]
                        
                        if new_trades:
                            from utils.trade_logger import TradeLogger
                            logger = TradeLogger(csv_path=f'live_trades_{symbol}.csv')
                            
                            for trade in new_trades:
                                pattern = trade.get('pattern', 'unknown')
                                entry_price = trade.get('entry_price', 0)
                                exit_price = trade.get('exit_price', 0)
                                pnl = trade.get('pnl', 0)
                                direction = trade.get('direction', 'long')
                                exit_reason = trade.get('exit_reason', 'open')
                                position_size = trade.get('position_size', 0)
                                invested_amount = position_size * entry_price if entry_price > 0 else 0
                                
                                # Update portfolio state
                                if exit_reason == 'open':
                                    # New trade opened
                                    portfolio['active_trades'] += 1
                                    portfolio['total_invested'] += invested_amount
                                elif exit_price > 0:
                                    # Trade closed
                                    portfolio['active_trades'] = max(0, portfolio['active_trades'] - 1)
                                    portfolio['total_invested'] = max(0, portfolio['total_invested'] - invested_amount)
                                    portfolio['capital'] += pnl
                                    portfolio['total_pnl'] += pnl
                                    portfolio['closed_trades'] += 1
                                    if pnl > 0:
                                        portfolio['winning_trades'] += 1
                                    else:
                                        portfolio['losing_trades'] += 1
                                
                                # Log pattern detection
                                if exit_reason == 'open' or 'entry' in str(trade.get('entry_time', '')):
                                    print(f'🔍 {symbol} PATTERN DETECTED: {pattern.upper()} | Entry: ${entry_price:.2f} | Direction: {direction.upper()}', flush=True)
                                
                                # Execute trade via Binance API
                                if exit_reason == 'open' and client:
                                    # Place BUY order via API
                                    try:
                                        # Calculate quantity based on position size
                                        quantity = position_size
                                        
                                        # Round to appropriate precision
                                        if symbol == 'BTCUSDT':
                                            quantity = round(quantity, 5)
                                        else:
                                            quantity = round(quantity, 3)
                                        
                                        # Place market order
                                        order = client.create_order(
                                            symbol=symbol,
                                            side='BUY',
                                            type='MARKET',
                                            quantity=quantity
                                        )
                                        
                                        order_id = order.get('orderId', 'N/A')
                                        fills = order.get('fills', [])
                                        avg_price = sum(float(f['price']) * float(f['qty']) for f in fills) / sum(float(f['qty']) for f in fills) if fills else entry_price
                                        
                                        print(f'\n📈 {symbol} BUY ORDER EXECUTED: {pattern} @ ${avg_price:.2f}', flush=True)
                                        print(f'   Order ID: {order_id}', flush=True)
                                        print(f'   Quantity: {quantity}', flush=True)
                                        print(f'   SL: ${trade.get("stop_loss", 0):.2f} | TP: ${trade.get("take_profit", 0):.2f}', flush=True)
                                        print(f'   Size: ${invested_amount:.2f}', flush=True)
                                        
                                        # Update portfolio from API
                                        try:
                                            account = client.get_account()
                                            for asset in account['balances']:
                                                if asset['asset'] == 'USDT':
                                                    portfolio['capital'] = float(asset['free'])
                                                    break
                                        except:
                                            pass
                                        
                                        print(f'💰 Portfolio [{symbol}]:', flush=True)
                                        print(f'   Capital: ${portfolio["capital"]:.2f}', flush=True)
                                        print(f'   Active Trades: {portfolio["active_trades"]}', flush=True)
                                        print(f'   Invested: ${portfolio["total_invested"]:.2f}', flush=True)
                                        print(f'   Total P&L: ${portfolio["total_pnl"]:.2f}\n', flush=True)
                                        
                                    except Exception as e:
                                        print(f'⚠️  [{symbol}] Failed to execute BUY order: {e}', flush=True)
                                
                                elif exit_reason == 'open':
                                    # Simulation mode (no API)
                                    print(f'\n📈 {symbol} BUY (SIM): {pattern} @ ${entry_price:.2f}', flush=True)
                                    print(f'   SL: ${trade.get("stop_loss", 0):.2f} | TP: ${trade.get("take_profit", 0):.2f}', flush=True)
                                    print(f'   Size: ${invested_amount:.2f}', flush=True)
                                    print(f'💰 Portfolio [{symbol}]:', flush=True)
                                    print(f'   Capital: ${portfolio["capital"]:.2f}', flush=True)
                                    print(f'   Active Trades: {portfolio["active_trades"]}', flush=True)
                                    print(f'   Invested: ${portfolio["total_invested"]:.2f}', flush=True)
                                    print(f'   Total P&L: ${portfolio["total_pnl"]:.2f}\n', flush=True)
                                
                                # Execute SELL via API
                                elif exit_price > 0 and client:
                                    try:
                                        quantity = position_size
                                        if symbol == 'BTCUSDT':
                                            quantity = round(quantity, 5)
                                        else:
                                            quantity = round(quantity, 3)
                                        
                                        order = client.create_order(
                                            symbol=symbol,
                                            side='SELL',
                                            type='MARKET',
                                            quantity=quantity
                                        )
                                        
                                        order_id = order.get('orderId', 'N/A')
                                        fills = order.get('fills', [])
                                        avg_price = sum(float(f['price']) * float(f['qty']) for f in fills) / sum(float(f['qty']) for f in fills) if fills else exit_price
                                        
                                        pnl_pct = ((avg_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                                        actual_pnl = (avg_price - entry_price) * quantity
                                        status = '✅ PROFIT' if actual_pnl > 0 else '❌ LOSS'
                                        win_rate = (portfolio['winning_trades'] / portfolio['closed_trades'] * 100) if portfolio['closed_trades'] > 0 else 0
                                        total_return = (portfolio['total_pnl'] / portfolio['initial_capital'] * 100) if portfolio['initial_capital'] > 0 else 0
                                        
                                        print(f'\n📊 {symbol} SELL ORDER EXECUTED: {pattern} @ ${avg_price:.2f} {status}', flush=True)
                                        print(f'   Order ID: {order_id}', flush=True)
                                        print(f'   P&L: ${actual_pnl:.2f} ({pnl_pct:+.2f}%)', flush=True)
                                        print(f'   Reason: {exit_reason}', flush=True)
                                        
                                        # Update portfolio from API
                                        try:
                                            account = client.get_account()
                                            for asset in account['balances']:
                                                if asset['asset'] == 'USDT':
                                                    portfolio['capital'] = float(asset['free'])
                                                    break
                                        except:
                                            pass
                                        
                                        print(f'💰 Portfolio [{symbol}]:', flush=True)
                                        print(f'   Capital: ${portfolio["capital"]:.2f}', flush=True)
                                        print(f'   Active Trades: {portfolio["active_trades"]}', flush=True)
                                        print(f'   Invested: ${portfolio["total_invested"]:.2f}', flush=True)
                                        print(f'   Total P&L: ${portfolio["total_pnl"]:.2f} ({total_return:+.2f}%)', flush=True)
                                        print(f'   Win Rate: {win_rate:.1f}%\n', flush=True)
                                        
                                    except Exception as e:
                                        print(f'⚠️  [{symbol}] Failed to execute SELL order: {e}', flush=True)
                                
                                # Log trade close (exit) - simulation mode
                                elif exit_price > 0:
                                    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                                    status = '✅ PROFIT' if pnl > 0 else '❌ LOSS'
                                    win_rate = (portfolio['winning_trades'] / portfolio['closed_trades'] * 100) if portfolio['closed_trades'] > 0 else 0
                                    total_return = (portfolio['total_pnl'] / portfolio['initial_capital'] * 100) if portfolio['initial_capital'] > 0 else 0
                                    print(f'\n📊 {symbol} SELL (SIM): {pattern} @ ${exit_price:.2f} {status}', flush=True)
                                    print(f'   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)', flush=True)
                                    print(f'   Reason: {exit_reason}', flush=True)
                                    print(f'💰 Portfolio [{symbol}]:', flush=True)
                                    print(f'   Capital: ${portfolio["capital"]:.2f}', flush=True)
                                    print(f'   Active Trades: {portfolio["active_trades"]}', flush=True)
                                    print(f'   Invested: ${portfolio["total_invested"]:.2f}', flush=True)
                                    print(f'   Total P&L: ${portfolio["total_pnl"]:.2f} ({total_return:+.2f}%)', flush=True)
                                    print(f'   Win Rate: {win_rate:.1f}%\n', flush=True)
                                
                                # Save to CSV
                                logger.log_trade(symbol, 'realtime-15s', trade)
                            
                            last_signal_time = pd.Timestamp.now()
                            
                except Exception as e:
                    if tick_count % 1000 == 0:  # Don't spam errors
                        print(f'⚠️  {symbol} processing error: {e}', flush=True)
                
                # Periodic portfolio status update (every 30 seconds)
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    win_rate = (portfolio['winning_trades'] / portfolio['closed_trades'] * 100) if portfolio['closed_trades'] > 0 else 0
                    total_return_pct = (portfolio['total_pnl'] / portfolio['initial_capital'] * 100) if portfolio['initial_capital'] > 0 else 0
                    print(f'📊 [{symbol}] Portfolio Update | Capital: ${portfolio["capital"]:.2f} | Active: {portfolio["active_trades"]} | Invested: ${portfolio["total_invested"]:.2f} | Total P&L: ${portfolio["total_pnl"]:.2f} ({total_return_pct:+.2f}%) | Closed: {portfolio["closed_trades"]} trades | Win Rate: {win_rate:.1f}%', flush=True)
                    last_status_time = current_time
                
                # Keep buffer from getting too large
                buffer = buffer[-200:]
                
            except Exception as e:
                print(f'❌ {symbol} aggregation error: {e}', flush=True)
                buffer = []
