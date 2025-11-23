"""
Hedge Manager - Közös hedging logika backtest és WebSocket számára
Dynamic hedging with volatility-based threshold adjustment
"""
import numpy as np
from datetime import datetime


class HedgeManager:
    """
    Dinamikus hedge kezelő - capital protection downtrend-ben
    Használható backtest és live trading esetén is
    """
    
    def __init__(self, config):
        """
        Args:
            config: Config modul referencia (HEDGING dict-tel) VAGY közvetlenül dict
        """
        # Handle both: config module (with .HEDGING) or direct dict
        if isinstance(config, dict):
            self.config = config  # Direct dict from backtest
        else:
            self.config = config.HEDGING  # Config module reference
        
        self.coin_overrides = self.config.get('coin_overrides', {})
        
        # Tracking
        self.equity_curve = []
        self.hedge_activations = 0
        
    def get_config_for_coin(self, coin):
        """Coin-specific vagy default config visszaadása"""
        if coin in self.coin_overrides:
            # Merge coin override with defaults
            cfg = self.config.copy()
            cfg.update(self.coin_overrides[coin])
            return cfg
        return self.config
    
    def compute_dynamic_threshold(self, volatility_window=None, min_thresh=None, max_thresh=None):
        """
        Dinamikus hedge threshold volatilitás alapján
        
        Logika:
        - Alacsony volatilitás → magasabb threshold (ritkább hedge)
        - Magas volatilitás → alacsonyabb threshold (gyakoribb hedge)
        
        Returns:
            float: Dinamikus threshold (0.0-1.0 között)
        """
        volatility_window = volatility_window or self.config['volatility_window']
        min_thresh = min_thresh or self.config['min_hedge_threshold']
        max_thresh = max_thresh or self.config['max_hedge_threshold']
        
        if len(self.equity_curve) < volatility_window + 2:
            # Nincs elég adat → középérték
            return (min_thresh + max_thresh) / 2
        
        # Számoljuk a volatilitást az equity curve-ből
        equity = np.array(self.equity_curve[-(volatility_window + 1):])
        returns = np.diff(equity) / equity[:-1]
        vol = np.std(returns)
        
        # Normalize volatility (0-1 skálára)
        # 3% volatilitás = max (crypto esetén)
        norm_vol = min(1.0, max(0.0, vol / 0.03))
        
        # Magasabb volatilitás → alacsonyabb threshold
        dynamic_threshold = max_thresh - norm_vol * (max_thresh - min_thresh)
        
        return dynamic_threshold
    
    def should_hedge(self, capital, peak_capital, equity=None, peak_equity=None, coin=None):
        """
        Eldönti hogy aktiválni kell-e hedge-et
        
        Args:
            capital: Jelenlegi szabad tőke
            peak_capital: Peak capital (all-time high)
            equity: Total equity (capital + unrealized P&L) - opcionális
            peak_equity: Peak equity - opcionális
            coin: Coin neve (coin-specific config-hoz) - opcionális
        
        Returns:
            tuple: (should_hedge: bool, drawdown: float)
        """
        cfg = self.get_config_for_coin(coin) if coin else self.config
        
        if not cfg['enable'] or peak_capital <= 0:
            return False, 0.0
        
        # Dinamikus threshold számítás
        if cfg['dynamic_hedge']:
            threshold = self.compute_dynamic_threshold(
                cfg['volatility_window'],
                cfg['min_hedge_threshold'],
                cfg['max_hedge_threshold']
            )
        else:
            threshold = cfg['hedge_threshold']
        
        # Drawdown számítás (equity vagy capital alapon)
        if cfg['drawdown_basis'] == 'equity' and equity is not None and peak_equity:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        else:
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
        
        return drawdown > threshold, drawdown
    
    def should_close_hedge(self, capital, peak_capital, equity=None, peak_equity=None, coin=None):
        """
        Eldönti hogy zárni kell-e a hedge-et (recovery)
        
        Returns:
            bool: True ha zárni kell a hedge-et
        """
        cfg = self.get_config_for_coin(coin) if coin else self.config
        
        if peak_capital <= 0:
            return False
        
        # Drawdown számítás
        if cfg['drawdown_basis'] == 'equity' and equity is not None and peak_equity:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        else:
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
        
        # Recovery threshold alatt → zárjuk a hedge-et
        return drawdown < cfg['hedge_recovery_threshold']
    
    def create_hedge_trade(self, active_trades, current_price, coin=None, entry_time=None):
        """
        Hedge trade létrehozása a jelenlegi LONG exposure alapján
        
        Args:
            active_trades: Aktív trade-ek listája
            current_price: Jelenlegi market ár
            coin: Coin neve
            entry_time: Entry időpont (datetime vagy index)
        
        Returns:
            dict: Hedge trade vagy None ha nincs LONG exposure
        """
        cfg = self.get_config_for_coin(coin) if coin else self.config
        
        if not active_trades:
            return None
        
        # BUG #49 FIX: Total LONG exposure calculation
        # DO NOT use fallback with current_price - it's the HEDGE coin price, not trade coin!
        # All trades MUST have position_value stored at entry time
        total_long_exposure = sum(
            t['position_value']  # Use stored value, no fallback!
            for t in active_trades
            if t['direction'] == 'long' and not t.get('is_hedge', False)
        )
        
        if total_long_exposure <= 0:
            return None
        
        # Hedge size (ratio alapján)
        hedge_size = total_long_exposure * cfg['hedge_ratio']
        
        if current_price <= 0:
            return None
        
        # Hedge position size
        hedge_position_size = hedge_size / current_price
        
        # Hedge trade (SHORT pozíció)
        hedge_trade = {
            'coin': coin,
            'entry_price': current_price,
            'position_size': hedge_position_size,
            'position_value': hedge_size,
            'stop_loss': current_price * 1.03,  # 3% SL (felfelé mozgás ellen)
            'take_profit': current_price * 0.97,  # 3% TP (lefelé mozgás profit)
            'direction': 'short',
            'pattern': 'hedge',
            'is_hedge': True,
            'status': 'open',
            'entry_time': entry_time or datetime.now(),
            'hedge_size': hedge_size,
            'probability': 1.0,  # Hedge mindig 100% konfidencia
            'strength': 1.0,
            'timeframe': 'hedge'
        }
        
        self.hedge_activations += 1
        
        return hedge_trade
    
    def check_hedge_exit(self, hedge_trade, current_candle):
        """
        Ellenőrzi hogy a hedge trade-et zárni kell-e (SL/TP hit)
        
        Args:
            hedge_trade: Hedge trade dict
            current_candle: Jelenlegi candle (pandas Series)
        
        Returns:
            tuple: (should_close: bool, exit_price: float, exit_reason: str)
        """
        if hedge_trade.get('direction') != 'short':
            return False, None, None
        
        high = current_candle['high']
        low = current_candle['low']
        close = current_candle['close']
        
        # SHORT pozíció: SL = high felett, TP = low alatt
        
        # Stop Loss check (ár felfelé megy)
        if high >= hedge_trade['stop_loss']:
            return True, hedge_trade['stop_loss'], 'hedge_stop_loss'
        
        # Take Profit check (ár lefelé megy)
        if low <= hedge_trade['take_profit']:
            return True, hedge_trade['take_profit'], 'hedge_take_profit'
        
        return False, None, None
    
    def calculate_hedge_pnl(self, hedge_trade, exit_price):
        """
        Hedge trade P&L számítása
        
        SHORT pozíció: profit ha az ár csökken
        
        Args:
            hedge_trade: Hedge trade dict
            exit_price: Exit ár
        
        Returns:
            float: P&L (USDT)
        """
        # SHORT: entry_price - exit_price (profit ha csökken az ár)
        pnl = (hedge_trade['entry_price'] - exit_price) * hedge_trade['position_size']
        return pnl
    
    def update_equity_curve(self, equity):
        """
        Frissíti az equity curve-öt (dinamikus threshold számításhoz)
        
        Args:
            equity: Jelenlegi total equity (capital + unrealized P&L)
        """
        self.equity_curve.append(equity)
        
        # Memory optimization: csak az utolsó N értéket tároljuk
        max_history = max(
            self.config['volatility_window'] + 10,
            100  # Min 100 pont
        )
        if len(self.equity_curve) > max_history:
            self.equity_curve = self.equity_curve[-max_history:]
    
    def get_statistics(self):
        """
        Hedge manager statisztikák
        
        Returns:
            dict: Stats
        """
        return {
            'hedge_activations': self.hedge_activations,
            'equity_curve_length': len(self.equity_curve),
            'current_dynamic_threshold': self.compute_dynamic_threshold() if self.config['dynamic_hedge'] else self.config['hedge_threshold']
        }
