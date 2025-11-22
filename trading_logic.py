"""
Közös trading logika modul
Backtest és WebSocket is ezt használja vásárláskor és eladáskor
"""
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import os


class TradingLogic:
    """
    Központi trading logika - használja mind a backtest, mind a websocket modul
    """
    
    def __init__(self, config, initial_capital=None):
        """
        Inicializálás konfigurációval
        
        Args:
            config: Config modul
            initial_capital: Kezdő tőke (opcionális)
        """
        self.config = config
        self.initial_capital = initial_capital or config.BACKTEST_INITIAL_CAPITAL
        self.capital = self.initial_capital
        self.active_trades = []
        self.closed_trades = []
        self.total_pnl = 0.0
        
        # Risk management
        self.risk_per_trade = config.RISK_PER_TRADE
        self.use_tiered_risk = config.USE_TIERED_RISK
        self.risk_tiers = config.RISK_TIERS
        
        # Pattern targets
        self.pattern_targets = config.PATTERN_TARGETS
        
        # Filters
        self.trend_alignment = config.TREND_ALIGNMENT
        self.volatility_filter = config.VOLATILITY_FILTER
        self.pattern_filters = config.PATTERN_FILTERS
        
        # Advanced strategies
        self.trailing_stop = config.TRAILING_STOP
        self.partial_tp = config.PARTIAL_TP
        self.breakeven_stop = config.BREAKEVEN_STOP
        self.ml_confidence_weighting = config.ML_CONFIDENCE_WEIGHTING
        self.losing_streak_protection = config.LOSING_STREAK_PROTECTION
        
        # Losing streak tracking
        self.consecutive_losses = 0
        self.cooldown_until_candle = 0
        
    def get_tiered_risk_percentage(self, current_capital):
        """
        Számítja a kockázatot a tőke alapján (tiered compounding)
        """
        if not self.use_tiered_risk:
            return self.risk_per_trade
        
        capital_ratio = current_capital / self.initial_capital
        
        for tier in self.risk_tiers:
            if capital_ratio < tier['max_capital_ratio']:
                return tier['risk']
        
        return self.risk_tiers[-1]['risk']
    
    def calculate_pattern_targets(self, pattern_type, entry_price, candle_data, recent_data=None):
        """
        Számítja a stop loss és take profit szinteket pattern alapján
        
        Args:
            pattern_type: Pattern neve (pl. 'ascending_triangle')
            entry_price: Belépési ár
            candle_data: Aktuális candle (high, low, close, stb.)
            recent_data: Recent DataFrame trend számításhoz
            
        Returns:
            tuple: (stop_loss, take_profit, direction) vagy (0, 0, 'skip')
        """
        base_pattern = pattern_type.split('_')[0] if '_' in pattern_type else pattern_type
        
        # Default targets
        params = self.pattern_targets.get(
            pattern_type,
            {'sl_pct': 0.015, 'tp_pct': 0.03}
        )
        
        # Trend alignment check
        direction = 'skip'
        
        if recent_data is not None and len(recent_data) >= self.trend_alignment['lookback_period']:
            # Get current price early
            current_price = recent_data['close'].iloc[-1]
            
            # EMA filter
            if self.trend_alignment['use_ema_filter']:
                ema_period = self.trend_alignment['ema_period']
                if len(recent_data) >= ema_period:
                    ema_50 = recent_data['close'].ewm(span=ema_period, adjust=False).iloc[-1]
                    
                    # LONG-ONLY: skip ha ár EMA alatt van
                    if current_price < ema_50:
                        return 0, 0, 'skip', None
            
            # ATR volatility filter
            if self.volatility_filter['enable']:
                atr_period = self.volatility_filter['atr_period']
                if len(recent_data) >= atr_period:
                    high = recent_data['high'].values[-atr_period:]
                    low = recent_data['low'].values[-atr_period:]
                    close = recent_data['close'].values[-atr_period:]
                    
                    tr1 = high - low
                    tr2 = np.abs(high - np.roll(close, 1))
                    tr3 = np.abs(low - np.roll(close, 1))
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    atr = np.mean(tr)
                    atr_pct = (atr / current_price) * 100
                    
                    # Skip ha túl alacsony volatilitás
                    if atr_pct < self.volatility_filter['min_atr_pct']:
                        return 0, 0, 'skip', None
            
            # Trend számítás
            closes = recent_data['close'].values[-self.trend_alignment['lookback_period']:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            trend = 'up' if slope > 0 else 'down'
            
            # Pattern classification
            bullish_patterns = [p.split('_')[0] for p in self.trend_alignment['bullish_patterns']]
            bearish_patterns = [p.split('_')[0] for p in self.trend_alignment['bearish_patterns']]
            
            # Blacklist check
            if pattern_type in self.pattern_filters.get('blacklist_patterns', []):
                return 0, 0, 'skip', None
            
            is_bullish = any(bp in base_pattern for bp in bullish_patterns)
            is_bearish = any(bp in base_pattern for bp in bearish_patterns)
            
            # Skip bearish patterns (LONG-ONLY)
            if is_bearish:
                return 0, 0, 'skip', None
            
            # LONG-ONLY: bullish patterns uptrend-ben
            if trend == 'up' and is_bullish:
                direction = 'long'
            else:
                return 0, 0, 'skip', None
        else:
            # Nincs elég adat, skip
            return 0, 0, 'skip', None
        
        # Számítsd a stop loss és take profit szinteket
        stop_loss = entry_price * (1 - params['sl_pct'])
        take_profit = entry_price * (1 + params['tp_pct'])
        
        return stop_loss, take_profit, direction, params
    
    def calculate_position_size(self, entry_price, stop_loss, current_capital, risk_multiplier=1.0):
        """
        Számítja a pozíció méretet risk management alapján
        
        Args:
            entry_price: Belépési ár
            stop_loss: Stop loss szint
            current_capital: Aktuális tőke
            risk_multiplier: Kockázat szorzó (pl. drawdown esetén csökkentett pozíció)
            
        Returns:
            float: Pozíció méret (quantity)
        """
        # Tiered risk
        current_risk_pct = self.get_tiered_risk_percentage(current_capital)
        
        # Losing streak protection
        if self.losing_streak_protection['enable']:
            if self.consecutive_losses >= self.losing_streak_protection['reduce_risk_after']:
                risk_multiplier *= self.losing_streak_protection['risk_multiplier']
        
        risk_amount = current_capital * current_risk_pct * risk_multiplier
        
        # LONG pozíció
        risk_per_unit = entry_price - stop_loss
        
        if risk_per_unit <= 0:
            return 0.0
        
        position_size = risk_amount / risk_per_unit
        
        # Cap position size to prevent over-leveraging
        # With MAX_CONCURRENT_TRADES=3, max 33% per trade ensures 100% total usage
        max_position_value = current_capital * self.config.MAX_POSITION_SIZE_PCT
        position_value = position_size * entry_price
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size
    
    def should_open_trade(self, pattern, probability, strength):
        """
        Eldönti, hogy nyisson-e trade-et a pattern alapján
        
        Args:
            pattern: Pattern neve
            probability: ML konfidencia
            strength: Pattern erősség
            
        Returns:
            bool: True ha nyisson trade-et
        """
        if pattern == 'no_pattern':
            return False
        
        if probability < self.pattern_filters['min_probability']:
            return False
        
        if strength < self.pattern_filters['min_strength']:
            return False
        
        # Max concurrent trades check
        if len(self.active_trades) >= self.config.MAX_CONCURRENT_TRADES:
            return False
        
        # Losing streak cooldown check
        if self.losing_streak_protection['enable']:
            if self.consecutive_losses >= self.losing_streak_protection['stop_trading_after']:
                if self.cooldown_until_candle > 0:
                    return False
        
        return True
    
    def open_trade(self, coin, pattern, entry_price, stop_loss, take_profit, 
                   position_size, probability, strength, timeframe, entry_time=None):
        """
        Megnyit egy új trade-et
        
        Returns:
            dict: Trade objektum
        """
        # CRITICAL FIX: Deduct position value from capital
        position_value = entry_price * position_size
        
        # ML Confidence Weighting - adjust position size
        if self.ml_confidence_weighting['enable']:
            for tier in self.ml_confidence_weighting['tiers']:
                if probability >= tier['min_prob']:
                    position_size *= tier['multiplier']
                    position_value = entry_price * position_size
                    break
        
        trade = {
            'coin': coin,
            'pattern': pattern,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'position_value': position_value,  # Track position value
            'probability': probability,
            'strength': strength,
            'timeframe': timeframe,
            'entry_time': entry_time or datetime.now(),
            'direction': 'long',
            'status': 'open',
            'pnl': 0.0,
            'trailing_stop': None,  # For trailing stop
            'partial_closed': 0.0,  # Track partial closures
            'breakeven_activated': False,  # Track breakeven status
        }
        
        # Deduct capital used for this trade
        self.capital -= position_value
        
        self.active_trades.append(trade)
        
        # Log trade open
        self._log_trade(trade, action='OPEN')
        
        return trade
    
    def check_trade_exit(self, trade, current_candle):
        """
        Ellenőrzi, hogy exit-elni kell-e a trade-et (SL, TP, Trailing Stop, Breakeven, Partial TP)
        
        Args:
            trade: Trade objektum
            current_candle: Aktuális candle (dict with high, low, close)
            
        Returns:
            tuple: (should_close, exit_price, exit_reason, partial_ratio) vagy (False, None, None, None)
        """
        if trade['status'] != 'open':
            return False, None, None, None
        
        current_price = current_candle['close']
        
        # LONG trade
        if trade['direction'] == 'long':
            # Calculate current profit %
            profit_pct = (current_price - trade['entry_price']) / trade['entry_price']
            
            # ========================================
            # BREAKEVEN STOP
            # ========================================
            if self.breakeven_stop['enable'] and not trade['breakeven_activated']:
                if profit_pct >= self.breakeven_stop['activation_pct']:
                    # Move SL to breakeven + buffer
                    trade['stop_loss'] = trade['entry_price'] * (1 + self.breakeven_stop['buffer_pct'])
                    trade['breakeven_activated'] = True
            
            # ========================================
            # TRAILING STOP
            # ========================================
            if self.trailing_stop['enable']:
                if profit_pct >= self.trailing_stop['activation_pct']:
                    # Calculate trailing stop price
                    trail_price = current_price * (1 - self.trailing_stop['trail_pct'])
                    
                    # Update trailing stop if higher than current
                    if trade['trailing_stop'] is None or trail_price > trade['trailing_stop']:
                        trade['trailing_stop'] = trail_price
                    
                    # Check if trailing stop hit
                    if current_candle['low'] <= trade['trailing_stop']:
                        return True, trade['trailing_stop'], 'trailing_stop', 1.0
            
            # ========================================
            # PARTIAL TAKE PROFIT
            # ========================================
            if self.partial_tp['enable']:
                for level in self.partial_tp['levels']:
                    if profit_pct >= level['pct']:
                        # Check if this level already executed
                        if trade['partial_closed'] < level['close_ratio']:
                            close_ratio = level['close_ratio'] - trade['partial_closed']
                            return True, current_price, f"partial_tp_{level['pct']*100:.1f}%", close_ratio
            
            # ========================================
            # REGULAR STOP LOSS
            # ========================================
            if current_candle['low'] <= trade['stop_loss']:
                return True, trade['stop_loss'], 'stop_loss', 1.0
            
            # ========================================
            # REGULAR TAKE PROFIT
            # ========================================
            if current_candle['high'] >= trade['take_profit']:
                return True, trade['take_profit'], 'take_profit', 1.0
        
        return False, None, None, None
    
    def close_trade(self, trade, exit_price, exit_reason, exit_time=None, partial_ratio=1.0):
        """
        Bezár egy trade-et (teljes vagy részleges) és számítja a P&L-t
        
        Args:
            partial_ratio: Milyen hányadot zárjon (1.0 = teljes, 0.5 = 50%)
        
        Returns:
            float: P&L (USDT)
        """
        # Calculate closing quantity
        close_size = trade['position_size'] * partial_ratio
        
        # Calculate P&L
        if trade['direction'] == 'long':
            pnl = (exit_price - trade['entry_price']) * close_size
        else:  # short (jelenleg nem használt)
            pnl = (trade['entry_price'] - exit_price) * close_size
        
        # Calculate position value being closed
        position_value_closed = trade['entry_price'] * close_size
        
        # CRITICAL FIX: Return position value + PnL
        self.capital += position_value_closed + pnl
        self.total_pnl += pnl
        
        # Update losing streak tracking
        if self.losing_streak_protection['enable']:
            if pnl < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.losing_streak_protection['stop_trading_after']:
                    self.cooldown_until_candle = self.losing_streak_protection['cooldown_candles']
            else:
                self.consecutive_losses = 0  # Reset on win
        
        # Partial close or full close?
        if partial_ratio < 1.0:
            # Partial close - update trade
            trade['position_size'] -= close_size
            trade['position_value'] -= position_value_closed
            trade['partial_closed'] += partial_ratio
            
            # Log partial close
            partial_trade = trade.copy()
            partial_trade['exit_price'] = exit_price
            partial_trade['exit_reason'] = exit_reason
            partial_trade['exit_time'] = exit_time or datetime.now()
            partial_trade['pnl'] = pnl
            partial_trade['position_size'] = close_size
            self._log_trade(partial_trade, action='PARTIAL_CLOSE')
            
            return pnl
        else:
            # Full close
            trade['exit_price'] = exit_price
            trade['exit_reason'] = exit_reason
            trade['exit_time'] = exit_time or datetime.now()
            trade['pnl'] = pnl
            trade['status'] = 'closed'
            
            # Move to closed trades
            self.closed_trades.append(trade)
            if trade in self.active_trades:
                self.active_trades.remove(trade)
            
            # Log trade close
            self._log_trade(trade, action='CLOSE')
            
            return pnl
    
    def _log_trade(self, trade, action='OPEN'):
        """
        Logol egy trade-et CSV-be
        """
        log_file = self.config.TRADES_LOG_FILE
        
        # Create file with headers if doesn't exist
        file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                # Write header
                writer.writerow([
                    'timestamp', 'coin', 'action', 'direction', 'pattern', 'timeframe',
                    'entry_price', 'exit_price', 'stop_loss', 'take_profit',
                    'position_size', 'probability', 'strength',
                    'exit_reason', 'pnl_usdt', 'total_pnl'
                ])
            
            # Write trade
            writer.writerow([
                trade.get('exit_time', trade.get('entry_time', datetime.now())).strftime('%Y-%m-%d %H:%M:%S'),
                trade['coin'],
                action,
                trade['direction'],
                trade['pattern'],
                trade['timeframe'],
                trade['entry_price'],
                trade.get('exit_price', ''),
                trade['stop_loss'],
                trade['take_profit'],
                trade['position_size'],
                trade['probability'],
                trade['strength'],
                trade.get('exit_reason', ''),
                trade.get('pnl', 0.0),
                self.total_pnl
            ])
    
    def get_statistics(self):
        """
        Visszaadja a trading statisztikákat
        
        Returns:
            dict: Statisztikák
        """
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_capital': self.capital,
                'return_pct': 0.0
            }
        
        df = pd.DataFrame(self.closed_trades)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'final_capital': self.capital,
            'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
            'trades': self.closed_trades  # Add trade list for pattern statistics
        }
