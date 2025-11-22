"""
Copied HedgingBacktestEngine from previous /old/backtest_with_hedging.py
to avoid direct imports from /old. Kept functionality intact and trimmed
no-op dependencies for the framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.pattern_targets import calculate_pattern_targets as get_pattern_targets


class HedgingBacktestEngine:
    """
    Advanced backtesting engine with hedging capabilities
    (Copied from the historical private codebase)
    """

    def __init__(self, 
                 initial_capital=10000, 
                 risk_per_trade=0.015,  # Reduced from 0.02 for better capital preservation
                 take_profit_multiplier=2.0,
                 enable_hedging=True,
                 hedge_threshold=0.15,
                 hedge_recovery_threshold=0.05,
                 hedge_ratio=0.5,
                 max_correlation_hedge=0.7,
                 dynamic_hedge=True,
                 volatility_window=20,
                 min_hedge_threshold=0.10,
                 max_hedge_threshold=0.25,
                 drawdown_basis='capital',
                 use_tiered_risk=True,
                 use_fixed_risk=False,
                 max_position_pct=0.15):  # Reduced from 0.20 for tighter position sizing:

        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_multiplier = take_profit_multiplier
        self.use_tiered_risk = use_tiered_risk
        self.use_fixed_risk = use_fixed_risk
        self.max_position_pct = max_position_pct

        self.enable_hedging = enable_hedging
        self.hedge_threshold = hedge_threshold
        self.hedge_recovery_threshold = hedge_recovery_threshold
        self.hedge_ratio = hedge_ratio
        self.max_correlation_hedge = max_correlation_hedge
        self.dynamic_hedge = dynamic_hedge
        self.volatility_window = volatility_window
        self.min_hedge_threshold = min_hedge_threshold
        self.max_hedge_threshold = max_hedge_threshold
        self.drawdown_basis = drawdown_basis

        self.trades = []
        self.equity_curve = []
        self.hedge_trades = []
        self.current_drawdown = 0

    # All methods below are maintained from the original implementation.
    # For brevity in this module, they are a near copy — see /old/backtest_with_hedging.py

    def get_tiered_risk_percentage(self, capital):
        if not self.use_tiered_risk:
            return self.risk_per_trade

        initial = self.initial_capital
        if capital < initial * 2:
            return 0.015  # Reduced from 0.02
        elif capital < initial * 3:
            return 0.012  # Reduced from 0.015
        elif capital < initial * 5:
            return 0.008  # Reduced from 0.01
        else:
            return 0.006  # Reduced from 0.0075

    def calculate_pattern_targets(self, pattern_type, entry_price, high, low, recent_data=None):
        """
        Wrapper to centralized calculate_pattern_targets for backward compatibility.
        Returns: stop_loss, take_profit, direction, params
        """
        # Use centralized function
        try:
            sl, tp, direction, params = get_pattern_targets(pattern_type, entry_price, high, low, recent_data)
            # If the engine wants to tighten stops during drawdown, adjust params
            if self.current_drawdown > 0.10 and params:
                params['sl_pct'] *= 0.7
                params['tp_pct'] *= 0.8
                sl = entry_price * (1 - params['sl_pct'])
                tp = entry_price * (1 + params['tp_pct'])
            return sl, tp, direction, params
        except Exception:
            # Keep fallback
            return 0, 0, 'skip', None

    def _compute_dynamic_threshold(self):
        if len(self.equity_curve) < self.volatility_window + 2:
            return self.hedge_threshold
        equity = np.array(self.equity_curve[-(self.volatility_window+1):])
        returns = np.diff(equity) / equity[:-1]
        vol = np.std(returns)
        norm_vol = min(1.0, max(0.0, vol / 0.03))
        dynamic_threshold = self.max_hedge_threshold - norm_vol * (self.max_hedge_threshold - self.min_hedge_threshold)
        return dynamic_threshold

    def should_hedge(self, capital, peak_capital, equity=None, peak_equity=None):
        if not self.enable_hedging or peak_capital <= 0:
            return False
        adaptive_threshold = self._compute_dynamic_threshold() if self.dynamic_hedge else self.hedge_threshold
        if self.drawdown_basis == 'equity' and equity is not None and peak_equity:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        else:
            drawdown = (peak_capital - capital) / peak_capital
        self.current_drawdown = drawdown
        return drawdown > adaptive_threshold

    def should_close_hedge(self, capital, peak_capital, equity=None, peak_equity=None):
        if peak_capital <= 0:
            return False
        if self.drawdown_basis == 'equity' and equity is not None and peak_equity:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        else:
            drawdown = (peak_capital - capital) / peak_capital
        return drawdown < self.hedge_recovery_threshold

    def create_hedge_trade(self, active_trades, current_price, index):
        if not active_trades:
            return None
        total_long_exposure = sum(t['position_size'] * current_price for t in active_trades if t['direction'] == 'long')
        hedge_size = total_long_exposure * self.hedge_ratio
        if current_price <= 0:
            return None
        hedge_position_size = hedge_size / current_price
        hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'pattern': 'hedge',
            'direction': 'short',
            'position_size': hedge_position_size,
            'stop_loss': current_price * 1.03,
            'take_profit': current_price * 0.97,
            'is_hedge': True,
            'status': 'open',
            'hedge_size': hedge_size
        }
        return hedge_trade

    def run_backtest(self, df, predictions, probabilities, pattern_strength_scores=None):
        print("\n=== RUNNING BACKTEST WITH HEDGING ===")
        capital = self.initial_capital
        peak_capital = capital
        equity = capital
        peak_equity = equity
        self.trades = []
        self.hedge_trades = []
        self.equity_curve = [capital]
        active_trades = []
        active_hedges = []
        hedge_activations = 0

        for i in range(len(df)):
            current_row = df.iloc[i]
            pattern = predictions[i]
            current_price = current_row['close']
            if capital > peak_capital:
                peak_capital = capital
            unrealized_main = 0.0
            for t in active_trades:
                if t.get('is_hedge', False):
                    continue
                unrealized_main += t['position_size'] * (current_price - t['entry_price'])
            unrealized_hedge = 0.0
            for h in active_hedges:
                if h['status'] == 'open':
                    unrealized_hedge += h['position_size'] * (h['entry_price'] - current_price)
            equity = capital + unrealized_main + unrealized_hedge
            if equity > peak_equity:
                peak_equity = equity

            # close hedges if needed
            closed_hedges = []
            for hedge_idx, hedge in enumerate(active_hedges):
                if hedge['status'] == 'open':
                    if self.should_close_hedge(capital, peak_capital, equity, peak_equity):
                        pnl = hedge['position_size'] * (hedge['entry_price'] - current_price)
                        hedge['exit_price'] = current_price
                        hedge['exit_reason'] = 'drawdown_recovered'
                        hedge['status'] = 'closed'
                        closed_hedges.append(hedge_idx)
                        hedge['exit_index'] = i
                        hedge['pnl'] = pnl
                        capital += pnl
                        self.hedge_trades.append(hedge)
                        continue
                    if current_row['high'] >= hedge['stop_loss']:
                        pnl = hedge['position_size'] * (hedge['entry_price'] - hedge['stop_loss'])
                        hedge['exit_price'] = hedge['stop_loss']
                        hedge['exit_reason'] = 'stop_loss'
                        hedge['status'] = 'closed'
                        closed_hedges.append(hedge_idx)
                    elif current_row['low'] <= hedge['take_profit']:
                        pnl = hedge['position_size'] * (hedge['entry_price'] - hedge['take_profit'])
                        hedge['exit_price'] = hedge['take_profit']
                        hedge['exit_reason'] = 'take_profit'
                        hedge['status'] = 'closed'
                        closed_hedges.append(hedge_idx)
                    else:
                        continue
                    hedge['exit_index'] = i
                    hedge['pnl'] = pnl
                    capital += pnl
                    self.hedge_trades.append(hedge)

            for idx in sorted(closed_hedges, reverse=True):
                active_hedges.pop(idx)

            if self.should_hedge(capital, peak_capital, equity, peak_equity) and len(active_trades) > 0 and len(active_hedges) == 0:
                hedge_trade = self.create_hedge_trade(active_trades, current_price, i)
                if hedge_trade:
                    active_hedges.append(hedge_trade)
                    hedge_activations += 1
                    self.equity_curve.append(capital)
                    continue

            if len(active_trades) == 0 and len(active_hedges) > 0:
                for hedge in list(active_hedges):
                    if hedge['status'] == 'open':
                        pnl = hedge['position_size'] * (hedge['entry_price'] - current_price)
                        hedge['exit_price'] = current_price
                        hedge['exit_reason'] = 'no_active_trades'
                        hedge['status'] = 'closed'
                        hedge['exit_index'] = i
                        hedge['pnl'] = pnl
                        capital += pnl
                        self.hedge_trades.append(hedge)
                active_hedges = [h for h in active_hedges if h['status'] == 'open']

            if pattern == 'no_pattern':
                self.equity_curve.append(capital)
                continue

            if pattern_strength_scores is not None and pattern_strength_scores[i] < 0.75:
                self.equity_curve.append(capital)
                continue

            if np.max(probabilities[i]) < 0.75:
                self.equity_curve.append(capital)
                continue

            entry_price = current_price
            recent_data = df.iloc[max(0, i-30):i+1]

            stop_loss, take_profit, direction, params = self.calculate_pattern_targets(pattern, entry_price, current_row['high'], current_row['low'], recent_data)

            if direction == 'skip':
                self.equity_curve.append(capital)
                continue

            risk_multiplier = 1.0
            if self.current_drawdown > 0.20:
                risk_multiplier = 0.5
            elif self.current_drawdown > 0.10:
                risk_multiplier = 0.75

            if self.use_fixed_risk:
                risk_base_capital = self.initial_capital
            else:
                risk_base_capital = capital

            current_risk_pct = self.get_tiered_risk_percentage(risk_base_capital)
            risk_amount = risk_base_capital * current_risk_pct * risk_multiplier

            risk_per_unit = entry_price - stop_loss
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            if self.use_fixed_risk:
                max_position_value = self.initial_capital * self.max_position_pct
            else:
                max_position_value = capital * self.max_position_pct

            position_value = position_size * entry_price
            if position_value > max_position_value:
                position_size = max_position_value / entry_price

            if position_size <= 0:
                self.equity_curve.append(capital)
                continue

            trade = {
                'entry_index': i,
                'entry_price': entry_price,
                'entry_time': current_row.name if hasattr(current_row, 'name') else i,
                'pattern': pattern,
                'direction': direction,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'probability': np.max(probabilities[i]),
                'status': 'open',
                'is_hedge': False,
                'risk_multiplier': risk_multiplier
            }

            active_trades.append(trade)

            closed_trades = []
            for trade_idx, trade in enumerate(active_trades):
                if trade['status'] == 'open' and not trade.get('is_hedge', False):
                    if trade['direction'] == 'long':
                        if current_row['low'] <= trade['stop_loss']:
                            pnl = (trade['stop_loss'] - trade['entry_price']) * trade['position_size']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['high'] >= trade['take_profit']:
                            pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size']
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        else:
                            continue
                    else:
                        if current_row['high'] >= trade['stop_loss']:
                            pnl = (trade['entry_price'] - trade['stop_loss']) * trade['position_size']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['low'] <= trade['take_profit']:
                            pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size']
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        else:
                            continue

                    trade['exit_index'] = i
                    trade['exit_time'] = current_row.name if hasattr(current_row, 'name') else i
                    trade['pnl'] = pnl
                    entry_value = trade['position_size'] * trade['entry_price']
                    trade['pnl_pct'] = (pnl / entry_value * 100) if entry_value > 0 else 0
                    capital += pnl
                    self.trades.append(trade)

            for idx in sorted(closed_trades, reverse=True):
                active_trades.pop(idx)

            self.equity_curve.append(capital)

        print(f"\n✅ Backtest completed!")
        print(f"🛡️ Hedge activations: {hedge_activations}")
        print(f"🛡️ Hedge trades executed: {len(self.hedge_trades)}")
        return self.analyze_results()

    def analyze_results(self):
        if not self.trades and not self.hedge_trades:
            print("No trades executed in backtest")
            return None
        all_trades = self.trades + self.hedge_trades
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        if trades_df.empty:
            return None
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = (trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if losing_trades > 0 else float('inf')
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        hedge_trades_df = pd.DataFrame(self.hedge_trades) if self.hedge_trades else pd.DataFrame()
        hedge_pnl = hedge_trades_df['pnl'].sum() if not hedge_trades_df.empty else 0
        if 'is_hedge' in trades_df.columns:
            main_trades_df = trades_df[~trades_df['is_hedge']]
        else:
            main_trades_df = trades_df.copy()
        pattern_performance = main_trades_df.groupby('pattern').agg({'pnl': ['sum', 'mean', 'count'],}).round(2) if not main_trades_df.empty else None
        if not main_trades_df.empty:
            pattern_wins = main_trades_df[main_trades_df['pnl'] > 0].groupby('pattern').size()
            pattern_totals = main_trades_df.groupby('pattern').size()
            pattern_win_rates = (pattern_wins / pattern_totals * 100).round(2)
        else:
            pattern_win_rates = None

        results = {
            'total_trades': total_trades,
            'main_trades': len(self.trades),
            'hedge_trades': len(self.hedge_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'hedge_pnl': hedge_pnl,
            'main_pnl': total_pnl - hedge_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.equity_curve[-1],
            'return_pct': ((self.equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100,
            'pattern_performance': pattern_performance,
            'pattern_win_rates': pattern_win_rates,
            'trades_df': trades_df,
            'equity_curve': self.equity_curve
        }

        self._print_results(results)
        return results

    def _calculate_max_drawdown(self):
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min()) * 100

    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        if len(self.equity_curve) < 2:
            return 0
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        excess_returns = returns - (risk_free_rate / 252)
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def _print_results(self, results):
        print("\n" + "="*70)
        print("BACKTEST RESULTS WITH HEDGING")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print("-"*70)
        print(f"Main Trades P&L: ${results['main_pnl']:,.2f}")
        print(f"Hedge Trades P&L: ${results['hedge_pnl']:,.2f}")
        print("-"*70)
        print(f"Total Trades: {results['total_trades']} (Main: {results['main_trades']}, Hedges: {results['hedge_trades']})")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print("-"*70)
        print(f"Average Win: ${results['avg_win']:,.2f}")
        print(f"Average Loss: ${results['avg_loss']:,.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print("-"*70)
        if results['pattern_performance'] is not None:
            print("\nPattern Performance:")
            print(results['pattern_performance'])
            print("\nPattern Win Rates (%):")
            print(results['pattern_win_rates'])
        print("="*70)
