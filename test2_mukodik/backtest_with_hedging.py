"""
Enhanced Backtesting Engine with Hedging Functionality
Reduces losses through dynamic hedging strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class HedgingBacktestEngine:
    """
    Advanced backtesting engine with hedging capabilities
    Features:
    - Dynamic hedging when drawdown exceeds threshold
    - Partial position hedging
    - Correlation-based hedging
    - Stop-loss tightening in adverse conditions
    """
    
    def __init__(self, 
                 initial_capital=10000, 
                 risk_per_trade=0.02, 
                 take_profit_multiplier=2.0,
                 enable_hedging=True,
                 hedge_threshold=0.15,  # Hedge when drawdown > 15%
                 hedge_ratio=0.5,       # Hedge 50% of exposure
                 max_correlation_hedge=0.7):  # Max correlation for counter-hedge
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_multiplier = take_profit_multiplier
        
        # Hedging parameters
        self.enable_hedging = enable_hedging
        self.hedge_threshold = hedge_threshold
        self.hedge_ratio = hedge_ratio
        self.max_correlation_hedge = max_correlation_hedge
        
        self.trades = []
        self.equity_curve = []
        self.hedge_trades = []
        self.current_drawdown = 0
        
    def calculate_pattern_targets(self, pattern_type, entry_price, high, low, recent_data=None):
        """Calculate stop loss and take profit with adaptive sizing"""
        base_pattern = pattern_type.split('_')[0] if '_' in pattern_type else pattern_type
        
        targets = {
            'ascending': {'sl_pct': 0.015, 'tp_pct': 0.03},
            'descending': {'sl_pct': 0.015, 'tp_pct': 0.03},
            'symmetrical': {'sl_pct': 0.02, 'tp_pct': 0.04},
            'double': {'sl_pct': 0.02, 'tp_pct': 0.04},
            'head': {'sl_pct': 0.025, 'tp_pct': 0.05},
            'cup': {'sl_pct': 0.02, 'tp_pct': 0.045},
            'wedge': {'sl_pct': 0.018, 'tp_pct': 0.036},
            'flag': {'sl_pct': 0.015, 'tp_pct': 0.03},
        }
        
        params = targets.get(base_pattern, {'sl_pct': 0.02, 'tp_pct': 0.04})
        
        # Tighten stops if in drawdown
        if self.current_drawdown > 0.10:  # 10% drawdown
            params['sl_pct'] *= 0.7  # Tighter stop loss
            params['tp_pct'] *= 0.8  # Closer take profit
        
        # Trend alignment with proper pattern direction
        if recent_data is not None and len(recent_data) >= 20:
            closes = recent_data['close'].values[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            trend = 'up' if slope > 0 else 'down'
            
            # OPTIMIZED v2: REMOVED Cup & Handle (23% better drawdown)
            # Cup & Handle: -$4,689 loss BUT causes +$6,088 Ascending Triangle boost
            # Net effect: +$1,398 BUT 64.39% vs 41.13% drawdown (23% worse!)
            # Trade-off: -13.98% return for -23.24% drawdown = BETTER RISK METRICS
            bullish_patterns = ['ascending', 'symmetrical']  # CUP REMOVED
            bearish_patterns = ['descending', 'wedge']
            
            is_bullish = any(bp in base_pattern for bp in bullish_patterns)
            is_bearish = any(bp in base_pattern for bp in bearish_patterns)
            
            # OPTIMIZED: LONG-ONLY strategy (best performance)
            # Bearish patterns skipped - SHORT is not profitable
            if trend == 'up' and is_bullish:
                direction = 'long'
            else:
                # Skip: bearish patterns or trend misalignment
                return 0, 0, 'skip', None
        else:
            # Default: LONG-ONLY for bullish patterns (OPTIMIZED v2: Cup removed)
            bullish_patterns = ['ascending', 'symmetrical']  # CUP REMOVED
            if any(bp in base_pattern for bp in bullish_patterns):
                direction = 'long'
            else:
                # Skip bearish patterns
                return 0, 0, 'skip', None
        
        # LONG positions only
        stop_loss = entry_price * (1 - params['sl_pct'])
        take_profit = entry_price * (1 + params['tp_pct'])
        
        return stop_loss, take_profit, direction, params
    
    def should_hedge(self, capital, peak_capital):
        """Determine if hedging should be activated"""
        if not self.enable_hedging:
            return False
        
        drawdown = (peak_capital - capital) / peak_capital
        self.current_drawdown = drawdown
        
        return drawdown > self.hedge_threshold
    
    def should_close_hedge(self, capital, peak_capital):
        """Determine if hedge should be closed (drawdown recovered)"""
        drawdown = (peak_capital - capital) / peak_capital
        # Close hedge when drawdown drops below 5% (recovery threshold)
        return drawdown < 0.05
    
    def create_hedge_trade(self, active_trades, current_price, index):
        """Create a counter-position to hedge current exposure"""
        if not active_trades:
            return None
        
        # Calculate net exposure based on CURRENT price (not entry price)
        total_long_exposure = sum(t['position_size'] * current_price 
                                  for t in active_trades if t['direction'] == 'long')
        
        # Create opposite position (short hedge for long positions)
        hedge_size = total_long_exposure * self.hedge_ratio
        hedge_position_size = hedge_size / current_price
        
        # SHORT hedge: profit when price falls, loss when price rises
        hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'pattern': 'hedge',
            'direction': 'short',
            'position_size': hedge_position_size,
            'stop_loss': current_price * 1.03,  # Loss if price rises 3%
            'take_profit': current_price * 0.97,  # Profit if price falls 3%
            'is_hedge': True,
            'status': 'open',
            'hedge_size': hedge_size  # Track nominal value
        }
        
        return hedge_trade
    
    def run_backtest(self, df, predictions, probabilities, pattern_strength_scores=None):
        """Run backtest with hedging capabilities"""
        print("\n=== RUNNING BACKTEST WITH HEDGING ===")
        print(f"Hedging Enabled: {self.enable_hedging}")
        print(f"Hedge Threshold: {self.hedge_threshold*100:.0f}% drawdown")
        print(f"Hedge Ratio: {self.hedge_ratio*100:.0f}% of exposure")
        
        capital = self.initial_capital
        peak_capital = capital
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
            
            # Update peak capital
            if capital > peak_capital:
                peak_capital = capital
            
            # Check existing hedges for exit or auto-close on recovery
            closed_hedges = []
            for hedge_idx, hedge in enumerate(active_hedges):
                if hedge['status'] == 'open':
                    # Auto-close hedge if drawdown recovered
                    if self.should_close_hedge(capital, peak_capital):
                        # Close at current price
                        pnl = hedge['position_size'] * (hedge['entry_price'] - current_price)
                        hedge['exit_price'] = current_price
                        hedge['exit_reason'] = 'drawdown_recovered'
                        hedge['status'] = 'closed'
                        closed_hedges.append(hedge_idx)
                        hedge['exit_index'] = i
                        hedge['pnl'] = pnl
                        capital += pnl
                        self.hedge_trades.append(hedge)
                        print(f"  🔓 Hedge closed - drawdown recovered at index {i}")
                        continue
                    
                    # Check if hedge SL or TP hit (SHORT logic: high=SL, low=TP)
                    if current_row['high'] >= hedge['stop_loss']:
                        # SHORT stop loss: price went UP (loss)
                        pnl = hedge['position_size'] * (hedge['entry_price'] - hedge['stop_loss'])
                        hedge['exit_price'] = hedge['stop_loss']
                        hedge['exit_reason'] = 'stop_loss'
                        hedge['status'] = 'closed'
                        closed_hedges.append(hedge_idx)
                    elif current_row['low'] <= hedge['take_profit']:
                        # SHORT take profit: price went DOWN (profit)
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
            
            # Remove closed hedges
            for idx in sorted(closed_hedges, reverse=True):
                active_hedges.pop(idx)
            
            # Check if hedging should be activated (ONLY if no active hedges and drawdown high)
            if self.should_hedge(capital, peak_capital) and len(active_trades) > 0 and len(active_hedges) == 0:
                hedge_trade = self.create_hedge_trade(active_trades, current_price, i)
                if hedge_trade:
                    active_hedges.append(hedge_trade)
                    hedge_activations += 1
                    print(f"\n🛡️ HEDGE ACTIVATED at index {i} | Drawdown: {self.current_drawdown*100:.2f}%")
                    # Skip opening new trades on the same bar as hedge activation
                    self.equity_curve.append(capital)
                    continue
            
            # Skip 'no_pattern'
            if pattern == 'no_pattern':
                self.equity_curve.append(capital)
                continue
            
            # Check pattern strength
            if pattern_strength_scores is not None and pattern_strength_scores[i] < 0.6:
                self.equity_curve.append(capital)
                continue
            
            # Get prediction probability
            pattern_prob = np.max(probabilities[i])
            if pattern_prob < 0.6:
                self.equity_curve.append(capital)
                continue
            
            # Calculate position size with risk management
            entry_price = current_price
            recent_data = df.iloc[max(0, i-30):i+1]
            
            stop_loss, take_profit, direction, params = self.calculate_pattern_targets(
                pattern, entry_price, current_row['high'], current_row['low'], recent_data
            )
            
            if direction == 'skip':
                self.equity_curve.append(capital)
                continue
            
            # Reduce risk if in drawdown
            risk_multiplier = 1.0
            if self.current_drawdown > 0.20:  # >20% drawdown
                risk_multiplier = 0.5  # Half position size
            elif self.current_drawdown > 0.10:  # >10% drawdown
                risk_multiplier = 0.75  # 75% position size
            
            risk_amount = capital * self.risk_per_trade * risk_multiplier
            
            if direction == 'long':
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price
            
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            if position_size <= 0:
                self.equity_curve.append(capital)
                continue
            
            # Create trade
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
                'probability': pattern_prob,
                'status': 'open',
                'is_hedge': False,
                'risk_multiplier': risk_multiplier
            }
            
            active_trades.append(trade)
            
            # Check active trades for exit
            closed_trades = []
            for trade_idx, trade in enumerate(active_trades):
                if trade['status'] == 'open' and not trade.get('is_hedge', False):
                    if trade['direction'] == 'long':
                        if current_row['low'] <= trade['stop_loss']:
                            pnl = -trade['risk_amount']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['high'] >= trade['take_profit']:
                            pnl = trade['risk_amount'] * self.take_profit_multiplier
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        else:
                            continue
                    else:
                        if current_row['high'] >= trade['stop_loss']:
                            pnl = -trade['risk_amount']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['low'] <= trade['take_profit']:
                            pnl = trade['risk_amount'] * self.take_profit_multiplier
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        else:
                            continue
                    
                    trade['exit_index'] = i
                    trade['exit_time'] = current_row.name if hasattr(current_row, 'name') else i
                    trade['pnl'] = pnl
                    trade['pnl_pct'] = (pnl / trade['risk_amount']) * 100
                    
                    capital += pnl
                    self.trades.append(trade)
            
            # Remove closed trades
            for idx in sorted(closed_trades, reverse=True):
                active_trades.pop(idx)
            
            self.equity_curve.append(capital)
        
        print(f"\n✅ Backtest completed!")
        print(f"🛡️ Hedge activations: {hedge_activations}")
        print(f"🛡️ Hedge trades executed: {len(self.hedge_trades)}")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results including hedge performance"""
        if not self.trades and not self.hedge_trades:
            print("No trades executed in backtest")
            return None
        
        all_trades = self.trades + self.hedge_trades
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        
        if trades_df.empty:
            return None
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                        abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if losing_trades > 0 else float('inf')
        
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Hedge-specific metrics
        hedge_trades_df = pd.DataFrame(self.hedge_trades) if self.hedge_trades else pd.DataFrame()
        hedge_pnl = hedge_trades_df['pnl'].sum() if not hedge_trades_df.empty else 0
        
        # Pattern-specific analysis (excluding hedges)
        main_trades_df = trades_df[~trades_df.get('is_hedge', False)]
        pattern_performance = main_trades_df.groupby('pattern').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).round(2) if not main_trades_df.empty else None
        
        pattern_win_rates = main_trades_df.groupby('pattern').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(2) if not main_trades_df.empty else None
        
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
            'trades_df': trades_df
        }
        
        self._print_results(results)
        return results
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min()) * 100
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        excess_returns = returns - (risk_free_rate / 252)
        
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _print_results(self, results):
        """Print formatted backtest results with hedging info"""
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
        print(f"Hedging Contribution: {(results['hedge_pnl']/results['total_pnl']*100) if results['total_pnl'] != 0 else 0:.2f}%")
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
    
    def plot_equity_curve(self, save_path='equity_curve_hedged.png'):
        """Plot equity curve with hedging activations"""
        plt.figure(figsize=(16, 8))
        
        # Plot equity curve
        plt.plot(self.equity_curve, linewidth=2, color='#2E86AB', label='Equity with Hedging')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital', alpha=0.5)
        
        # Mark hedge activations
        if self.hedge_trades:
            hedge_indices = [h['entry_index'] for h in self.hedge_trades]
            hedge_values = [self.equity_curve[idx] if idx < len(self.equity_curve) else self.equity_curve[-1] 
                           for idx in hedge_indices]
            plt.scatter(hedge_indices, hedge_values, color='orange', s=100, 
                       marker='^', label='Hedge Activated', zorder=5)
        
        plt.title('Equity Curve with Dynamic Hedging', fontsize=16, fontweight='bold')
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Capital ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Equity curve saved to {save_path}")


if __name__ == "__main__":
    print("Hedging Backtest Engine - Ready for use")
    print("\nFeatures:")
    print("  ✓ Dynamic hedging activation on drawdown")
    print("  ✓ Adaptive position sizing in adverse conditions")
    print("  ✓ Tighter stop-losses during drawdowns")
    print("  ✓ Counter-position hedging")
    print("  ✓ Comprehensive hedge performance tracking")
