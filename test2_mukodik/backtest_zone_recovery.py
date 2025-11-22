"""
Zone Recovery Backtesting Engine
Advanced loss recovery using zone-based grid strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


class ZoneRecoveryBacktestEngine:
    """
    Advanced backtesting engine with Zone Recovery loss management
    
    Zone Recovery Strategy:
    - When a losing position is detected, creates recovery zones
    - Places grid orders at predefined intervals below/above entry
    - Averages down/up the position to break even faster
    - Uses smaller position sizes for recovery trades
    - Exit all positions when break-even is reached
    """
    
    def __init__(self, 
                 initial_capital=10000, 
                 risk_per_trade=0.02, 
                 take_profit_multiplier=2.0,
                 enable_zone_recovery=True,
                 recovery_zone_size=0.01,      # 1% zones
                 max_recovery_zones=5,          # Max 5 recovery levels
                 recovery_position_multiplier=0.5):  # Half size for recovery
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_multiplier = take_profit_multiplier
        
        # Zone Recovery parameters
        self.enable_zone_recovery = enable_zone_recovery
        self.recovery_zone_size = recovery_zone_size
        self.max_recovery_zones = max_recovery_zones
        self.recovery_position_multiplier = recovery_position_multiplier
        
        self.trades = []
        self.equity_curve = []
        self.recovery_trades = []
        self.recovery_groups = []  # Track recovery trade groups
        
    def calculate_pattern_targets(self, pattern_type, entry_price, high, low, recent_data=None):
        """Calculate stop loss and take profit"""
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
        
        # Trend alignment (LONG-ONLY strategy)
        if recent_data is not None and len(recent_data) >= 20:
            closes = recent_data['close'].values[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            trend = 'up' if slope > 0 else 'down'
            
            bullish_patterns = ['ascending', 'symmetrical', 'cup']
            bearish_patterns = ['descending', 'wedge']
            
            is_bullish = any(bp in base_pattern for bp in bullish_patterns)
            is_bearish = any(bp in base_pattern for bp in bearish_patterns)
            
            if (trend == 'up' and is_bullish) or (trend == 'down' and is_bearish):
                direction = 'long'
            else:
                return 0, 0, 'skip', None
        else:
            direction = 'long'
        
        stop_loss = entry_price * (1 - params['sl_pct'])
        take_profit = entry_price * (1 + params['tp_pct'])
        
        return stop_loss, take_profit, direction, params
    
    def create_recovery_zones(self, losing_trade, current_price, group_id):
        """Create zone recovery grid for a losing position"""
        if not self.enable_zone_recovery:
            return []
        
        recovery_zones = []
        entry_price = losing_trade['entry_price']
        direction = losing_trade['direction']
        
        if direction == 'long':
            # Price is below entry - create buy zones below current price
            for i in range(1, self.max_recovery_zones + 1):
                zone_price = current_price * (1 - i * self.recovery_zone_size)
                
                # Calculate position size (smaller than original)
                original_size = losing_trade['position_size']
                recovery_size = original_size * self.recovery_position_multiplier
                
                recovery_zone = {
                    'zone_level': i,
                    'zone_price': zone_price,
                    'position_size': recovery_size,
                    'direction': 'long',
                    'status': 'pending',
                    'group_id': group_id,
                    'original_trade': losing_trade
                }
                recovery_zones.append(recovery_zone)
        
        return recovery_zones
    
    def calculate_breakeven_price(self, group_trades):
        """Calculate break-even price for a group of trades"""
        if not group_trades:
            return 0
        
        total_cost = sum(t['entry_price'] * t['position_size'] for t in group_trades)
        total_size = sum(t['position_size'] for t in group_trades)
        
        if total_size == 0:
            return 0
        
        breakeven = total_cost / total_size
        
        # Add small profit margin (0.5%)
        breakeven *= 1.005
        
        return breakeven
    
    def check_recovery_zone_triggers(self, recovery_zones, current_price, index):
        """Check if any recovery zones should be triggered"""
        triggered_zones = []
        
        for zone in recovery_zones:
            if zone['status'] == 'pending' and current_price <= zone['zone_price']:
                # Zone triggered - create recovery trade
                recovery_trade = {
                    'entry_index': index,
                    'entry_price': current_price,
                    'pattern': 'zone_recovery',
                    'direction': zone['direction'],
                    'position_size': zone['position_size'],
                    'zone_level': zone['zone_level'],
                    'group_id': zone['group_id'],
                    'is_recovery': True,
                    'status': 'open'
                }
                triggered_zones.append(recovery_trade)
                zone['status'] = 'triggered'
        
        return triggered_zones
    
    def run_backtest(self, df, predictions, probabilities, pattern_strength_scores=None):
        """Run backtest with Zone Recovery"""
        print("\n=== RUNNING BACKTEST WITH ZONE RECOVERY ===")
        print(f"Zone Recovery Enabled: {self.enable_zone_recovery}")
        print(f"Recovery Zone Size: {self.recovery_zone_size*100:.1f}%")
        print(f"Max Recovery Zones: {self.max_recovery_zones}")
        print(f"Recovery Position Multiplier: {self.recovery_position_multiplier}")
        
        capital = self.initial_capital
        self.trades = []
        self.recovery_trades = []
        self.equity_curve = [capital]
        
        active_trades = []
        active_recovery_groups = {}  # group_id -> list of trades
        pending_recovery_zones = {}  # group_id -> list of zones
        recovery_activations = 0
        successful_recoveries = 0
        next_group_id = 0
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            pattern = predictions[i]
            current_price = current_row['close']
            
            # Check for failed recovery groups (price dropped below all zones)
            failed_groups = []
            for group_id, zones in list(pending_recovery_zones.items()):
                if zones:
                    lowest_zone = min(z['zone_price'] for z in zones)
                    # If price drops 2% below lowest zone, recovery failed
                    if current_price < lowest_zone * 0.98:
                        # Close all trades in this group as failures
                        group_trades = active_recovery_groups.get(group_id, [])
                        total_loss = 0
                        for trade in group_trades:
                            if trade['status'] in ['open', 'stopped']:
                                if trade['status'] == 'stopped':
                                    # Already recorded the loss, just mark as failed
                                    pass
                                else:
                                    # This is a recovery trade that failed
                                    exit_price = current_price
                                    trade['exit_price'] = exit_price
                                    pnl = (exit_price - trade['entry_price']) * trade['position_size']
                                    trade['pnl'] = pnl
                                    total_loss += pnl
                                    capital += pnl
                                
                                trade['exit_index'] = i
                                trade['exit_reason'] = 'recovery_failed'
                                trade['status'] = 'closed'
                                
                                if trade.get('is_recovery', False):
                                    self.recovery_trades.append(trade)
                                elif not any(t.get('group_id') == trade.get('group_id') and 
                                           t.get('entry_index') == trade.get('entry_index') 
                                           for t in self.trades):
                                    self.trades.append(trade)
                        
                        failed_groups.append(group_id)
                        print(f"❌ RECOVERY FAILED at index {i} | Group {group_id} | Additional Loss: ${total_loss:.2f}")
            
            # Remove failed groups
            for group_id in failed_groups:
                active_recovery_groups.pop(group_id, None)
                pending_recovery_zones.pop(group_id, None)
                active_trades = [t for t in active_trades if t.get('group_id') != group_id]
            
            # Check for recovery zone triggers
            triggered_recoveries = []
            for group_id, zones in list(pending_recovery_zones.items()):
                triggered = self.check_recovery_zone_triggers(zones, current_price, i)
                if triggered:
                    triggered_recoveries.extend(triggered)
                    for recovery_trade in triggered:
                        active_recovery_groups[group_id].append(recovery_trade)
                        active_trades.append(recovery_trade)
            
            # Check if any recovery group reached break-even
            groups_to_close = []
            for group_id, group_trades in active_recovery_groups.items():
                if group_trades:
                    breakeven = self.calculate_breakeven_price(group_trades)
                    
                    # Check if current price reached break-even
                    if current_price >= breakeven:
                        # Close all trades in this group
                        total_pnl = 0
                        for trade in group_trades:
                            if trade['status'] in ['open', 'stopped']:  # Include stopped trades
                                exit_price = current_price
                                
                                # If trade was already stopped, it has PnL from stop loss
                                if trade['status'] == 'stopped':
                                    # This trade already has a loss recorded
                                    # We need to calculate the recovery PnL
                                    recovery_pnl = (exit_price - trade['exit_price']) * trade['position_size']
                                    total_pnl += recovery_pnl
                                    trade['recovery_pnl'] = recovery_pnl
                                    trade['final_exit_price'] = exit_price
                                else:
                                    # This is a recovery trade or open original trade
                                    trade['exit_price'] = exit_price
                                    pnl = (exit_price - trade['entry_price']) * trade['position_size']
                                    trade['pnl'] = pnl
                                    total_pnl += pnl
                                
                                trade['exit_index'] = i
                                trade['exit_reason'] = 'breakeven_recovery'
                                trade['status'] = 'closed'
                                
                                if trade.get('is_recovery', False):
                                    self.recovery_trades.append(trade)
                                else:
                                    # Only add to main trades if not already there
                                    if not any(t.get('group_id') == trade.get('group_id') and 
                                             t.get('entry_index') == trade.get('entry_index') 
                                             for t in self.trades):
                                        self.trades.append(trade)
                        
                        capital += total_pnl
                        groups_to_close.append(group_id)
                        successful_recoveries += 1
                        print(f"✅ RECOVERY SUCCESS at index {i} | Group {group_id} | PnL: ${total_pnl:.2f}")
            
            # Remove closed groups
            for group_id in groups_to_close:
                active_recovery_groups.pop(group_id, None)
                pending_recovery_zones.pop(group_id, None)
                active_trades = [t for t in active_trades if t.get('group_id') != group_id]
            
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
            
            # Calculate position size
            entry_price = current_price
            recent_data = df.iloc[max(0, i-30):i+1]
            
            stop_loss, take_profit, direction, params = self.calculate_pattern_targets(
                pattern, entry_price, current_row['high'], current_row['low'], recent_data
            )
            
            if direction == 'skip':
                self.equity_curve.append(capital)
                continue
            
            risk_amount = capital * self.risk_per_trade
            
            if direction == 'long':
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price
            
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            if position_size <= 0:
                self.equity_curve.append(capital)
                continue
            
            # Create trade with group ID
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
                'is_recovery': False,
                'group_id': next_group_id
            }
            
            active_trades.append(trade)
            active_recovery_groups[next_group_id] = [trade]
            next_group_id += 1
            
            # Check active trades for exit or recovery activation
            closed_trades = []
            for trade_idx, trade in enumerate(active_trades):
                if trade['status'] == 'open' and not trade.get('is_recovery', False):
                    if trade['direction'] == 'long':
                        # Check stop loss
                        if current_row['low'] <= trade['stop_loss']:
                            # Stop loss hit - mark the loss
                            exit_price = trade['stop_loss']
                            pnl = (exit_price - trade['entry_price']) * trade['position_size']
                            trade['exit_price'] = exit_price
                            trade['exit_index'] = i
                            trade['exit_reason'] = 'stop_loss'
                            trade['pnl'] = pnl
                            trade['status'] = 'stopped'  # Mark as stopped, not closed yet
                            
                            # Deduct the loss from capital immediately
                            capital += pnl
                            
                            # Activate zone recovery if enabled
                            if self.enable_zone_recovery and trade['group_id'] not in pending_recovery_zones:
                                recovery_zones = self.create_recovery_zones(trade, current_price, trade['group_id'])
                                if recovery_zones:
                                    pending_recovery_zones[trade['group_id']] = recovery_zones
                                    recovery_activations += 1
                                    print(f"\n🔄 ZONE RECOVERY ACTIVATED at index {i} | Group {trade['group_id']} | {len(recovery_zones)} zones | Initial Loss: ${pnl:.2f}")
                            else:
                                # No recovery - close the trade
                                trade['status'] = 'closed'
                                self.trades.append(trade)
                                closed_trades.append(trade_idx)
                                if trade['group_id'] in active_recovery_groups:
                                    active_recovery_groups.pop(trade['group_id'], None)
                        
                        # Check take profit
                        elif current_row['high'] >= trade['take_profit']:
                            pnl = trade['risk_amount'] * self.take_profit_multiplier
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                            
                            trade['exit_index'] = i
                            trade['pnl'] = pnl
                            capital += pnl
                            self.trades.append(trade)
                            
                            # Remove from recovery group
                            if trade['group_id'] in active_recovery_groups:
                                active_recovery_groups.pop(trade['group_id'], None)
                                pending_recovery_zones.pop(trade['group_id'], None)
            
            # Remove closed trades (only take profit exits)
            for idx in sorted(closed_trades, reverse=True):
                active_trades.pop(idx)
            
            self.equity_curve.append(capital)
        
        print(f"\n✅ Backtest completed!")
        print(f"🔄 Recovery activations: {recovery_activations}")
        print(f"✅ Successful recoveries: {successful_recoveries}")
        print(f"📊 Recovery success rate: {(successful_recoveries/recovery_activations*100) if recovery_activations > 0 else 0:.1f}%")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results including zone recovery performance"""
        if not self.trades and not self.recovery_trades:
            print("No trades executed in backtest")
            return None
        
        all_trades = self.trades + self.recovery_trades
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
        
        # Zone Recovery specific metrics
        recovery_trades_df = pd.DataFrame(self.recovery_trades) if self.recovery_trades else pd.DataFrame()
        recovery_pnl = recovery_trades_df['pnl'].sum() if not recovery_trades_df.empty else 0
        
        # Pattern-specific analysis (excluding recoveries)
        main_trades_df = trades_df[~trades_df.get('is_recovery', False)]
        pattern_performance = main_trades_df.groupby('pattern').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).round(2) if not main_trades_df.empty else None
        
        pattern_win_rates = main_trades_df.groupby('pattern').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(2) if not main_trades_df.empty else None
        
        results = {
            'total_trades': total_trades,
            'main_trades': len(self.trades),
            'recovery_trades': len(self.recovery_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'recovery_pnl': recovery_pnl,
            'main_pnl': total_pnl - recovery_pnl,
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
        """Print formatted backtest results with zone recovery info"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS WITH ZONE RECOVERY")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print("-"*70)
        print(f"Main Trades P&L: ${results['main_pnl']:,.2f}")
        print(f"Recovery Trades P&L: ${results['recovery_pnl']:,.2f}")
        print(f"Recovery Contribution: {(results['recovery_pnl']/results['total_pnl']*100) if results['total_pnl'] != 0 else 0:.2f}%")
        print("-"*70)
        print(f"Total Trades: {results['total_trades']} (Main: {results['main_trades']}, Recovery: {results['recovery_trades']})")
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
    
    def plot_equity_curve(self, save_path='equity_curve_zone_recovery.png'):
        """Plot equity curve with recovery activations"""
        plt.figure(figsize=(16, 8))
        
        # Plot equity curve
        plt.plot(self.equity_curve, linewidth=2, color='#2E86AB', label='Equity with Zone Recovery')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital', alpha=0.5)
        
        # Mark recovery activations
        if self.recovery_trades:
            recovery_indices = list(set([r['entry_index'] for r in self.recovery_trades]))
            recovery_values = [self.equity_curve[idx] if idx < len(self.equity_curve) else self.equity_curve[-1] 
                              for idx in recovery_indices]
            plt.scatter(recovery_indices, recovery_values, color='orange', s=80, 
                       marker='s', label='Recovery Zone Activated', zorder=5, alpha=0.6)
        
        plt.title('Equity Curve with Zone Recovery', fontsize=16, fontweight='bold')
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Capital ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Equity curve saved to {save_path}")


if __name__ == "__main__":
    print("Zone Recovery Backtest Engine - Ready for use")
    print("\nFeatures:")
    print("  ✓ Grid-based zone recovery on losing trades")
    print("  ✓ Automatic break-even calculation")
    print("  ✓ Multiple recovery levels (1-5 zones)")
    print("  ✓ Reduced position sizing for recovery trades")
    print("  ✓ Comprehensive recovery performance tracking")
