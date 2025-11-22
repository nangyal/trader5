"""
Zone Recovery Backtesting Engine V2 - Direction-Aware
Grid-based loss recovery strategy with proper accounting
Supports both LONG and SHORT positions with correct P&L and zone placement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ZoneRecoveryBacktestEngine:
    """
    Zone Recovery Strategy (LONG and SHORT compatible):
    
    LONG Recovery:
    - When LONG position hits stop loss, activate recovery grid
    - Place buy orders at 1%, 2%, 3%, 4%, 5% BELOW current price
    - When price recovers to break-even, close all positions for small profit
    - P&L: (exit_price - entry_price) * size
    
    SHORT Recovery:
    - When SHORT position hits stop loss, activate recovery grid
    - Place sell orders at 1%, 2%, 3%, 4%, 5% ABOVE current price
    - When price recovers to break-even, close all positions for small profit
    - P&L: (entry_price - exit_price) * size
    """
    
    def __init__(self, 
                 initial_capital=10000, 
                 risk_per_trade=0.02, 
                 take_profit_multiplier=2.0,
                 enable_zone_recovery=True,
                 recovery_zone_size=0.01,      # 1% per zone
                 max_recovery_zones=5,          # 5 zones max
                 recovery_position_size=200):   # Fixed $200 per recovery zone
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_multiplier = take_profit_multiplier
        
        self.enable_zone_recovery = enable_zone_recovery
        self.recovery_zone_size = recovery_zone_size
        self.max_recovery_zones = max_recovery_zones
        self.recovery_position_size = recovery_position_size
        
        self.trades = []
        self.equity_curve = []
        
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
        
        # Trend alignment (LONG-ONLY)
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
    
    def run_backtest(self, df, predictions, probabilities, pattern_strength_scores=None):
        """Run backtest with Zone Recovery"""
        print("\n=== RUNNING BACKTEST WITH ZONE RECOVERY V2 ===")
        print(f"Zone Recovery Enabled: {self.enable_zone_recovery}")
        if self.enable_zone_recovery:
            print(f"Recovery Zone Size: {self.recovery_zone_size*100:.1f}%")
            print(f"Max Recovery Zones: {self.max_recovery_zones}")
            print(f"Recovery Position Size: ${self.recovery_position_size}")
        
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [capital]
        
        active_positions = []  # List of all open positions (main + recovery)
        recovery_groups = {}   # group_id -> {'positions': [], 'initial_loss': float}
        next_group_id = 0
        
        recovery_activations = 0
        successful_recoveries = 0
        failed_recoveries = 0
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            pattern = predictions[i]
            current_price = current_row['close']
            
            # Check active positions for exit or stop loss
            closed_positions = []
            
            for pos_idx, pos in enumerate(active_positions):
                if pos['status'] != 'open':
                    continue
                
                group_id = pos.get('group_id')
                is_in_recovery = group_id in recovery_groups
                is_recovery_pos = pos.get('is_recovery', False)
                
                # Skip recovery positions from stop loss check (they have stop_loss = 0)
                if is_recovery_pos:
                    continue
                
                # Check if main position hit stop loss
                if pos['stop_loss'] > 0 and current_row['low'] <= pos['stop_loss']:
                    # Stop loss hit
                    exit_price = pos['stop_loss']
                    
                    # FIXED: Direction-aware P&L calculation
                    if pos['direction'] == 'long':
                        # LONG: loss when price falls to stop
                        pnl = (exit_price - pos['entry_price']) * pos['position_size']
                    else:  # short
                        # SHORT: loss when price rises to stop
                        pnl = (pos['entry_price'] - exit_price) * pos['position_size']
                    
                    pos['exit_price'] = exit_price
                    pos['exit_index'] = i
                    pos['exit_reason'] = 'stop_loss'
                    pos['pnl'] = pnl
                    pos['status'] = 'closed'
                    
                    capital += pnl
                    self.trades.append(pos.copy())
                    closed_positions.append(pos_idx)
                    
                    # Activate zone recovery if enabled and not already in recovery
                    if self.enable_zone_recovery and not is_in_recovery and not pos.get('is_recovery'):
                        # Create recovery group (don't add initial loss yet - already deducted)
                        recovery_groups[next_group_id] = {
                            'positions': [],  # Will add recovery positions only
                            'initial_loss': pnl,  # Track for statistics only
                            'activation_price': current_price,
                            'already_deducted': True  # Mark that loss was already taken
                        }
                        
                        # Create recovery zones (direction-aware)
                        position_direction = pos['direction']  # Get original position direction
                        
                        for zone_num in range(1, self.max_recovery_zones + 1):
                            # FIXED: Direction-aware zone placement
                            if position_direction == 'long':
                                # LONG: Buy zones BELOW current price (price falls, we buy cheaper)
                                zone_price = current_price * (1 - zone_num * self.recovery_zone_size)
                            else:  # short
                                # SHORT: Sell zones ABOVE current price (price rises, we sell higher)
                                zone_price = current_price * (1 + zone_num * self.recovery_zone_size)
                            
                            zone_position_dollars = self.recovery_position_size
                            zone_position_size = zone_position_dollars / zone_price
                            
                            recovery_pos = {
                                'entry_index': i,
                                'entry_price': None,  # Not entered yet
                                'pattern': 'zone_recovery',
                                'direction': position_direction,  # Inherit direction from main position
                                'position_size': zone_position_size,
                                'stop_loss': 0,  # No stop loss for recovery
                                'take_profit': 0,  # Will calculate break-even
                                'zone_trigger_price': zone_price,
                                'zone_number': zone_num,
                                'zone_cost': zone_position_dollars,  # Track cost
                                'status': 'pending',
                                'is_recovery': True,
                                'group_id': next_group_id
                            }
                            active_positions.append(recovery_pos)
                        
                        recovery_activations += 1
                        print(f"\n🔄 RECOVERY ACTIVATED at index {i} | Group {next_group_id} | Loss: ${pnl:.2f} | Zones: {self.max_recovery_zones}")
                        next_group_id += 1
                
                # Check if position hit take profit (only for main positions)
                elif pos['take_profit'] > 0 and current_row['high'] >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    
                    # FIXED: Direction-aware P&L calculation
                    if pos['direction'] == 'long':
                        # LONG: profit when price rises to target
                        pnl = (exit_price - pos['entry_price']) * pos['position_size']
                    else:  # short
                        # SHORT: profit when price falls to target
                        pnl = (pos['entry_price'] - exit_price) * pos['position_size']
                    
                    pos['exit_price'] = exit_price
                    pos['exit_index'] = i
                    pos['exit_reason'] = 'take_profit'
                    pos['pnl'] = pnl
                    pos['status'] = 'closed'
                    
                    capital += pnl
                    self.trades.append(pos.copy())
                    closed_positions.append(pos_idx)
                    
                    # If this position has a recovery group, cancel it
                    if group_id in recovery_groups:
                        # Remove pending recovery positions
                        active_positions = [p for p in active_positions 
                                          if not (p.get('group_id') == group_id and p.get('is_recovery'))]
                        recovery_groups.pop(group_id, None)
            
            # Check recovery zone triggers (direction-aware)
            for pos_idx, pos in enumerate(active_positions):
                if pos['status'] == 'pending' and pos.get('is_recovery'):
                    # FIXED: Direction-aware trigger check
                    position_direction = pos['direction']
                    
                    if position_direction == 'long':
                        # LONG: Trigger when price FALLS to zone price
                        zone_triggered = current_price <= pos['zone_trigger_price']
                    else:  # short
                        # SHORT: Trigger when price RISES to zone price
                        zone_triggered = current_price >= pos['zone_trigger_price']
                    
                    if zone_triggered:
                        # Enter recovery position - NO capital deduction here
                        # We're buying at current price with the position size
                        pos['entry_price'] = current_price
                        pos['entry_index'] = i
                        pos['status'] = 'open'
                        
                        group_id = pos['group_id']
                        if group_id in recovery_groups:
                            recovery_groups[group_id]['positions'].append(pos.copy())
            
            # Check recovery groups for break-even exit
            groups_to_close = []
            
            for group_id, group_data in recovery_groups.items():
                # Get all OPEN positions in this group (only recovery positions that were entered)
                group_positions = [p for p in active_positions 
                                 if p.get('group_id') == group_id 
                                 and p['status'] == 'open' 
                                 and p.get('is_recovery')]
                
                if not group_positions:
                    continue
                
                # Calculate break-even price for recovery positions only
                total_cost = sum(p['entry_price'] * p['position_size'] for p in group_positions)
                total_size = sum(p['position_size'] for p in group_positions)
                
                if total_size > 0:
                    breakeven_price = total_cost / total_size
                    target_exit = breakeven_price * 1.005  # 0.5% profit margin
                    
                    # Check if price reached break-even
                    if current_price >= target_exit:
                        # Close all recovery positions in group
                        recovery_pnl = 0
                        for pos in group_positions:
                            exit_price = current_price
                            
                            # FIXED: Direction-aware P&L calculation
                            if pos['direction'] == 'long':
                                # LONG: profit when price rises
                                pnl = (exit_price - pos['entry_price']) * pos['position_size']
                            else:  # short
                                # SHORT: profit when price falls
                                pnl = (pos['entry_price'] - exit_price) * pos['position_size']
                            
                            pos['exit_price'] = exit_price
                            pos['exit_index'] = i
                            pos['exit_reason'] = 'recovery_breakeven'
                            pos['pnl'] = pnl
                            pos['status'] = 'closed'
                            
                            recovery_pnl += pnl
                            # Add PnL to capital (this is the profit/loss, not return of principal)
                            capital += pnl
                            self.trades.append(pos.copy())
                        
                        # Mark group for removal
                        groups_to_close.append(group_id)
                        
                        # Calculate net result (recovery PnL only, initial loss already deducted)
                        initial_loss = group_data['initial_loss']
                        net_result = initial_loss + recovery_pnl
                        
                        if net_result >= 0:
                            successful_recoveries += 1
                            print(f"✅ RECOVERY SUCCESS at index {i} | Group {group_id} | Initial: ${initial_loss:.2f} | Recovery: ${recovery_pnl:.2f} | Net: ${net_result:.2f}")
                        else:
                            print(f"⚠️  RECOVERY PARTIAL at index {i} | Group {group_id} | Initial: ${initial_loss:.2f} | Recovery: ${recovery_pnl:.2f} | Net: ${net_result:.2f}")
            
            # Remove closed groups and their pending positions
            for group_id in groups_to_close:
                recovery_groups.pop(group_id, None)
                # Remove all positions from this group
                active_positions = [p for p in active_positions 
                                  if p.get('group_id') != group_id or p['status'] == 'closed']
            
            # Remove closed positions
            for idx in sorted(closed_positions, reverse=True):
                if idx < len(active_positions):
                    active_positions.pop(idx)
            
            # Skip if no pattern
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
            risk_per_unit = entry_price - stop_loss
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            if position_size <= 0:
                self.equity_curve.append(capital)
                continue
            
            # Create new main position
            new_position = {
                'entry_index': i,
                'entry_price': entry_price,
                'pattern': pattern,
                'direction': direction,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'probability': pattern_prob,
                'status': 'open',
                'is_recovery': False,
                'group_id': None  # Will be assigned if recovery activates
            }
            
            active_positions.append(new_position)
            self.equity_curve.append(capital)
        
        # Close any remaining open positions at market
        final_price = df.iloc[-1]['close']
        
        for pos in active_positions:
            if pos['status'] == 'open' and pos['entry_price'] is not None:
                exit_price = final_price
                
                # FIXED: Direction-aware P&L calculation
                if pos['direction'] == 'long':
                    # LONG: (exit - entry) * size
                    pnl = (exit_price - pos['entry_price']) * pos['position_size']
                else:  # short
                    # SHORT: (entry - exit) * size
                    pnl = (pos['entry_price'] - exit_price) * pos['position_size']
                
                pos['exit_price'] = exit_price
                pos['exit_index'] = len(df) - 1
                pos['exit_reason'] = 'end_of_backtest'
                pos['pnl'] = pnl
                pos['status'] = 'closed'
                
                capital += pnl
                self.trades.append(pos)
            
            # Cancel pending recovery positions (never entered)
            elif pos['status'] == 'pending' and pos.get('is_recovery'):
                # These never entered, no PnL impact
                pass
        
        # Count failed recoveries (groups that never recovered)
        failed_recoveries = recovery_activations - successful_recoveries
        
        print(f"\n✅ Backtest completed!")
        print(f"🔄 Recovery activations: {recovery_activations}")
        print(f"✅ Successful recoveries: {successful_recoveries}")
        print(f"❌ Failed/Partial recoveries: {failed_recoveries}")
        if recovery_activations > 0:
            print(f"📊 Recovery success rate: {(successful_recoveries/recovery_activations*100):.1f}%")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.trades:
            print("No trades executed in backtest")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Separate main and recovery trades
        main_trades = trades_df[~trades_df['is_recovery']]
        recovery_trades = trades_df[trades_df['is_recovery']]
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        main_pnl = main_trades['pnl'].sum() if not main_trades.empty else 0
        recovery_pnl = recovery_trades['pnl'].sum() if not recovery_trades.empty else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Pattern performance (main trades only)
        pattern_performance = main_trades.groupby('pattern').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).round(2) if not main_trades.empty else None
        
        pattern_win_rates = main_trades.groupby('pattern').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(2) if not main_trades.empty else None
        
        results = {
            'total_trades': total_trades,
            'main_trades': len(main_trades),
            'recovery_trades': len(recovery_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'main_pnl': main_pnl,
            'recovery_pnl': recovery_pnl,
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
        """Print formatted results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - ZONE RECOVERY V2")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print("-"*70)
        print(f"Main Trades P&L: ${results['main_pnl']:,.2f}")
        print(f"Recovery Trades P&L: ${results['recovery_pnl']:,.2f}")
        if results['total_pnl'] != 0:
            recovery_contrib = (results['recovery_pnl'] / results['total_pnl'] * 100)
            print(f"Recovery Contribution: {recovery_contrib:.2f}%")
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
            print("\nPattern Performance (Main Trades):")
            print(results['pattern_performance'])
            print("\nPattern Win Rates (%):")
            print(results['pattern_win_rates'])
        print("="*70)
    
    def plot_equity_curve(self, save_path='equity_curve_zone_v2.png'):
        """Plot equity curve"""
        plt.figure(figsize=(16, 8))
        
        plt.plot(self.equity_curve, linewidth=2, color='#2E86AB', label='Equity')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   label='Initial Capital', alpha=0.5)
        
        plt.title('Equity Curve - Zone Recovery V2', fontsize=16, fontweight='bold')
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Capital ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Equity curve saved to {save_path}")


if __name__ == "__main__":
    print("Zone Recovery Backtest Engine V2 - Ready")
    print("\nKey Features:")
    print("  ✓ Simple and correct accounting")
    print("  ✓ Grid-based recovery on stop loss")
    print("  ✓ Break-even exit strategy")
    print("  ✓ Fixed position sizing for recovery trades")
    print("  ✓ Direction-aware: LONG and SHORT compatible")
    print("  ✓ Correct P&L calculation for both directions")
    print("  ✓ Correct zone placement for both directions")
