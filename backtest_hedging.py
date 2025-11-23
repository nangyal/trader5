"""
Hedging Backtest modul - Dinamikus hedging-gel kiegészített backtest
JAVÍTOTT capital management-tel (position value tracking)
"""
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import traceback
import warnings
warnings.filterwarnings('ignore')

import config
from trading_logic import TradingLogic


def compute_dynamic_threshold(equity_curve, volatility_window, min_thresh, max_thresh):
    """Dinamikus hedge threshold volatilitás alapján"""
    if len(equity_curve) < volatility_window + 2:
        return (min_thresh + max_thresh) / 2
        
    equity = np.array(equity_curve[-(volatility_window+1):])
    returns = np.diff(equity) / equity[:-1]
    vol = np.std(returns)
    
    # Normalize volatility
    norm_vol = min(1.0, max(0.0, vol / 0.03))
    
    # Higher vol → lower threshold
    dynamic_threshold = max_thresh - norm_vol * (max_thresh - min_thresh)
    
    return dynamic_threshold


def should_hedge(capital, peak_capital, equity, peak_equity, config_dict, equity_curve):
    """Eldönti, hogy aktiválni kell-e hedge-et"""
    if not config_dict['enable_hedging'] or peak_capital <= 0:
        return False, 0
        
    if config_dict['dynamic_hedge']:
        threshold = compute_dynamic_threshold(
            equity_curve,
            config_dict['volatility_window'],
            config_dict['min_hedge_threshold'],
            config_dict['max_hedge_threshold']
        )
    else:
        threshold = config_dict['hedge_threshold']
    
    if config_dict['drawdown_basis'] == 'equity' and equity is not None and peak_equity:
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
    else:
        drawdown = (peak_capital - capital) / peak_capital
        
    return drawdown > threshold, drawdown


def should_close_hedge(capital, peak_capital, equity, peak_equity, config_dict):
    """Eldönti, hogy zárni kell-e a hedge-et"""
    if peak_capital <= 0:
        return False
        
    if config_dict['drawdown_basis'] == 'equity' and equity is not None and peak_equity:
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
    else:
        drawdown = (peak_capital - capital) / peak_capital
        
    return drawdown < config_dict['hedge_recovery_threshold']


def create_hedge_trade(active_trades, current_price, index, hedge_ratio):
    """Hedge trade létrehozása"""
    if not active_trades:
        return None
        
    # Total LONG exposure
    total_long_exposure = sum(
        t.get('position_value', t['position_size'] * current_price) 
        for t in active_trades 
        if t['direction'] == 'long' and not t.get('is_hedge', False)
    )
    
    hedge_size = total_long_exposure * hedge_ratio
    
    if current_price <= 0:
        return None
        
    hedge_position_size = hedge_size / current_price
    
    # Hedge trade (SHORT)
    hedge_trade = {
        'entry_price': current_price,
        'position_size': hedge_position_size,
        'position_value': hedge_size,
        'stop_loss': current_price * 1.03,  # 3% SL
        'take_profit': current_price * 0.97,  # 3% TP
        'direction': 'short',
        'pattern': 'hedge',
        'is_hedge': True,
        'status': 'open',
        'entry_index': index,
        'hedge_size': hedge_size
    }
    
    return hedge_trade


def run_single_coin_backtest_worker(args):
    """Worker function for multiprocessing"""
    coin, timeframes, data_path_template, model_path, worker_id, config_dict = args
    
    try:
        print(f"\n[Worker {worker_id}] 🛡️  {coin} HEDGING backtest indítása...")
        
        # Load ML model
        from old.forex_pattern_classifier import EnhancedForexPatternClassifier
        
        classifier = EnhancedForexPatternClassifier()
        try:
            classifier.load_model(str(model_path))
            print(f"  ✅ Model betöltve")
        except Exception as e:
            print(f"  ❌ Model betöltési hiba: {e}")
            return {
                'coin': coin,
                'status': 'model_load_error',
                'total_trades': 0,
                'return_pct': 0.0
            }
        
        # Initialize trading logic
        import config as config_module
        trading = TradingLogic(config_module)
        
        # Initialize hedging tracking
        initial_capital = config_dict['initial_capital']
        capital = initial_capital
        peak_capital = capital
        equity = capital
        peak_equity = equity
        
        active_trades = []
        active_hedges = []
        all_trades = []
        all_hedges = []
        hedge_activations = 0
        equity_curve = [capital]
        current_drawdown = 0
        
        # Process each timeframe
        for timeframe in timeframes:
            print(f"  📊 {coin} - {timeframe} (HEDGING)...")
            
            # Load timeframe-specific data
            from backtest import load_timeframe_data, resample_tick_to_timeframe
            
            df_tick = load_timeframe_data(coin, timeframe, data_path_template)
            
            if df_tick is None or len(df_tick) < 1000:
                print(f"    ⚠️  Nincs elég adat")
                continue
            
            df_ohlcv = resample_tick_to_timeframe(df_tick, timeframe)
            
            if len(df_ohlcv) < 100:
                continue
            
            # Run predictions
            try:
                predictions, probabilities = classifier.predict(df_ohlcv)
            except Exception as e:
                print(f"    ❌ Prediction hiba: {e}")
                continue
            
            # Pattern strength
            pattern_strengths = np.array([
                probabilities[i][np.argmax(probabilities[i])] if predictions[i] != 'no_pattern' else 0.0
                for i in range(len(predictions))
            ])
            
            # Simulate trading with hedging
            for i in range(len(df_ohlcv)):
                current_candle = df_ohlcv.iloc[i]
                current_price = current_candle['close']
                pattern = predictions[i]
                pattern_prob = np.max(probabilities[i])
                pattern_strength = pattern_strengths[i]
                
                # Update peaks
                if capital > peak_capital:
                    peak_capital = capital
                
                # Calculate unrealized PnL
                unrealized_main = sum(
                    t['position_size'] * (current_price - t['entry_price'])
                    for t in active_trades
                    if not t.get('is_hedge', False)
                )
                
                unrealized_hedge = sum(
                    h['position_size'] * (h['entry_price'] - current_price)
                    for h in active_hedges
                    if h['status'] == 'open'
                )
                
                equity = capital + unrealized_main + unrealized_hedge
                
                if equity > peak_equity:
                    peak_equity = equity
                
                # ========================================
                # HEDGE MANAGEMENT
                # ========================================
                
                # Close hedges if drawdown recovered
                closed_hedges = []
                for hedge_idx, hedge in enumerate(active_hedges):
                    if hedge['status'] == 'open':
                        # Check recovery
                        if should_close_hedge(capital, peak_capital, equity, peak_equity, config_dict):
                            pnl = hedge['position_size'] * (hedge['entry_price'] - current_price)
                            
                            # CRITICAL FIX: Return position value + PnL
                            capital += hedge['position_value'] + pnl
                            
                            hedge['exit_price'] = current_price
                            hedge['exit_reason'] = 'drawdown_recovered'
                            hedge['exit_time'] = current_candle.name if hasattr(current_candle, 'name') else i
                            hedge['status'] = 'closed'
                            hedge['pnl'] = pnl
                            hedge['coin'] = coin
                            hedge['timeframe'] = timeframe
                            
                            all_hedges.append(hedge)
                            closed_hedges.append(hedge_idx)
                            continue
                        
                        # Check SL/TP
                        if current_candle['high'] >= hedge['stop_loss']:
                            pnl = hedge['position_size'] * (hedge['entry_price'] - hedge['stop_loss'])
                            capital += hedge['position_value'] + pnl
                            
                            hedge['exit_price'] = hedge['stop_loss']
                            hedge['exit_reason'] = 'stop_loss'
                            hedge['status'] = 'closed'
                            closed_hedges.append(hedge_idx)
                        elif current_candle['low'] <= hedge['take_profit']:
                            pnl = hedge['position_size'] * (hedge['entry_price'] - hedge['take_profit'])
                            capital += hedge['position_value'] + pnl
                            
                            hedge['exit_price'] = hedge['take_profit']
                            hedge['exit_reason'] = 'take_profit'
                            hedge['status'] = 'closed'
                            closed_hedges.append(hedge_idx)
                        else:
                            continue
                        
                        hedge['exit_time'] = current_candle.name if hasattr(current_candle, 'name') else i
                        hedge['pnl'] = pnl
                        hedge['coin'] = coin
                        hedge['timeframe'] = timeframe
                        all_hedges.append(hedge)
                
                for idx in sorted(closed_hedges, reverse=True):
                    active_hedges.pop(idx)
                
                # Activate hedge if needed - CSAK 5min vagy nagyobb timeframe-eken
                # 15s, 30s, 1min timeframe-eken NEM használunk hedging-et
                timeframe_allows_hedge = timeframe not in ['15s', '30s', '1min']
                
                should_hedge_now, current_drawdown = should_hedge(
                    capital, peak_capital, equity, peak_equity, config_dict, equity_curve
                )
                
                if should_hedge_now and timeframe_allows_hedge:
                    if len(active_trades) > 0 and len(active_hedges) == 0:
                        hedge_trade = create_hedge_trade(
                            active_trades, current_price, i, config_dict['hedge_ratio']
                        )
                        if hedge_trade:
                            # CRITICAL FIX: Deduct hedge position value
                            capital -= hedge_trade['position_value']
                            active_hedges.append(hedge_trade)
                            hedge_activations += 1
                
                # Close all hedges if no active trades
                if len(active_trades) == 0 and len(active_hedges) > 0:
                    for hedge in list(active_hedges):
                        if hedge['status'] == 'open':
                            pnl = hedge['position_size'] * (hedge['entry_price'] - current_price)
                            capital += hedge['position_value'] + pnl
                            
                            hedge['exit_price'] = current_price
                            hedge['exit_reason'] = 'no_active_trades'
                            hedge['status'] = 'closed'
                            hedge['pnl'] = pnl
                            hedge['coin'] = coin
                            hedge['timeframe'] = timeframe
                            all_hedges.append(hedge)
                    
                    active_hedges = [h for h in active_hedges if h['status'] == 'open']
                
                # ========================================
                # MAIN TRADE MANAGEMENT
                # ========================================
                
                # Check if we should open new trade
                if trading.should_open_trade(pattern, pattern_prob, pattern_strength):
                    recent_data = df_ohlcv.iloc[max(0, i-30):i+1]
                    entry_price = current_candle['close']
                    
                    sl, tp, direction, params = trading.calculate_pattern_targets(
                        pattern, entry_price, current_candle, recent_data
                    )
                    
                    if direction == 'skip':
                        equity_curve.append(capital)
                        continue
                    
                    # Reduce risk during drawdown
                    risk_multiplier = 1.0
                    if current_drawdown > 0.20:
                        risk_multiplier = 0.5
                    elif current_drawdown > 0.10:
                        risk_multiplier = 0.75
                    
                    # Calculate position size (with ML confidence weighting)
                    position_size = trading.calculate_position_size(
                        entry_price, sl, capital, risk_multiplier, ml_probability=pattern_prob
                    )
                    
                    if position_size <= 0:
                        equity_curve.append(capital)
                        continue
                    
                    # CRITICAL FIX: Calculate and deduct position value
                    position_value = position_size * entry_price
                    capital -= position_value
                    
                    trade = {
                        'coin': coin,
                        'pattern': pattern,
                        'entry_price': entry_price,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'position_size': position_size,
                        'position_value': position_value,
                        'probability': pattern_prob,
                        'strength': pattern_strength,
                        'timeframe': timeframe,
                        'entry_time': current_candle.name if hasattr(current_candle, 'name') else i,
                        'direction': direction,
                        'status': 'open',
                        'is_hedge': False,
                        # Advanced strategy tracking fields
                        'trailing_stop': None,
                        'partial_closed': 0.0,
                        'breakeven_activated': False
                    }
                    
                    active_trades.append(trade)
                
                # Check active trades for exit - ADVANCED STRATEGIES
                closed_trades = []
                for trade_idx, trade in enumerate(active_trades):
                    if not trade.get('is_hedge', False) and trade['status'] == 'open':
                        # Use advanced exit logic (trailing stop, partial TP, breakeven, etc.)
                        should_close, exit_price, exit_reason, partial_ratio = trading.check_trade_exit(trade, current_candle)
                        
                        if should_close:
                            # Calculate PnL
                            if trade['direction'] == 'long':
                                pnl = (exit_price - trade['entry_price']) * trade['position_size'] * partial_ratio
                            else:  # short
                                pnl = (trade['entry_price'] - exit_price) * trade['position_size'] * partial_ratio
                            
                            # Return capital
                            capital += trade['position_value'] * partial_ratio + pnl
                            
                            # Partial close vagy full close
                            if partial_ratio < 1.0:
                                # Partial close - reduce position
                                trade['position_size'] *= (1 - partial_ratio)
                                trade['position_value'] *= (1 - partial_ratio)
                                
                                # Log partial close
                                partial_trade = trade.copy()
                                partial_trade['exit_price'] = exit_price
                                partial_trade['exit_reason'] = exit_reason
                                partial_trade['exit_time'] = current_candle.name if hasattr(current_candle, 'name') else i
                                partial_trade['pnl'] = pnl
                                partial_trade['partial_close'] = True
                                partial_trade['partial_ratio'] = partial_ratio
                                all_trades.append(partial_trade)
                            else:
                                # Full close
                                trade['exit_price'] = exit_price
                                trade['exit_reason'] = exit_reason
                                trade['exit_time'] = current_candle.name if hasattr(current_candle, 'name') else i
                                trade['pnl'] = pnl
                                trade['status'] = 'closed'
                                
                                all_trades.append(trade)
                                closed_trades.append(trade_idx)
                
                for idx in sorted(closed_trades, reverse=True):
                    active_trades.pop(idx)
                
                equity_curve.append(capital)
        
        # Calculate statistics
        combined_trades = all_trades + all_hedges
        
        if not combined_trades:
            return {
                'coin': coin,
                'status': 'no_trades',
                'total_trades': 0,
                'return_pct': 0.0
            }
        
        trades_df = pd.DataFrame(combined_trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        
        main_trades = len(all_trades)
        hedge_trades_count = len(all_hedges)
        
        print(f"\n  ✅ {coin} HEDGING backtest befejezve:")
        print(f"     Main trades: {main_trades}")
        print(f"     Hedge trades: {hedge_trades_count}")
        print(f"     Hedge activations: {hedge_activations}")
        print(f"     Total trades: {total_trades}")
        print(f"     Win rate: {win_rate*100:.1f}%")
        print(f"     Return: {((capital - initial_capital) / initial_capital * 100):.2f}%")
        print(f"     Final capital: ${capital:.2f}")
        
        return {
            'coin': coin,
            'status': 'completed',
            'total_trades': total_trades,
            'main_trades': main_trades,
            'hedge_trades': hedge_trades_count,
            'hedge_activations': hedge_activations,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': capital,
            'return_pct': ((capital - initial_capital) / initial_capital) * 100,
            'equity_curve': equity_curve,
            'trades': all_trades,
            'hedges': all_hedges
        }
        
    except Exception as e:
        print(f"\n  ❌ {coin} hiba: {e}")
        traceback.print_exc()
        return {
            'coin': coin,
            'status': f'error: {str(e)}',
            'total_trades': 0,
            'return_pct': 0.0
        }


def run_hedging_backtest(coins, timeframes, num_workers=None):
    """Fő hedging backtest függvény"""
    print("\n" + "="*80)
    print("🛡️  HEDGING BACKTEST INDÍTÁSA")
    print("="*80)
    
    print(f"\nCoinok: {len(coins)}")
    for coin in coins:
        print(f"  • {coin}")
    
    print(f"\nTimeframes: {timeframes}")
    print(f"Kezdő tőke: ${config.BACKTEST_INITIAL_CAPITAL}")
    print(f"\nHedging beállítások:")
    print(f"  Enabled: {config.HEDGING['enable']}")
    print(f"  Threshold: {config.HEDGING['hedge_threshold']*100}%")
    print(f"  Recovery: {config.HEDGING['hedge_recovery_threshold']*100}%")
    print(f"  Hedge ratio: {config.HEDGING['hedge_ratio']*100}%")
    print(f"  Dynamic: {config.HEDGING['dynamic_hedge']}")
    
    if num_workers is None:
        num_workers = min(cpu_count(), len(coins))
    
    print(f"\nWorkers: {num_workers}")
    
    # Prepare args with coin-specific config
    args_list = []
    for idx, coin in enumerate(coins):
        # Base config
        config_dict = {
            'initial_capital': config.BACKTEST_INITIAL_CAPITAL,
            'enable_hedging': config.HEDGING['enable'],
            'hedge_threshold': config.HEDGING['hedge_threshold'],
            'hedge_recovery_threshold': config.HEDGING['hedge_recovery_threshold'],
            'hedge_ratio': config.HEDGING['hedge_ratio'],
            'dynamic_hedge': config.HEDGING['dynamic_hedge'],
            'volatility_window': config.HEDGING['volatility_window'],
            'min_hedge_threshold': config.HEDGING['min_hedge_threshold'],
            'max_hedge_threshold': config.HEDGING['max_hedge_threshold'],
            'drawdown_basis': config.HEDGING['drawdown_basis'],
        }
        
        # Apply coin-specific overrides if available
        if 'coin_overrides' in config.HEDGING and coin in config.HEDGING['coin_overrides']:
            overrides = config.HEDGING['coin_overrides'][coin]
            config_dict.update(overrides)
            print(f"  ⚙️  {coin}: Custom hedging params - threshold={overrides.get('hedge_threshold', config_dict['hedge_threshold'])*100:.0f}%, ratio={overrides.get('hedge_ratio', config_dict['hedge_ratio'])*100:.0f}%")
        
        args_list.append((
            coin,
            timeframes,
            config.BACKTEST_DATA_PATH_TEMPLATE,
            config.MODEL_PATH,
            idx + 1,
            config_dict
        ))
    
    # Run parallel
    print(f"\n{'='*80}")
    print("PÁRHUZAMOS FELDOLGOZÁS INDÍTÁSA")
    print(f"{'='*80}\n")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_coin_backtest_worker, args_list)
    
    print(f"\n{'='*80}")
    print("✅ HEDGING BACKTEST BEFEJEZVE")
    print(f"{'='*80}\n")
    
    # Print summary
    successful = [r for r in results if r['status'] == 'completed']
    
    if successful:
        total_trades = sum(r['total_trades'] for r in successful)
        total_hedge_activations = sum(r.get('hedge_activations', 0) for r in successful)
        avg_return = np.mean([r['return_pct'] for r in successful])
        
        print(f"Sikeres backtestek: {len(successful)}/{len(coins)}")
        print(f"Összes trade: {total_trades}")
        print(f"Hedge aktivációk: {total_hedge_activations}")
        print(f"Átlagos hozam: {avg_return:.2f}%")
    else:
        print("⚠️  Nincs sikeres backtest")
    
    return results


if __name__ == '__main__':
    config.ensure_dirs()
    
    results = run_hedging_backtest(
        coins=config.COINS,
        timeframes=config.TIMEFRAMES,
        num_workers=config.NUM_WORKERS
    )
    
    print("\n=== EREDMÉNYEK ===")
    for result in results:
        print(f"\n{result['coin']}: {result['status']}")
        if result['status'] == 'completed':
            print(f"  Trades: {result['total_trades']} (Main: {result['main_trades']}, Hedge: {result['hedge_trades']})")
            print(f"  Return: {result['return_pct']:.2f}%")
