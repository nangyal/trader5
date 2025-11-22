"""
Excel statisztika generátor
Részletes Excel riportot készít backtest eredményekről
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import config


def generate_excel_report(backtest_results):
    """
    Generál részletes Excel riportot a backtest eredményekből
    
    Args:
        backtest_results: list of dict, minden coin eredménye
        
    Returns:
        str: Excel file path
    """
    print("📊 Excel riport generálása...")
    
    # Ensure stat directory exists
    config.STAT_DIR.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = config.EXCEL_FILENAME_TEMPLATE.format(timestamp=timestamp)
    excel_path = config.STAT_DIR / filename
    
    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        _write_summary_sheet(writer, backtest_results)
        
        # Sheet 2: Detailed Results
        _write_detailed_results_sheet(writer, backtest_results)
        
        # Sheet 3: Per Coin Statistics
        _write_per_coin_stats_sheet(writer, backtest_results)
        
        # Sheet 4: Per Timeframe Statistics
        _write_per_timeframe_stats_sheet(writer, backtest_results)
        
        # Sheet 5: Top Performers
        _write_top_performers_sheet(writer, backtest_results)
    
    print(f"✅ Excel riport mentve: {excel_path}")
    
    return str(excel_path)


def _write_summary_sheet(writer, results):
    """
    Summary sheet - összefoglaló statisztikák
    """
    successful = [r for r in results if r.get('status') == 'completed']
    
    if not successful:
        summary_data = {
            'Metric': ['No successful backtests'],
            'Value': ['N/A']
        }
    else:
        total_trades = sum(r.get('total_trades', 0) for r in successful)
        total_winning = sum(r.get('winning_trades', 0) for r in successful)
        total_losing = sum(r.get('losing_trades', 0) for r in successful)
        
        overall_win_rate = total_winning / total_trades if total_trades > 0 else 0
        
        avg_return = sum(r.get('return_pct', 0) for r in successful) / len(successful)
        best_return = max(r.get('return_pct', 0) for r in successful)
        worst_return = min(r.get('return_pct', 0) for r in successful)
        
        summary_data = {
            'Metric': [
                'Backtest időpont',
                'Összes coin',
                'Sikeres backtest-ek',
                'Sikertelen backtest-ek',
                'Kezdő tőke (USDT)',
                '',
                'Összes trade',
                'Nyerő trade-ek',
                'Vesztő trade-ek',
                'Átlagos win rate (%)',
                '',
                'Átlagos hozam (%)',
                'Legjobb hozam (%)',
                'Legrosszabb hozam (%)',
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(results),
                len(successful),
                len(results) - len(successful),
                f"${config.BACKTEST_INITIAL_CAPITAL:,.2f}",
                '',
                total_trades,
                total_winning,
                total_losing,
                f"{overall_win_rate*100:.2f}%",
                '',
                f"{avg_return:.2f}%",
                f"{best_return:.2f}%",
                f"{worst_return:.2f}%",
            ]
        }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print("  ✓ Summary sheet")


def _write_detailed_results_sheet(writer, results):
    """
    Detailed Results sheet - minden coin részletes eredménye
    """
    detailed_data = []
    
    for result in results:
        coin = result.get('coin', 'N/A')
        status = result.get('status', 'N/A')
        
        row = {
            'Coin': coin,
            'Status': status,
            'Total Trades': result.get('total_trades', 0),
            'Winning Trades': result.get('winning_trades', 0),
            'Losing Trades': result.get('losing_trades', 0),
            'Win Rate (%)': result.get('win_rate', 0) * 100,
            'Total P&L (USDT)': result.get('total_pnl', 0),
            'Return (%)': result.get('return_pct', 0),
            'Final Capital (USDT)': result.get('final_capital', config.BACKTEST_INITIAL_CAPITAL),
            'Avg Win (USDT)': result.get('avg_win', 0),
            'Avg Loss (USDT)': result.get('avg_loss', 0),
        }
        
        # Timeframe info
        timeframes_info = result.get('timeframes', {})
        for tf, tf_data in timeframes_info.items():
            row[f'{tf}_candles'] = tf_data.get('candles', 0)
            row[f'{tf}_signals'] = tf_data.get('quality_signals', 0)
            row[f'{tf}_trades'] = tf_data.get('trades', 0)
        
        detailed_data.append(row)
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)
    
    print("  ✓ Detailed Results sheet")


def _write_per_coin_stats_sheet(writer, results):
    """
    Per Coin Statistics - coin-onként aggregált statisztikák
    """
    successful = [r for r in results if r.get('status') == 'completed' and r.get('total_trades', 0) > 0]
    
    if not successful:
        df_empty = pd.DataFrame({'Message': ['No data available']})
        df_empty.to_excel(writer, sheet_name='Per Coin Stats', index=False)
        print("  ✓ Per Coin Stats sheet (empty)")
        return
    
    coin_stats = []
    
    for result in successful:
        coin_stats.append({
            'Coin': result['coin'],
            'Total Trades': result.get('total_trades', 0),
            'Win Rate (%)': result.get('win_rate', 0) * 100,
            'Total P&L (USDT)': result.get('total_pnl', 0),
            'Return (%)': result.get('return_pct', 0),
            'Avg Win (USDT)': result.get('avg_win', 0),
            'Avg Loss (USDT)': result.get('avg_loss', 0),
        })
    
    df_coin_stats = pd.DataFrame(coin_stats)
    df_coin_stats = df_coin_stats.sort_values('Return (%)', ascending=False)
    df_coin_stats.to_excel(writer, sheet_name='Per Coin Stats', index=False)
    
    print("  ✓ Per Coin Stats sheet")


def _write_per_timeframe_stats_sheet(writer, results):
    """
    Per Timeframe Statistics - timeframe-enként aggregált statisztikák
    """
    successful = [r for r in results if r.get('status') == 'completed']
    
    if not successful:
        df_empty = pd.DataFrame({'Message': ['No data available']})
        df_empty.to_excel(writer, sheet_name='Per Timeframe Stats', index=False)
        print("  ✓ Per Timeframe Stats sheet (empty)")
        return
    
    # Collect timeframe data
    tf_data = {}
    
    for result in successful:
        timeframes_info = result.get('timeframes', {})
        
        for tf, tf_info in timeframes_info.items():
            if tf not in tf_data:
                tf_data[tf] = {
                    'total_candles': 0,
                    'total_signals': 0,
                    'total_trades': 0,
                    'coins': 0
                }
            
            tf_data[tf]['total_candles'] += tf_info.get('candles', 0)
            tf_data[tf]['total_signals'] += tf_info.get('quality_signals', 0)
            tf_data[tf]['total_trades'] += tf_info.get('trades', 0)
            tf_data[tf]['coins'] += 1
    
    # Convert to DataFrame
    tf_stats = []
    for tf, data in tf_data.items():
        tf_stats.append({
            'Timeframe': tf,
            'Coins Processed': data['coins'],
            'Total Candles': data['total_candles'],
            'Avg Candles per Coin': data['total_candles'] / data['coins'] if data['coins'] > 0 else 0,
            'Total Quality Signals': data['total_signals'],
            'Total Trades': data['total_trades'],
            'Avg Trades per Coin': data['total_trades'] / data['coins'] if data['coins'] > 0 else 0,
        })
    
    df_tf_stats = pd.DataFrame(tf_stats)
    df_tf_stats = df_tf_stats.sort_values('Total Trades', ascending=False)
    df_tf_stats.to_excel(writer, sheet_name='Per Timeframe Stats', index=False)
    
    print("  ✓ Per Timeframe Stats sheet")


def _write_top_performers_sheet(writer, results):
    """
    Top Performers - legjobb és legrosszabb eredmények
    """
    successful = [r for r in results if r.get('status') == 'completed' and r.get('total_trades', 0) > 0]
    
    if not successful:
        df_empty = pd.DataFrame({'Message': ['No data available']})
        df_empty.to_excel(writer, sheet_name='Top Performers', index=False)
        print("  ✓ Top Performers sheet (empty)")
        return
    
    # Sort by return
    sorted_results = sorted(successful, key=lambda x: x.get('return_pct', 0), reverse=True)
    
    # Top 10 and Bottom 10
    top_10 = sorted_results[:10]
    bottom_10 = sorted_results[-10:]
    
    top_data = []
    
    # Top performers
    for idx, result in enumerate(top_10, 1):
        top_data.append({
            'Rank': f"Top {idx}",
            'Coin': result['coin'],
            'Return (%)': result.get('return_pct', 0),
            'Total Trades': result.get('total_trades', 0),
            'Win Rate (%)': result.get('win_rate', 0) * 100,
            'P&L (USDT)': result.get('total_pnl', 0),
        })
    
    # Add separator
    top_data.append({
        'Rank': '',
        'Coin': '--- WORST PERFORMERS ---',
        'Return (%)': '',
        'Total Trades': '',
        'Win Rate (%)': '',
        'P&L (USDT)': '',
    })
    
    # Bottom performers
    for idx, result in enumerate(reversed(bottom_10), 1):
        top_data.append({
            'Rank': f"Bottom {idx}",
            'Coin': result['coin'],
            'Return (%)': result.get('return_pct', 0),
            'Total Trades': result.get('total_trades', 0),
            'Win Rate (%)': result.get('win_rate', 0) * 100,
            'P&L (USDT)': result.get('total_pnl', 0),
        })
    
    df_top = pd.DataFrame(top_data)
    df_top.to_excel(writer, sheet_name='Top Performers', index=False)
    
    print("  ✓ Top Performers sheet")


if __name__ == '__main__':
    # Test with dummy data
    dummy_results = [
        {
            'coin': 'BTCUSDT',
            'status': 'completed',
            'total_trades': 50,
            'winning_trades': 30,
            'losing_trades': 20,
            'win_rate': 0.6,
            'total_pnl': 15.5,
            'return_pct': 7.75,
            'final_capital': 215.5,
            'avg_win': 1.2,
            'avg_loss': -0.8,
            'timeframes': {
                '15s': {'candles': 5000, 'quality_signals': 20, 'trades': 15},
                '30s': {'candles': 2500, 'quality_signals': 18, 'trades': 20},
                '1min': {'candles': 1250, 'quality_signals': 22, 'trades': 15},
            }
        },
        {
            'coin': 'ETHUSDT',
            'status': 'completed',
            'total_trades': 40,
            'winning_trades': 22,
            'losing_trades': 18,
            'win_rate': 0.55,
            'total_pnl': -5.2,
            'return_pct': -2.6,
            'final_capital': 194.8,
            'avg_win': 1.0,
            'avg_loss': -1.1,
            'timeframes': {
                '15s': {'candles': 4800, 'quality_signals': 15, 'trades': 12},
                '30s': {'candles': 2400, 'quality_signals': 16, 'trades': 18},
                '1min': {'candles': 1200, 'quality_signals': 14, 'trades': 10},
            }
        }
    ]
    
    excel_file = generate_excel_report(dummy_results)
    print(f"\nTest Excel riport: {excel_file}")
