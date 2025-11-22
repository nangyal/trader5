import pandas as pd


class TradingLogic:
    """Simple logic: SMA crossover per timeframe. For demo/testing only."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    def run_backtest(self, coin, timeframe, df, initial_capital):
        df = df.copy()
        df['sma_short'] = df['close'].rolling(5).mean()
        df['sma_long'] = df['close'].rolling(20).mean()

        trades = []
        position = None
        capital = initial_capital

        for i in range(len(df)):
            if i == 0:
                continue
            row = df.iloc[i]
            prev = df.iloc[i-1]

            # Buy signal
            if prev['sma_short'] <= prev['sma_long'] and row['sma_short'] > row['sma_long'] and position is None:
                position = {'entry_index': i, 'entry_price': row['close'], 'direction': 'long'}
            # Exit on cross down
            if position is not None and prev['sma_short'] >= prev['sma_long'] and row['sma_short'] < row['sma_long']:
                entry_price = position['entry_price']
                exit_price = row['close']
                pnl = exit_price - entry_price
                trade = {
                    'entry_index': position['entry_index'],
                    'entry_price': entry_price,
                    'exit_index': i,
                    'exit_price': exit_price,
                    'direction': position['direction'],
                    'pnl': pnl,
                    'exit_reason': 'sma_cross'
                }
                trades.append(trade)
                capital += pnl
                position = None

        results = {
            'total_trades': len(trades),
            'trades': trades,
            'total_pnl': sum([t['pnl'] for t in trades]) if trades else 0,
            'final_capital': capital,
            'return_pct': (capital - initial_capital) / initial_capital * 100
        }
        return results

    def run_multicoin(self, timeframe, ohlc_dict, initial_capital):
        """Example multi-coin logic that checks simple cross-sectional momentum
        and logs a coin to buy if it outperforms the group.
        """
        import numpy as np

        closes = {c: df['close'].pct_change().fillna(0).cumsum().iloc[-1] if len(df) > 0 else 0 for c, df in ohlc_dict.items()}
        # choose top coin
        top_coin = max(closes, key=closes.get)

        # create a fake trade result
        trade = {
            'entry_index': 0,
            'entry_price': ohlc_dict[top_coin]['close'].iloc[0] if len(ohlc_dict[top_coin]) > 0 else 0,
            'exit_price': ohlc_dict[top_coin]['close'].iloc[-1] if len(ohlc_dict[top_coin]) > 0 else 0,
            'pnl': ohlc_dict[top_coin]['close'].iloc[-1] - ohlc_dict[top_coin]['close'].iloc[0] if len(ohlc_dict[top_coin]) > 1 else 0,
            'exit_reason': 'multicoin_momentum',
            'coin': top_coin
        }

        results = {
            'trades': [trade],
            'total_pnl': trade['pnl'],
            'final_capital': initial_capital + trade['pnl'],
            'timeframe': timeframe
        }

        return results
