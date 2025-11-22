import csv
from datetime import datetime
from pathlib import Path


class TradeLogger:
    def __init__(self, csv_path='trades_log.csv'):
        self.csv_path = Path(csv_path)
        # ensure header
        if not self.csv_path.exists():
            with open(self.csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'coin', 'timeframe', 'trade_id', 'entry_price', 'exit_price', 'pnl', 'pnl_usdt', 'reason'])

    def log_trade(self, coin, timeframe, trade: dict):
        ts = datetime.utcnow().isoformat()
        trade_id = trade.get('entry_index', '')
        entry = trade.get('entry_price', '')
        exitp = trade.get('exit_price', '')
        pnl = trade.get('pnl', '')
        pnl_usdt = trade.get('pnl', '')  # keep same unit
        reason = trade.get('exit_reason', trade.get('reason', ''))

        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([ts, coin, timeframe, trade_id, entry, exitp, pnl, pnl_usdt, reason])

        print(f"Trade logged: {coin} {timeframe} {trade_id} PnL: {pnl}")
