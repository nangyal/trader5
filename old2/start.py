from multiprocessing import Process
import importlib
import config
from classes.training import BacktestRunner
from classes.classification import Classifier
from utils.trade_logger import TradeLogger


def run_backtest():
    print('Starting backtest...')
    cfg = config
    # initialize logger
    logger = TradeLogger(csv_path='trades_log.csv')

    # DON'T load classifier in main process - causes CUDA fork issues
    # Each worker will load its own copy
    clf = None

    # Don't need trading logic in main either - workers create their own
    runner = BacktestRunner(cfg, 'trading_logic1', logger)
    runner.run()


def run_realtime():
    print('Starting realtime (websocket) runner...')
    module = importlib.import_module('trading_logics.' + config.TRADING_LOGIC)
    # trading logic will be created per-process if needed inside engine

    from classes.realtime import RealtimeRunner
    runner = RealtimeRunner(config, module.TradingLogic, TradeLogger())
    runner.run()


def main():
    config.ensure_dirs()
    if config.DATA_SOURCE == 'backtest':
        run_backtest()
    elif config.DATA_SOURCE == 'websocket':
        run_realtime()
    else:
        raise ValueError('Unknown data source: ' + config.DATA_SOURCE)


if __name__ == '__main__':
    main()
