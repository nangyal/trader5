from utils.backtest_engine import HedgingBacktestEngine
import config
import pandas as pd
import numpy as np


class TradingLogic:
    def __init__(self, classifier=None):
        self.classifier = classifier

    def detect_live_patterns(self, coin, timeframe, df):
        """
        Detect patterns in live data WITHOUT running full backtest
        Returns only pattern signals with entry prices and targets
        """
        if len(df) < 50:
            return []
        
        try:
            if self.classifier:
                predictions, probabilities = self.classifier.predict(df)
            else:
                return []
        except Exception as e:
            print(f'⚠️  Classifier error: {e}', flush=True)
            return []
        
        # Filter for valid patterns with confidence > 0.75
        signals = []
        for i in range(len(df)):
            pattern = predictions[i]
            if pattern == 'no_pattern':
                continue
            
            confidence = np.max(probabilities[i]) if len(probabilities[i]) > 0 else 0
            if confidence < 0.75:  # Minimum confidence threshold
                continue
            
            # Get current bar data
            current_bar = df.iloc[i]
            entry_price = current_bar['close']
            
            # Calculate SL/TP using centralized helper (no full backtest)
            from utils.pattern_targets import calculate_pattern_targets
            recent_data = df.iloc[max(0, i-50):i+1]
            
            # Skip descending triangles
            if 'descending' in pattern.lower():
                continue
            
            stop_loss, take_profit, direction, params = calculate_pattern_targets(pattern, entry_price, current_bar['high'], current_bar['low'], recent_data)
            if direction == 'skip':
                continue
            
            # Trend filter
            if len(recent_data) >= 50:
                ema_50 = recent_data['close'].ewm(span=50, adjust=False).iloc[-1]
                if entry_price < ema_50:
                    continue
                
                closes = recent_data['close'].values[-20:]
                x = np.arange(len(closes))
                slope = np.polyfit(x, closes, 1)[0]
                trend = 'up' if slope > 0 else 'down'
                
                bullish_patterns = ['ascending', 'symmetrical', 'cup', 'flag']
                is_bullish = any(bp in base_pattern for bp in bullish_patterns)
                
                if not (trend == 'up' and is_bullish):
                    continue
            
            # stop_loss, take_profit already returned from calculate_pattern_targets
            
            # Calculate position size (15% max of capital)
            risk_amount = 200 * 0.015  # 1.5% risk per trade
            risk_per_unit = entry_price - stop_loss
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Max position check
            max_position_value = 200 * 0.15  # 15% of initial capital
            position_value = position_size * entry_price
            if position_value > max_position_value:
                position_size = max_position_value / entry_price
            
            if position_size <= 0:
                continue
            
            signal = {
                'pattern': pattern,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'direction': 'long',
                'entry_time': current_bar.name if hasattr(current_bar, 'name') else i,
                'exit_reason': 'open'
            }
            signals.append(signal)
        
        return signals

    def run_backtest(self, coin, timeframe, df, initial_capital):
        # This logic uses the HedgingBacktestEngine
        engine = HedgingBacktestEngine(initial_capital=initial_capital, use_fixed_risk=True, max_position_pct=0.1)

        # predictions: use classifier if available
        try:
            if self.classifier:
                predictions, probabilities = self.classifier.predict(df)
            else:
                predictions = ['no_pattern'] * len(df)
                probabilities = [[1.0]] * len(df)
            pattern_strengths = None
        except Exception as e:
            print('Classifier error - proceeding without classifier:', e)
            predictions = ['no_pattern'] * len(df)
            probabilities = [[1.0]] * len(df)
            pattern_strengths = None

        results = engine.run_backtest(df, predictions, probabilities, pattern_strength_scores=pattern_strengths)
        # Include trades for logging
        if results and 'trades_df' in results and results['trades_df'] is not None:
            trades = results['trades_df'].to_dict('records')
            results['trades'] = trades
        else:
            results['trades'] = []

        return results
