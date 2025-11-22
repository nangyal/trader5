"""
Centralized pattern target calculation
Used by all backtesting and trading components
"""
import numpy as np


def calculate_pattern_targets(pattern_type, entry_price, high, low, recent_data=None, atr_threshold=0.8, min_bars_for_trend=20):
    """
    Calculate stop loss and take profit based on pattern type
    V2.3: LONG-ONLY strategy - only take LONG trades when pattern aligns with trend
    
    Args:
        pattern_type: Pattern name (e.g., 'ascending_triangle', 'descending_triangle')
        entry_price: Entry price for the trade
        high: Current high price
        low: Current low price
        recent_data: Optional DataFrame with recent price data for trend calculation
        
    Returns:
        tuple: (stop_loss, take_profit, direction)
               direction can be 'long', 'short', or 'skip'
    """
    base_pattern = pattern_type.split('_')[0] if '_' in pattern_type else pattern_type
    
    targets = {
        'ascending': {'sl_pct': 0.005, 'tp_pct': 0.020},  # 1:4 risk/reward
        'symmetrical': {'sl_pct': 0.006, 'tp_pct': 0.024},
        'double': {'sl_pct': 0.006, 'tp_pct': 0.024},
        'head': {'sl_pct': 0.007, 'tp_pct': 0.028},
        'cup': {'sl_pct': 0.006, 'tp_pct': 0.024},
        'wedge': {'sl_pct': 0.006, 'tp_pct': 0.024},
        'flag': {'sl_pct': 0.005, 'tp_pct': 0.020},
    }
    
    params = targets.get(base_pattern, {'sl_pct': 0.02, 'tp_pct': 0.04})
    
    # V2.3: LONG-ONLY ALIGNED STRATEGY
    # Winners from 10K-bar test (all LONG):
    # - ascending_triangle in uptrend (54.3% win, +0.069%)
    # - descending_triangle in downtrend (52.8% win, +0.036%)
    # - wedge_falling in downtrend (58.4% win, +0.051%)
    if recent_data is not None and len(recent_data) >= min_bars_for_trend:
        closes = recent_data['close'].values[-20:]
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        trend = 'up' if slope > 0 else 'down'
        
        # Align with backtests: Cup & Handle removed from trading set
        bullish_patterns = ['ascending', 'symmetrical']
        bearish_patterns = ['descending', 'wedge']
        
        is_bullish = any(bp in base_pattern for bp in bullish_patterns)
        is_bearish = any(bp in base_pattern for bp in bearish_patterns)
        
        # Only trade LONG when pattern aligns with trend
        if (trend == 'up' and is_bullish) or (trend == 'down' and is_bearish):
            direction = 'long'
        else:
            return 0, 0, 'skip'
    else:
        direction = 'long'
    
    # Volatility/ATR filter (prevent trading in very low-volatility markets)
    if recent_data is not None and len(recent_data) >= 14:
        high_vals = recent_data['high'].values[-14:]
        low_vals = recent_data['low'].values[-14:]
        close_vals = recent_data['close'].values[-14:]
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)
        atr_pct = (atr / entry_price) * 100
        if atr_pct < atr_threshold:
            return 0, 0, 'skip', params

    stop_loss = entry_price * (1 - params['sl_pct'])
    take_profit = entry_price * (1 + params['tp_pct'])

    return stop_loss, take_profit, direction, params
