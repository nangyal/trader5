# forex_pattern_classifier_csv.py
import os

# Suppress all XGBoost warnings at C++ level
os.environ['XGBOOST_VERBOSITY'] = '0'

import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import warnings
import logging
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Suppress XGBoost parameter warnings
logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class AdvancedPatternDetector:
    """
    Advanced pattern detection with confidence scores and volume validation
    Returns confidence scores (0-1) instead of binary True/False
    """
    
    @staticmethod
    def _find_peaks_and_troughs(prices, prominence=0.02):
        """
        Efficient peak and trough detection using scipy
        prominence: minimum height difference from surrounding values (as % of mean price)
        """
        mean_price = np.mean(prices)
        min_prominence = mean_price * prominence
        
        # Find peaks (local maxima)
        peaks, peak_props = find_peaks(prices, prominence=min_prominence)
        
        # Find troughs (local minima) - invert the array
        troughs, trough_props = find_peaks(-prices, prominence=min_prominence)
        
        return peaks, troughs, peak_props, trough_props
    
    @staticmethod
    def _calculate_volume_confirmation(volumes, window_start, window_end):
        """
        Calculate volume confirmation score (0-1)
        Higher volume during pattern formation = higher score
        """
        if volumes is None or len(volumes) == 0:
            return 0.5  # Neutral if no volume data
        
        pattern_volume = np.mean(volumes[window_start:window_end])
        avg_volume = np.mean(volumes)
        
        if avg_volume == 0:
            return 0.5
        
        # Volume ratio: 1.0 = same as average, >1.0 = higher than average
        volume_ratio = pattern_volume / avg_volume
        
        # Convert to 0-1 score (capped at 2x average volume = 1.0)
        return min(volume_ratio / 2.0, 1.0)
    
    @staticmethod
    def _get_adaptive_window(timeframe='1min'):
        """
        Get adaptive window sizes based on timeframe
        """
        windows = {
            '1min': {'triangle': 20, 'double': 30, 'h_s': 40, 'cup': 50},
            '5min': {'triangle': 40, 'double': 60, 'h_s': 80, 'cup': 100},
            '15min': {'triangle': 60, 'double': 90, 'h_s': 120, 'cup': 150},
            '1h': {'triangle': 100, 'double': 150, 'h_s': 200, 'cup': 250},
        }
        return windows.get(timeframe, windows['1min'])
    
    @staticmethod
    def detect_ascending_triangle(highs, lows, volumes=None, window=20, tolerance=0.02, 
                                  timeframe='1min'):
        """
        Ascending Triangle with confidence score
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['triangle']
        
        if len(highs) < window:
            return False, 0.0
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        recent_volumes = volumes[-window:] if volumes is not None else None
        
        # Find resistance level (horizontal top)
        resistance_level = np.max(recent_highs)
        resistance_touches = np.sum(np.abs(recent_highs - resistance_level) / resistance_level < tolerance)
        
        # Check for rising lows (support trend) with linear regression
        x = np.arange(len(recent_lows))
        low_slope, low_intercept = np.polyfit(x, recent_lows, 1)
        
        # Calculate R-squared for trendline quality
        predicted_lows = low_slope * x + low_intercept
        ss_res = np.sum((recent_lows - predicted_lows) ** 2)
        ss_tot = np.sum((recent_lows - np.mean(recent_lows)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate confidence score
        confidence = 0.0
        
        # Resistance touches (max 0.3)
        touch_score = min(resistance_touches / 3.0, 1.0) * 0.3
        confidence += touch_score
        
        # Rising support slope (0.2)
        slope_score = min(low_slope / (resistance_level * 0.01), 1.0) * 0.2 if low_slope > 0 else 0
        confidence += slope_score
        
        # Trendline quality (0.2)
        confidence += r_squared * 0.2
        
        # Volume confirmation (0.15)
        if recent_volumes is not None:
            volume_score = AdvancedPatternDetector._calculate_volume_confirmation(
                volumes, len(highs) - window, len(highs)
            )
            confidence += volume_score * 0.15
        else:
            confidence += 0.075  # Neutral if no volume
        
        # Price proximity to resistance (0.15)
        current_price = recent_highs[-1]
        proximity = 1 - abs(current_price - resistance_level) / resistance_level
        confidence += proximity * 0.15
        
        detected = resistance_touches >= 2 and low_slope > 0 and confidence > 0.65
        
        return detected, confidence

    @staticmethod
    def detect_descending_triangle(highs, lows, volumes=None, window=20, tolerance=0.02,
                                   timeframe='1min'):
        """
        Descending Triangle with confidence score
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['triangle']
        
        if len(highs) < window:
            return False, 0.0
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        recent_volumes = volumes[-window:] if volumes is not None else None
        
        # Find support level (horizontal bottom)
        support_level = np.min(recent_lows)
        support_touches = np.sum(np.abs(recent_lows - support_level) / support_level < tolerance)
        
        # Check for declining highs (resistance trend)
        x = np.arange(len(recent_highs))
        high_slope, high_intercept = np.polyfit(x, recent_highs, 1)
        
        # Calculate R-squared
        predicted_highs = high_slope * x + high_intercept
        ss_res = np.sum((recent_highs - predicted_highs) ** 2)
        ss_tot = np.sum((recent_highs - np.mean(recent_highs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate confidence score
        confidence = 0.0
        
        # Support touches
        touch_score = min(support_touches / 3.0, 1.0) * 0.3
        confidence += touch_score
        
        # Declining resistance slope
        slope_score = min(abs(high_slope) / (support_level * 0.01), 1.0) * 0.2 if high_slope < 0 else 0
        confidence += slope_score
        
        # Trendline quality
        confidence += r_squared * 0.2
        
        # Volume confirmation
        if recent_volumes is not None:
            volume_score = AdvancedPatternDetector._calculate_volume_confirmation(
                volumes, len(highs) - window, len(highs)
            )
            confidence += volume_score * 0.15
        else:
            confidence += 0.075
        
        # Price proximity to support
        current_price = recent_lows[-1]
        proximity = 1 - abs(current_price - support_level) / support_level
        confidence += proximity * 0.15
        
        detected = support_touches >= 2 and high_slope < 0 and confidence > 0.65
        
        return detected, confidence

    @staticmethod
    def detect_symmetrical_triangle(highs, lows, volumes=None, window=20, timeframe='1min'):
        """
        Symmetrical Triangle with confidence score
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['triangle']
        
        if len(highs) < window:
            return False, 0.0
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        
        # Calculate slopes
        x = np.arange(len(recent_highs))
        high_slope, _ = np.polyfit(x, recent_highs, 1)
        low_slope, _ = np.polyfit(x, recent_lows, 1)
        
        # Check convergence
        convergence = high_slope < 0 and low_slope > 0
        
        if not convergence:
            return False, 0.0
        
        # Calculate slope symmetry
        slope_ratio = abs(high_slope / low_slope) if low_slope != 0 else 0
        symmetry_score = 1 - abs(slope_ratio - 1.0)  # Perfect symmetry = 1.0
        symmetry_score = max(0, min(1, symmetry_score))
        
        # Calculate confidence
        confidence = 0.0
        
        # Convergence quality (0.4)
        if 0.5 < slope_ratio < 2.0:
            confidence += 0.4
        elif 0.3 < slope_ratio < 3.0:
            confidence += 0.2
        
        # Symmetry score (0.3)
        confidence += symmetry_score * 0.3
        
        # Volume (0.15)
        if volumes is not None:
            volume_score = AdvancedPatternDetector._calculate_volume_confirmation(
                volumes, len(highs) - window, len(highs)
            )
            confidence += volume_score * 0.15
        else:
            confidence += 0.075
        
        # Range compression (0.15)
        early_range = np.mean(recent_highs[:window//3] - recent_lows[:window//3])
        late_range = np.mean(recent_highs[-window//3:] - recent_lows[-window//3:])
        compression = 1 - (late_range / early_range) if early_range > 0 else 0
        confidence += max(0, min(1, compression)) * 0.15
        
        detected = convergence and confidence > 0.70
        
        return detected, confidence


    @staticmethod
    def detect_double_top(highs, lows, volumes=None, window=30, tolerance=0.015, timeframe='1min'):
        """
        Double Top with confidence score using efficient peak detection
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['double']
        
        if len(highs) < window:
            return False, 0.0
        
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        
        # Use scipy to find peaks efficiently
        peaks, _, peak_props, _ = AdvancedPatternDetector._find_peaks_and_troughs(recent_highs)
        
        if len(peaks) < 2:
            return False, 0.0
        
        # Get two highest peaks
        peak_heights = recent_highs[peaks]
        sorted_indices = np.argsort(peak_heights)[-2:]  # Two highest
        peak_indices = peaks[sorted_indices]
        peak_indices.sort()  # Chronological order
        
        if len(peak_indices) < 2:
            return False, 0.0
        
        peak1_idx, peak2_idx = peak_indices[0], peak_indices[1]
        peak1_val, peak2_val = recent_highs[peak1_idx], recent_highs[peak2_idx]
        
        # Peak similarity
        peak_similarity = abs(peak1_val - peak2_val) / ((peak1_val + peak2_val) / 2)
        
        if peak_similarity >= tolerance:
            return False, 0.0
        
        # Find trough between peaks
        trough_range = recent_lows[peak1_idx:peak2_idx]
        if len(trough_range) == 0:
            return False, 0.0
        
        trough = np.min(trough_range)
        trough_depth = min(peak1_val, peak2_val) - trough
        
        # Calculate confidence
        confidence = 0.0
        
        # Peak similarity (0.3)
        similarity_score = 1 - (peak_similarity / tolerance)
        confidence += max(0, min(1, similarity_score)) * 0.3
        
        # Trough depth (0.25) - should be significant
        avg_peak = (peak1_val + peak2_val) / 2
        depth_ratio = trough_depth / avg_peak
        depth_score = min(depth_ratio / 0.05, 1.0)  # 5% depth = full score
        confidence += depth_score * 0.25
        
        # Peak spacing (0.15) - not too close, not too far
        peak_distance = peak2_idx - peak1_idx
        ideal_distance = window // 3
        distance_score = 1 - abs(peak_distance - ideal_distance) / ideal_distance
        confidence += max(0, min(1, distance_score)) * 0.15
        
        # Volume confirmation (0.15)
        if volumes is not None:
            # Volume should decrease on second peak (bearish sign)
            vol1 = np.mean(volumes[len(volumes) - window + peak1_idx - 2:len(volumes) - window + peak1_idx + 2])
            vol2 = np.mean(volumes[len(volumes) - window + peak2_idx - 2:len(volumes) - window + peak2_idx + 2])
            volume_divergence = (vol1 - vol2) / vol1 if vol1 > 0 else 0
            confidence += max(0, min(1, volume_divergence)) * 0.15
        else:
            confidence += 0.075
        
        # Neckline level (0.15)
        neckline = trough
        current_price = recent_highs[-1]
        below_neckline = current_price < neckline
        if below_neckline:
            confidence += 0.15  # Breakout confirmed
        else:
            proximity = 1 - (current_price - neckline) / neckline
            confidence += max(0, proximity) * 0.075
        
        detected = peak_similarity < tolerance and trough_depth > 0 and confidence > 0.70
        
        return detected, confidence

    @staticmethod
    def detect_double_bottom(lows, highs, volumes=None, window=30, tolerance=0.015, timeframe='1min'):
        """
        Double Bottom with confidence score using efficient peak detection
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['double']
        
        if len(lows) < window:
            return False, 0.0
        
        recent_lows = lows[-window:]
        recent_highs = highs[-window:]
        
        # Use scipy to find troughs efficiently
        _, troughs, _, trough_props = AdvancedPatternDetector._find_peaks_and_troughs(recent_lows)
        
        if len(troughs) < 2:
            return False, 0.0
        
        # Get two lowest troughs
        trough_depths = recent_lows[troughs]
        sorted_indices = np.argsort(trough_depths)[:2]  # Two lowest
        trough_indices = troughs[sorted_indices]
        trough_indices.sort()  # Chronological order
        
        if len(trough_indices) < 2:
            return False, 0.0
        
        trough1_idx, trough2_idx = trough_indices[0], trough_indices[1]
        trough1_val, trough2_val = recent_lows[trough1_idx], recent_lows[trough2_idx]
        
        # Trough similarity
        trough_similarity = abs(trough1_val - trough2_val) / ((trough1_val + trough2_val) / 2)
        
        if trough_similarity >= tolerance:
            return False, 0.0
        
        # Find peak between troughs
        peak_range = recent_highs[trough1_idx:trough2_idx]
        if len(peak_range) == 0:
            return False, 0.0
        
        peak = np.max(peak_range)
        peak_height = peak - max(trough1_val, trough2_val)
        
        # Calculate confidence
        confidence = 0.0
        
        # Trough similarity (0.3)
        similarity_score = 1 - (trough_similarity / tolerance)
        confidence += max(0, min(1, similarity_score)) * 0.3
        
        # Peak height (0.25)
        avg_trough = (trough1_val + trough2_val) / 2
        height_ratio = peak_height / avg_trough
        height_score = min(height_ratio / 0.05, 1.0)
        confidence += height_score * 0.25
        
        # Trough spacing (0.15)
        trough_distance = trough2_idx - trough1_idx
        ideal_distance = window // 3
        distance_score = 1 - abs(trough_distance - ideal_distance) / ideal_distance
        confidence += max(0, min(1, distance_score)) * 0.15
        
        # Volume confirmation (0.15)
        if volumes is not None:
            # Volume should increase on second trough (bullish sign)
            vol1 = np.mean(volumes[len(volumes) - window + trough1_idx - 2:len(volumes) - window + trough1_idx + 2])
            vol2 = np.mean(volumes[len(volumes) - window + trough2_idx - 2:len(volumes) - window + trough2_idx + 2])
            volume_increase = (vol2 - vol1) / vol1 if vol1 > 0 else 0
            confidence += max(0, min(1, volume_increase)) * 0.15
        else:
            confidence += 0.075
        
        # Neckline level (0.15)
        neckline = peak
        current_price = recent_lows[-1]
        above_neckline = current_price > neckline
        if above_neckline:
            confidence += 0.15  # Breakout confirmed
        else:
            proximity = 1 - (neckline - current_price) / neckline
            confidence += max(0, proximity) * 0.075
        
        detected = trough_similarity < tolerance and peak_height > 0 and confidence > 0.70
        
        return detected, confidence

    @staticmethod
    def detect_head_shoulders(highs, lows, volumes=None, window=40, timeframe='1min'):
        """
        Head and Shoulders with confidence score
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['h_s']
        
        if len(highs) < window:
            return False, 0.0
        
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        
        # Find peaks using scipy
        peaks, _, peak_props, _ = AdvancedPatternDetector._find_peaks_and_troughs(recent_highs)
        
        if len(peaks) < 3:
            return False, 0.0
        
        # Get top 3 peaks
        peak_heights = recent_highs[peaks]
        sorted_indices = np.argsort(peak_heights)[-3:]
        peak_indices = peaks[sorted_indices]
        peak_indices.sort()  # Chronological order
        
        if len(peak_indices) < 3:
            return False, 0.0
        
        left_idx, head_idx, right_idx = peak_indices[0], peak_indices[1], peak_indices[2]
        left_val, head_val, right_val = recent_highs[left_idx], recent_highs[head_idx], recent_highs[right_idx]
        
        # Check structure: head should be highest
        head_highest = head_val > left_val and head_val > right_val
        
        if not head_highest:
            return False, 0.0
        
        # Shoulder similarity
        shoulder_similarity = abs(left_val - right_val) / ((left_val + right_val) / 2)
        
        # Calculate confidence
        confidence = 0.0
        
        # Head prominence (0.3)
        head_height = head_val - max(left_val, right_val)
        prominence_ratio = head_height / head_val
        confidence += min(prominence_ratio / 0.05, 1.0) * 0.3
        
        # Shoulder similarity (0.25)
        similarity_score = 1 - shoulder_similarity
        confidence += max(0, min(1, similarity_score * 10)) * 0.25  # Allow up to 10% difference
        
        # Pattern symmetry (0.2)
        left_distance = head_idx - left_idx
        right_distance = right_idx - head_idx
        symmetry = 1 - abs(left_distance - right_distance) / max(left_distance, right_distance)
        confidence += symmetry * 0.2
        
        # Neckline (0.15)
        left_trough = np.min(recent_lows[left_idx:head_idx]) if head_idx > left_idx else recent_lows[left_idx]
        right_trough = np.min(recent_lows[head_idx:right_idx]) if right_idx > head_idx else recent_lows[head_idx]
        neckline = (left_trough + right_trough) / 2
        current_price = recent_highs[-1]
        
        if current_price < neckline:
            confidence += 0.15  # Neckline break confirmed
        else:
            proximity = 1 - (current_price - neckline) / neckline
            confidence += max(0, proximity) * 0.075
        
        # Volume (0.1)
        if volumes is not None:
            recent_volumes = volumes[-window:]
            vol_left = np.mean(recent_volumes[max(0, left_idx-2):left_idx+2])
            vol_head = np.mean(recent_volumes[max(0, head_idx-2):head_idx+2])
            vol_right = np.mean(recent_volumes[max(0, right_idx-2):right_idx+2])
            
            # Decreasing volume pattern (bearish)
            if vol_left > vol_head > vol_right:
                confidence += 0.1
            elif vol_head > vol_right:
                confidence += 0.05
        else:
            confidence += 0.05
        
        detected = head_highest and shoulder_similarity < 0.1 and confidence > 0.70
        
        return detected, confidence


    @staticmethod
    def detect_cup_and_handle(highs, lows, closes, volumes=None, window=50, timeframe='1min'):
        """
        Cup and Handle with confidence score and improved detection
        Returns: (detected: bool, confidence: float 0-1)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['cup']
        
        if len(highs) < window:
            return False, 0.0
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        recent_closes = closes[-window:]
        
        # Cup formation (60% of pattern)
        cup_end = int(window * 0.6)
        cup_highs = recent_highs[:cup_end]
        cup_lows = recent_lows[:cup_end]
        
        # Handle formation (40% of pattern)
        handle_highs = recent_highs[cup_end:]
        handle_lows = recent_lows[cup_end:]
        
        if len(cup_highs) < 10 or len(handle_highs) < 5:
            return False, 0.0
        
        # Calculate confidence
        confidence = 0.0
        
        # Cup depth (0.25) - should be U-shaped, not too deep
        cup_high = np.max(cup_highs)
        cup_low = np.min(cup_lows)
        cup_depth = cup_high - cup_low
        cup_depth_ratio = cup_depth / cup_high
        
        if 0.12 < cup_depth_ratio < 0.33:  # Ideal depth: 12-33%
            confidence += 0.25
        elif 0.08 < cup_depth_ratio < 0.50:
            confidence += 0.15
        
        # Cup roundness (0.25) - should be U-shaped
        cup_middle = len(cup_lows) // 2
        left_low = np.min(cup_lows[:cup_middle])
        right_low = np.min(cup_lows[cup_middle:])
        bottom_low = cup_low
        
        # Both sides should be similar depth
        left_depth = cup_high - left_low
        right_depth = cup_high - right_low
        symmetry = 1 - abs(left_depth - right_depth) / max(left_depth, right_depth)
        confidence += symmetry * 0.25
        
        # Handle characteristics (0.2)
        handle_high = np.max(handle_highs)
        handle_low = np.min(handle_lows)
        handle_depth = handle_high - handle_low
        
        # Handle should be smaller than cup
        handle_to_cup_ratio = handle_depth / cup_depth if cup_depth > 0 else 0
        if handle_to_cup_ratio < 0.5:  # Handle < 50% of cup
            confidence += 0.2
        elif handle_to_cup_ratio < 0.7:
            confidence += 0.1
        
        # Handle should slope slightly downward or sideways
        x = np.arange(len(handle_highs))
        handle_slope, _ = np.polyfit(x, handle_highs, 1)
        if handle_slope <= 0:  # Downward or flat
            confidence += 0.1
        
        # Volume pattern (0.15)
        if volumes is not None:
            recent_volumes = volumes[-window:]
            cup_volume = np.mean(recent_volumes[:cup_end])
            handle_volume = np.mean(recent_volumes[cup_end:])
            
            # Volume should decrease in handle (consolidation)
            if handle_volume < cup_volume:
                vol_decrease = 1 - (handle_volume / cup_volume)
                confidence += min(vol_decrease, 1.0) * 0.15
        else:
            confidence += 0.075
        
        # Breakout potential (0.05)
        current_price = recent_closes[-1]
        rim_level = cup_high
        if current_price >= rim_level * 0.98:  # Near breakout
            confidence += 0.05
        
        detected = cup_depth_ratio > 0.08 and handle_to_cup_ratio < 0.7 and confidence > 0.70
        
        return detected, confidence

    @staticmethod
    def detect_wedge(highs, lows, volumes=None, window=30, timeframe='1min'):
        """
        Wedge pattern (rising or falling) with improved detection
        Returns: (detected: bool, confidence: float 0-1, wedge_type: str)
        """
        adaptive_window = AdvancedPatternDetector._get_adaptive_window(timeframe)
        window = adaptive_window['triangle']
        
        if len(highs) < window:
            return False, 0.0, 'none'
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        
        # Calculate trendlines
        x = np.arange(len(recent_highs))
        high_slope, high_intercept = np.polyfit(x, recent_highs, 1)
        low_slope, low_intercept = np.polyfit(x, recent_lows, 1)
        
        # Both slopes should be in same direction
        rising_wedge = high_slope > 0 and low_slope > 0
        falling_wedge = high_slope < 0 and low_slope < 0
        
        if not (rising_wedge or falling_wedge):
            return False, 0.0, 'none'
        
        wedge_type = 'rising' if rising_wedge else 'falling'
        
        # Calculate R-squared for both trendlines
        predicted_highs = high_slope * x + high_intercept
        predicted_lows = low_slope * x + low_intercept
        
        ss_res_high = np.sum((recent_highs - predicted_highs) ** 2)
        ss_tot_high = np.sum((recent_highs - np.mean(recent_highs)) ** 2)
        r_squared_high = 1 - (ss_res_high / ss_tot_high) if ss_tot_high != 0 else 0
        
        ss_res_low = np.sum((recent_lows - predicted_lows) ** 2)
        ss_tot_low = np.sum((recent_lows - np.mean(recent_lows)) ** 2)
        r_squared_low = 1 - (ss_res_low / ss_tot_low) if ss_tot_low != 0 else 0
        
        # Calculate confidence
        confidence = 0.0
        
        # Trendline quality (0.4)
        avg_r_squared = (r_squared_high + r_squared_low) / 2
        confidence += max(0, min(1, avg_r_squared)) * 0.4
        
        # Convergence (0.25) - lines should be converging
        initial_range = predicted_highs[0] - predicted_lows[0]
        final_range = predicted_highs[-1] - predicted_lows[-1]
        
        if initial_range > 0:
            convergence_ratio = final_range / initial_range
            if convergence_ratio < 0.8:  # At least 20% narrowing
                confidence += (1 - convergence_ratio) * 0.25
        
        # Touch points (0.2)
        high_tolerance = np.mean(recent_highs) * 0.02
        low_tolerance = np.mean(recent_lows) * 0.02
        
        high_touches = np.sum(np.abs(recent_highs - predicted_highs) < high_tolerance)
        low_touches = np.sum(np.abs(recent_lows - predicted_lows) < low_tolerance)
        
        touch_score = min((high_touches + low_touches) / 6.0, 1.0)  # Ideal: 3 touches each
        confidence += touch_score * 0.2
        
        # Volume pattern (0.15)
        if volumes is not None:
            recent_volumes = volumes[-window:]
            early_volume = np.mean(recent_volumes[:window//3])
            late_volume = np.mean(recent_volumes[-window//3:])
            
            # Volume should decrease (consolidation)
            if late_volume < early_volume:
                vol_decrease = 1 - (late_volume / early_volume)
                confidence += min(vol_decrease, 1.0) * 0.15
        else:
            confidence += 0.075
        
        detected = (rising_wedge or falling_wedge) and confidence > 0.70
        
        return detected, confidence, wedge_type

    @staticmethod
    def detect_flag(highs, lows, closes, volumes=None, window=20, timeframe='1min'):
        """
        Flag pattern with improved detection and pole requirement
        Returns: (detected: bool, confidence: float 0-1, flag_type: str)
        """
        if len(highs) < window * 2:  # Need room for pole + flag
            return False, 0.0, 'none'
            
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        recent_closes = closes[-window:]
        
        # Look for strong move (pole) before the flag
        pole_window = window
        pole_highs = highs[-window*2:-window]
        pole_lows = lows[-window*2:-window]
        pole_closes = closes[-window*2:-window]
        
        if len(pole_closes) < 5:
            return False, 0.0, 'none'
        
        # Calculate pole strength
        pole_start = pole_closes[0]
        pole_end = pole_closes[-1]
        pole_move = (pole_end - pole_start) / pole_start
        
        # Need at least 3% move to qualify as pole
        if abs(pole_move) < 0.03:
            return False, 0.0, 'none'
        
        flag_type = 'bullish' if pole_move > 0 else 'bearish'
        
        # Calculate confidence
        confidence = 0.0
        
        # Pole strength (0.3)
        pole_strength = min(abs(pole_move) / 0.10, 1.0)  # 10% move = full score
        confidence += pole_strength * 0.3
        
        # Flag consolidation (0.25)
        flag_range = np.max(recent_highs) - np.min(recent_lows)
        flag_range_ratio = flag_range / np.mean(recent_closes)
        
        if flag_range_ratio < 0.03:  # Tight consolidation
            confidence += 0.25
        elif flag_range_ratio < 0.05:
            confidence += 0.15
        
        # Flag angle (0.2) - should slope against the pole slightly
        x = np.arange(len(recent_closes))
        flag_slope, _ = np.polyfit(x, recent_closes, 1)
        
        if flag_type == 'bullish' and flag_slope <= 0:
            # Bullish flag should slope down or sideways
            confidence += 0.2
        elif flag_type == 'bearish' and flag_slope >= 0:
            # Bearish flag should slope up or sideways
            confidence += 0.2
        else:
            confidence += 0.1  # Parallel to pole is okay too
        
        # Duration (0.15) - flag should be shorter than pole
        flag_duration_ratio = window / pole_window
        if flag_duration_ratio <= 0.5:
            confidence += 0.15
        elif flag_duration_ratio <= 0.7:
            confidence += 0.1
        
        # Volume (0.1)
        if volumes is not None:
            pole_volume = np.mean(volumes[-window*2:-window])
            flag_volume = np.mean(volumes[-window:])
            
            # Volume should decrease in flag
            if flag_volume < pole_volume:
                vol_ratio = 1 - (flag_volume / pole_volume)
                confidence += min(vol_ratio, 1.0) * 0.1
        else:
            confidence += 0.05
        
        detected = abs(pole_move) >= 0.03 and flag_range_ratio < 0.05 and confidence > 0.70
        
        return detected, confidence, flag_type



class EnhancedFeatureExtractor:
    """Enhanced feature extraction with professional technical indicators"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = {}
        
    def add_advanced_price_features(self):
        """Advanced price action features with professional metrics"""
        df = self.df
        
        # Basic price features with safe calculations
        self.features['price_range'] = df['high'] - df['low']
        self.features['body_size'] = abs(df['close'] - df['open'])
        self.features['body_to_range_ratio'] = self._safe_divide(
            self.features['body_size'], self.features['price_range'], 0.5
        )
        
        # Professional candlestick patterns
        self.features['upper_shadow_ratio'] = self._safe_divide(
            df['high'] - df[['open', 'close']].max(axis=1), self.features['price_range'], 0
        )
        self.features['lower_shadow_ratio'] = self._safe_divide(
            df[['open', 'close']].min(axis=1) - df['low'], self.features['price_range'], 0
        )
        
        # Price position features
        self.features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        return self
        
    def add_professional_technical_indicators(self):
        """Professional-grade technical indicators"""
        df = self.df
        close, high, low = df['close'].values, df['high'].values, df['low'].values
        
        # Trend indicators
        self.features['sma_10'] = talib.SMA(close, timeperiod=10)
        self.features['sma_20'] = talib.SMA(close, timeperiod=20)
        self.features['sma_50'] = talib.SMA(close, timeperiod=50)
        self.features['ema_12'] = talib.EMA(close, timeperiod=12)
        self.features['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum indicators
        self.features['rsi'] = talib.RSI(close, timeperiod=14)
        self.features['stoch_k'], self.features['stoch_d'] = talib.STOCH(high, low, close)
        self.features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Volume-based indicators (if volume available)
        if 'volume' in df.columns:
            self.features['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            self.features['volume_ratio'] = self._safe_divide(
                df['volume'], self.features['volume_sma'], 1.0
            )
            self.features['obv'] = talib.OBV(close, df['volume'])
        
        # Volatility indicators
        self.features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        self.features['natr'] = talib.NATR(high, low, close, timeperiod=14)
        self.features['volatility'] = talib.STDDEV(close, timeperiod=20, nbdev=1)
        
        # Advanced indicators
        macd, macd_signal, macd_hist = talib.MACD(close)
        self.features['macd'] = macd
        self.features['macd_signal'] = macd_signal
        self.features['macd_hist'] = macd_hist
        
        # Bollinger Bands with position
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.features['bb_upper'] = bb_upper
        self.features['bb_lower'] = bb_lower
        self.features['bb_position'] = self._safe_divide(
            close - bb_lower, bb_upper - bb_lower, 0.5
        )
        self.features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        return self
        
    def add_pattern_specific_features(self, window=30):
        """Pattern-specific features based on professional definitions"""
        df = self.df
        close, high, low = df['close'].values, df['high'].values, df['low'].values
        
        # Support and resistance features
        self.features['resistance_level'] = talib.MAX(high, timeperiod=window)
        self.features['support_level'] = talib.MIN(low, timeperiod=window)
        self.features['price_vs_resistance'] = self._safe_divide(
            self.features['resistance_level'] - close, self.features['resistance_level'], 0
        )
        self.features['price_vs_support'] = self._safe_divide(
            close - self.features['support_level'], close, 0
        )
        
        # Trend strength and direction
        self.features['trend_strength'] = talib.LINEARREG_SLOPE(close, timeperiod=window)
        self.features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Pattern probability features
        self.features['triangle_probability'] = self._calculate_triangle_probability(high, low, window)
        self.features['reversal_probability'] = self._calculate_reversal_probability(high, low, close, window)
        self.features['consolidation_strength'] = self._calculate_consolidation(high, low, window)
        
        return self
        
    def _calculate_triangle_probability(self, highs, lows, window):
        """Calculate probability of triangle patterns"""
        result = np.zeros(len(highs))
        for i in range(window, len(highs)):
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            
            if len(recent_highs) < 10:
                continue
                
            # Calculate convergence
            high_slope, _ = np.polyfit(range(len(recent_highs)), recent_highs, 1)
            low_slope, _ = np.polyfit(range(len(recent_lows)), recent_lows, 1)
            
            # Triangle probability based on convergence
            if high_slope < 0 and low_slope > 0:  # Symmetrical
                result[i] = 0.7
            elif high_slope < 0 and abs(low_slope) < 0.001:  # Descending
                result[i] = 0.6
            elif low_slope > 0 and abs(high_slope) < 0.001:  # Ascending
                result[i] = 0.6
            else:
                result[i] = 0.1
                
        return result
        
    def _calculate_reversal_probability(self, highs, lows, closes, window):
        """Calculate probability of reversal patterns"""
        result = np.zeros(len(highs))
        for i in range(window, len(highs)):
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            recent_closes = closes[i-window:i]
            
            # Check for double top/bottom characteristics
            if len(recent_highs) >= 15:
                # Double top check
                first_half_high = np.max(recent_highs[:len(recent_highs)//2])
                second_half_high = np.max(recent_highs[len(recent_highs)//2:])
                high_similarity = 1 - abs(first_half_high - second_half_high) / ((first_half_high + second_half_high) / 2)
                
                # Double bottom check
                first_half_low = np.min(recent_lows[:len(recent_lows)//2])
                second_half_low = np.min(recent_lows[len(recent_lows)//2:])
                low_similarity = 1 - abs(first_half_low - second_half_low) / ((first_half_low + second_half_low) / 2)
                
                result[i] = max(high_similarity, low_similarity) * 0.8
                
        return result
        
    def _calculate_consolidation(self, highs, lows, window):
        """Calculate consolidation strength"""
        result = np.zeros(len(highs))
        for i in range(window, len(highs)):
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            
            high_range = np.std(recent_highs) / np.mean(recent_highs)
            low_range = np.std(recent_lows) / np.mean(recent_lows)
            
            # Lower volatility indicates stronger consolidation
            consolidation = 1 - (high_range + low_range) / 2
            result[i] = max(0, min(1, consolidation))
            
        return result
        
    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division avoiding division by zero"""
        return np.where(denominator == 0, default, numerator / denominator)
        
    def get_features_df(self):
        """Get final features DataFrame with robust data cleaning"""
        features_df = pd.DataFrame(self.features)
        
        # Robust data cleaning
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values strategically
        for col in features_df.columns:
            if features_df[col].isna().any():
                # For technical indicators, use forward then backward fill
                # FIXED: Pandas 2.0+ compatible (no method= parameter)
                features_df[col] = features_df[col].ffill().bfill()
        
        # Final cleanup of any remaining NaN
        features_df = features_df.fillna(0)
        
        # Remove any remaining infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Optimize data types for GPU processing
        for col in features_df.select_dtypes(include=['float64']).columns:
            features_df[col] = features_df[col].astype(np.float32)
            
        return features_df


class EnhancedForexPatternClassifier:
    """Enhanced classifier with GPU optimization and professional features"""
    
    def __init__(self, output_dir=None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pattern_detector = AdvancedPatternDetector()
        self.output_dir = output_dir if output_dir else '.'
        
    def prepare_data(self, df: pd.DataFrame, pattern_labels: pd.Series):
        """Prepare data with enhanced feature extraction"""
        print("Extracting advanced features...")
        
        # Enhanced feature extraction
        extractor = (EnhancedFeatureExtractor(df)
                    .add_advanced_price_features()
                    .add_professional_technical_indicators()
                    .add_pattern_specific_features())
        
        features_df = extractor.get_features_df()
        self.feature_names = features_df.columns.tolist()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(pattern_labels)
        
        return features_df, encoded_labels
        
    def train(self, df: pd.DataFrame, pattern_labels: pd.Series, test_size=0.2, optimize_hyperparams=True):
        """Train with GPU optimization and hyperparameter tuning"""
        # Prepare data
        X, y = self.prepare_data(df, pattern_labels)
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            best_params = {
                'n_estimators': 500,  # Reduced for faster training
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        # GPU-optimized XGBoost model (falls back to CPU if GPU unavailable)
        try:
            # Try GPU first with gpu_hist (fastest)
            self.model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='mlogloss',
                tree_method='gpu_hist',  # GPU-accelerated histogram algorithm
                device='cuda',           # Use CUDA GPU
                predictor='gpu_predictor'  # GPU prediction
            )
            print("✅ Training model with GPU acceleration (gpu_hist)...")
        except Exception as e:
            # Fall back to CPU if GPU fails
            print(f"⚠️  GPU not available ({e}), using CPU...")
            self.model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='mlogloss',
                tree_method='hist',
                device='cpu'
            )
        
        print("Training model...")
        # JAVÍTOTT: early_stopping_rounds paraméter eltávolítva
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=True
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Plot feature importance
        self._plot_feature_importance()
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        return self.model
        
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using randomized search"""
        param_dist = {
            'n_estimators': [300, 500, 800],  # Reduced for faster optimization
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
        }
        
        # Try GPU for hyperparameter search (much faster)
        try:
            model = xgb.XGBClassifier(
                random_state=42,
                tree_method='gpu_hist',
                device='cuda',
                predictor='gpu_predictor'
            )
            print("🔍 Hyperparameter search using GPU...")
        except:
            model = xgb.XGBClassifier(
                random_state=42,
                tree_method='hist',
                device='cpu'
            )
            print("🔍 Hyperparameter search using CPU...")
        
        search = RandomizedSearchCV(
            model, param_dist, n_iter=10, cv=3, scoring='accuracy',  # Reduced iterations
            random_state=42, n_jobs=1, verbose=1
        )
        
        search.fit(X, y)
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        return search.best_params_
        
    def _plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.model is None:
            return
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTop {top_n} features saved to '{save_path}'")
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to '{save_path}'")
        
    def predict(self, df: pd.DataFrame):
        """Predict patterns with GPU acceleration"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
            
        # Extract features
        extractor = (EnhancedFeatureExtractor(df)
                    .add_advanced_price_features()
                    .add_professional_technical_indicators()
                    .add_pattern_specific_features())
        
        features_df = extractor.get_features_df()
        features_scaled = self.scaler.transform(features_df)
        
        # Predict with GPU
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
        
    def save_model(self, filepath: str):
        """Save model with all components"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load model with all components"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        # Force model to use CPU for predictions (avoid GPU/CPU mismatch)
        if hasattr(self.model, 'set_params'):
            try:
                self.model.set_params(device='cpu')
            except:
                pass  # Ignore if parameters not supported
        
        print(f"Model loaded from {filepath} (device: CPU)")


def load_and_preprocess_data(csv_file_path, sample_size=None):
    """
    Load and preprocess data from CSV file
    Expected columns: datetime, open, high, low, close, volume (optional)
    """
    print(f"Loading data from {csv_file_path}...")
    
    # Load CSV file
    df = pd.read_csv(csv_file_path)
    print(f"Original data shape: {df.shape}")
    
    # Sample data if specified
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} rows...")
        df = df.sample(sample_size, random_state=42).sort_index()
    
    # Identify datetime column
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower()]
    if datetime_cols:
        datetime_col = datetime_cols[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)
        print(f"Using datetime column: {datetime_col}")
    else:
        print("No datetime column found, using index as time reference")
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean data
    print("Cleaning data...")
    df_clean = df.copy()
    
    # Remove rows with invalid prices
    initial_count = len(df_clean)
    df_clean = df_clean[
        (df_clean['high'] >= df_clean['low']) & 
        (df_clean['high'] >= df_clean['close']) & 
        (df_clean['high'] >= df_clean['open']) &
        (df_clean['low'] <= df_clean['close']) & 
        (df_clean['low'] <= df_clean['open'])
    ]
    
    # Remove extreme outliers
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        q1 = df_clean[col].quantile(0.01)
        q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q3)]
    
    final_count = len(df_clean)
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {initial_count - final_count} invalid rows ({((initial_count - final_count) / initial_count * 100):.1f}%)")
    
    return df_clean


def create_labels_from_data(df, pattern_detector):
    """
    Create pattern labels using the advanced pattern detector
    This uses rule-based detection to create training labels
    Now supports confidence scores from improved pattern detection
    """
    print("Creating pattern labels using advanced detection...")
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else None
    
    patterns = ['no_pattern'] * len(df)
    confidences = [0.0] * len(df)  # Store confidence scores
    
    # Minimum points needed for pattern detection
    min_points = 50
    
    total_rows = len(df)
    processed_rows = 0
    
    print("Pattern detection progress:")
    
    for i in range(min_points, total_rows):
        # Progress reporting
        if i % max(1, (total_rows - min_points) // 100) == 0 or i == min_points or i == total_rows - 1:
            progress = (i - min_points) / (total_rows - min_points) * 100
            print(f"  Progress: {progress:.1f}% ({i}/{total_rows})", end='\r')
        
        window_high = highs[i-min_points:i]
        window_low = lows[i-min_points:i]
        window_close = closes[i-min_points:i]
        window_volume = volumes[i-min_points:i] if volumes is not None else None
        
        # Check patterns in order of complexity
        # All methods now return (detected: bool, confidence: float, [optional: type])
        detected, conf = pattern_detector.detect_head_shoulders(window_high, window_low, window_volume)
        if detected:
            patterns[i] = 'head_and_shoulders'
            confidences[i] = conf
            continue
            
        detected, conf = pattern_detector.detect_double_top(window_high, window_low, window_volume)
        if detected:
            patterns[i] = 'double_top'
            confidences[i] = conf
            continue
            
        detected, conf = pattern_detector.detect_double_bottom(window_low, window_high, window_volume)
        if detected:
            patterns[i] = 'double_bottom'
            confidences[i] = conf
            continue
            
        detected, conf = pattern_detector.detect_ascending_triangle(window_high, window_low, window_volume)
        if detected:
            patterns[i] = 'ascending_triangle'
            confidences[i] = conf
            continue
            
        detected, conf = pattern_detector.detect_descending_triangle(window_high, window_low, window_volume)
        if detected:
            patterns[i] = 'descending_triangle'
            confidences[i] = conf
            continue
            
        detected, conf = pattern_detector.detect_symmetrical_triangle(window_high, window_low, window_volume)
        if detected:
            patterns[i] = 'symmetrical_triangle'
            confidences[i] = conf
            continue
            
        # OPTIMIZED v2: Cup & Handle DISABLED (23% worse drawdown, removed from trading)
        # detected, conf = pattern_detector.detect_cup_and_handle(window_high, window_low, window_close, window_volume)
        # if detected:
        #     patterns[i] = 'cup_and_handle'
        #     confidences[i] = conf
        #     continue
            
        detected, conf, wedge_type = pattern_detector.detect_wedge(window_high, window_low, window_volume)
        if detected:
            patterns[i] = f'wedge_{wedge_type}'  # wedge_rising or wedge_falling
            confidences[i] = conf
            continue
            
        detected, conf, flag_type = pattern_detector.detect_flag(window_high, window_low, window_close, window_volume)
        if detected:
            patterns[i] = f'flag_{flag_type}'  # flag_bullish or flag_bearish
            confidences[i] = conf
            continue
    
    print("\nPattern detection completed!")
    
    # Add confidence column to dataframe
    df['pattern_confidence'] = confidences
    
    # Analyze pattern distribution
    pattern_counts = pd.Series(patterns).value_counts()
    print("\nPattern distribution in data:")
    for pattern, count in pattern_counts.items():
        percentage = count / len(patterns) * 100
        # Calculate average confidence for this pattern
        pattern_mask = pd.Series(patterns) == pattern
        avg_conf = pd.Series(confidences)[pattern_mask].mean()
        if pattern != 'no_pattern':
            print(f"  {pattern}: {count} ({percentage:.2f}%) - avg confidence: {avg_conf:.3f}")
        else:
            print(f"  {pattern}: {count} ({percentage:.2f}%)")
    
    return patterns


def main():
    """Main function to train model on CSV data"""
    print("=== ENHANCED FOREX PATTERN CLASSIFIER ===")
    print("Training on real CSV data with GPU acceleration")
    
    # Configuration
    csv_file_path = 'data/DOGEUSDT-trades-2025-08.csv'  # Change this to your CSV file path
    sample_size = None  # Use all data for full month
    
    start_time = time.time()
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(csv_file_path, sample_size)
        
        # Create pattern labels using advanced detection
        pattern_detector = AdvancedPatternDetector()
        pattern_labels = create_labels_from_data(df, pattern_detector)
        pattern_series = pd.Series(pattern_labels, index=df.index)
        
        # Check if we have enough patterns for training
        pattern_counts = pattern_series.value_counts()
        min_patterns = 30  # Reduced minimum for smaller datasets
        valid_patterns = pattern_counts[pattern_counts >= min_patterns].index.tolist()
        
        if 'no_pattern' in valid_patterns:
            valid_patterns.remove('no_pattern')
        
        if len(valid_patterns) < 2:
            print("Warning: Not enough pattern diversity for training.")
            print("Consider using a larger dataset or different pattern detection parameters.")
            return
        
        # Filter data to only include valid patterns
        mask = pattern_series.isin(valid_patterns + ['no_pattern'])
        df_filtered = df[mask]
        patterns_filtered = pattern_series[mask]
        
        print(f"\nTraining data shape: {df_filtered.shape}")
        print(f"Patterns for training: {valid_patterns}")
        
        # Initialize and train enhanced classifier
        classifier = EnhancedForexPatternClassifier()
        
        print("\n--- Training Enhanced Model ---")
        model = classifier.train(df_filtered, patterns_filtered, optimize_hyperparams=True)
        
        # Save the enhanced model
        print("\n--- Saving Enhanced Model ---")
        classifier.save_model('enhanced_forex_pattern_model.pkl')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n=== Enhanced Model Training Complete ===")
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print("Model features:")
        print(f"- Professional pattern definitions")
        print(f"- GPU acceleration enabled") 
        print(f"- Advanced technical indicators")
        print(f"- Hyperparameter optimization")
        print(f"- Feature importance visualization")
        
        return classifier, df_filtered
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


class BacktestingEngine:
    """
    Backtesting engine for pattern-based trading strategies
    Tests profitability of detected patterns
    """
    
    def __init__(self, initial_capital=10000, risk_per_trade=0.02, take_profit_multiplier=2.0):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.take_profit_multiplier = take_profit_multiplier  # 2:1 reward-to-risk
        self.trades = []
        self.equity_curve = []
        
    def calculate_pattern_targets(self, pattern_type, entry_price, high, low, recent_data=None):
        """
        Calculate stop loss and take profit based on pattern type
        V2.3: LONG-ONLY strategy - only take LONG trades when pattern aligns with trend
        """
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
        
        # V2.3: LONG-ONLY ALIGNED STRATEGY
        # Winners from 10K-bar test (all LONG):
        # - ascending_triangle in uptrend (54.3% win, +0.069%)
        # - descending_triangle in downtrend (52.8% win, +0.036%)
        # - wedge_falling in downtrend (58.4% win, +0.051%)
        if recent_data is not None and len(recent_data) >= 20:
            from numpy import polyfit, arange
            closes = recent_data['close'].values[-20:]
            x = arange(len(closes))
            slope = polyfit(x, closes, 1)[0]
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
        
        stop_loss = entry_price * (1 - params['sl_pct'])
        take_profit = entry_price * (1 + params['tp_pct'])
            
        return stop_loss, take_profit, direction
    
    def run_backtest(self, df, predictions, probabilities, pattern_strength_scores=None):
        """
        Run backtest on predicted patterns
        """
        print("\n=== RUNNING BACKTEST ===")
        
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [capital]
        
        active_trades = []
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            pattern = predictions[i]
            
            # Skip 'no_pattern'
            if pattern == 'no_pattern':
                self.equity_curve.append(capital)
                continue
            
            # Check pattern strength if available
            if pattern_strength_scores is not None and pattern_strength_scores[i] < 0.6:
                self.equity_curve.append(capital)
                continue
            
            # Get prediction probability
            pattern_prob = np.max(probabilities[i])
            if pattern_prob < 0.6:  # Minimum confidence threshold
                self.equity_curve.append(capital)
                continue
            
            # Calculate position size
            entry_price = current_row['close']
            
            # Pass recent data for trend calculation (V2.3)
            recent_data = df.iloc[max(0, i-30):i+1]  # Last 30 bars including current
            
            stop_loss, take_profit, direction = self.calculate_pattern_targets(
                pattern, entry_price, current_row['high'], current_row['low'], recent_data
            )
            
            # Skip misaligned setups (V2.3)
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
                'status': 'open'
            }
            
            active_trades.append(trade)
            
            # Check active trades for exit
            closed_trades = []
            for trade_idx, trade in enumerate(active_trades):
                if trade['status'] == 'open':
                    # Check if SL or TP hit
                    if trade['direction'] == 'long':
                        if current_row['low'] <= trade['stop_loss']:
                            # Stop loss hit
                            pnl = -trade['risk_amount']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['high'] >= trade['take_profit']:
                            # Take profit hit
                            pnl = trade['risk_amount'] * self.take_profit_multiplier
                            trade['exit_price'] = trade['take_profit']
                            trade['exit_reason'] = 'take_profit'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        else:
                            continue
                    else:  # short
                        if current_row['high'] >= trade['stop_loss']:
                            # Stop loss hit
                            pnl = -trade['risk_amount']
                            trade['exit_price'] = trade['stop_loss']
                            trade['exit_reason'] = 'stop_loss'
                            trade['status'] = 'closed'
                            closed_trades.append(trade_idx)
                        elif current_row['low'] <= trade['take_profit']:
                            # Take profit hit
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
            
            # Remove closed trades from active
            for idx in sorted(closed_trades, reverse=True):
                active_trades.pop(idx)
            
            self.equity_curve.append(capital)
        
        return self.analyze_results()
    
    def analyze_results(self):
        """
        Analyze backtest results and generate metrics
        """
        if not self.trades:
            print("No trades executed in backtest")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
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
        
        # Pattern-specific analysis
        pattern_performance = trades_df.groupby('pattern').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).round(2)
        
        pattern_win_rates = trades_df.groupby('pattern').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(2)
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
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
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _print_results(self, results):
        """Print formatted backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print("-"*60)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print("-"*60)
        print(f"Average Win: ${results['avg_win']:,.2f}")
        print(f"Average Loss: ${results['avg_loss']:,.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print("-"*60)
        print("\nPattern Performance:")
        print(results['pattern_performance'])
        print("\nPattern Win Rates (%):")
        print(results['pattern_win_rates'])
        print("="*60)
    
    def plot_equity_curve(self, save_path='equity_curve.png'):
        """Plot equity curve"""
        plt.figure(figsize=(14, 7))
        plt.plot(self.equity_curve, linewidth=2, color='#2E86AB')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        plt.title('Equity Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Capital ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Equity curve saved to {save_path}")


class MultiTimeframeAnalyzer:
    """
    Analyze patterns across multiple timeframes
    Provides consensus-based signals
    """
    
    def __init__(self, timeframes=['1min', '5min', '15min', '1h']):
        self.timeframes = timeframes
        self.classifiers = {}
        
    def resample_data(self, df, timeframe):
        """
        Resample data to different timeframe
        """
        # Map timeframe strings to pandas offset aliases
        tf_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        freq = tf_map.get(timeframe, '1T')
        
        # Assume df has datetime index or column
        df_copy = df.copy()
        
        # Find datetime column
        datetime_cols = [col for col in df_copy.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols and datetime_cols[0] in df_copy.columns:
            df_copy['datetime'] = pd.to_datetime(df_copy[datetime_cols[0]])
            df_copy = df_copy.set_index('datetime')
        
        # Resample OHLCV data
        resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df_copy.columns else lambda x: 0
        }).dropna()
        
        return resampled
    
    def analyze_multi_timeframe(self, df, classifier):
        """
        Analyze patterns across multiple timeframes
        Returns consensus signals
        """
        print(f"\n=== MULTI-TIMEFRAME ANALYSIS ===")
        
        results = {}
        
        for tf in self.timeframes:
            print(f"Analyzing {tf} timeframe...")
            
            # Resample data
            df_resampled = self.resample_data(df, tf)
            
            if len(df_resampled) < 100:
                print(f"  Not enough data for {tf} timeframe")
                continue
            
            # Predict patterns
            predictions, probabilities = classifier.predict(df_resampled)
            
            # Count patterns
            pattern_counts = pd.Series(predictions).value_counts()
            
            results[tf] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'pattern_counts': pattern_counts,
                'df': df_resampled
            }
            
            print(f"  Patterns found: {len(predictions[predictions != 'no_pattern'])}")
        
        # Generate consensus signals
        consensus = self._generate_consensus(results)
        
        self._print_consensus(consensus)
        
        return results, consensus
    
    def _generate_consensus(self, results):
        """
        Generate consensus signals from multiple timeframes
        """
        consensus_signals = []
        
        # Get patterns from each timeframe
        all_patterns = {}
        for tf, data in results.items():
            patterns = pd.Series(data['predictions'])
            all_patterns[tf] = patterns[patterns != 'no_pattern']
        
        # Find patterns that appear in multiple timeframes
        if all_patterns:
            # Get most common patterns across timeframes
            all_pattern_list = []
            for patterns in all_patterns.values():
                all_pattern_list.extend(patterns.tolist())
            
            if all_pattern_list:
                pattern_freq = pd.Series(all_pattern_list).value_counts()
                
                for pattern, count in pattern_freq.items():
                    if count >= 2:  # Pattern appears in at least 2 timeframes
                        consensus_signals.append({
                            'pattern': pattern,
                            'timeframes': count,
                            'strength': count / len(self.timeframes)
                        })
        
        return sorted(consensus_signals, key=lambda x: x['strength'], reverse=True)
    
    def _print_consensus(self, consensus):
        """Print consensus signals"""
        print("\n" + "="*60)
        print("MULTI-TIMEFRAME CONSENSUS")
        print("="*60)
        
        if not consensus:
            print("No consensus patterns found across timeframes")
        else:
            for signal in consensus:
                print(f"Pattern: {signal['pattern']}")
                print(f"  Appears in {signal['timeframes']} timeframes")
                print(f"  Strength: {signal['strength']*100:.1f}%")
                print("-"*60)


class PatternStrengthScorer:
    """
    Calculate pattern strength and reliability scores
    """
    
    @staticmethod
    def calculate_pattern_strength(df, pattern_type, pattern_index, window=50):
        """
        Calculate pattern strength score (0-1)
        Based on multiple factors:
        - Volume confirmation
        - Trend alignment
        - Pattern clarity
        - Market conditions
        """
        if pattern_index < window:
            return 0.5  # Default score
        
        scores = []
        
        # 1. Volume confirmation (20% weight)
        if 'volume' in df.columns:
            volume_score = PatternStrengthScorer._volume_confirmation(
                df, pattern_index, window
            )
            scores.append(volume_score * 0.2)
        
        # 2. Trend alignment (25% weight)
        trend_score = PatternStrengthScorer._trend_alignment(
            df, pattern_type, pattern_index, window
        )
        scores.append(trend_score * 0.25)
        
        # 3. Pattern clarity (30% weight)
        clarity_score = PatternStrengthScorer._pattern_clarity(
            df, pattern_index, window
        )
        scores.append(clarity_score * 0.3)
        
        # 4. Market volatility (15% weight)
        volatility_score = PatternStrengthScorer._volatility_check(
            df, pattern_index, window
        )
        scores.append(volatility_score * 0.15)
        
        # 5. Support/Resistance alignment (10% weight)
        sr_score = PatternStrengthScorer._support_resistance_alignment(
            df, pattern_index, window
        )
        scores.append(sr_score * 0.1)
        
        total_score = sum(scores)
        return np.clip(total_score, 0, 1)
    
    @staticmethod
    def _volume_confirmation(df, index, window):
        """Check if volume confirms the pattern"""
        if 'volume' not in df.columns:
            return 0.5
        
        recent_volume = df['volume'].iloc[max(0, index-10):index].mean()
        avg_volume = df['volume'].iloc[max(0, index-window):index].mean()
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = recent_volume / avg_volume
        
        # Higher volume is better (up to 2x average)
        return min(volume_ratio / 2.0, 1.0)
    
    @staticmethod
    def _trend_alignment(df, pattern_type, index, window):
        """Check if pattern aligns with trend"""
        closes = df['close'].iloc[max(0, index-window):index].values
        
        if len(closes) < 10:
            return 0.5
        
        # Calculate trend
        slope, _ = np.polyfit(range(len(closes)), closes, 1)
        trend = 'up' if slope > 0 else 'down'
        
        # UNIFIED PATTERN CLASSIFICATION (matches backtest_with_hedging.py)
        # Bullish patterns (LONG-ONLY optimized - Cup & Handle removed)
        bullish_patterns = ['ascending', 'symmetrical', 'double_bottom', 'flag_bullish']
        # Bearish patterns (skip in LONG-ONLY strategy)
        bearish_patterns = ['descending', 'wedge', 'double_top', 'head_shoulders']
        
        # Check if pattern matches any bullish/bearish pattern
        is_bullish = any(bp in pattern_type for bp in bullish_patterns)
        is_bearish = any(bp in pattern_type for bp in bearish_patterns)
        
        if is_bullish and trend == 'up':
            return 0.9
        elif is_bearish and trend == 'down':
            return 0.9
        elif pattern_type == 'symmetrical_triangle' or pattern_type.startswith('wedge'):
            return 0.7  # Neutral patterns
        else:
            return 0.3  # Counter-trend
    
    @staticmethod
    def _pattern_clarity(df, index, window):
        """Measure how clear/well-formed the pattern is"""
        highs = df['high'].iloc[max(0, index-window):index].values
        lows = df['low'].iloc[max(0, index-window):index].values
        
        if len(highs) < 10:
            return 0.5
        
        # Check price action consistency
        high_volatility = np.std(highs) / np.mean(highs)
        low_volatility = np.std(lows) / np.mean(lows)
        
        # Lower volatility = clearer pattern
        avg_volatility = (high_volatility + low_volatility) / 2
        clarity = 1 - min(avg_volatility * 10, 1)
        
        return clarity
    
    @staticmethod
    def _volatility_check(df, index, window):
        """Check if volatility is in acceptable range"""
        closes = df['close'].iloc[max(0, index-window):index].values
        
        if len(closes) < 10:
            return 0.5
        
        volatility = np.std(closes) / np.mean(closes)
        
        # Optimal volatility range: 0.01 - 0.05
        if 0.01 <= volatility <= 0.05:
            return 1.0
        elif volatility < 0.01:
            return 0.5  # Too low volatility
        else:
            return max(0.2, 1 - (volatility - 0.05) * 10)  # Too high volatility
    
    @staticmethod
    def _support_resistance_alignment(df, index, window):
        """Check alignment with support/resistance levels"""
        highs = df['high'].iloc[max(0, index-window):index].values
        lows = df['low'].iloc[max(0, index-window):index].values
        current_price = df['close'].iloc[index]
        
        if len(highs) < 10:
            return 0.5
        
        # Find key levels
        resistance = np.max(highs)
        support = np.min(lows)
        
        # Check if price is near key levels
        price_range = resistance - support
        if price_range == 0:
            return 0.5
        
        dist_to_resistance = abs(current_price - resistance) / price_range
        dist_to_support = abs(current_price - support) / price_range
        
        # Better score if near key levels
        min_dist = min(dist_to_resistance, dist_to_support)
        
        return 1 - min_dist


class AlertSystem:
    """
    Alert system for pattern detection
    Logs alerts and displays them in console
    """
    
    def __init__(self, alert_threshold=0.75, enable_log=True):
        self.alert_threshold = alert_threshold
        self.enable_log = enable_log
        self.alerts = []
        self.log_file = 'pattern_alerts.json'
        
    def check_and_alert(self, pattern, strength_score, probability, price, timestamp=None):
        """
        Check if pattern meets alert criteria and send notification
        """
        if strength_score >= self.alert_threshold and probability >= 0.7:
            alert = {
                'timestamp': timestamp or datetime.now().isoformat(),
                'pattern': pattern,
                'strength_score': float(strength_score),
                'probability': float(probability),
                'price': float(price),
                'alert_level': self._get_alert_level(strength_score, probability)
            }
            
            self.alerts.append(alert)
            
            # Log alert
            if self.enable_log:
                self._log_alert(alert)
            
            # Print to console
            self._print_alert(alert)
            
            return True
        
        return False
    
    def _get_alert_level(self, strength, probability):
        """Determine alert priority level"""
        score = (strength + probability) / 2
        
        if score >= 0.9:
            return 'CRITICAL'
        elif score >= 0.8:
            return 'HIGH'
        elif score >= 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _print_alert(self, alert):
        """Print alert to console"""
        print(f"\n{'='*60}")
        print(f"🚨 PATTERN ALERT - {alert['alert_level']}")
        print(f"{'='*60}")
        print(f"Pattern: {alert['pattern']}")
        print(f"Strength: {alert['strength_score']:.2%}")
        print(f"Probability: {alert['probability']:.2%}")
        print(f"Price: ${alert['price']:.4f}")
        print(f"Time: {alert['timestamp']}")
        print(f"{'='*60}\n")
    
    def _log_alert(self, alert):
        """Log alert to JSON file"""
        try:
            # Load existing alerts
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            # Add new alert
            alerts.append(alert)
            
            # Save back
            with open(self.log_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            print(f"Failed to log alert: {e}")
    
    def get_alert_summary(self):
        """Get summary of all alerts"""
        if not self.alerts:
            return "No alerts generated"
        
        df = pd.DataFrame(self.alerts)
        
        summary = {
            'total_alerts': len(df),
            'by_pattern': df['pattern'].value_counts().to_dict(),
            'by_level': df['alert_level'].value_counts().to_dict(),
            'avg_strength': df['strength_score'].mean(),
            'avg_probability': df['probability'].mean()
        }
        
        return summary
    
    def print_alert_summary(self):
        """Print formatted alert summary"""
        summary = self.get_alert_summary()
        
        if isinstance(summary, str):
            print(summary)
            return
        
        print("\n" + "="*60)
        print("ALERT SYSTEM SUMMARY")
        print("="*60)
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Average Strength: {summary['avg_strength']:.2%}")
        print(f"Average Probability: {summary['avg_probability']:.2%}")
        print("\nAlerts by Pattern:")
        for pattern, count in summary['by_pattern'].items():
            print(f"  {pattern}: {count}")
        print("\nAlerts by Level:")
        for level, count in summary['by_level'].items():
            print(f"  {level}: {count}")
        print("="*60)


class InteractiveDashboard:
    """
    Interactive dashboard using Plotly for pattern visualization
    """
    
    def __init__(self):
        self.figures = []
        # Create HTML output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.html_dir = f'html_{timestamp}'
        os.makedirs(self.html_dir, exist_ok=True)
        
    def create_candlestick_chart(self, df, predictions, probabilities, 
                                 strength_scores=None, save_html=True):
        """
        Create interactive candlestick chart with pattern annotations
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None
        
        # Prepare data
        df_viz = df.copy()
        if not isinstance(df_viz.index, pd.DatetimeIndex):
            # Try to create datetime index
            datetime_cols = [col for col in df_viz.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols:
                df_viz.index = pd.to_datetime(df_viz[datetime_cols[0]])
            else:
                df_viz.index = pd.date_range(start='2025-01-01', periods=len(df_viz), freq='1min')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Action with Patterns', 'Pattern Probability', 'Pattern Strength'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_viz.index,
                open=df_viz['open'],
                high=df_viz['high'],
                low=df_viz['low'],
                close=df_viz['close'],
                name='Price',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ),
            row=1, col=1
        )
        
        # Add pattern annotations
        pattern_colors = {
            'ascending_triangle': '#4CAF50',
            'descending_triangle': '#F44336',
            'symmetrical_triangle': '#2196F3',
            'double_top': '#E91E63',
            'double_bottom': '#9C27B0',
            'head_and_shoulders': '#FF9800',
            'cup_and_handle': '#00BCD4',
            'wedge': '#795548',
            'flag': '#607D8B'
        }
        
        for i, pattern in enumerate(predictions):
            if pattern != 'no_pattern':
                color = pattern_colors.get(pattern, '#FFC107')
                prob = probabilities[i].max()
                
                annotation_text = f"{pattern}<br>Prob: {prob:.1%}"
                if strength_scores is not None:
                    annotation_text += f"<br>Strength: {strength_scores[i]:.1%}"
                
                fig.add_annotation(
                    x=df_viz.index[i],
                    y=df_viz['high'].iloc[i],
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=0,
                    ay=-40,
                    bgcolor=color,
                    opacity=0.8,
                    font=dict(size=10, color='white'),
                    row=1, col=1
                )
        
        # Pattern probability chart
        max_probs = [p.max() for p in probabilities]
        fig.add_trace(
            go.Scatter(
                x=df_viz.index,
                y=max_probs,
                mode='lines',
                name='Max Probability',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.3)'
            ),
            row=2, col=1
        )
        
        # Add probability threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Threshold", row=2, col=1)
        
        # Pattern strength chart (if available)
        if strength_scores is not None:
            fig.add_trace(
                go.Scatter(
                    x=df_viz.index,
                    y=strength_scores,
                    mode='lines',
                    name='Pattern Strength',
                    line=dict(color='#4CAF50', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(76, 175, 80, 0.3)'
                ),
                row=3, col=1
            )
            
            # Add strength threshold line
            fig.add_hline(y=0.75, line_dash="dash", line_color="orange",
                         annotation_text="Alert Threshold", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title='Pattern Detection Dashboard',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        fig.update_yaxes(title_text="Strength", row=3, col=1)
        
        if save_html:
            filename = f'{self.html_dir}/pattern_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(filename)
            print(f"\nInteractive dashboard saved to: {filename}")
        
        return fig
    
    def create_pattern_distribution_chart(self, predictions, save_html=True):
        """Create pattern distribution pie chart"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        pattern_counts = pd.Series(predictions).value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=pattern_counts.index,
                values=pattern_counts.values,
                hole=0.3,
                marker=dict(
                    colors=['#4CAF50', '#F44336', '#2196F3', '#E91E63', 
                           '#9C27B0', '#FF9800', '#00BCD4', '#795548', '#607D8B']
                )
            )
        ])
        
        fig.update_layout(
            title='Pattern Distribution',
            template='plotly_dark'
        )
        
        if save_html:
            filename = f'{self.html_dir}/pattern_distribution.html'
            fig.write_html(filename)
            print(f"Pattern distribution chart saved to: {filename}")
        
        return fig
    
    def create_backtest_dashboard(self, backtest_results, save_html=True):
        """Create comprehensive backtest dashboard"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None
        
        if backtest_results is None:
            return None
        
        trades_df = backtest_results['trades_df']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curve', 'Win/Loss Distribution', 
                          'Pattern Performance', 'Trade Timeline'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Equity curve
        equity = backtest_results.get('equity_curve', [])
        if equity:
            fig.add_trace(
                go.Scatter(
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#4CAF50', width=3)
                ),
                row=1, col=1
            )
        
        # Win/Loss distribution
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        fig.add_trace(
            go.Bar(
                x=['Wins', 'Losses'],
                y=[len(wins), len(losses)],
                marker_color=['#4CAF50', '#F44336'],
                name='Count'
            ),
            row=1, col=2
        )
        
        # Pattern performance
        pattern_pnl = trades_df.groupby('pattern')['pnl'].sum().sort_values()
        
        fig.add_trace(
            go.Bar(
                x=pattern_pnl.values,
                y=pattern_pnl.index,
                orientation='h',
                marker_color=['#F44336' if x < 0 else '#4CAF50' for x in pattern_pnl.values],
                name='P&L'
            ),
            row=2, col=1
        )
        
        # Trade timeline
        fig.add_trace(
            go.Scatter(
                x=trades_df.index,
                y=trades_df['pnl'].cumsum(),
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#2196F3', width=2),
                marker=dict(
                    size=8,
                    color=['#4CAF50' if x > 0 else '#F44336' for x in trades_df['pnl']],
                    line=dict(width=1, color='white')
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Backtest Results - Total P&L: ${backtest_results["total_pnl"]:,.2f}',
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        if save_html:
            filename = f'{self.html_dir}/backtest_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(filename)
            print(f"Backtest dashboard saved to: {filename}")
        
        return fig


class MLflowTracker:
    """
    MLflow integration for model versioning and experiment tracking
    """
    
    def __init__(self, experiment_name="forex_pattern_detection"):
        self.experiment_name = experiment_name
        self.mlflow_available = False
        
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
            
            # Set experiment
            self.mlflow.set_experiment(experiment_name)
            print(f"MLflow experiment set: {experiment_name}")
            
        except ImportError:
            print("MLflow not available. Install with: pip install mlflow")
    
    def log_training_run(self, params, metrics, model=None, artifacts=None):
        """
        Log a training run to MLflow
        """
        if not self.mlflow_available:
            return
        
        with self.mlflow.start_run():
            # Log parameters
            for key, value in params.items():
                self.mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(key, value)
            
            # Log model
            if model is not None:
                self.mlflow.sklearn.log_model(model, "model")
            
            # Log artifacts
            if artifacts is not None:
                for artifact_path in artifacts:
                    if os.path.exists(artifact_path):
                        self.mlflow.log_artifact(artifact_path)
            
            run_id = self.mlflow.active_run().info.run_id
            print(f"MLflow run logged: {run_id}")
            
            return run_id
    
    def log_backtest_results(self, backtest_results):
        """
        Log backtest results to MLflow
        """
        if not self.mlflow_available or backtest_results is None:
            return
        
        with self.mlflow.start_run():
            # Log backtest metrics
            self.mlflow.log_metric("total_trades", backtest_results['total_trades'])
            self.mlflow.log_metric("win_rate", backtest_results['win_rate'])
            self.mlflow.log_metric("total_pnl", backtest_results['total_pnl'])
            self.mlflow.log_metric("profit_factor", backtest_results['profit_factor'])
            self.mlflow.log_metric("max_drawdown", backtest_results['max_drawdown'])
            self.mlflow.log_metric("sharpe_ratio", backtest_results['sharpe_ratio'])
            self.mlflow.log_metric("return_pct", backtest_results['return_pct'])
            
            # Log pattern performance
            for pattern, win_rate in backtest_results['pattern_win_rates'].items():
                self.mlflow.log_metric(f"win_rate_{pattern}", win_rate)
            
            print("Backtest results logged to MLflow")


if __name__ == "__main__":
    """
    Interactive dashboard using Plotly for pattern visualization
    """
    
    def __init__(self):
        self.figures = []
        
    def create_candlestick_chart(self, df, predictions, probabilities, 
                                 strength_scores=None, save_html=True):
        """
        Create interactive candlestick chart with pattern annotations
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None
        
        # Prepare data
        df_viz = df.copy()
        if not isinstance(df_viz.index, pd.DatetimeIndex):
            # Try to create datetime index
            datetime_cols = [col for col in df_viz.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols:
                df_viz.index = pd.to_datetime(df_viz[datetime_cols[0]])
            else:
                df_viz.index = pd.date_range(start='2025-01-01', periods=len(df_viz), freq='1min')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Action with Patterns', 'Pattern Probability', 'Pattern Strength'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_viz.index,
                open=df_viz['open'],
                high=df_viz['high'],
                low=df_viz['low'],
                close=df_viz['close'],
                name='Price',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ),
            row=1, col=1
        )
        
        # Add pattern annotations
        pattern_colors = {
            'ascending_triangle': '#4CAF50',
            'descending_triangle': '#F44336',
            'symmetrical_triangle': '#2196F3',
            'double_top': '#E91E63',
            'double_bottom': '#9C27B0',
            'head_and_shoulders': '#FF9800',
            'cup_and_handle': '#00BCD4',
            'wedge': '#795548',
            'flag': '#607D8B'
        }
        
        for i, pattern in enumerate(predictions):
            if pattern != 'no_pattern':
                color = pattern_colors.get(pattern, '#FFC107')
                prob = probabilities[i].max()
                
                annotation_text = f"{pattern}<br>Prob: {prob:.1%}"
                if strength_scores is not None:
                    annotation_text += f"<br>Strength: {strength_scores[i]:.1%}"
                
                fig.add_annotation(
                    x=df_viz.index[i],
                    y=df_viz['high'].iloc[i],
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=0,
                    ay=-40,
                    bgcolor=color,
                    opacity=0.8,
                    font=dict(size=10, color='white'),
                    row=1, col=1
                )
        
        # Pattern probability chart
        max_probs = [p.max() for p in probabilities]
        fig.add_trace(
            go.Scatter(
                x=df_viz.index,
                y=max_probs,
                mode='lines',
                name='Max Probability',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.3)'
            ),
            row=2, col=1
        )
        
        # Add probability threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Threshold", row=2, col=1)
        
        # Pattern strength chart (if available)
        if strength_scores is not None:
            fig.add_trace(
                go.Scatter(
                    x=df_viz.index,
                    y=strength_scores,
                    mode='lines',
                    name='Pattern Strength',
                    line=dict(color='#4CAF50', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(76, 175, 80, 0.3)'
                ),
                row=3, col=1
            )
            
            # Add strength threshold line
            fig.add_hline(y=0.75, line_dash="dash", line_color="orange",
                         annotation_text="Alert Threshold", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title='Pattern Detection Dashboard',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        fig.update_yaxes(title_text="Strength", row=3, col=1)
        
        if save_html:
            filename = f'{self.html_dir}/pattern_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(filename)
            print(f"\nInteractive dashboard saved to: {filename}")
        
        return fig
    
    def create_pattern_distribution_chart(self, predictions, save_html=True):
        """Create pattern distribution pie chart"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        pattern_counts = pd.Series(predictions).value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=pattern_counts.index,
                values=pattern_counts.values,
                hole=0.3,
                marker=dict(
                    colors=['#4CAF50', '#F44336', '#2196F3', '#E91E63', 
                           '#9C27B0', '#FF9800', '#00BCD4', '#795548', '#607D8B']
                )
            )
        ])
        
        fig.update_layout(
            title='Pattern Distribution',
            template='plotly_dark'
        )
        
        if save_html:
            filename = f'{self.html_dir}/pattern_distribution.html'
            fig.write_html(filename)
            print(f"Pattern distribution chart saved to: {filename}")
        
        return fig
    
    def create_backtest_dashboard(self, backtest_results, save_html=True):
        """Create comprehensive backtest dashboard"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None
        
        if backtest_results is None:
            return None
        
        trades_df = backtest_results['trades_df']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curve', 'Win/Loss Distribution', 
                          'Pattern Performance', 'Trade Timeline'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Equity curve
        equity = backtest_results.get('equity_curve', [])
        if equity:
            fig.add_trace(
                go.Scatter(
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#4CAF50', width=3)
                ),
                row=1, col=1
            )
        
        # Win/Loss distribution
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        fig.add_trace(
            go.Bar(
                x=['Wins', 'Losses'],
                y=[len(wins), len(losses)],
                marker_color=['#4CAF50', '#F44336'],
                name='Count'
            ),
            row=1, col=2
        )
        
        # Pattern performance
        pattern_pnl = trades_df.groupby('pattern')['pnl'].sum().sort_values()
        
        fig.add_trace(
            go.Bar(
                x=pattern_pnl.values,
                y=pattern_pnl.index,
                orientation='h',
                marker_color=['#F44336' if x < 0 else '#4CAF50' for x in pattern_pnl.values],
                name='P&L'
            ),
            row=2, col=1
        )
        
        # Trade timeline
        fig.add_trace(
            go.Scatter(
                x=trades_df.index,
                y=trades_df['pnl'].cumsum(),
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#2196F3', width=2),
                marker=dict(
                    size=8,
                    color=['#4CAF50' if x > 0 else '#F44336' for x in trades_df['pnl']],
                    line=dict(width=1, color='white')
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Backtest Results - Total P&L: ${backtest_results["total_pnl"]:,.2f}',
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        if save_html:
            filename = f'{self.html_dir}/backtest_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(filename)
            print(f"Backtest dashboard saved to: {filename}")
        
        return fig


if __name__ == "__main__":
    # Check for GPU availability
    try:
        import xgboost as xgb
        gpu_info = xgb.get_config()
        print("GPU support available:", 'cuda' in str(gpu_info).lower())
    except:
        print("GPU check failed - running on CPU")
    
    # Check dependencies
    try:
        import talib
        import xgboost
        import sklearn
        print("All dependencies available")
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install talib-binary xgboost scikit-learn pandas numpy matplotlib seaborn scipy")