"""
Pattern Detection Configuration
Settings for rule-based pattern detection
"""

# ============================================================================
# Pattern Detection Thresholds
# ============================================================================

# Minimum confidence scores for pattern detection (0-1 scale)
PATTERN_DETECTION_THRESHOLDS = {
    'ascending_triangle': 0.65,
    'descending_triangle': 0.65,
    'symmetrical_triangle': 0.70,
    'double_top': 0.70,
    'double_bottom': 0.70,
    'head_and_shoulders': 0.70,
    'cup_and_handle': 0.70,
    'wedge': 0.70,
    'flag': 0.70
}

# ============================================================================
# Pattern-Specific Parameters
# ============================================================================

# Ascending/Descending Triangle
TRIANGLE_CONFIG = {
    'tolerance': 0.02,          # 2% tolerance for horizontal levels
    'min_touches': 2,           # Minimum resistance/support touches
    'min_bars': 20,             # Minimum bars to form pattern
    'use_volume': True          # Use volume confirmation
}

# Double Top/Bottom
DOUBLE_PATTERN_CONFIG = {
    'tolerance': 0.015,         # 1.5% tolerance for peak/trough similarity
    'min_bars': 30,             # Minimum bars between peaks/troughs
    'peak_prominence': 0.02,    # 2% minimum prominence for peaks
    'trough_depth_min': 0.02    # 2% minimum trough depth
}

# Head and Shoulders
HEAD_SHOULDERS_CONFIG = {
    'shoulder_tolerance': 0.10, # 10% tolerance for shoulder similarity
    'min_bars': 40,             # Minimum bars for full pattern
    'neckline_touches': 2       # Minimum neckline touches
}

# Cup and Handle
CUP_HANDLE_CONFIG = {
    'cup_depth_min': 0.12,      # 12% minimum cup depth
    'cup_depth_max': 0.33,      # 33% maximum cup depth
    'handle_to_cup_max': 0.7,   # Handle max 70% of cup depth
    'min_bars': 50,             # Minimum bars for pattern
    'cup_ratio': 0.6            # Cup forms 60% of pattern
}

# Wedge Patterns
WEDGE_CONFIG = {
    'min_bars': 30,             # Minimum bars for wedge
    'convergence_min': 0.2,     # 20% minimum range narrowing
    'min_touches': 3,           # Minimum touches per trendline
    'slope_tolerance': 0.001    # Slope tolerance for parallel lines
}

# Flag Patterns
FLAG_CONFIG = {
    'pole_min_move': 0.03,      # 3% minimum pole movement
    'flag_max_range': 0.05,     # 5% maximum flag consolidation range
    'flag_to_pole_max': 0.7,    # Flag max 70% of pole duration
    'min_bars': 20              # Minimum bars for pattern
}

# ============================================================================
# Adaptive Window Sizes (by Timeframe)
# ============================================================================

ADAPTIVE_WINDOWS = {
    '1min': {
        'triangle': 20,
        'double': 30,
        'head_shoulders': 40,
        'cup': 50,
        'wedge': 30,
        'flag': 20
    },
    '5min': {
        'triangle': 40,
        'double': 60,
        'head_shoulders': 80,
        'cup': 100,
        'wedge': 60,
        'flag': 40
    },
    '15min': {
        'triangle': 60,
        'double': 90,
        'head_shoulders': 120,
        'cup': 150,
        'wedge': 90,
        'flag': 60
    },
    '1h': {
        'triangle': 100,
        'double': 150,
        'head_shoulders': 200,
        'cup': 250,
        'wedge': 150,
        'flag': 100
    },
    '4h': {
        'triangle': 150,
        'double': 200,
        'head_shoulders': 250,
        'cup': 300,
        'wedge': 200,
        'flag': 150
    }
}

# ============================================================================
# Pattern Strength Scoring Weights
# ============================================================================

# Weights for pattern strength calculation (must sum to 1.0)
STRENGTH_WEIGHTS = {
    'volume_confirmation': 0.20,        # Volume alignment
    'trend_alignment': 0.25,            # Pattern/trend match
    'pattern_clarity': 0.30,            # Pattern shape quality
    'volatility': 0.15,                 # Market volatility check
    'support_resistance': 0.10          # S/R alignment
}

# Volume confirmation settings
VOLUME_CONFIG = {
    'use_volume': True,                 # Enable volume analysis
    'volume_ma_period': 20,             # Volume moving average period
    'volume_increase_threshold': 1.2,   # 20% volume increase = strong
    'volume_decrease_threshold': 0.8    # 20% volume decrease = weak
}

# ============================================================================
# Peak/Trough Detection (scipy.signal)
# ============================================================================

PEAK_DETECTION_CONFIG = {
    'prominence': 0.02,         # 2% minimum prominence (% of mean price)
    'distance': 5,              # Minimum distance between peaks (bars)
    'width': None,              # Minimum peak width (None = auto)
    'rel_height': 0.5           # Relative height for width calculation
}

# ============================================================================
# Pattern Classification
# ============================================================================

# Bullish patterns (expect upward breakout)
BULLISH_PATTERNS = [
    'ascending_triangle',
    'double_bottom',
    'cup_and_handle',
    'flag_bullish',
    'wedge_falling'
]

# Bearish patterns (expect downward breakout)
BEARISH_PATTERNS = [
    'descending_triangle',
    'double_top',
    'head_and_shoulders',
    'flag_bearish',
    'wedge_rising'
]

# Neutral patterns (direction depends on breakout)
NEUTRAL_PATTERNS = [
    'symmetrical_triangle'
]

# ============================================================================
# Pattern Priority (for multi-pattern detection)
# ============================================================================

# Patterns checked in this order (higher priority first)
PATTERN_PRIORITY = [
    'head_and_shoulders',   # Most complex, check first
    'cup_and_handle',
    'double_top',
    'double_bottom',
    'ascending_triangle',
    'descending_triangle',
    'symmetrical_triangle',
    'wedge',
    'flag'
]

# ============================================================================
# Label Creation Settings
# ============================================================================

# Minimum data points before pattern detection starts
MIN_DATA_POINTS = 50

# Progress reporting interval (every N bars)
PROGRESS_REPORT_INTERVAL = 100  # Print progress every 100 bars

# Store confidence scores with labels
STORE_CONFIDENCE_SCORES = True

# ============================================================================
# Pattern Validation Rules
# ============================================================================

# Validate pattern formation rules
VALIDATE_PATTERNS = True

# Rules
VALIDATION_RULES = {
    'max_pattern_age': 100,         # Max bars since pattern start
    'min_pattern_duration': 10,     # Min bars for pattern formation
    'require_volume_data': False,   # Require volume for detection
    'check_price_validity': True    # Validate OHLC relationships
}
