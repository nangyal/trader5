"""
Data Configuration
Data loading, preprocessing, and storage settings
"""

import os

# ============================================================================
# Data Paths
# ============================================================================

# Base data directory
DATA_DIR = 'data'

# Subdirectories
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, 'historical')
LIVE_DATA_DIR = os.path.join(DATA_DIR, 'live')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Data sources
DATA_SOURCES = {
    'binance_csv': os.path.join(DATA_DIR, '*.csv'),
    'custom_csv': os.path.join(DATA_DIR, 'custom', '*.csv')
}

# ============================================================================
# CSV File Configuration
# ============================================================================

# Default CSV file for training
DEFAULT_TRAINING_DATA = os.path.join(DATA_DIR, 'DOGEUSDT-1h-2025-09.csv')

# CSV parsing settings
CSV_CONFIG = {
    'delimiter': ',',
    'decimal': '.',
    'encoding': 'utf-8',
    'parse_dates': True,
    'date_column': None,        # Auto-detect datetime column
    'index_col': None           # Auto-detect index
}

# Required CSV columns (lowercase)
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
OPTIONAL_COLUMNS = ['volume', 'datetime', 'timestamp', 'date', 'time']

# Column name mapping (for non-standard CSVs)
COLUMN_MAPPING = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Date': 'datetime',
    'Time': 'datetime',
    'Timestamp': 'datetime'
}

# ============================================================================
# Data Preprocessing
# ============================================================================

# Data cleaning
DATA_CLEANING = {
    'remove_duplicates': True,          # Remove duplicate timestamps
    'remove_invalid_ohlc': True,        # Remove invalid OHLC rows
    'remove_zero_volume': False,        # Keep zero volume candles
    'forward_fill_gaps': True,          # Fill gaps in time series
    'max_gap_minutes': 60               # Max gap to fill (minutes)
}

# Invalid OHLC conditions
INVALID_OHLC_CHECKS = {
    'high_gte_low': True,       # high >= low
    'high_gte_open': True,      # high >= open
    'high_gte_close': True,     # high >= close
    'low_lte_open': True,       # low <= open
    'low_lte_close': True,      # low <= close
    'positive_prices': True     # all prices > 0
}

# Outlier detection
OUTLIER_DETECTION = {
    'enable': True,
    'method': 'quantile',       # 'quantile', 'zscore', or 'iqr'
    'quantile_range': (0.01, 0.99),     # Remove bottom 1% and top 1%
    'zscore_threshold': 3.0,            # Z-score threshold
    'iqr_multiplier': 1.5               # IQR multiplier
}

# Missing data handling
MISSING_DATA = {
    'strategy': 'ffill_bfill',  # 'ffill', 'bfill', 'ffill_bfill', 'drop', 'interpolate'
    'interpolate_method': 'linear',     # For interpolate strategy
    'max_missing_pct': 0.05             # Max 5% missing data allowed
}

# Data type optimization
DATA_TYPE_OPTIMIZATION = {
    'enable': True,
    'float_to_float32': True,   # Convert float64 → float32
    'int_to_int32': True,       # Convert int64 → int32
    'category_encoding': True   # Use category dtype for patterns
}

# ============================================================================
# Data Validation
# ============================================================================

# Validation rules
DATA_VALIDATION = {
    'min_rows': 100,            # Minimum rows required
    'max_rows': None,           # Maximum rows (None = unlimited)
    'check_continuity': True,   # Check time series continuity
    'check_sorted': True,       # Check if data is sorted by time
    'check_duplicates': True    # Check for duplicate timestamps
}

# Price validation
PRICE_VALIDATION = {
    'min_price': 0.0,           # Minimum allowed price
    'max_price': None,          # Maximum allowed price (None = unlimited)
    'max_price_change_pct': 0.5,  # Max 50% price change between candles
    'warn_on_anomalies': True   # Print warnings for anomalies
}

# Volume validation
VOLUME_VALIDATION = {
    'min_volume': 0.0,          # Minimum allowed volume
    'max_volume': None,         # Maximum allowed volume
    'warn_on_zero_volume': False  # Warn on zero volume candles
}

# ============================================================================
# Data Sampling
# ============================================================================

# Data sampling for training (reduce dataset size)
DATA_SAMPLING = {
    'enable': False,            # Enable sampling
    'method': 'random',         # 'random', 'systematic', 'stratified'
    'sample_size': None,        # Number of samples (None = use sample_pct)
    'sample_pct': 0.5,          # Percentage of data to use (50%)
    'random_state': 42          # Seed for reproducibility
}

# Time-based sampling
TIME_SAMPLING = {
    'enable': False,
    'start_date': None,         # Start date (YYYY-MM-DD)
    'end_date': None,           # End date (YYYY-MM-DD)
    'recent_days': None         # Use only last N days
}

# ============================================================================
# Data Resampling (Timeframe Conversion)
# ============================================================================

# Resampling configuration
RESAMPLING = {
    'enable': False,
    'target_timeframe': '1h',   # Target timeframe
    'aggregation': {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
}

# Timeframe mapping
TIMEFRAME_MAPPING = {
    '1min': '1T',
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1h': '1H',
    '4h': '4H',
    '1d': '1D',
    '1w': '1W'
}

# ============================================================================
# Data Caching
# ============================================================================

# Cache processed data
DATA_CACHING = {
    'enable': True,
    'cache_dir': 'cache',
    'cache_format': 'pickle',   # 'pickle', 'parquet', 'feather'
    'use_compression': True,
    'cache_expiry_hours': 24    # Refresh cache after 24 hours
}

# ============================================================================
# Data Export
# ============================================================================

# Export processed data
DATA_EXPORT = {
    'enable': False,
    'export_format': 'csv',     # 'csv', 'parquet', 'feather', 'json'
    'export_dir': 'exports',
    'include_features': True,   # Export engineered features
    'include_labels': True      # Export pattern labels
}

# CSV export settings
CSV_EXPORT_CONFIG = {
    'index': True,
    'float_format': '%.8f',
    'compression': None         # None, 'gzip', 'bz2', 'zip'
}

# ============================================================================
# Historical Data Download (Binance)
# ============================================================================

# Download settings
DOWNLOAD_CONFIG = {
    'symbol': 'DOGEUSDT',
    'interval': '1h',           # Kline interval
    'start_date': '2025-01-01', # Start date (YYYY-MM-DD)
    'end_date': None,           # End date (None = today)
    'limit': 1000,              # Max candles per request
    'save_to_csv': True,
    'csv_filename': 'DOGEUSDT-1h-{year}-{month:02d}.csv'
}

# Supported Binance intervals
BINANCE_INTERVALS = [
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
]

# ============================================================================
# Real-Time Data (WebSocket)
# ============================================================================

# Kline stream configuration
KLINE_STREAM_CONFIG = {
    'symbol': 'DOGEUSDT',
    'intervals': ['1m', '5m', '15m', '1h'],
    'buffer_size': 1000,        # Max candles to store per timeframe
    'save_to_disk': False,      # Periodically save to disk
    'save_interval_minutes': 60 # Save every N minutes
}

# ============================================================================
# Data Quality Checks
# ============================================================================

# Quality metrics
QUALITY_CHECKS = {
    'check_on_load': True,
    'print_summary': True,
    'checks': [
        'row_count',
        'missing_values',
        'duplicates',
        'outliers',
        'data_types',
        'date_range',
        'price_validity',
        'volume_validity'
    ]
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'max_missing_pct': 0.05,    # Fail if >5% missing
    'max_duplicates': 10,       # Fail if >10 duplicates
    'max_outliers_pct': 0.01,   # Warn if >1% outliers
    'min_data_completeness': 0.95  # Fail if <95% complete
}
