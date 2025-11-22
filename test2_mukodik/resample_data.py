"""
Resample tick data to proper candlestick timeframes for pattern trading
"""
import pandas as pd
import sys

def resample_to_timeframe(input_file, output_file, timeframe='1h'):
    """
    Resample tick/trade data to OHLCV candles
    
    Args:
        input_file: Path to CSV with timestamp,open,high,low,close,volume
        output_file: Path to save resampled data
        timeframe: Pandas offset string ('1min', '5min', '15min', '1h', '4h', '1D')
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Timestamp range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    df.set_index('timestamp', inplace=True)
    
    print(f"\nResampling to {timeframe} candles...")
    
    # Resample OHLCV data
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum'
    }).dropna()
    
    print(f"Resampled shape: {resampled.shape}")
    print(f"\nFirst few candles:")
    print(resampled.head())
    
    print(f"\nSaving to {output_file}...")
    resampled.reset_index().to_csv(output_file, index=False)
    
    print(f"✅ Done! Created {len(resampled)} {timeframe} candles")
    
    # Statistics
    print(f"\nPrice statistics:")
    print(f"  Range: ${resampled['low'].min():.4f} - ${resampled['high'].max():.4f}")
    print(f"  Avg candle range: {((resampled['high'] - resampled['low']) / resampled['close'] * 100).mean():.3f}%")
    print(f"  Total volume: {resampled['volume'].sum():,.0f}")
    
    return resampled

if __name__ == "__main__":
    # Create multiple timeframes
    timeframes = {
        '15min': 'data/DOGEUSDT-15min-2025-08.csv',
        '1h': 'data/DOGEUSDT-1h-2025-08.csv',
        '4h': 'data/DOGEUSDT-4h-2025-08.csv',
    }
    
    input_file = 'data/DOGEUSDT-trades-2025-08.csv'
    
    for tf, output in timeframes.items():
        print(f"\n{'='*80}")
        print(f"Creating {tf} timeframe")
        print('='*80)
        resample_to_timeframe(input_file, output, tf)
