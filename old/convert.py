from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob

def process_csv_to_candles(input_csv_path, output_csv_path, timeframe):
    """Feldolgoz egy CSV fájlt és megadott időintervallumú gyertyákat készít
    
    Args:
        input_csv_path: Bemeneti CSV fájl útvonala
        output_csv_path: Kimeneti CSV fájl útvonala
        timeframe: Időintervallum (pl. '1min', '5min', '15min', '30min', '1h')
    """
    try:
        # CSV beolvasása
        df = pd.read_csv(input_csv_path)
        
        # Idő konvertálása (milliszekundumból datetime objektumba)
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        
        # Időintervallumok létrehozása
        df['interval'] = df['datetime'].dt.floor(timeframe)
        
        # Csoportosítás intervallumok szerint és OHLCV számítás
        ohlcv = df.groupby('interval').agg({
            'price': ['first', 'max', 'min', 'last'],
            'qty': 'sum',
            'quote_qty': 'sum'
        }).round(6)
        
        # Oszlopnevek átnevezése
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        
        # Reset index hogy az időintervallum oszlop legyen
        ohlcv = ohlcv.reset_index()
        
        # Időbélyeg MÁSODPERCBEN (milliszekundum osztva 1000-al)
        ohlcv['timestamp'] = (ohlcv['interval'].astype(np.int64) // 10**9).astype(np.int64)
        
        # Végleges oszloprend
        result_df = ohlcv[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
        
        # Könyvtár létrehozása ha nem létezik
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Eredmény mentése
        result_df.to_csv(output_csv_path, index=False)
        # print(f"{timeframe} gyertyák elkészítve: {output_csv_path}")
        
        return True
        
    except Exception as e:
        print(f"Hiba a fájl feldolgozása közben {input_csv_path}: {e}")
        return False

def collect_all_tasks():
    """Összegyűjti az összes feldolgozandó feladatot"""
    base_path = Path("data")
    
    if not base_path.exists():
        print("A data/ könyvtár nem létezik!")
        return []
    
    # Időintervallumok meghatározása
    timeframes = {
        '15s': '15s',    # 15 másodperc
        '30s': '30s',    # 30 másodperc
        '1min': '1min',  # 1 perc
        '5min': '5min',  # 5 perc
    }
    
    all_tasks = []
    
    # Minden coin mappa megkeresése
    for coin_path in base_path.iterdir():
        if coin_path.is_dir():
            coin_name = coin_path.name
            
            # Monthly adatok feldolgozása
            monthly_path = coin_path / "monthly"
            if monthly_path.exists():
                csv_files = list(monthly_path.glob("*.csv"))
                
                for tf_name, tf_value in timeframes.items():
                    for csv_file in csv_files:
                        output_dir = Path("data") / coin_name / tf_name / "monthly"
                        output_filename = csv_file.name.replace('.csv', f'_{tf_name}.csv')
                        output_path = output_dir / output_filename
                        
                        # Csak akkor adjuk hozzá, ha még nem létezik
                        if not output_path.exists():
                            all_tasks.append((str(csv_file), str(output_path), tf_value))
    
    return all_tasks

def find_and_process_crypto_data():
    """Megkeresi és feldolgozza az összes crypto adatfájlt párhuzamosan"""
    print("Feladatok összegyűjtése...")
    all_tasks = collect_all_tasks()
    
    if not all_tasks:
        print("Nincs feldolgozandó feladat!")
        return
    
    print(f"Összesen {len(all_tasks)} feladat feldolgozása {cpu_count()} processzorral...")
    
    processed_count = 0
    error_count = 0
    
    # Párhuzamos feldolgozás az összes processzoron
    with Pool(processes=cpu_count()) as pool:
        for success in pool.imap_unordered(_process_csv_task, all_tasks):
            if success:
                processed_count += 1
            else:
                error_count += 1
    
    print(f"\nÖsszesen - Feldolgozva: {processed_count}, Hibák: {error_count}")

def _process_csv_task(task):
    """Segédfeladat a multiprocessing Pool számára"""
    input_csv_path, output_csv_path, timeframe_value = task
    try:
        success = process_csv_to_candles(Path(input_csv_path), Path(output_csv_path), timeframe_value)
        return success
    except Exception as exc:
        print(f"Multiprocessing hiba {input_csv_path}: {exc}")
        return False


def verify_timestamp_format():
    """Ellenőrzi, hogy a timestamp másodpercben van-e"""
    print("\nIdőbélyeg formátum ellenőrzése...")
    
    # Példa ellenőrzés a létrehozott fájlokra
    timeframe_dirs = ['5s', '15s', '30s', '1min', '5min', '15min', '30min', '1h']
    all_files = []
    
    for tf in timeframe_dirs:
        tf_files = list(Path("data").rglob(f"{tf}/**/*_{tf}.csv"))
        all_files.extend(tf_files)
    
    if not all_files:
        print("Nincsenek feldolgozott fájlok az ellenőrzéshez")
        return
    
    checked_count = 0
    correct_count = 0
    
    for csv_file in all_files[:10]:  # Csak az első 10 fájlt ellenőrizzük
        try:
            df = pd.read_csv(csv_file)
            if 'timestamp' in df.columns and len(df) > 0:
                checked_count += 1
                # Ellenőrizzük, hogy másodpercben van-e (normál Unix timestamp)
                sample_timestamp = df['timestamp'].iloc[0]
                current_timestamp_seconds = int(datetime.now().timestamp())
                
                # Ha a timestamp nagyságrendileg másodperc (kb 1-2 milliárd), akkor jó
                if 1_000_000_000 < sample_timestamp < 2_000_000_000:
                    correct_count += 1
                    print(f"✓ {csv_file}: timestamp másodpercben ({sample_timestamp})")
                else:
                    print(f"✗ {csv_file}: timestamp nem másodpercben? ({sample_timestamp})")
                    
        except Exception as e:
            print(f"! {csv_file}: ellenőrzési hiba - {e}")
    
    print(f"Ellenőrzés eredménye: {correct_count}/{checked_count} fájl helyes formátumú")

def main():
    print("Gyertyák generálása több időintervallumban...")
    print("Időintervallumok: 5s, 15s, 30s, 1min, 5min, 15min, 30min, 1h")
    print("A már létező fájlok kihagyásra kerülnek...")
    
    start_time = datetime.now()
    find_and_process_crypto_data()
    end_time = datetime.now()
    
    duration = end_time - start_time
    
    # Időbélyeg formátum ellenőrzése
    verify_timestamp_format()

    print(f"\nA feldolgozás befejeződött! Időtartam: {duration}")

if __name__ == "__main__":
    main()