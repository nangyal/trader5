import os
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def read_crypto_pairs(filename):
    """Beolvassa a crypto valutapárok listáját"""
    with open(filename, 'r') as file:
        pairs = [line.strip() for line in file if line.strip()]
    return pairs

def read_months_count(filename):
    """Beolvassa a letöltendő hónapok számát"""
    with open(filename, 'r') as file:
        months = int(file.read().strip())
    return months

def get_previous_months(months_count):
    """Visszaadja a megelőző hónapok listáját"""
    current_date = datetime.now()
    months = []
    
    for i in range(months_count, 0, -1):
        # Az aktuális hónap előtti i. hónap
        target_date = current_date - timedelta(days=30*i)
        year = target_date.year
        month = target_date.month
        months.append((year, month))
    
    return months

def download_and_extract(pair, year, month):
    """Letölti és kicsomagolja a megadott valutapár adatait"""
    # Formázott dátumok
    month_str = str(month).zfill(2)
    
    # Fájlnevek
    zip_filename = f"{pair}-trades-{year}-{month_str}.zip"
    csv_filename = f"{pair}-trades-{year}-{month_str}.csv"
    
    # Könyvtár struktúra
    base_dir = Path("data") / pair / "monthly"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = base_dir / csv_filename
    
    # Ha már létezik a CSV, nem kell letölteni
    if csv_path.exists():
        print(f"A {csv_filename} már létezik, kihagyás...")
        return True
    
    # URL összeállítása
    url = f"https://data.binance.vision/data/futures/um/monthly/trades/{pair}/{zip_filename}"
    
    try:
        print(f"Letöltés: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Ideiglenes zip fájl mentése
        temp_zip = base_dir / zip_filename
        with open(temp_zip, 'wb') as f:
            f.write(response.content)
        
        # Zip kicsomagolása
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Zip fájl törlése
        os.remove(temp_zip)
        
        print(f"Sikeresen letöltve és kicsomagolva: {csv_filename}")
        return True
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"A fájl nem található: {zip_filename}")
        else:
            print(f"HTTP hiba: {e}")
        return False
    except Exception as e:
        print(f"Hiba történt: {e}")
        return False

def main():
    # Fájlok ellenőrzése
    if not os.path.exists("crypto.txt"):
        print("Hiba: crypto.txt fájl nem található!")
        return
    
    if not os.path.exists("months.txt"):
        print("Hiba: months.txt fájl nem található!")
        return
    
    try:
        # Adatok beolvasása
        crypto_pairs = read_crypto_pairs("crypto.txt")
        months_count = read_months_count("months.txt")
        
        print(f"Kripto valutapárok: {crypto_pairs}")
        print(f"Letöltendő hónapok száma: {months_count}")
        
        # Hónapok meghatározása
        target_months = get_previous_months(months_count)
        print(f"Letöltendő hónapok: {target_months}")
        
        # Letöltés minden valutapárra és hónapra
        for pair in crypto_pairs:
            print(f"\nFeldolgozás alatt: {pair}")
            
            for year, month in target_months:
                success = download_and_extract(pair, year, month)
                
                if not success:
                    print(f"Sikertelen letöltés: {pair} {year}-{month}")
        
        print("\nA letöltés befejeződött!")
        
    except Exception as e:
        print(f"Váratlan hiba: {e}")

if __name__ == "__main__":
    main()