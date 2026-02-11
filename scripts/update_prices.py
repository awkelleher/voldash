"""
Daily Price Data Update Script

Updates all commodity price files with new data.
Run once per morning after exporting fresh prices from Excel.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def parse_commodity_prices(csv_path, commodity_name):
    """Parse wide-format price data into long format"""
    print(f"  Parsing {commodity_name}...")
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip().split(',') for line in f.readlines()]
    
    if len(lines) < 3:
        print(f"    ❌ File too short")
        return pd.DataFrame()
    
    # Find all "Date" columns (start of each contract block)
    header_line = lines[0]
    contract_starts = [i for i, val in enumerate(header_line) if val == 'Date']
    
    # Parse data rows (skip first 2 header rows)
    cleaned_data = []
    
    for line in lines[2:]:
        if len(line) < 10:
            continue
        
        for col_start in contract_starts:
            try:
                date_str = line[col_start]
                if not date_str or date_str == '':
                    continue
                
                date = pd.to_datetime(date_str)
                high = float(line[col_start + 1]) if line[col_start + 1] else np.nan
                low = float(line[col_start + 2]) if line[col_start + 2] else np.nan
                open_price = float(line[col_start + 3]) if line[col_start + 3] else np.nan
                close = float(line[col_start + 4]) if line[col_start + 4] else np.nan
                code = line[col_start + 5] if len(line) > col_start + 5 else ''
                
                if pd.isna(close):
                    continue
                
                cleaned_data.append({
                    'date': date,
                    'contract_code': code,
                    'high': high,
                    'low': low,
                    'open': open_price,
                    'close': close,
                    'commodity': commodity_name
                })
                
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(cleaned_data)
    
    if len(df) > 0:
        print(f"    ✓ {len(df):,} records")
    
    return df


def update_prices(new_data_folder):
    """
    Update all commodity prices with new data
    
    Args:
        new_data_folder: Folder containing today's CSV exports
    """
    print("="*70)
    print("DAILY PRICE DATA UPDATE")
    print("="*70)
    print(f"Reading from: {new_data_folder}\n")
    
    # Expected file names (adjust if yours are different)
    price_files = {
        'SOY': 'soy_price.csv',
        'MEAL': 'meal_price.csv',
        'OIL': 'oil_price.csv',
        'CORN': 'corn_price.csv',
        'WHEAT': 'wheat_price.csv',
        'KW': 'kw_price.csv',
    }
    
    # Parse all new data
    new_prices = []
    
    for commodity, filename in price_files.items():
        filepath = os.path.join(new_data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  {commodity}: File not found - {filename}")
            continue
        
        try:
            df = parse_commodity_prices(filepath, commodity)
            if len(df) > 0:
                new_prices.append(df)
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
    
    if len(new_prices) == 0:
        print("\n❌ No new data parsed!")
        return False
    
    # Combine new data
    new_combined = pd.concat(new_prices, ignore_index=True)
    print(f"\nNew data: {len(new_combined):,} records")
    
    # Load existing data
    historical_file = 'data/all_commodity_prices.csv'
    
    if os.path.exists(historical_file):
        print(f"\nLoading existing data from {historical_file}...")
        existing_df = pd.read_csv(historical_file, parse_dates=['date'])
        print(f"  Existing: {len(existing_df):,} records")
        
        # Combine
        combined = pd.concat([existing_df, new_combined], ignore_index=True)
        
        # Remove duplicates (keep last = keep new data)
        combined = combined.sort_values('date').drop_duplicates(
            subset=['date', 'commodity', 'contract_code'], 
            keep='last'
        )
        
        print(f"  Combined: {len(combined):,} records")
        
        # Check what's new
        new_dates = set(new_combined['date'].unique()) - set(existing_df['date'].unique())
        if new_dates:
            print(f"\n  ✓ New dates added: {sorted([d.date() for d in new_dates])}")
        else:
            print(f"\n  ℹ️  No new dates (data updated for existing dates)")
        
        # Backup old file
        backup_file = historical_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
        if not os.path.exists(backup_file):
            existing_df.to_csv(backup_file, index=False)
            print(f"  ✓ Backup saved: {backup_file}")
        
    else:
        print(f"\nNo existing file found - creating new {historical_file}")
        combined = new_combined
    
    # Save updated data
    combined.to_csv(historical_file, index=False)
    print(f"\n✓ Updated data saved to {historical_file}")
    
    print("\n" + "="*70)
    print("✅ PRICE UPDATE COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    print("Daily Price Update Tool\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python update_prices.py <folder_with_new_csvs>")
        print("\nExample:")
        print("  python update_prices.py ./today_prices")
        print("\nExpected files in folder:")
        print("  - soy_price.csv")
        print("  - meal_price.csv")
        print("  - oil_price.csv")
        print("  - corn_price.csv")
        print("  - wheat_price.csv")
        print("  - kw_price.csv")
        sys.exit(1)
    
    folder = sys.argv[1]
    
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)
    
    success = update_prices(folder)
    sys.exit(0 if success else 1)
