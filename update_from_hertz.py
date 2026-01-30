"""
Update Prices from HertzDR.xlsm

Reads directly from your daily price file instead of requiring CSV exports.
Much faster - just run this script after the file updates.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def parse_price_sheet(xl_file, sheet_name, commodity_name):
    """Parse a commodity sheet from the Excel file"""
    print(f"  Parsing {commodity_name}...")
    
    # Read the sheet
    df = pd.read_excel(xl_file, sheet_name=sheet_name, header=None)
    
    # Convert to list of lists for easier parsing
    data = df.values.tolist()
    
    # Find "Date" columns (start of each contract block)
    if len(data) < 2:
        print(f"    ❌ Sheet too short")
        return pd.DataFrame()
    
    header_row = data[0]
    contract_starts = [i for i, val in enumerate(header_row) if val == 'Date']
    
    if len(contract_starts) == 0:
        print(f"    ❌ No 'Date' columns found")
        return pd.DataFrame()
    
    # Parse data rows (skip first row which is header)
    cleaned_data = []
    
    for row in data[1:]:
        if len(row) < 10:
            continue
        
        for col_start in contract_starts:
            try:
                date_val = row[col_start]
                if pd.isna(date_val) or date_val == '':
                    continue
                
                # Try to parse date
                try:
                    date = pd.to_datetime(date_val)
                except:
                    continue
                
                # Extract OHLC
                high = float(row[col_start + 1]) if not pd.isna(row[col_start + 1]) else np.nan
                low = float(row[col_start + 2]) if not pd.isna(row[col_start + 2]) else np.nan
                open_price = float(row[col_start + 3]) if not pd.isna(row[col_start + 3]) else np.nan
                close = float(row[col_start + 4]) if not pd.isna(row[col_start + 4]) else np.nan
                code = row[col_start + 5] if len(row) > col_start + 5 else ''
                
                # Skip if no close price
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
                
            except (ValueError, IndexError, TypeError):
                continue
    
    df = pd.DataFrame(cleaned_data)
    
    if len(df) > 0:
        print(f"    ✓ {len(df):,} records")
    else:
        print(f"    ❌ No data extracted")
    
    return df


def update_from_hertz_file(hertz_path='C:/Users/AdamKelleher/OneDrive - Prime Trading/DR files/HertzDR.xlsm'):
    """
    Update all commodity prices from HertzDR.xlsm file
    
    Args:
        hertz_path: Path to HertzDR.xlsm file
    """
    print("="*70)
    print("UPDATE PRICES FROM HERTZDR.XLSM")
    print("="*70)
    
    if not os.path.exists(hertz_path):
        print(f"\n❌ File not found: {hertz_path}")
        print("\nPlease update the path in the script or provide it as argument:")
        print("  python update_from_hertz.py <path_to_HertzDR.xlsm>")
        return False
    
    print(f"\nReading from: {hertz_path}")
    
    # Open Excel file
    try:
        xl_file = pd.ExcelFile(hertz_path)
        print(f"  ✓ File opened")
    except Exception as e:
        print(f"  ❌ Error opening file: {str(e)}")
        return False
    
    # Commodity sheets to parse
    commodities = {
        'SOY': 'SOY',
        'MEAL': 'MEAL',
        'OIL': 'OIL',
        'CORN': 'CORN',
        'WHEAT': 'WHEAT',
        'KW': 'KW'
    }
    
    # Parse all sheets
    all_prices = []
    
    for commodity, sheet_name in commodities.items():
        if sheet_name not in xl_file.sheet_names:
            print(f"  ⚠️  {commodity}: Sheet '{sheet_name}' not found")
            continue
        
        try:
            df = parse_price_sheet(xl_file, sheet_name, commodity)
            if len(df) > 0:
                all_prices.append(df)
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
    
    if len(all_prices) == 0:
        print("\n❌ No data parsed from any sheets!")
        return False
    
    # Combine new data
    new_combined = pd.concat(all_prices, ignore_index=True)
    print(f"\n✓ Total new data: {len(new_combined):,} records")
    print(f"  Date range: {new_combined['date'].min().date()} to {new_combined['date'].max().date()}")
    
    # Load existing historical data
    historical_file = 'all_commodity_prices.csv'
    
    if os.path.exists(historical_file):
        print(f"\nLoading existing data from {historical_file}...")
        existing_df = pd.read_csv(historical_file, parse_dates=['date'])
        print(f"  Existing: {len(existing_df):,} records")
        print(f"  Date range: {existing_df['date'].min().date()} to {existing_df['date'].max().date()}")
        
        # Combine
        combined = pd.concat([existing_df, new_combined], ignore_index=True)
        
        # Remove duplicates (keep last = keep new data)
        combined = combined.sort_values('date').drop_duplicates(
            subset=['date', 'commodity', 'contract_code'], 
            keep='last'
        )
        
        print(f"\nCombined dataset:")
        print(f"  Total records: {len(combined):,}")
        print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
        
        # Check what's new
        new_dates = set(new_combined['date'].unique()) - set(existing_df['date'].unique())
        if new_dates:
            new_dates_sorted = sorted([d.date() for d in new_dates])
            if len(new_dates_sorted) <= 5:
                print(f"\n  ✓ New dates added: {new_dates_sorted}")
            else:
                print(f"\n  ✓ New dates added: {new_dates_sorted[0]} to {new_dates_sorted[-1]} ({len(new_dates_sorted)} dates)")
        else:
            print(f"\n  ℹ️  No new dates (data updated for existing dates)")
        
        # Backup old file (only once per day)
        backup_file = historical_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
        if not os.path.exists(backup_file):
            existing_df.to_csv(backup_file, index=False)
            print(f"\n  ✓ Backup saved: {backup_file}")
        
    else:
        print(f"\nNo existing file found - creating new {historical_file}")
        combined = new_combined
    
    # Save updated data
    combined.to_csv(historical_file, index=False)
    print(f"\n✓ Updated data saved to {historical_file}")
    
    # Show summary by commodity
    print(f"\n" + "="*70)
    print("SUMMARY BY COMMODITY")
    print("="*70)
    for commodity in sorted(combined['commodity'].unique()):
        comm_data = combined[combined['commodity'] == commodity]
        date_range = comm_data['date']
        print(f"  {commodity:6} {len(comm_data):,} records  ({date_range.min().date()} to {date_range.max().date()})")
    
    print("\n" + "="*70)
    print("✅ PRICE UPDATE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Refresh your Streamlit dashboard (press 'R')")
    print("2. New price data will be available for variance calculations")
    
    return True


if __name__ == "__main__":
    import sys
    
    print("HertzDR.xlsm Price Update Tool\n")
    
    # Default path
    default_path = 'C:/Users/AdamKelleher/OneDrive - Prime Trading/DR files/HertzDR.xlsm'
    
    # Allow custom path as argument
    if len(sys.argv) > 1:
        hertz_path = sys.argv[1]
    else:
        hertz_path = default_path
    
    success = update_from_hertz_file(hertz_path)
    
    if not success:
        print("\n💡 Tip: Make sure HertzDR.xlsm is closed before running this script")
        print("        (Excel locks the file when it's open)")
    
    sys.exit(0 if success else 1)
