"""
Vol/Skew Data Parser (update_volskew_clean.py)
===============================================
Called by update_prices.bat as:  python update_volskew_clean.py eod_vol_snap.csv

Parses the wide-format vol dump CSV and appends new data to master_vol_skew.csv.
Deduplicates on (date, commodity, expiry) so re-runs are safe.
"""

import csv
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, 'data', 'master_vol_skew.csv')
LOG_FILE = os.path.join(PROJECT_DIR, 'logs', 'vol_parser.log')

EXCEL_EPOCH = datetime(1899, 12, 30)

COMMODITY_BLOCKS = {
    'SOY':   {'row_num': 1,   'expiry': 2,   'dirty_vol': 4,   'fwd_vol': 6,   'skew_m1_5': 11,  'skew_m0_5': 12,  'skew_p0_5': 13,  'skew_p1_5': 14,  'skew_p3': 15,  'trading_dte': 16},
    'MEAL':  {'row_num': 19,  'expiry': 20,  'dirty_vol': 22,  'fwd_vol': 24,  'skew_m1_5': 29,  'skew_m0_5': 30,  'skew_p0_5': 31,  'skew_p1_5': 32,  'skew_p3': 33,  'trading_dte': 34},
    'OIL':   {'row_num': 37,  'expiry': 38,  'dirty_vol': 40,  'fwd_vol': 42,  'skew_m1_5': 47,  'skew_m0_5': 48,  'skew_p0_5': 49,  'skew_p1_5': 50,  'skew_p3': 51,  'trading_dte': 52},
    'CORN':  {'row_num': 55,  'expiry': 56,  'dirty_vol': 58,  'fwd_vol': 60,  'skew_m1_5': 65,  'skew_m0_5': 66,  'skew_p0_5': 67,  'skew_p1_5': 68,  'skew_p3': 69,  'trading_dte': 70},
    'WHEAT': {'row_num': 83,  'expiry': 84,  'dirty_vol': 86,  'fwd_vol': 88,  'skew_m1_5': 93,  'skew_m0_5': 94,  'skew_p0_5': 95,  'skew_p1_5': 96,  'skew_p3': 97,  'trading_dte': 98},
    'KW':    {'row_num': 101, 'expiry': 102, 'dirty_vol': 104, 'fwd_vol': 106, 'skew_m1_5': 111, 'skew_m0_5': 112, 'skew_p0_5': 113, 'skew_p1_5': 114, 'skew_p3': 115, 'trading_dte': 116},
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ]
    )


def serial_to_date(serial):
    if serial is None:
        return None
    serial = str(serial).strip()
    if not serial:
        return None
    # Try Excel serial number first
    try:
        s = int(float(serial))
        if 30000 < s < 60000:
            return (EXCEL_EPOCH + timedelta(days=s)).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        pass
    # Try common date formats (M/D/YYYY, YYYY-MM-DD)
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y'):
        try:
            return datetime.strptime(serial, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def parse_float(val):
    if val is None:
        return None
    val = str(val).strip()
    if not val or val.startswith('#') or val.lower() in ('n/a', 'na'):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_csv_file(filepath):
    records = []
    current_date = None

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 20:
                continue

            col0 = row[0].strip() if row[0] else ''
            col1 = row[1].strip() if len(row) > 1 else ''

            # Header row: date serial in col0, 'SOY' in col1
            if col0 and col1 == 'SOY':
                current_date = serial_to_date(col0)
                continue

            if current_date is None:
                continue

            # Data row: row number (1-11) in col1
            row_num = parse_float(col1)
            if row_num is None or row_num < 1 or row_num > 15:
                continue

            for commodity, cols in COMMODITY_BLOCKS.items():
                try:
                    expiry_raw = row[cols['expiry']] if cols['expiry'] < len(row) else ''
                    dirty_vol = parse_float(row[cols['dirty_vol']] if cols['dirty_vol'] < len(row) else '')

                    expiry = serial_to_date(expiry_raw)
                    if expiry is None or dirty_vol is None or dirty_vol == 0:
                        continue

                    records.append({
                        'date': current_date,
                        'commodity': commodity,
                        'expiry': expiry,
                        'dirty_vol': round(dirty_vol, 4),
                        'fwd_vol': parse_float(row[cols['fwd_vol']] if cols['fwd_vol'] < len(row) else ''),
                        'skew_m1.5': parse_float(row[cols['skew_m1_5']] if cols['skew_m1_5'] < len(row) else ''),
                        'skew_m0.5': parse_float(row[cols['skew_m0_5']] if cols['skew_m0_5'] < len(row) else ''),
                        'skew_p0.5': parse_float(row[cols['skew_p0_5']] if cols['skew_p0_5'] < len(row) else ''),
                        'skew_p1.5': parse_float(row[cols['skew_p1_5']] if cols['skew_p1_5'] < len(row) else ''),
                        'skew_p3.0': parse_float(row[cols['skew_p3']] if cols['skew_p3'] < len(row) else ''),
                        'trading_dte': parse_float(row[cols['trading_dte']] if cols['trading_dte'] < len(row) else ''),
                    })
                except (IndexError, KeyError):
                    continue

    return records


def main():
    setup_logging()
    log = logging.getLogger()

    # Accept input file as positional arg (matches bat: python update_volskew_clean.py eod_vol_snap.csv)
    if len(sys.argv) < 2:
        input_file = os.path.join(PROJECT_DIR, 'data', 'eod_vol_snap.csv')
    else:
        input_file = sys.argv[1]
        if not os.path.isabs(input_file):
            input_file = os.path.join(PROJECT_DIR, 'data', input_file)

    output_file = DEFAULT_OUTPUT

    log.info('=' * 50)
    log.info('Vol/skew parser started')

    if not os.path.exists(input_file):
        log.error(f'Input not found: {input_file}')
        sys.exit(1)

    # Determine cutoff: only parse dates after the latest in master
    cutoff_date = None
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_df['date'] = pd.to_datetime(existing_df['date'], format='mixed').dt.strftime('%Y-%m-%d')
        cutoff_date = max(existing_df['date'])
        log.info(f'Existing master: {len(existing_df)} records (latest: {cutoff_date})')
    else:
        existing_df = None

    # Parse
    log.info(f'Parsing {os.path.basename(input_file)}...')
    all_records = parse_csv_file(input_file)

    if not all_records:
        log.warning('No valid data found. Exiting.')
        sys.exit(1)

    # Filter to only new dates
    if cutoff_date:
        new_records = [r for r in all_records if r['date'] > cutoff_date]
        all_dates = sorted(set(r['date'] for r in all_records))
        log.info(f'  Parsed {len(all_records)} records, {len(all_dates)} dates ({all_dates[0]} to {all_dates[-1]})')
    else:
        new_records = all_records

    # Filter out bad rows where expiry < date
    clean_records = []
    bad_count = 0
    for r in new_records:
        if r['expiry'] and r['date'] and r['expiry'] < r['date']:
            bad_count += 1
        else:
            clean_records.append(r)

    if bad_count:
        log.info(f'  Filtered {bad_count} rows where expiry < date')

    new_df = pd.DataFrame(clean_records)

    if new_df.empty:
        log.info('No new dates to add. Master is up to date.')
        sys.exit(0)

    new_dates = sorted(new_df['date'].unique())
    log.info(f'  +{len(new_df)} new records for {len(new_dates)} dates ({new_dates[0]} to {new_dates[-1]})')

    # Append to master
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'commodity', 'expiry'], keep='last')
    else:
        combined = new_df.drop_duplicates(subset=['date', 'commodity', 'expiry'], keep='last')
        log.info(f'Created new master with {len(combined)} records')

    combined = combined.sort_values(['date', 'commodity', 'expiry'], ascending=[False, True, True]).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    log.info(f'Saved {len(combined)} records -> {os.path.basename(output_file)} ({combined["date"].min()} to {combined["date"].max()})')
    log.info('Done.')


if __name__ == '__main__':
    main()