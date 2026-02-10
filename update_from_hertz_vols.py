"""
Update Vol/Skew Data from hertzsoy VOLS Workbook

Reads the Parameters and Monthly&BEs sheets from your hertzsoy.XX.VOLS.xlsm file
and appends a new daily snapshot to historical_vol_skew_all_commodities.csv.

Usage:
    python update_from_hertz_vols.py
    python update_from_hertz_vols.py "C:/path/to/hertzsoy.22.VOLS.xlsm"
    python update_from_hertz_vols.py --date 2026-02-07
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import glob


# ── Column layout in the Parameters sheets ──────────────────────────────
# Col 2  = contract number (1-based ordinal)
# Col 3  = product code (S, SM, BO, C, W, KW)
# Col 4  = Volatility (dirty vol)
# Col 6  = InterestRate
# Col 7  = PivotValue
# Col 8  = VolDiffA  -> left_slope
# Col 9  = VolDiffB  -> right_slope
# Col 10 = VolDiffD  -> skew_neg15
# Col 11 = VolDiffE  -> skew_neg05
# Col 12 = VolDiffF  -> skew_pos05
# Col 13 = VolDiffG  -> skew_pos15
# Col 14 = VolDiffH  -> skew_pos3
# Col 19 = OptionExpiration

PARAM_SHEETS = {
    'SOY':   'SOY Parameters',
    'MEAL':  'MEAL Parameters',
    'OIL':   'OIL Parameters',
    'CORN':  'CORN Parameters',
    'WHEAT': 'WHEAT Parameters',
    'KW':    'KW Parameters',
}

# Commodity header labels in the Monthly&BEs sheet
MONTHLY_LABELS = {
    'SOY':   'SOY Implied Fwd vols by Month',
    'MEAL':  'MEAL Implied Fwd vols by Month',
    'OIL':   'OIL Implied Fwd vols by Month',
    'CORN':  'CORN Implied Fwd vols by Month',
    'WHEAT': 'WHEAT Implied Fwd vols by Month',
    'KW':    'KW Implied Fwd vols by Month',
}

BE_LABELS = {
    'SOY':   'SOY BreakEvens',
    'MEAL':  'MEAL BreakEvens',
    'OIL':   'OIL BreakEvens',
    'CORN':  'CORN BreakEvens',
    'WHEAT': 'WHEAT BreakEvens',
    'KW':    'KW BreakEvens',
}


def _safe_float(val):
    """Convert to float, returning NaN on failure."""
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _parse_expiry(val):
    """Parse expiry from either a datetime or an Excel serial number."""
    if pd.isna(val):
        return pd.NaT
    # If already a datetime-like, use directly
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val)
    # Try Excel serial date first (int or float in the 30000-60000 range)
    try:
        serial = int(float(val))
        if 30000 < serial < 60000:
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=serial)
    except (ValueError, TypeError):
        pass
    # Fallback: try string parsing
    try:
        return pd.to_datetime(val)
    except Exception:
        pass
    return pd.NaT


def parse_parameters(xl, commodity, sheet_name):
    """
    Parse a Parameters sheet into a list of row dicts.

    Returns list of dicts, one per contract month.
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    rows = []

    for i in range(1, min(14, len(df))):
        cm = df.iloc[i, 2]
        try:
            cm = int(cm)
        except (ValueError, TypeError):
            continue
        if cm < 1 or cm > 12:
            continue

        vol = _safe_float(df.iloc[i, 4])
        if np.isnan(vol) or vol == 0:
            continue  # skip months without vol or zero vol

        pivot = _safe_float(df.iloc[i, 7])
        expiry = _parse_expiry(df.iloc[i, 19])

        rows.append({
            'commodity': commodity,
            'contract_month': cm,
            'dirty_vol': vol,
            'pivot': pivot,
            'expiry': expiry,
            'left_slope': _safe_float(df.iloc[i, 8]),
            'right_slope': _safe_float(df.iloc[i, 9]),
            'skew_neg15': _safe_float(df.iloc[i, 10]),
            'skew_neg05': _safe_float(df.iloc[i, 11]),
            'skew_pos05': _safe_float(df.iloc[i, 12]),
            'skew_pos15': _safe_float(df.iloc[i, 13]),
            'skew_pos3': _safe_float(df.iloc[i, 14]),
        })

    return rows


def parse_monthly_bes(xl, commodity_rows_map):
    """
    Parse the Monthly&BEs sheet to extract forward vols, clean vols,
    and breakevens for each commodity.

    commodity_rows_map: dict {commodity: num_contracts} from Parameters parsing.

    Returns dict {(commodity, contract_month): {fwd_vol, clean_vol, be, fwd_be}}
    """
    try:
        df = pd.read_excel(xl, 'Monthly&BEs', header=None)
    except Exception:
        return {}

    result = {}

    # Find where each commodity's fwd vol block starts
    for commodity, label in MONTHLY_LABELS.items():
        if commodity not in commodity_rows_map:
            continue

        # Find the header row
        header_row = None
        for i in range(len(df)):
            val = df.iloc[i, 0]
            if pd.notna(val) and str(val).strip() == label:
                header_row = i
                break

        if header_row is None:
            continue

        # Find the corresponding BE header
        be_label = BE_LABELS.get(commodity, '')
        be_col_start = None
        for c in range(len(df.columns)):
            val = df.iloc[header_row, c]
            if pd.notna(val) and str(val).strip() == be_label:
                be_col_start = c
                break

        # Read each contract row after the header
        num_contracts = commodity_rows_map[commodity]
        for offset in range(1, num_contracts + 1):
            row_idx = header_row + offset
            if row_idx >= len(df):
                break

            cm = offset  # contract_month ordinal

            # Forward vol = diagonal element (last non-NaN in the fwd vol columns)
            # The matrix is triangular: row has values in cols 1..offset
            # Clean vol = col 1 value (cumulative vol from now to this expiry)
            # Fwd vol = col offset value (marginal vol for this specific period)
            clean_vol = _safe_float(df.iloc[row_idx, 1])
            fwd_vol = _safe_float(df.iloc[row_idx, offset])

            # Breakevens
            be = np.nan
            fwd_be = np.nan
            if be_col_start is not None:
                be = _safe_float(df.iloc[row_idx, be_col_start + 1])  # +1 to skip the label col

            result[(commodity, cm)] = {
                'clean_vol': clean_vol,
                'fwd_vol': fwd_vol,
                'be': be,
                'fwd_be': np.nan,  # Will derive below if data available
            }

    # Try to get fwd_be from the "Month N" columns (cols 27+)
    # These appear to be individual month breakevens
    # fwd_be for month N = the Nth "Month" column breakeven
    try:
        for commodity, label in MONTHLY_LABELS.items():
            if commodity not in commodity_rows_map:
                continue
            header_row = None
            for i in range(len(df)):
                val = df.iloc[i, 0]
                if pd.notna(val) and str(val).strip() == label:
                    header_row = i
                    break
            if header_row is None:
                continue

            # Find "Month 1", "Month 2" etc headers in the header row
            month_cols = {}
            for c in range(len(df.columns)):
                val = df.iloc[header_row, c]
                if pd.notna(val) and str(val).startswith('Month '):
                    try:
                        mn = int(str(val).replace('Month ', ''))
                        month_cols[mn] = c
                    except ValueError:
                        pass

            num_contracts = commodity_rows_map[commodity]
            for offset in range(1, num_contracts + 1):
                row_idx = header_row + offset
                cm = offset
                key = (commodity, cm)
                if key in result and cm in month_cols:
                    # The fwd_be for month cm is in the first data row, column for Month cm
                    fwd_be_val = _safe_float(df.iloc[header_row + 1, month_cols[cm]])
                    if not np.isnan(fwd_be_val):
                        result[key]['fwd_be'] = fwd_be_val
    except Exception:
        pass

    return result


def calculate_dte(expiry, today):
    """Calculate trading DTE and clean DTE from expiry date."""
    if pd.isna(expiry):
        return np.nan, np.nan

    delta = (expiry - today).days
    if delta < 0:
        return np.nan, np.nan

    # Trading DTE ≈ calendar days × (252/365)
    trading_dte = round(delta * 252 / 365, 1)
    # Clean DTE = slightly adjusted (accounts for weekends etc.)
    clean_dte = round(delta * 252 / 365 * 1.045, 1)

    return trading_dte, clean_dte


def build_daily_snapshot(hertz_path, date=None):
    """
    Build a complete daily snapshot from the hertzsoy VOLS workbook.

    Args:
        hertz_path: Path to the .xlsm file
        date: Date for this snapshot (default: today)

    Returns:
        DataFrame matching historical_vol_skew_all_commodities.csv schema
    """
    if date is None:
        date = pd.Timestamp(datetime.now().date())
    else:
        date = pd.to_datetime(date)

    print(f"\nReading: {hertz_path}")
    xl = pd.ExcelFile(hertz_path)
    available_sheets = xl.sheet_names

    # ── Parse Parameters sheets ──
    all_rows = []
    commodity_counts = {}

    for commodity, sheet in PARAM_SHEETS.items():
        if sheet not in available_sheets:
            print(f"  [!] {commodity}: sheet '{sheet}' not found, skipping")
            continue

        rows = parse_parameters(xl, commodity, sheet)
        if rows:
            all_rows.extend(rows)
            commodity_counts[commodity] = len(rows)
            print(f"  [OK] {commodity}: {len(rows)} months")
        else:
            print(f"  [!] {commodity}: no data parsed")

    if not all_rows:
        print("\n[ERROR] No data parsed from any Parameters sheet!")
        return pd.DataFrame()

    # ── Parse Monthly&BEs ──
    monthly_data = {}
    if 'Monthly&BEs' in available_sheets:
        monthly_data = parse_monthly_bes(xl, commodity_counts)
        print(f"  [OK] Monthly&BEs: {len(monthly_data)} entries")
    else:
        print("  [!] Monthly&BEs sheet not found - fwd_vol/clean_vol will be estimated")

    # ── Merge and build final rows ──
    final_rows = []

    for row in all_rows:
        commodity = row['commodity']
        cm = row['contract_month']

        # Merge Monthly&BEs data
        mb = monthly_data.get((commodity, cm), {})
        clean_vol = mb.get('clean_vol', row['dirty_vol'])  # fallback to dirty
        fwd_vol = mb.get('fwd_vol', row['dirty_vol'])
        be = mb.get('be', np.nan)
        fwd_be = mb.get('fwd_be', np.nan)

        # Calculate DTE
        trading_dte, clean_dte = calculate_dte(row['expiry'], date)

        # Skip rows with expired contracts (expiry before snapshot date)
        if pd.notna(row['expiry']) and row['expiry'] < date:
            continue

        final_rows.append({
            'date': date,
            'commodity': commodity,
            'contract_month': cm,
            'expiry': row['expiry'],
            'pivot': row['pivot'],
            'dirty_vol': row['dirty_vol'],
            'clean_vol': clean_vol,
            'fwd_vol': fwd_vol,
            'be': be,
            'fwd_be': fwd_be,
            'left_slope': row['left_slope'],
            'right_slope': row['right_slope'],
            'skew_neg15': row['skew_neg15'],
            'skew_neg05': row['skew_neg05'],
            'skew_pos05': row['skew_pos05'],
            'skew_pos15': row['skew_pos15'],
            'skew_pos3': row['skew_pos3'],
            'trading_dte': trading_dte,
            'clean_dte': clean_dte,
        })

    snapshot = pd.DataFrame(final_rows)
    return snapshot


def update_historical_csv(snapshot, csv_path='historical_vol_skew_all_commodities.csv'):
    """
    Append a daily snapshot to the historical CSV.
    Handles deduplication (same date+commodity+contract_month = replace).
    Creates a dated backup before modifying.
    """
    date = snapshot['date'].iloc[0]
    date_str = date.strftime('%Y-%m-%d')

    if os.path.exists(csv_path):
        print(f"\nLoading existing: {csv_path}")
        existing = pd.read_csv(csv_path, parse_dates=['date', 'expiry'])
        print(f"  Existing: {len(existing):,} rows, {existing['date'].min().date()} to {existing['date'].max().date()}")

        # Check if this date already exists
        existing_dates = existing['date'].dt.date.unique()
        if date.date() in existing_dates:
            existing_count = len(existing[(existing['date'].dt.date == date.date())])
            print(f"  [!] {date_str} already has {existing_count} rows -- will be REPLACED")

        # Backup (once per day)
        backup = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
        if not os.path.exists(backup):
            existing.to_csv(backup, index=False)
            print(f"  [OK] Backup: {backup}")

        # Remove existing rows for this date, then append new
        combined = existing[existing['date'].dt.date != date.date()]
        combined = pd.concat([combined, snapshot], ignore_index=True)
        combined = combined.sort_values(['date', 'commodity', 'contract_month']).reset_index(drop=True)
    else:
        print(f"\nNo existing file — creating new {csv_path}")
        combined = snapshot

    combined.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved: {len(combined):,} total rows")
    print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

    # Summary
    new_data = combined[combined['date'].dt.date == date.date()]
    print(f"\n  {date_str} snapshot: {len(new_data)} rows")
    for comm in sorted(new_data['commodity'].unique()):
        comm_data = new_data[new_data['commodity'] == comm]
        months = sorted(comm_data['contract_month'].unique())
        print(f"    {comm:6} M1-M{max(months)} ({len(months)} months)")

    return combined


def find_hertz_vols_file():
    """Find the hertzsoy VOLS file in common locations."""
    search_dirs = [
        os.path.expanduser('~/Desktop'),
        os.path.expanduser('~/Downloads'),
        os.path.expanduser('~/Documents'),
        os.getcwd(),
        'C:/Users/AdamKelleher/OneDrive - Prime Trading/DR files',
    ]

    patterns = ['hertzsoy*.VOLS*.xlsm', 'hertzsoy*.vols*.xlsm']

    found = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(d, pat)))

    if not found:
        return None

    # Return the most recently modified
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]


def main():
    print("=" * 70)
    print("UPDATE VOLS/SKEW FROM HERTZSOY VOLS WORKBOOK")
    print("=" * 70)

    # Determine file path
    hertz_path = None
    date_override = None

    for arg in sys.argv[1:]:
        if arg.startswith('--date'):
            # --date 2026-02-07  or  --date=2026-02-07
            if '=' in arg:
                date_override = arg.split('=', 1)[1]
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    date_override = sys.argv[idx + 1]
        elif not arg.startswith('--'):
            hertz_path = arg

    if hertz_path is None:
        print("\nSearching for hertzsoy VOLS file...")
        hertz_path = find_hertz_vols_file()

    if hertz_path is None or not os.path.exists(hertz_path):
        print(f"\n[ERROR] File not found: {hertz_path or 'no file specified'}")
        print("\nUsage:")
        print('  python update_from_hertz_vols.py "C:/path/to/hertzsoy.22.VOLS.xlsm"')
        print('  python update_from_hertz_vols.py --date 2026-02-07')
        print("\nOr place the file on your Desktop/Downloads and run without arguments.")
        return False

    print(f"\nFile: {hertz_path}")
    print(f"Size: {os.path.getsize(hertz_path) / 1024 / 1024:.1f} MB")

    # Parse date
    date = pd.Timestamp(datetime.now().date()) if date_override is None else pd.to_datetime(date_override)
    print(f"Date: {date.date()}")

    # Build snapshot
    snapshot = build_daily_snapshot(hertz_path, date)

    if len(snapshot) == 0:
        print("\n[ERROR] No data extracted!")
        return False

    print(f"\n[OK] Snapshot: {len(snapshot)} rows for {date.date()}")

    # Preview
    print("\n── PREVIEW ──")
    for comm in sorted(snapshot['commodity'].unique()):
        comm_data = snapshot[snapshot['commodity'] == comm].sort_values('contract_month')
        m1 = comm_data[comm_data['contract_month'] == 1]
        if len(m1) > 0:
            r = m1.iloc[0]
            print(f"  {comm:6} M1: dirty={r['dirty_vol']:.2f}  clean={r['clean_vol']:.2f}  "
                  f"fwd={r['fwd_vol']:.2f}  pivot={r['pivot']:.2f}  "
                  f"DTE={r['trading_dte']}")

    # Confirm
    print()
    confirm = input("Append to historical CSV? [Y/n] ").strip().lower()
    if confirm in ('', 'y', 'yes'):
        update_historical_csv(snapshot)
        print("\n" + "=" * 70)
        print("VOL UPDATE COMPLETE")
        print("=" * 70)
        print("\nRefresh your Streamlit dashboard (press R) to see the new data.")
        return True
    else:
        print("\nCancelled.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
