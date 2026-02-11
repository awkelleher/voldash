"""
Precompute Realized Volatility (Latest Snapshot)

Calculates 5, 10, 20, 50-day realized volatility (annualized) for each
individual futures contract, per commodity. Outputs only the most recent
RV values per contract. Caps at 12 nearest-expiry contracts per commodity.

Output: cache/realized_vol_precomputed.csv
Run once per day after update_from_hertz.py

Usage:
    python scripts/precompute_realized_vol.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICE_FILE = 'data/all_commodity_prices.csv'
OUTPUT_FILE = 'cache/realized_vol_precomputed.csv'
COMMODITIES = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW']
RV_WINDOWS = [5, 10, 20, 50]
MAX_CONTRACTS = 12
ANNUALIZATION_FACTOR = np.sqrt(252)

# Contract month ordering (maps letter -> calendar rank for sorting)
MONTH_ORDER = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def contract_sort_key(code):
    """
    Parse contract code (e.g. 'H26') into a sortable tuple (year, month_rank).
    Nearest expiry sorts first.
    """
    if pd.isna(code) or len(code) < 2:
        return (9999, 99)
    month_letter = code[0]
    try:
        year = int('20' + code[1:])
    except ValueError:
        year = 2099
    month_rank = MONTH_ORDER.get(month_letter, 99)
    return (year, month_rank)


def calc_realized_vol(closes, window):
    """
    Realized volatility = annualised standard deviation of log returns
    over a rolling window.

    Returns a Series the same length as closes (NaN where insufficient data).
    """
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * ANNUALIZATION_FACTOR


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def select_nearest_contracts(df, commodity, reference_date, max_n=MAX_CONTRACTS):
    """
    For a given commodity, return the contract codes of the max_n nearest
    contracts that have price data on or near the reference date.
    """
    comm = df[df['commodity'] == commodity]
    recent_cutoff = reference_date - pd.Timedelta(days=7)
    recent = comm[comm['date'] >= recent_cutoff]

    contracts = recent['contract_code'].dropna().unique()
    ranked = sorted(contracts, key=contract_sort_key)
    return ranked[:max_n]


def compute_latest_rv_for_commodity(df, commodity):
    """
    Compute the latest realized vol for each contract of a commodity.
    Returns one row per contract with the most recent RV values only.
    """
    comm = df[df['commodity'] == commodity].copy()
    reference_date = comm['date'].max()

    target_contracts = select_nearest_contracts(df, commodity, reference_date)
    if len(target_contracts) == 0:
        print(f"  ⚠ {commodity}: no contracts found")
        return pd.DataFrame()

    print(f"  {commodity}: {len(target_contracts)} contracts → {', '.join(target_contracts)}")

    rows = []

    for code in target_contracts:
        contract_df = comm[comm['contract_code'] == code].sort_values('date')
        closes = contract_df['close']

        if len(closes) < RV_WINDOWS[0]:
            continue

        latest_date = contract_df['date'].iloc[-1]
        latest_close = closes.iloc[-1]

        row = {
            'date': latest_date,
            'commodity': commodity,
            'contract_code': code,
            'close': latest_close,
        }

        for w in RV_WINDOWS:
            if len(closes) >= w + 1:
                # Use last w+1 closes to get w log returns
                recent_closes = closes.iloc[-(w + 1):].values
                log_ret = np.log(recent_closes[1:] / recent_closes[:-1])
                # RMS realized vol: sqrt(mean(r^2)) * sqrt(252), as percentage
                row[f'rv_{w}d'] = round(np.sqrt(np.mean(log_ret ** 2)) * ANNUALIZATION_FACTOR * 100, 2)
            else:
                row[f'rv_{w}d'] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_precompute(price_file=PRICE_FILE, output_file=OUTPUT_FILE):
    """Full precompute pipeline - latest snapshot only."""
    start = time.time()

    print("=" * 60)
    print("PRECOMPUTE REALIZED VOLATILITY (latest snapshot)")
    print("=" * 60)

    # Load prices
    print(f"\nLoading {price_file} ...")
    df = pd.read_csv(price_file)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values(['commodity', 'contract_code', 'date'])
    print(f"  {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")

    # Compute per commodity
    print(f"\nCalculating RV (windows: {RV_WINDOWS}) ...\n")
    all_results = []
    for commodity in COMMODITIES:
        result = compute_latest_rv_for_commodity(df, commodity)
        if len(result) > 0:
            all_results.append(result)

    if not all_results:
        print("\n⚠ No results generated.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Save
    combined.to_csv(output_file, index=False)
    elapsed = time.time() - start
    print(f"\n✓ Saved {len(combined)} rows to {output_file}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Print snapshot
    print_snapshot(combined)


def print_snapshot(combined):
    """Print the RV snapshot table."""
    print("\n" + "=" * 80)
    print("LATEST REALIZED VOLATILITY (annualized %)")
    print("=" * 80)

    for commodity in COMMODITIES:
        comm = combined[combined['commodity'] == commodity]
        if comm.empty:
            continue
        latest_date = comm['date'].max()
        rows = comm.sort_values('contract_code', key=lambda s: s.map(contract_sort_key))
        print(f"\n{commodity}  ({pd.to_datetime(latest_date).date()})")
        print(f"  {'Contract':<10} {'Close':>10} {'5D RV':>8} {'10D RV':>8} {'20D RV':>8} {'50D RV':>8}")
        print(f"  {'-'*54}")
        for _, row in rows.iterrows():
            rv5  = f"{row['rv_5d']:.3f}" if pd.notna(row['rv_5d']) else '-'
            rv10 = f"{row['rv_10d']:.3f}" if pd.notna(row['rv_10d']) else '-'
            rv20 = f"{row['rv_20d']:.3f}" if pd.notna(row['rv_20d']) else '-'
            rv50 = f"{row['rv_50d']:.3f}" if pd.notna(row['rv_50d']) else '-'
            print(f"  {row['contract_code']:<10} {row['close']:>10.2f} {rv5:>8} {rv10:>8} {rv20:>8} {rv50:>8}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_precompute()
