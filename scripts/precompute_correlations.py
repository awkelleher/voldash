"""
Precompute Front Month Correlation Matrices

Calculates pairwise correlations between front-month futures contracts
for all commodities using log returns over multiple lookback windows.

Output: cache/correlation_matrices.csv
Run once per day after update_from_hertz.py

Usage:
    python scripts/precompute_correlations.py
    python scripts/precompute_correlations.py --force
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICE_FILE = 'data/all_commodity_prices.csv'
OUTPUT_FILE = 'cache/correlation_matrices.csv'
COMMODITIES = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW']
WINDOWS = [10, 20, 30, 50, 100, 200]

MONTH_ORDER = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}


def get_front_month_series(df):
    """
    Build a continuous front-month close price series for each commodity.
    Front month = earliest-expiry contract trading on each date.
    Returns DataFrame with columns: date, commodity, close
    """
    # Parse contract code into a sortable contract date
    df = df.copy()
    df['month_letter'] = df['contract_code'].str[0]
    df['year_num'] = df['contract_code'].str[1:].astype(int) + 2000
    df['month_num'] = df['month_letter'].map(MONTH_ORDER)

    # Drop rows with unmapped month letters
    df = df.dropna(subset=['month_num'])
    df['month_num'] = df['month_num'].astype(int)

    # Build a sortable contract expiry proxy (1st of the delivery month)
    df['contract_date'] = pd.to_datetime(
        df['year_num'].astype(str) + '-' + df['month_num'].astype(str).str.zfill(2) + '-01'
    )

    # Front month = earliest contract_date per (date, commodity)
    idx = df.groupby(['date', 'commodity'])['contract_date'].idxmin()
    front = df.loc[idx, ['date', 'commodity', 'close', 'contract_code']].copy()

    return front


def compute_correlation_matrices(front_df, windows):
    """
    Compute pairwise correlation matrices for each lookback window.
    Uses log returns of front-month close prices.
    """
    # Pivot to wide format: date x commodity
    wide = front_df.pivot_table(index='date', columns='commodity', values='close')
    wide = wide.sort_index()

    # Log returns
    log_ret = np.log(wide / wide.shift(1)).dropna(how='all')

    results = []

    for window in windows:
        if len(log_ret) < window:
            print(f"  Skipping {window}D: not enough data ({len(log_ret)} days)", file=sys.stderr)
            continue

        # Use the most recent N days
        recent = log_ret.iloc[-window:]

        # Correlation matrix
        corr = recent.corr()

        # Store each pair
        for c1 in COMMODITIES:
            for c2 in COMMODITIES:
                if c1 in corr.columns and c2 in corr.columns:
                    val = corr.loc[c1, c2]
                    if pd.notna(val):
                        results.append({
                            'window': window,
                            'commodity_1': c1,
                            'commodity_2': c2,
                            'correlation': round(val, 4),
                        })

    return pd.DataFrame(results)


def main():
    start = time.time()

    # Check --force flag
    force = '--force' in sys.argv

    if os.path.exists(OUTPUT_FILE) and not force:
        mod_time = os.path.getmtime(OUTPUT_FILE)
        age_hours = (time.time() - mod_time) / 3600
        if age_hours < 20:
            print(f"Cache is {age_hours:.1f}h old (< 20h). Use --force to rebuild.", file=sys.stderr)
            sys.exit(0)

    print("Loading price data...", file=sys.stderr)
    df = pd.read_csv(PRICE_FILE)
    df['date'] = pd.to_datetime(df['date'], format='mixed')

    # Filter to our commodities
    df = df[df['commodity'].isin(COMMODITIES)].copy()

    print("Identifying front month contracts...", file=sys.stderr)
    front = get_front_month_series(df)
    print(f"  {len(front)} front-month observations across {front['commodity'].nunique()} commodities", file=sys.stderr)

    # Show current front months
    latest_date = front['date'].max()
    latest_front = front[front['date'] == latest_date].sort_values('commodity')
    print(f"\n  Front months as of {latest_date.strftime('%Y-%m-%d')}:", file=sys.stderr)
    for _, row in latest_front.iterrows():
        print(f"    {row['commodity']:6s}  {row['contract_code']}  {row['close']}", file=sys.stderr)

    print(f"\nComputing correlation matrices ({', '.join(str(w)+'D' for w in WINDOWS)})...", file=sys.stderr)
    result = compute_correlation_matrices(front, WINDOWS)

    if result.empty:
        print("No correlations computed.", file=sys.stderr)
        sys.exit(1)

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start
    print(f"\nSaved {len(result)} entries to {OUTPUT_FILE} ({elapsed:.1f}s)", file=sys.stderr)

    # Print a sample matrix (10D)
    sample_window = WINDOWS[0]
    sample = result[result['window'] == sample_window]
    if not sample.empty:
        matrix = sample.pivot(index='commodity_1', columns='commodity_2', values='correlation')
        matrix = matrix.reindex(index=COMMODITIES, columns=COMMODITIES)
        print(f"\n{sample_window}D Correlation Matrix:", file=sys.stderr)
        print(matrix.to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
