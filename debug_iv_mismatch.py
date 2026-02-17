"""
Debug script: Compare iv_hist_median in parquet snapshot vs median_iv in CSV.
"""
import pandas as pd
import numpy as np
import os

os.chdir(r"C:\Users\AdamKelleher\ags_book_streamlit")

print("=" * 80)
print("STEP 1: Load parquet snapshot CORN_H_all.parquet")
print("=" * 80)

parquet_path = "cache/iv_percentiles/snapshots/CORN_H_all.parquet"
if not os.path.exists(parquet_path):
    print(f"  ERROR: {parquet_path} does not exist!")
else:
    snap = pd.read_parquet(parquet_path)
    print(f"\n  Shape: {snap.shape}")
    print(f"  All columns: {list(snap.columns)}\n")

    cols_to_show = ['options_month', 'atm_iv', 'iv_hist_median', 'iv_percentile', 'curve_position']
    missing_cols = [c for c in cols_to_show if c not in snap.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
        cols_to_show = [c for c in cols_to_show if c in snap.columns]

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}')
    print(snap[cols_to_show].to_string(index=True))

    # Also show as_of_date
    if 'as_of_date' in snap.columns:
        print(f"\n  as_of_date values: {snap['as_of_date'].unique()}")
    else:
        print("\n  WARNING: 'as_of_date' column not found!")
        # Check for any date-like columns
        date_cols = [c for c in snap.columns if 'date' in c.lower()]
        print(f"  Date-like columns: {date_cols}")
        for dc in date_cols:
            print(f"    {dc}: {snap[dc].unique()}")

print("\n" + "=" * 80)
print("STEP 2: Load cache/median_iv.csv, filter for CORN / FRONT_OPTIONS=H")
print("=" * 80)

csv_path = "cache/median_iv.csv"
if not os.path.exists(csv_path):
    print(f"  ERROR: {csv_path} does not exist!")
else:
    miv = pd.read_csv(csv_path)
    print(f"\n  Full CSV shape: {miv.shape}")
    print(f"  All columns: {list(miv.columns)}")

    corn_h = miv[(miv['commodity'] == 'CORN') & (miv['FRONT_OPTIONS'] == 'H')]
    print(f"\n  Filtered (CORN, H) shape: {corn_h.shape}")

    cols_csv = ['OPTIONS', 'median_iv', 'obs_count']
    missing_csv = [c for c in cols_csv if c not in corn_h.columns]
    if missing_csv:
        print(f"  WARNING: Missing columns: {missing_csv}")
        cols_csv = [c for c in cols_csv if c in corn_h.columns]

    print(corn_h[cols_csv].to_string(index=False))

print("\n" + "=" * 80)
print("STEP 3: Compare iv_hist_median (parquet) vs median_iv (CSV)")
print("=" * 80)

if os.path.exists(parquet_path) and os.path.exists(csv_path):
    snap = pd.read_parquet(parquet_path)
    miv = pd.read_csv(csv_path)
    corn_h = miv[(miv['commodity'] == 'CORN') & (miv['FRONT_OPTIONS'] == 'H')]

    csv_lookup = corn_h.set_index('OPTIONS')['median_iv'].to_dict()

    comparison_rows = []
    for _, row in snap.iterrows():
        om = row.get('options_month', None)
        parq_median = row.get('iv_hist_median', np.nan)
        csv_median = csv_lookup.get(om, np.nan)
        diff = np.nan
        if pd.notna(parq_median) and pd.notna(csv_median):
            diff = parq_median - csv_median
        comparison_rows.append({
            'options_month': om,
            'curve_position': row.get('curve_position', np.nan),
            'atm_iv': row.get('atm_iv', np.nan),
            'parquet_iv_hist_median': parq_median,
            'csv_median_iv': csv_median,
            'difference': diff,
            'match': 'YES' if (pd.notna(parq_median) and pd.notna(csv_median) and abs(diff) < 0.0001) else ('N/A' if pd.isna(csv_median) or pd.isna(parq_median) else 'NO'),
        })

    comp = pd.DataFrame(comparison_rows)
    print(comp.to_string(index=False))

    matches = (comp['match'] == 'YES').sum()
    mismatches = (comp['match'] == 'NO').sum()
    na_count = (comp['match'] == 'N/A').sum()
    print(f"\n  Summary: {matches} matches, {mismatches} mismatches, {na_count} N/A")

print("\n" + "=" * 80)
print("STEP 4: Check as_of_date freshness")
print("=" * 80)

if os.path.exists(parquet_path):
    snap = pd.read_parquet(parquet_path)
    if 'as_of_date' in snap.columns:
        as_of = snap['as_of_date'].iloc[0]
        print(f"  as_of_date in snapshot: {as_of}")
        print(f"  Today's date:           2026-02-13")
        try:
            as_of_dt = pd.to_datetime(as_of)
            today = pd.Timestamp('2026-02-13')
            delta = (today - as_of_dt).days
            print(f"  Days stale:             {delta}")
            if delta > 1:
                print(f"  WARNING: Snapshot is {delta} days old!")
            else:
                print("  Snapshot appears current.")
        except Exception as e:
            print(f"  Could not parse as_of_date: {e}")
    else:
        print("  No 'as_of_date' column found.")
        if 'date' in snap.columns:
            print(f"  'date' column max: {snap['date'].max()}")

print("\n" + "=" * 80)
print("STEP 5: Check lookback-specific snapshots for differences")
print("=" * 80)

# Also check a lookback-specific snapshot (e.g., CORN_H_5.parquet for 5-year lookback)
for lb in ['1', '3', '5', 'all']:
    lb_path = f"cache/iv_percentiles/snapshots/CORN_H_{lb}.parquet"
    if os.path.exists(lb_path):
        s = pd.read_parquet(lb_path)
        print(f"\n  CORN_H_{lb}.parquet:")
        if 'iv_hist_median' in s.columns and 'options_month' in s.columns:
            for _, row in s.iterrows():
                om = row['options_month']
                parq_med = row.get('iv_hist_median', np.nan)
                csv_med = csv_lookup.get(om, np.nan) if os.path.exists(csv_path) else np.nan
                match_str = ''
                if pd.notna(parq_med) and pd.notna(csv_med):
                    match_str = 'MATCH' if abs(parq_med - csv_med) < 0.0001 else f'DIFF={parq_med - csv_med:.6f}'
                print(f"    {om}: parquet={parq_med:.6f} csv={csv_med:.6f} {match_str}" if pd.notna(parq_med) else f"    {om}: parquet=NaN csv={csv_med}")

print("\nDone.")
