"""
IV Percentile Pre-Computation
==============================
Calculates historical IV percentile rankings conditioned on which options month
is front. Pre-computes to Parquet for instant Streamlit loading.

Core concept:
    "When H is the front options month, where does each contract's current IV
    rank vs all historical days when H was also front?"

Data source: master_vol_skew.csv
    Columns: date, commodity, expiry, dirty_vol, skew_m1.5, skew_m0.5,
             skew_p0.5, skew_p1.5, skew_p3.0, trading_dte

    dirty_vol = ATM IV
    Skew points are std devs from ATM: P2(-1.5σ), P1(-0.5σ), C1(+0.5σ), C2(+1.5σ), C3(+3.0σ)

Usage:
    python iv_percentiles_precompute.py                          # compute all
    python iv_percentiles_precompute.py --commodity SOY --month H # single combo
    python iv_percentiles_precompute.py --status                  # cache status
    python iv_percentiles_precompute.py --force                   # recompute all

Schedule (crontab - every Sunday at 6:30pm, after variance ratios):
    30 18 * * 0 cd /path/to/project && python iv_percentiles_precompute.py
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache/iv_percentiles")
VOL_SKEW_PATH = r"C:\Users\AdamKelleher\ags_book_streamlit\master_vol_skew.csv"
MAPPING_PATH = "mapping.csv"

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]

# Column renames for clarity
SKEW_RENAME = {
    "dirty_vol": "atm_iv",
    "skew_m1.5": "P2",    # -1.5σ put
    "skew_m0.5": "P1",    # -0.5σ put
    "skew_p0.5": "C1",    # +0.5σ call
    "skew_p1.5": "C2",    # +1.5σ call
    "skew_p3.0": "C3",    # +3.0σ call
}

COMMODITIES = ["SOY", "MEAL", "OIL", "CORN", "WHEAT", "KW"]
LOOKBACK_OPTIONS = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ── Mapping Helpers ────────────────────────────────────────────────────────────

_mapping_cache = None


def load_mapping() -> pd.DataFrame:
    global _mapping_cache
    if _mapping_cache is None:
        _mapping_cache = pd.read_csv(MAPPING_PATH)
        for col in ["OPTIONS", "FUTURES", "COMMODITY"]:
            _mapping_cache[col] = _mapping_cache[col].astype(str).str.upper()
    return _mapping_cache


def options_to_futures(option_code: str, commodity: str) -> str:
    mapping = load_mapping()
    row = mapping[
        (mapping["OPTIONS"] == option_code.upper())
        & (mapping["COMMODITY"] == commodity.upper())
    ]
    if row.empty:
        return option_code.upper()
    return row.iloc[0]["FUTURES"]


def expiry_month_to_options_code(expiry_month: int, commodity: str) -> str:
    """Map a calendar month (1-12) to the options month code for that commodity."""
    mapping = load_mapping()
    row = mapping[
        (mapping["EXPIRY_MONTH"] == expiry_month)
        & (mapping["COMMODITY"] == commodity.upper())
    ]
    if row.empty:
        return MONTH_CODES[expiry_month - 1]  # fallback
    return row.iloc[0]["OPTIONS"]


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_vol_skew(path: str = None) -> pd.DataFrame:
    """
    Load and normalize master_vol_skew.csv.

    Adds columns:
        - options_month: the options contract month code (derived from expiry month)
        - front_options_month: which options month was front on the observation date
        - contract_label: e.g. "H25" for display
        - curve_position: how far out from front (0 = front)
    """
    if path is None:
        path = VOL_SKEW_PATH

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["expiry"] = pd.to_datetime(df["expiry"], format="mixed")
    df["commodity"] = df["commodity"].str.upper()

    # Rename for clarity
    df = df.rename(columns=SKEW_RENAME)

    # Derive options month code from expiry date's month
    df["expiry_month"] = df["expiry"].dt.month
    df["expiry_year"] = df["expiry"].dt.year

    # Build lookup: (commodity, expiry_month) -> options_month
    mapping = load_mapping()
    lookup = mapping.set_index(["COMMODITY", "EXPIRY_MONTH"])["OPTIONS"].to_dict()

    # Vectorized mapping via merge instead of apply (much faster on 290k rows)
    df["options_month"] = df[["commodity", "expiry_month"]].apply(
        lambda r: lookup.get((r["commodity"], r["expiry_month"]), "?"), axis=1
    )

    # Front options month on each observation date
    df["obs_month"] = df["date"].dt.month
    df["front_options_month"] = df[["commodity", "obs_month"]].apply(
        lambda r: lookup.get((r["commodity"], r["obs_month"]), "?"), axis=1
    )

    # Contract label for display (e.g., "H25")
    df["contract_label"] = (
        df["options_month"] + (df["expiry_year"] % 100).astype(str).str.zfill(2)
    )

    # Curve position: how many months out from front (0 = front)
    front_idx = df["front_options_month"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else -1
    )
    opt_idx = df["options_month"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else -1
    )
    df["curve_position"] = (opt_idx - front_idx) % 12

    df.drop(columns=["obs_month"], inplace=True)

    return df


# ── IV Percentile Calculation ──────────────────────────────────────────────────

def calculate_iv_percentiles(
    df: pd.DataFrame,
    front_options_month: str,
    commodity: str,
    lookback_years: int | None = None,
) -> pd.DataFrame:
    """
    For all dates when `front_options_month` was front, calculate the ATM IV
    percentile rank of each available contract.

    Percentile = "what % of historical IV readings (when this same options month
    was front) were below the current IV?"

    Returns DataFrame with all original columns plus:
        iv_percentile, n_observations, and skew percentile columns
    """
    sub = df[
        (df["commodity"] == commodity.upper())
        & (df["front_options_month"] == front_options_month.upper())
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    # Apply lookback filter
    if lookback_years is not None and lookback_years > 0:
        cutoff = sub["date"].max() - pd.DateOffset(years=lookback_years)
        sub = sub[sub["date"] >= cutoff]

    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["options_month", "date"])

    # For each contract month, compute expanding percentile rank
    results = []
    for opt_month, group in sub.groupby("options_month"):
        g = group.sort_values("date").copy()
        if len(g) < 2:
            continue

        # ATM IV percentile
        g["iv_percentile"] = g["atm_iv"].expanding(min_periods=1).rank(pct=True)
        g["n_observations"] = range(1, len(g) + 1)

        # Skew percentiles
        for skew_col in ["P2", "P1", "C1", "C2", "C3"]:
            if skew_col in g.columns:
                g[f"{skew_col}_pctile"] = (
                    g[skew_col].expanding(min_periods=1).rank(pct=True)
                )

        results.append(g)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True).sort_values(["date", "curve_position"])


def build_snapshot(
    df: pd.DataFrame,
    front_options_month: str,
    commodity: str,
    lookback_years: int | None = None,
) -> pd.DataFrame:
    """
    Snapshot for the most recent date: each contract's IV and percentile.
    Primary Streamlit display table.
    """
    full = calculate_iv_percentiles(df, front_options_month, commodity, lookback_years)
    if full.empty:
        return pd.DataFrame()

    latest_date = full["date"].max()
    snapshot = full[full["date"] == latest_date].copy()

    # Historical summary stats per contract month
    hist_stats = (
        full.groupby("options_month")["atm_iv"]
        .agg(["min", "max", "mean", "median"])
        .rename(columns={
            "min": "iv_hist_min",
            "max": "iv_hist_max",
            "mean": "iv_hist_mean",
            "median": "iv_hist_median",
        })
    )
    snapshot = snapshot.merge(hist_stats, on="options_month", how="left")

    # Historical skew stats
    for sc in ["P2", "P1", "C1", "C2", "C3"]:
        if sc in full.columns:
            stats = (
                full.groupby("options_month")[sc]
                .agg(["min", "max", "mean", "median"])
                .rename(columns={
                    "min": f"{sc}_hist_min",
                    "max": f"{sc}_hist_max",
                    "mean": f"{sc}_hist_mean",
                    "median": f"{sc}_hist_median",
                })
            )
            snapshot = snapshot.merge(stats, on="options_month", how="left")

    snapshot["as_of_date"] = latest_date

    return snapshot.sort_values("curve_position")


# ── Cache Paths ────────────────────────────────────────────────────────────────

def _cache_key(commodity: str, options_month: str, lookback: int | None) -> str:
    lb = "all" if lookback is None else str(lookback)
    return f"{commodity}_{options_month}_{lb}"


def _snapshot_path(commodity: str, options_month: str, lookback: int | None) -> Path:
    return CACHE_DIR / "snapshots" / f"{_cache_key(commodity, options_month, lookback)}.parquet"


def _timeseries_path(commodity: str, options_month: str, lookback: int | None) -> Path:
    return CACHE_DIR / "timeseries" / f"{_cache_key(commodity, options_month, lookback)}.parquet"


def _metadata_path(commodity: str, options_month: str, lookback: int | None) -> Path:
    return CACHE_DIR / "metadata" / f"{_cache_key(commodity, options_month, lookback)}.json"


# ── Pre-Compute ────────────────────────────────────────────────────────────────

def precompute_single(
    vol_df: pd.DataFrame,
    commodity: str,
    options_month: str,
    lookback: int | None,
) -> bool:
    """Compute and cache IV + skew percentiles for one combo."""
    try:
        # 1. Snapshot (current percentiles for ATM + skew)
        snapshot = build_snapshot(vol_df, options_month, commodity, lookback)
        if snapshot.empty:
            return False

        # 2. Full timeseries (for charting)
        timeseries = calculate_iv_percentiles(vol_df, options_month, commodity, lookback)

        # 3. Metadata
        meta = {
            "commodity": commodity,
            "front_options_month": options_month,
            "front_futures_month": options_to_futures(options_month, commodity),
            "lookback_years": lookback,
            "as_of_date": str(snapshot["as_of_date"].iloc[0]),
            "contracts_available": sorted(snapshot["options_month"].unique().tolist()),
            "n_contracts": len(snapshot),
            "date_range_start": str(timeseries["date"].min()) if not timeseries.empty else None,
            "date_range_end": str(timeseries["date"].max()) if not timeseries.empty else None,
            "total_observations": len(timeseries),
            "computed_at": datetime.now().isoformat(),
        }

        # Save
        for p_func in [_snapshot_path, _timeseries_path, _metadata_path]:
            p_func(commodity, options_month, lookback).parent.mkdir(parents=True, exist_ok=True)

        snapshot.to_parquet(_snapshot_path(commodity, options_month, lookback))
        timeseries.to_parquet(_timeseries_path(commodity, options_month, lookback))

        with open(_metadata_path(commodity, options_month, lookback), "w") as f:
            json.dump(meta, f, indent=2)

        return True

    except Exception as e:
        print(f"  ERROR {commodity} {options_month} lb={lookback}: {e}")
        return False


def precompute_all(commodity_filter: str = None, month_filter: str = None):
    """Run full pre-computation across all commodities/months/lookbacks."""
    print(f"Loading vol/skew data from {VOL_SKEW_PATH}...")
    vol_df = load_vol_skew()
    print(f"  {len(vol_df):,} records loaded")
    print(f"  Date range: {vol_df['date'].min().date()} to {vol_df['date'].max().date()}")
    print(f"  Commodities: {sorted(vol_df['commodity'].unique())}")
    print()

    commodities = COMMODITIES
    if commodity_filter:
        commodities = [c for c in commodities if c == commodity_filter.upper()]

    total = 0
    success = 0
    skipped = 0

    for commodity in commodities:
        comm_df = vol_df[vol_df["commodity"] == commodity]
        if comm_df.empty:
            print(f"  No data for {commodity}, skipping")
            continue

        # Get all options months for this commodity
        mapping = load_mapping()
        months = mapping[mapping["COMMODITY"] == commodity]["OPTIONS"].unique().tolist()

        if month_filter:
            months = [m for m in months if m == month_filter.upper()]

        for month in months:
            for lookback in LOOKBACK_OPTIONS:
                total += 1
                lb_str = "all" if lookback is None else f"{lookback}y"
                label = f"{commodity} {month} ({lb_str})"

                # Skip if cache is fresh (< 7 days old)
                out = _snapshot_path(commodity, month, lookback)
                if out.exists():
                    age_days = (datetime.now().timestamp() - out.stat().st_mtime) / 86400
                    if age_days < 7:
                        skipped += 1
                        continue

                ok = precompute_single(vol_df, commodity, month, lookback)
                if ok:
                    success += 1
                    print(f"  ✓ {label}")
                else:
                    print(f"  ✗ {label} (no data)")

    print(f"\nDone: {success} computed, {skipped} cached (fresh), {total - success - skipped} no data")


# ── Fast Loaders for Streamlit ─────────────────────────────────────────────────

def load_iv_snapshot(
    commodity: str,
    front_options_month: str,
    lookback_years: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load pre-computed IV percentile snapshot.

    Returns:
        (snapshot_df, metadata_dict)

    Key snapshot_df columns:
        options_month, contract_label, atm_iv, iv_percentile, n_observations,
        curve_position, trading_dte,
        iv_hist_min, iv_hist_max, iv_hist_mean, iv_hist_median,
        P2, P1, C1, C2, C3 (current skew values),
        P2_pctile, P1_pctile, ... (skew percentiles),
        P2_hist_min, P2_hist_max, ... (skew historical stats)
    """
    sp = _snapshot_path(commodity.upper(), front_options_month.upper(), lookback_years)
    mp = _metadata_path(commodity.upper(), front_options_month.upper(), lookback_years)

    if not sp.exists():
        return pd.DataFrame(), {"error": "Not pre-computed. Run iv_percentiles_precompute.py"}

    snapshot = pd.read_parquet(sp)
    meta = {}
    if mp.exists():
        with open(mp) as f:
            meta = json.load(f)

    return snapshot, meta


def load_iv_timeseries(
    commodity: str,
    front_options_month: str,
    lookback_years: int | None = None,
    contract_months: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load pre-computed IV percentile timeseries for charting.

    Columns include: date, options_month, atm_iv, iv_percentile,
    P2, P1, C1, C2, C3, P2_pctile, ..., trading_dte, curve_position
    """
    tp = _timeseries_path(commodity.upper(), front_options_month.upper(), lookback_years)
    if not tp.exists():
        return pd.DataFrame()

    df = pd.read_parquet(tp)
    if contract_months:
        contract_months = [m.upper() for m in contract_months]
        df = df[df["options_month"].isin(contract_months)]

    return df


def get_current_front_month(commodity: str, date: pd.Timestamp = None) -> str:
    """Get the current front options month code."""
    if date is None:
        date = pd.Timestamp.now()
    return expiry_month_to_options_code(date.month, commodity)


def get_cache_status() -> pd.DataFrame:
    """Return summary of cached data."""
    if not CACHE_DIR.exists():
        return pd.DataFrame()

    rows = []
    snapshots_dir = CACHE_DIR / "snapshots"
    if snapshots_dir.exists():
        for f in snapshots_dir.glob("*.parquet"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                age_hours = (datetime.now().timestamp() - f.stat().st_mtime) / 3600
                rows.append({
                    "commodity": parts[0],
                    "month": parts[1],
                    "lookback": parts[2],
                    "age_hours": round(age_hours, 1),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    return (
        pd.DataFrame(rows).sort_values(["commodity", "month", "lookback"])
        if rows
        else pd.DataFrame()
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute IV percentile rankings")
    parser.add_argument("--commodity", type=str, help="Single commodity (e.g., SOY)")
    parser.add_argument("--month", type=str, help="Single options month (e.g., H)")
    parser.add_argument("--force", action="store_true", help="Recompute even if fresh")
    parser.add_argument("--status", action="store_true", help="Show cache status")
    args = parser.parse_args()

    if args.status:
        status = get_cache_status()
        if status.empty:
            print("No cached files found.")
        else:
            print(status.to_string(index=False))
    else:
        if args.force:
            import shutil
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                print("Cleared cache, recomputing all...")

        precompute_all(args.commodity, args.month)