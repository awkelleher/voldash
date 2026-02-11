"""
Variance Ratio Pre-Computation
===============================
Run this weekly (e.g., via cron or Task Scheduler) to pre-compute all variance
ratio matrices and save to Parquet. Streamlit then reads the cached results
instantly instead of recalculating.

Usage:
    python variance_ratios_precompute.py              # compute all combos
    python variance_ratios_precompute.py --commodity SOY --month H  # single combo

Schedule (crontab example - every Sunday at 6pm):
    0 18 * * 0 cd /path/to/project && python variance_ratios_precompute.py
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from lib.variance_ratios import (
    build_variance_ratio_matrix,
    identify_front_month_periods,
    options_to_futures,
    load_month_mapping,
    _load_prices,
)

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("cache/variance_ratios")
PRICES_PATH = "data/all_commodity_prices.csv"
MAPPING_PATH = "data/mapping.csv"

# All commodities and their options months to pre-compute
COMMODITIES = {
    "SOY":   ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "MEAL":  ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "OIL":   ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "CORN":  ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "WHEAT": ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "KW":    ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
}

LOOKBACK_OPTIONS = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # None = all history


# ── Helpers ────────────────────────────────────────────────────────────────────

def cache_key(commodity: str, options_month: str, lookback: int | None) -> str:
    lb = "all" if lookback is None else str(lookback)
    return f"{commodity}_{options_month}_{lb}"


def matrix_path(commodity: str, options_month: str, lookback: int | None) -> Path:
    return CACHE_DIR / f"{cache_key(commodity, options_month, lookback)}.parquet"


def metadata_path(commodity: str, options_month: str, lookback: int | None) -> Path:
    return CACHE_DIR / f"{cache_key(commodity, options_month, lookback)}_meta.json"


# ── Pre-compute ────────────────────────────────────────────────────────────────

def precompute_single(prices_df: pd.DataFrame,
                      commodity: str,
                      options_month: str,
                      lookback: int | None,
                      num_contracts: int = 12) -> bool:
    """Compute one variance ratio matrix and save to Parquet + JSON metadata."""
    try:
        matrix = build_variance_ratio_matrix(
            prices_df, options_month, commodity,
            num_contracts=num_contracts,
            lookback_years=lookback,
        )

        if matrix.empty:
            return False

        # Round for display
        display_df = matrix.round(2)

        # Build metadata
        periods = identify_front_month_periods(prices_df, options_month, commodity)
        if lookback is not None and lookback > 0:
            periods = periods.sort_values("year", ascending=False).head(lookback)

        front_futures = options_to_futures(options_month, commodity)
        meta = {
            "front_options_month": options_month,
            "front_futures_month": front_futures,
            "commodity": commodity,
            "lookback_years": lookback,
            "num_historical_periods": len(periods),
            "years_included": sorted(periods["year"].unique().tolist()) if len(periods) > 0 else [],
            "total_trading_days": int(periods["trading_days"].sum()) if len(periods) > 0 else 0,
            "computed_at": datetime.now().isoformat(),
        }

        # Save
        out = matrix_path(commodity, options_month, lookback)
        display_df.to_parquet(out)

        with open(metadata_path(commodity, options_month, lookback), "w") as f:
            json.dump(meta, f, indent=2)

        return True

    except Exception as e:
        print(f"  ERROR {commodity} {options_month} lb={lookback}: {e}")
        return False


def precompute_all(commodity_filter: str = None, month_filter: str = None):
    """Run full pre-computation. Optionally filter to one commodity/month."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading prices from {PRICES_PATH}...")
    prices_df = pd.read_csv(PRICES_PATH)
    prices_df["date"] = pd.to_datetime(prices_df["date"], format="mixed")
    print(f"  {len(prices_df):,} records loaded")

    commodities = COMMODITIES
    if commodity_filter:
        commodities = {k: v for k, v in commodities.items() if k == commodity_filter.upper()}

    total = 0
    success = 0
    skipped = 0

    for commodity, months in commodities.items():
        if month_filter:
            months = [m for m in months if m == month_filter.upper()]

        comm_prices = prices_df[prices_df["commodity"] == commodity].copy()
        if comm_prices.empty:
            print(f"  No price data for {commodity}, skipping")
            continue

        for month in months:
            for lookback in LOOKBACK_OPTIONS:
                total += 1
                lb_str = "all" if lookback is None else f"{lookback}y"
                label = f"{commodity} {month} ({lb_str})"

                # Skip if cache is fresh (< 7 days old)
                out = matrix_path(commodity, month, lookback)
                if out.exists():
                    age_days = (datetime.now().timestamp() - out.stat().st_mtime) / 86400
                    if age_days < 7:
                        skipped += 1
                        continue

                ok = precompute_single(comm_prices, commodity, month, lookback)
                if ok:
                    success += 1
                    print(f"  ✓ {label}")
                else:
                    print(f"  ✗ {label} (no data)")

    print(f"\nDone: {success} computed, {skipped} cached (fresh), {total - success - skipped} no data")


# ── Fast Loader for Streamlit ──────────────────────────────────────────────────

def load_cached_variance_ratios(commodity: str,
                                 options_month: str,
                                 lookback_years: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Load pre-computed variance ratio matrix. Call this from Streamlit instead of
    the slow build_variance_ratio_matrix().

    Returns:
        (matrix_df, metadata_dict) — same interface as get_variance_ratio_display()
    """
    mp = matrix_path(commodity.upper(), options_month.upper(), lookback_years)
    mdp = metadata_path(commodity.upper(), options_month.upper(), lookback_years)

    if not mp.exists():
        return pd.DataFrame(), {"error": "Not pre-computed. Run variance_ratios_precompute.py"}

    matrix = pd.read_parquet(mp)

    meta = {}
    if mdp.exists():
        with open(mdp) as f:
            meta = json.load(f)

    return matrix, meta


def get_cache_status() -> pd.DataFrame:
    """Return a summary of what's cached and how fresh it is."""
    if not CACHE_DIR.exists():
        return pd.DataFrame(columns=["file", "commodity", "month", "lookback", "age_hours"])

    rows = []
    for f in CACHE_DIR.glob("*.parquet"):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            age_hours = (datetime.now().timestamp() - f.stat().st_mtime) / 3600
            rows.append({
                "file": f.name,
                "commodity": parts[0],
                "month": parts[1],
                "lookback": parts[2],
                "age_hours": round(age_hours, 1),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })

    return pd.DataFrame(rows).sort_values(["commodity", "month", "lookback"])


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute variance ratio matrices")
    parser.add_argument("--commodity", type=str, help="Single commodity to compute (e.g., SOY)")
    parser.add_argument("--month", type=str, help="Single options month to compute (e.g., H)")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache is fresh")
    parser.add_argument("--status", action="store_true", help="Show cache status and exit")
    args = parser.parse_args()

    if args.status:
        status = get_cache_status()
        if status.empty:
            print("No cached files found.")
        else:
            print(status.to_string(index=False))
    else:
        if args.force:
            # Delete existing cache to force recompute
            import shutil
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                print("Cleared cache, recomputing all...")

        precompute_all(args.commodity, args.month)
