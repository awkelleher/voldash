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
VOL_SKEW_PATH = Path(__file__).resolve().parent.parent / "data" / "master_vol_skew.csv"
MAPPING_PATH = "data/mapping.csv"

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

MEDIAN_SKEW_PATH = Path("cache/median_skew.csv")
MEDIAN_IV_PATH = Path("cache/median_iv.csv")
IV_PCTILE_DIST_PATH = Path("cache/iv_percentile_dist.csv")
SKEW_PCTILE_DIST_PATH = Path("cache/skew_percentile_dist.csv")

# Percentile breakpoints at every 5% (0.05, 0.10, ..., 0.95, 1.00)
PCTILE_BREAKPOINTS = [round(i * 0.05, 2) for i in range(1, 21)]  # 0.05 to 1.00


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

    # Drop exact duplicate rows (can occur from data append/ingestion)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "commodity", "expiry"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate rows ({before} -> {after})")

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
    # IMPORTANT: Same options_month code in different years are DISTINCT contracts.
    # E.g., when H26 is front, H27 is 12 months out, not 0.
    # We use actual days-to-expiry to determine if a contract is the near or
    # deferred version when the month code matches.
    front_idx = df["front_options_month"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else -1
    )
    opt_idx = df["options_month"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else -1
    )
    month_offset = (opt_idx - front_idx) % 12

    # For each (date, commodity, options_month), find the nearest expiry.
    # If a contract is NOT the nearest (i.e., it's a deferred-year duplicate),
    # add 12 to its curve_position so H27 becomes position 12 instead of 0.
    df["_days_to_expiry"] = (df["expiry"] - df["date"]).dt.days
    nearest_expiry = (
        df.groupby(["date", "commodity", "options_month"])["_days_to_expiry"]
        .transform("min")
    )
    is_deferred = df["_days_to_expiry"] > nearest_expiry + 30  # >30 day gap = different year
    df["curve_position"] = month_offset + is_deferred.astype(int) * 12

    df.drop(columns=["obs_month", "_days_to_expiry"], inplace=True)

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

    sub = sub.sort_values(["options_month", "expiry_year", "date"])

    # For each distinct contract (month + year), compute expanding percentile rank.
    # IMPORTANT: Group by (options_month, expiry_year) so H26 and H27 are treated
    # as separate contracts. E.g., front-month H26 at 30 DTE should not be mixed
    # with deferred H27 at 395 DTE — they have structurally different vol levels.
    results = []
    for (opt_month, exp_year), group in sub.groupby(["options_month", "expiry_year"]):
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

    # Historical IV stats: use precomputed median_iv.csv when available,
    # fall back to inline computation otherwise.
    # NOTE: Median CSV stats represent the nearest-expiry contract for each month
    # code. Deferred-year duplicates (curve_position >= 12) won't have a match,
    # which is correct — they are structurally different contracts.
    median_iv_loaded = False
    if MEDIAN_IV_PATH.exists():
        try:
            miv = pd.read_csv(MEDIAN_IV_PATH)
            miv = miv[
                (miv["commodity"] == commodity.upper())
                & (miv["FRONT_OPTIONS"] == front_options_month.upper())
            ]
            if not miv.empty:
                iv_lookup = miv.set_index("OPTIONS")
                # Only map to non-deferred contracts (curve_position < 12)
                near_mask = snapshot["curve_position"] < 12
                snapshot.loc[near_mask, "iv_hist_median"] = snapshot.loc[near_mask, "options_month"].map(
                    iv_lookup["median_iv"].to_dict()
                )
                snapshot.loc[near_mask, "iv_hist_min"] = snapshot.loc[near_mask, "options_month"].map(
                    iv_lookup["iv_min"].to_dict()
                )
                snapshot.loc[near_mask, "iv_hist_max"] = snapshot.loc[near_mask, "options_month"].map(
                    iv_lookup["iv_max"].to_dict()
                )
                snapshot.loc[near_mask, "iv_hist_mean"] = snapshot.loc[near_mask, "options_month"].map(
                    iv_lookup["iv_mean"].to_dict()
                )
                median_iv_loaded = True
        except Exception:
            pass

    if not median_iv_loaded:
        # Fallback: compute inline from nearest-expiry data only
        full_near = full[full["curve_position"] < 12]
        hist_stats = (
            full_near.groupby("options_month")["atm_iv"]
            .agg(["min", "max", "mean", "median"])
            .rename(columns={
                "min": "iv_hist_min",
                "max": "iv_hist_max",
                "mean": "iv_hist_mean",
                "median": "iv_hist_median",
            })
        )
        # Only merge to near contracts
        near_snap = snapshot[snapshot["curve_position"] < 12][["options_month"]].drop_duplicates()
        near_stats = near_snap.merge(hist_stats, on="options_month", how="left")
        snapshot = snapshot.merge(near_stats, on="options_month", how="left", suffixes=("", "_dup"))
        # Drop any duplicate columns from merge
        dup_cols = [c for c in snapshot.columns if c.endswith("_dup")]
        if dup_cols:
            snapshot.drop(columns=dup_cols, inplace=True)

    # Historical skew stats: use precomputed median_skew.csv when available,
    # fall back to inline computation otherwise.
    # The median_skew.csv contains medians conditioned on (commodity, FRONT_OPTIONS, OPTIONS).
    skew_map_reverse = {"P2": "skew_m1.5", "P1": "skew_m0.5", "C1": "skew_p0.5",
                        "C2": "skew_p1.5", "C3": "skew_p3.0"}
    median_skew_loaded = False
    if MEDIAN_SKEW_PATH.exists():
        try:
            msdf = pd.read_csv(MEDIAN_SKEW_PATH)
            msdf = msdf[
                (msdf["commodity"] == commodity.upper())
                & (msdf["FRONT_OPTIONS"] == front_options_month.upper())
            ]
            if not msdf.empty:
                near_mask = snapshot["curve_position"] < 12
                for sc in ["P2", "P1", "C1", "C2", "C3"]:
                    csv_col = skew_map_reverse.get(sc, sc)
                    if csv_col in msdf.columns:
                        med_lookup = msdf.set_index("OPTIONS")[csv_col].to_dict()
                        snapshot.loc[near_mask, f"{sc}_hist_median"] = snapshot.loc[near_mask, "options_month"].map(med_lookup)
                median_skew_loaded = True
        except Exception:
            pass

    # Inline computation for min/max/mean (always) and median (fallback)
    # Use nearest-expiry data only to avoid contamination from deferred-year contracts
    full_near = full[full["curve_position"] < 12]
    for sc in ["P2", "P1", "C1", "C2", "C3"]:
        if sc in full_near.columns:
            agg_funcs = ["min", "max", "mean"]
            if not median_skew_loaded:
                agg_funcs.append("median")
            stats = (
                full_near.groupby("options_month")[sc]
                .agg(agg_funcs)
                .rename(columns={
                    "min": f"{sc}_hist_min",
                    "max": f"{sc}_hist_max",
                    "mean": f"{sc}_hist_mean",
                    "median": f"{sc}_hist_median",
                })
            )
            # Only merge columns we don't already have
            existing = [c for c in stats.columns if c in snapshot.columns]
            if existing:
                stats = stats.drop(columns=existing)
            if not stats.empty and len(stats.columns) > 0:
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


def _filter_nearest_expiry(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only the nearest expiry for each (date, commodity, options_month).

    When the same options month code exists in two years (e.g. H26 and H27 both
    present on the same date), we keep only the nearest-to-expiry contract.
    This ensures historical medians represent the primary contract for that month
    code, not a deferred-year duplicate 12 months further out.
    """
    days_to_exp = (vol_df["expiry"] - vol_df["date"]).dt.days
    min_dte = (
        vol_df.assign(_dte=days_to_exp)
        .groupby(["date", "commodity", "options_month"])["_dte"]
        .transform("min")
    )
    # Keep rows within 30 days of the nearest expiry (same contract cluster)
    return vol_df[days_to_exp <= min_dte + 30].copy()


def compute_median_skew_csv(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median skew for every (commodity, FRONT_OPTIONS, OPTIONS) combination
    and save to cache/median_skew.csv.

    Groups all historical observations by the front options month regime and the
    options contract, then takes the median of each skew column.

    IMPORTANT: Filters to nearest-expiry contracts only, so deferred-year duplicates
    (e.g. H27 when H26 is front) don't contaminate the medians.

    Output columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        skew_m1.5, skew_m0.5, skew_p0.5, skew_p1.5, skew_p3.0
    """
    print("Computing median skew by (commodity, FRONT_OPTIONS, OPTIONS)...")

    # Filter out deferred-year duplicates before computing medians
    filtered = _filter_nearest_expiry(vol_df)
    print(f"  Filtered {len(vol_df):,} -> {len(filtered):,} rows (nearest expiry only)")

    # The vol_df already has front_options_month and options_month from load_vol_skew().
    # Use the renamed columns (P2/P1/C1/C2/C3) for computation but output with
    # the original skew column names for compatibility.
    skew_map = {
        "P2": "skew_m1.5",
        "P1": "skew_m0.5",
        "C1": "skew_p0.5",
        "C2": "skew_p1.5",
        "C3": "skew_p3.0",
    }
    skew_internal = list(skew_map.keys())

    grouped = filtered.groupby(["commodity", "front_options_month", "options_month"])

    medians = grouped[skew_internal].median().reset_index()
    counts = grouped.size().reset_index(name="obs_count")
    result = medians.merge(counts, on=["commodity", "front_options_month", "options_month"])

    # Rename to match reference file format
    result = result.rename(columns={
        "front_options_month": "FRONT_OPTIONS",
        "options_month": "OPTIONS",
        **skew_map,
    })

    # Sort: commodity alpha, then month code order for both front and options
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
    result["_os"] = result["OPTIONS"].map(month_order)
    result = result.sort_values(["commodity", "_fs", "_os"]).drop(columns=["_fs", "_os"])

    # Reorder columns
    col_order = ["commodity", "FRONT_OPTIONS", "OPTIONS", "obs_count",
                 "skew_m1.5", "skew_m0.5", "skew_p0.5", "skew_p1.5", "skew_p3.0"]
    result = result[col_order]

    MEDIAN_SKEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(MEDIAN_SKEW_PATH, index=False)
    print(f"  {len(result)} rows -> {MEDIAN_SKEW_PATH}")

    return result


def compute_median_iv_csv(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median ATM IV for every (commodity, FRONT_OPTIONS, OPTIONS) combination
    and save to cache/median_iv.csv.

    Same front-month-conditioned logic as median_skew:
        "When H is front, what is the median IV for each options contract?"

    IMPORTANT: Filters to nearest-expiry contracts only, so deferred-year duplicates
    (e.g. H27 when H26 is front) don't contaminate the medians.

    Output columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count, median_iv,
        iv_min, iv_max, iv_mean, iv_p10, iv_p25, iv_p75, iv_p90
    """
    print("Computing median IV by (commodity, FRONT_OPTIONS, OPTIONS)...")

    # Filter out deferred-year duplicates before computing medians
    filtered = _filter_nearest_expiry(vol_df)
    print(f"  Filtered {len(vol_df):,} -> {len(filtered):,} rows (nearest expiry only)")

    grouped = filtered.groupby(["commodity", "front_options_month", "options_month"])

    stats = grouped["atm_iv"].agg(
        median_iv="median",
        iv_min="min",
        iv_max="max",
        iv_mean="mean",
        iv_p10=lambda x: np.nanpercentile(x, 10),
        iv_p25=lambda x: np.nanpercentile(x, 25),
        iv_p75=lambda x: np.nanpercentile(x, 75),
        iv_p90=lambda x: np.nanpercentile(x, 90),
    ).reset_index()

    counts = grouped.size().reset_index(name="obs_count")
    result = stats.merge(counts, on=["commodity", "front_options_month", "options_month"])

    # Rename to match convention
    result = result.rename(columns={
        "front_options_month": "FRONT_OPTIONS",
        "options_month": "OPTIONS",
    })

    # Sort: commodity alpha, then month code order
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
    result["_os"] = result["OPTIONS"].map(month_order)
    result = result.sort_values(["commodity", "_fs", "_os"]).drop(columns=["_fs", "_os"])

    # Reorder columns
    col_order = ["commodity", "FRONT_OPTIONS", "OPTIONS", "obs_count",
                 "median_iv", "iv_min", "iv_max", "iv_mean",
                 "iv_p10", "iv_p25", "iv_p75", "iv_p90"]
    result = result[col_order]

    MEDIAN_IV_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(MEDIAN_IV_PATH, index=False)
    print(f"  {len(result)} rows -> {MEDIAN_IV_PATH}")

    return result


def compute_iv_percentile_dist_csv(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full IV distribution at every 5th percentile for each
    (commodity, FRONT_OPTIONS, OPTIONS) combination.

    Output: one row per combo with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        iv_p05, iv_p10, iv_p15, ..., iv_p95, iv_p100

    Use case: take a live IV value, look up the row for that contract's
    (commodity, front_month, options_month), and find which bucket it falls in.
    """
    print("Computing IV percentile distributions (every 5%) ...")

    filtered = _filter_nearest_expiry(vol_df)
    print(f"  Filtered {len(vol_df):,} -> {len(filtered):,} rows (nearest expiry only)")

    grouped = filtered.groupby(["commodity", "front_options_month", "options_month"])

    # Build percentile columns
    pctile_aggs = {}
    for p in PCTILE_BREAKPOINTS:
        pct_int = int(p * 100)
        pctile_aggs[f"iv_p{pct_int:02d}"] = (
            "atm_iv", lambda x, pct=pct_int: np.nanpercentile(x, pct)
        )

    # Use apply to compute all percentiles at once
    rows = []
    for (comm, front, opt), grp in grouped:
        vals = grp["atm_iv"].dropna()
        if len(vals) < 2:
            continue
        row = {"commodity": comm, "front_options_month": front,
               "options_month": opt, "obs_count": len(vals)}
        for p in PCTILE_BREAKPOINTS:
            pct_int = int(p * 100)
            row[f"iv_p{pct_int:02d}"] = np.nanpercentile(vals, pct_int)
        rows.append(row)

    result = pd.DataFrame(rows)

    # Rename to match convention
    result = result.rename(columns={
        "front_options_month": "FRONT_OPTIONS",
        "options_month": "OPTIONS",
    })

    # Sort
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
    result["_os"] = result["OPTIONS"].map(month_order)
    result = result.sort_values(["commodity", "_fs", "_os"]).drop(columns=["_fs", "_os"])

    IV_PCTILE_DIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(IV_PCTILE_DIST_PATH, index=False)
    print(f"  {len(result)} rows -> {IV_PCTILE_DIST_PATH}")

    return result


def compute_skew_percentile_dist_csv(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full skew distribution at every 5th percentile for each
    (commodity, FRONT_OPTIONS, OPTIONS) combination, for each skew point.

    Output: one row per combo with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        skew_m1.5_p05, skew_m1.5_p10, ..., skew_m1.5_p100,
        skew_m0.5_p05, ..., skew_m0.5_p100,
        skew_p0.5_p05, ..., skew_p0.5_p100,
        skew_p1.5_p05, ..., skew_p1.5_p100,
        skew_p3.0_p05, ..., skew_p3.0_p100

    Use case: take a live skew value for a given strike point, look up the row
    for that contract, and find which bucket it falls in.
    """
    print("Computing skew percentile distributions (every 5%) ...")

    filtered = _filter_nearest_expiry(vol_df)
    print(f"  Filtered {len(vol_df):,} -> {len(filtered):,} rows (nearest expiry only)")

    skew_map = {
        "P2": "skew_m1.5",
        "P1": "skew_m0.5",
        "C1": "skew_p0.5",
        "C2": "skew_p1.5",
        "C3": "skew_p3.0",
    }
    skew_internal = list(skew_map.keys())

    grouped = filtered.groupby(["commodity", "front_options_month", "options_month"])

    rows = []
    for (comm, front, opt), grp in grouped:
        row = {"commodity": comm, "front_options_month": front,
               "options_month": opt, "obs_count": len(grp)}
        for internal_col, csv_name in skew_map.items():
            vals = grp[internal_col].dropna()
            if len(vals) < 2:
                for p in PCTILE_BREAKPOINTS:
                    pct_int = int(p * 100)
                    row[f"{csv_name}_p{pct_int:02d}"] = np.nan
            else:
                for p in PCTILE_BREAKPOINTS:
                    pct_int = int(p * 100)
                    row[f"{csv_name}_p{pct_int:02d}"] = np.nanpercentile(vals, pct_int)
        rows.append(row)

    result = pd.DataFrame(rows)

    # Rename to match convention
    result = result.rename(columns={
        "front_options_month": "FRONT_OPTIONS",
        "options_month": "OPTIONS",
    })

    # Sort
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
    result["_os"] = result["OPTIONS"].map(month_order)
    result = result.sort_values(["commodity", "_fs", "_os"]).drop(columns=["_fs", "_os"])

    SKEW_PCTILE_DIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SKEW_PCTILE_DIST_PATH, index=False)
    print(f"  {len(result)} rows -> {SKEW_PCTILE_DIST_PATH}")

    return result


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

                # Skip if cache is fresh (< 30 days old)
                out = _snapshot_path(commodity, month, lookback)
                if out.exists():
                    age_days = (datetime.now().timestamp() - out.stat().st_mtime) / 86400
                    if age_days < 30:
                        skipped += 1
                        continue

                ok = precompute_single(vol_df, commodity, month, lookback)
                if ok:
                    success += 1
                    print(f"  OK {label}")
                else:
                    print(f"  SKIP {label} (no data)")

    print(f"\nDone: {success} computed, {skipped} cached (fresh), {total - success - skipped} no data")

    # Generate median_iv.csv (front-month-conditioned IV medians + distribution stats)
    compute_median_iv_csv(vol_df)

    # Generate percentile distribution CSVs (every 5%) for live IV/skew lookup
    compute_iv_percentile_dist_csv(vol_df)
    compute_skew_percentile_dist_csv(vol_df)

    # Note: median_skew.csv is NOT regenerated here automatically.
    # It is maintained as an externally-validated file.
    # To regenerate from raw data, run:
    #     compute_median_skew_csv(load_vol_skew())


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


def load_median_skew(
    commodity: str = None,
    front_options_month: str = None,
) -> pd.DataFrame:
    """
    Load the pre-computed median skew CSV.

    Optionally filter by commodity and/or front options month.

    Returns DataFrame with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        skew_m1.5, skew_m0.5, skew_p0.5, skew_p1.5, skew_p3.0
    """
    if not MEDIAN_SKEW_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(MEDIAN_SKEW_PATH)

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if front_options_month:
        df = df[df["FRONT_OPTIONS"] == front_options_month.upper()]

    return df


def load_median_iv(
    commodity: str = None,
    front_options_month: str = None,
) -> pd.DataFrame:
    """
    Load the pre-computed median IV CSV.

    Optionally filter by commodity and/or front options month.

    Returns DataFrame with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count, median_iv,
        iv_min, iv_max, iv_mean, iv_p10, iv_p25, iv_p75, iv_p90
    """
    if not MEDIAN_IV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(MEDIAN_IV_PATH)

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if front_options_month:
        df = df[df["FRONT_OPTIONS"] == front_options_month.upper()]

    return df


def load_iv_percentile_dist(
    commodity: str = None,
    front_options_month: str = None,
) -> pd.DataFrame:
    """
    Load the pre-computed IV percentile distribution CSV.

    Returns DataFrame with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        iv_p05, iv_p10, iv_p15, ..., iv_p95, iv_p100
    """
    if not IV_PCTILE_DIST_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(IV_PCTILE_DIST_PATH)

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if front_options_month:
        df = df[df["FRONT_OPTIONS"] == front_options_month.upper()]

    return df


def load_skew_percentile_dist(
    commodity: str = None,
    front_options_month: str = None,
) -> pd.DataFrame:
    """
    Load the pre-computed skew percentile distribution CSV.

    Returns DataFrame with columns:
        commodity, FRONT_OPTIONS, OPTIONS, obs_count,
        skew_m1.5_p05, ..., skew_m1.5_p100,
        skew_m0.5_p05, ..., skew_p3.0_p100
    """
    if not SKEW_PCTILE_DIST_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(SKEW_PCTILE_DIST_PATH)

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if front_options_month:
        df = df[df["FRONT_OPTIONS"] == front_options_month.upper()]

    return df


def lookup_iv_percentile(
    live_iv: float,
    commodity: str,
    front_options_month: str,
    options_month: str,
) -> dict:
    """
    Look up where a live IV value falls in the historical distribution.

    Returns dict with:
        percentile: float (0.0 - 1.0) — interpolated percentile rank
        bracket_low: str — e.g. "p40" (the breakpoint just below)
        bracket_high: str — e.g. "p45" (the breakpoint just above)
        obs_count: int — number of historical observations
    """
    dist = load_iv_percentile_dist(commodity, front_options_month)
    if dist.empty:
        return {"percentile": None, "error": "No distribution data"}

    row = dist[dist["OPTIONS"] == options_month.upper()]
    if row.empty:
        return {"percentile": None, "error": f"No data for OPTIONS={options_month}"}

    row = row.iloc[0]
    obs = int(row["obs_count"])

    # Walk the breakpoints to find where live_iv falls
    pctile_cols = [f"iv_p{int(p * 100):02d}" for p in PCTILE_BREAKPOINTS]

    if live_iv <= row[pctile_cols[0]]:
        return {"percentile": 0.0, "bracket_low": "min", "bracket_high": "p05",
                "obs_count": obs}

    if live_iv >= row[pctile_cols[-1]]:
        return {"percentile": 1.0, "bracket_low": "p100", "bracket_high": "max",
                "obs_count": obs}

    for i in range(len(pctile_cols) - 1):
        lo_val = row[pctile_cols[i]]
        hi_val = row[pctile_cols[i + 1]]
        if live_iv <= hi_val:
            # Interpolate within the bracket
            lo_pct = PCTILE_BREAKPOINTS[i]
            hi_pct = PCTILE_BREAKPOINTS[i + 1]
            if hi_val == lo_val:
                pct = lo_pct
            else:
                frac = (live_iv - lo_val) / (hi_val - lo_val)
                pct = lo_pct + frac * (hi_pct - lo_pct)
            return {
                "percentile": round(pct, 4),
                "bracket_low": pctile_cols[i].replace("iv_", ""),
                "bracket_high": pctile_cols[i + 1].replace("iv_", ""),
                "obs_count": obs,
            }

    return {"percentile": None, "error": "Unexpected"}


def lookup_skew_percentile(
    live_skew: float,
    commodity: str,
    front_options_month: str,
    options_month: str,
    skew_point: str = "skew_p0.5",
) -> dict:
    """
    Look up where a live skew value falls in the historical distribution.

    skew_point: one of "skew_m1.5", "skew_m0.5", "skew_p0.5", "skew_p1.5", "skew_p3.0"

    Returns dict with:
        percentile: float (0.0 - 1.0) — interpolated percentile rank
        bracket_low: str — e.g. "p40"
        bracket_high: str — e.g. "p45"
        obs_count: int
    """
    dist = load_skew_percentile_dist(commodity, front_options_month)
    if dist.empty:
        return {"percentile": None, "error": "No distribution data"}

    row = dist[dist["OPTIONS"] == options_month.upper()]
    if row.empty:
        return {"percentile": None, "error": f"No data for OPTIONS={options_month}"}

    row = row.iloc[0]
    obs = int(row["obs_count"])

    pctile_cols = [f"{skew_point}_p{int(p * 100):02d}" for p in PCTILE_BREAKPOINTS]

    # Check columns exist
    if pctile_cols[0] not in row.index:
        return {"percentile": None, "error": f"Unknown skew_point={skew_point}"}

    if live_skew <= row[pctile_cols[0]]:
        return {"percentile": 0.0, "bracket_low": "min", "bracket_high": "p05",
                "obs_count": obs}

    if live_skew >= row[pctile_cols[-1]]:
        return {"percentile": 1.0, "bracket_low": "p100", "bracket_high": "max",
                "obs_count": obs}

    for i in range(len(pctile_cols) - 1):
        lo_val = row[pctile_cols[i]]
        hi_val = row[pctile_cols[i + 1]]
        if pd.isna(lo_val) or pd.isna(hi_val):
            continue
        if live_skew <= hi_val:
            lo_pct = PCTILE_BREAKPOINTS[i]
            hi_pct = PCTILE_BREAKPOINTS[i + 1]
            if hi_val == lo_val:
                pct = lo_pct
            else:
                frac = (live_skew - lo_val) / (hi_val - lo_val)
                pct = lo_pct + frac * (hi_pct - lo_pct)
            return {
                "percentile": round(pct, 4),
                "bracket_low": pctile_cols[i].replace(f"{skew_point}_", ""),
                "bracket_high": pctile_cols[i + 1].replace(f"{skew_point}_", ""),
                "obs_count": obs,
            }

    return {"percentile": None, "error": "Unexpected"}


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