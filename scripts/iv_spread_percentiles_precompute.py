"""
IV Spread Percentile Pre-Computation
=====================================
Calculates historical percentile distributions for consecutive-month IV spreads,
conditioned on which options month is front.

Core concept:
    "When H is the front options month, what is the percentile distribution of
    every consecutive-month IV spread (H-J, J-K, K-M, ..., G-H)?"

An IV spread = ATM IV of near month - ATM IV of far month (consecutive pairs).

When H is front, the pairs are: H-J, J-K, K-M, M-N, N-Q, Q-U, U-V, V-X, X-Z,
Z-F(next year), F-G, G-H.

IMPORTANT: Contracts are matched by year to avoid mixing H25 with H26. The near
and far legs of each spread are identified by (options_month, expiry_year) to
ensure consecutive-year wrapping (e.g., Z26 paired with F27, not F26).

Data source: master_vol_skew.csv (same as IV percentile precompute)

Output: cache/iv_spread_percentile_dist.csv
    Columns: commodity, FRONT_OPTIONS, SPREAD_PAIR, obs_count,
             spread_p05, spread_p10, ..., spread_p95, spread_p100

Usage:
    python iv_spread_percentiles_precompute.py                          # compute all
    python iv_spread_percentiles_precompute.py --commodity SOY          # single commodity
    python iv_spread_percentiles_precompute.py --commodity SOY --month H # single combo
    python iv_spread_percentiles_precompute.py --status                  # show cache info

Schedule: run after iv_percentiles_precompute.py (same weekly cadence)
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

VOL_SKEW_PATH = Path(__file__).resolve().parent.parent / "data" / "master_vol_skew.csv"
MAPPING_PATH = "data/mapping.csv"
OUTPUT_PATH = Path("cache/iv_spread_percentile_dist.csv")

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]

# Commodities to process
COMMODITIES = ["SOY", "MEAL", "OIL", "CORN", "WHEAT", "KW"]

# Cycle months per commodity — serial months outside this set rarely have
# enough liquidity for meaningful spread data.  Pairs are formed between
# consecutive *cycle* months so that e.g. SOY produces U-X (not U-V, V-X).
CYCLE_MONTHS = {
    "SOY":   ["F", "H", "K", "N", "Q", "U", "X"],
    "MEAL":  ["F", "H", "K", "N", "Q", "U", "V", "Z"],
    "OIL":   ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
    "CORN":  ["H", "K", "N", "U", "Z"],
    "WHEAT": ["H", "K", "N", "U", "Z"],
    "KW":    ["H", "K", "N", "U", "Z"],
}

# Percentile breakpoints at every 5% (0.05, 0.10, ..., 0.95, 1.00)
PCTILE_BREAKPOINTS = [round(i * 0.05, 2) for i in range(1, 21)]  # 0.05 to 1.00

# Column renames for clarity (matching iv_percentiles_precompute.py)
SKEW_RENAME = {
    "dirty_vol": "atm_iv",
    "skew_m1.5": "P2",
    "skew_m0.5": "P1",
    "skew_p0.5": "C1",
    "skew_p1.5": "C2",
    "skew_p3.0": "C3",
}

# Month that wraps into the next year: F expires in December but the contract
# is for the FOLLOWING year (e.g., December 2026 expiry = F27).
# Calendar month -> options code lookup is commodity-specific via mapping.csv.
# For consecutive-month pairing, the year increments when we wrap from Z -> F.

# ── Mapping Helpers ────────────────────────────────────────────────────────────

_mapping_cache = None


def load_mapping() -> pd.DataFrame:
    global _mapping_cache
    if _mapping_cache is None:
        _mapping_cache = pd.read_csv(MAPPING_PATH)
        for col in ["OPTIONS", "FUTURES", "COMMODITY"]:
            _mapping_cache[col] = _mapping_cache[col].astype(str).str.upper()
    return _mapping_cache


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
        - options_month: the options contract month code
        - front_options_month: which options month was front on the observation date
        - contract_label: e.g. "H25"
        - expiry_year: the year component of the expiry date
    """
    if path is None:
        path = VOL_SKEW_PATH

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["expiry"] = pd.to_datetime(df["expiry"], format="mixed")
    df["commodity"] = df["commodity"].str.upper()

    # Drop exact duplicate rows
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

    df["options_month"] = df[["commodity", "expiry_month"]].apply(
        lambda r: lookup.get((r["commodity"], r["expiry_month"]), "?"), axis=1
    )

    # Front options month: derived from actual front contract's expiry month
    df["_dte"] = (df["expiry"] - df["date"]).dt.days
    front_expiry_month = (
        df.loc[df.groupby(["date", "commodity"])["_dte"].idxmin()]
        [["date", "commodity", "expiry_month"]]
        .rename(columns={"expiry_month": "_front_expiry_month"})
    )
    df = df.merge(front_expiry_month, on=["date", "commodity"], how="left")
    df["front_options_month"] = df[["commodity", "_front_expiry_month"]].apply(
        lambda r: lookup.get((r["commodity"], int(r["_front_expiry_month"])), "?"), axis=1
    )
    df.drop(columns=["_dte", "_front_expiry_month"], inplace=True)

    # Contract label for display (e.g., "H25")
    df["contract_label"] = (
        df["options_month"] + (df["expiry_year"] % 100).astype(str).str.zfill(2)
    )

    return df


# ── Consecutive Month Pairs ───────────────────────────────────────────────────

def get_consecutive_pairs(front_month: str, commodity: str = None) -> list[tuple[str, str]]:
    """
    Generate ALL consecutive-month spread pairs starting from the front month.

    Produces pairs from the full 12-month code list (every consecutive month)
    PLUS cycle-month pairs (consecutive cycle months, skipping serials) when
    the commodity has a cycle month definition.

    For SOY front=H this returns:
        Serial pairs:  (H,J), (J,K), (K,M), (M,N), (N,Q), (Q,U), (U,V),
                       (V,X), (X,Z), (Z,F), (F,G)
        Cycle pairs:   (H,K), (K,N), (N,Q), (Q,U), (U,X), (X,F)
        Combined (deduplicated, ordered from front).

    This ensures the precompute covers serial-month spreads (H-J) when data
    exists, while also covering cycle-month spreads (H-K, U-X) that skip
    over illiquid serial months.
    """
    if front_month not in MONTH_CODES:
        return []

    # Always generate all 12-code consecutive pairs
    start_idx = MONTH_CODES.index(front_month)
    pairs = []
    seen = set()
    for i in range(11):
        near_idx = (start_idx + i) % 12
        far_idx = (start_idx + i + 1) % 12
        pair = (MONTH_CODES[near_idx], MONTH_CODES[far_idx])
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)

    # Add cycle-month pairs (these skip serial months, e.g. H-K for SOY)
    if commodity and commodity.upper() in CYCLE_MONTHS:
        codes = CYCLE_MONTHS[commodity.upper()]
        if front_month in codes:
            n = len(codes)
            c_start = codes.index(front_month)
            for i in range(n - 1):
                near_idx = (c_start + i) % n
                far_idx = (c_start + i + 1) % n
                pair = (codes[near_idx], codes[far_idx])
                if pair not in seen:
                    pairs.append(pair)
                    seen.add(pair)

    # Sort all pairs by near-month distance from front
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    front_pos = month_order[front_month]
    pairs.sort(key=lambda p: (
        (month_order[p[0]] - front_pos) % 12,
        (month_order[p[1]] - front_pos) % 12,
    ))

    return pairs


def _filter_nearest_expiry(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only the nearest expiry for each (date, commodity, options_month).

    When the same options month code exists in two years (e.g. H26 and H27 both
    present on the same date), we keep only the nearest-to-expiry contract.
    """
    days_to_exp = (vol_df["expiry"] - vol_df["date"]).dt.days
    min_dte = (
        vol_df.assign(_dte=days_to_exp)
        .groupby(["date", "commodity", "options_month"])["_dte"]
        .transform("min")
    )
    return vol_df[days_to_exp <= min_dte + 30].copy()


# ── Spread Computation ─────────────────────────────────────────────────────────

def compute_iv_spreads(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ATM IV spreads for every consecutive-month pair on each date,
    conditioned on the front options month.

    For each date and front month, pairs the near and far legs by matching
    them as consecutive contracts. Handles year wrapping: if the near month
    is Z (Nov expiry) and far month is F, the far contract's expiry_year
    must be near_expiry_year + 1 (since F expires in December of the
    FOLLOWING calendar year relative to Z).

    Returns DataFrame with columns:
        date, commodity, front_options_month, near_month, far_month,
        spread_pair, near_iv, far_iv, iv_spread, near_label, far_label
    """
    print("  Computing IV spreads for all consecutive pairs...")

    # For each (date, commodity), we know the front_options_month.
    # We need to pair contracts by consecutive months, matching by year properly.

    # Step 1: For each date/commodity, get the unique front options month
    # Step 2: For each pair (near_month, far_month), find the nearest-expiry
    #         contract for each month code and compute the spread.

    # Keep only rows with valid IV
    df = vol_df.dropna(subset=["atm_iv"]).copy()
    df = df[df["atm_iv"] > 0]

    # For pairing purposes, we need to handle year wrapping correctly.
    # Key insight: contracts are identified by (options_month, expiry_year).
    # When pairing near Z and far F, the far F should be the one with
    # expiry_year = near.expiry_year + 1 (F expires in Dec of NEXT year).
    #
    # Strategy: for each (date, commodity, front_options_month), iterate
    # through all 11 consecutive pairs. For each pair, find the nearest
    # available contracts and compute the spread.

    # Build a lookup: (date, commodity, options_month) -> list of (expiry_year, atm_iv)
    # Then for each pair, match the correct year.

    # Pivot: for each (date, commodity, options_month, expiry_year), keep nearest expiry IV
    df["_dte"] = (df["expiry"] - df["date"]).dt.days
    # Keep the nearest expiry for each (date, commodity, options_month, expiry_year)
    idx = df.groupby(["date", "commodity", "options_month", "expiry_year"])["_dte"].idxmin()
    nearest = df.loc[idx, ["date", "commodity", "options_month", "expiry_year",
                           "atm_iv", "front_options_month", "contract_label"]].copy()

    all_spreads = []

    for commodity in nearest["commodity"].unique():
        comm_df = nearest[nearest["commodity"] == commodity]
        front_months = comm_df["front_options_month"].unique()

        for front_month in front_months:
            if front_month not in MONTH_CODES:
                continue

            front_df = comm_df[comm_df["front_options_month"] == front_month]
            pairs = get_consecutive_pairs(front_month, commodity)
            dates = front_df["date"].unique()

            for dt in dates:
                day_df = front_df[front_df["date"] == dt]

                # Build available contracts for this date: {(options_month, expiry_year): row}
                contracts = {}
                for _, row in day_df.iterrows():
                    key = (row["options_month"], row["expiry_year"])
                    contracts[key] = row

                # For each consecutive pair, find the matching near/far contracts
                for near_month, far_month in pairs:
                    # Find all available years for each month
                    near_options = [
                        (om, ey) for (om, ey) in contracts
                        if om == near_month
                    ]
                    far_options = [
                        (om, ey) for (om, ey) in contracts
                        if om == far_month
                    ]

                    if not near_options or not far_options:
                        continue

                    # Sort by expiry year
                    near_options.sort(key=lambda x: x[1])
                    far_options.sort(key=lambda x: x[1])

                    # For each near contract, find the correct far contract
                    # The far month should be the next consecutive contract after near.
                    near_idx = MONTH_CODES.index(near_month)
                    far_idx = MONTH_CODES.index(far_month)

                    for near_key in near_options:
                        near_year = near_key[1]
                        near_row = contracts[near_key]

                        # Determine what year the far contract should be:
                        # If far_month comes AFTER near_month in the code order
                        # (i.e., far_idx > near_idx), same expiry year.
                        # If far wraps around (far_idx <= near_idx), far is next year.
                        #
                        # BUT: F is special — F expires in DECEMBER but is labeled
                        # for the following year. E.g., F27 expires Dec 2026.
                        # The expiry_year in our data is the EXPIRY date's year.
                        # So F27 has expiry_year=2026 (Dec 2026 expiry).
                        #
                        # Wait — let's verify: F27 expiry is December 2026.
                        # expiry.dt.year for Dec 2026 = 2026. So expiry_year=2026.
                        #
                        # For Z->F wrapping: Z has expiry in November (month 11).
                        # Z26 expiry = Nov 2026, so expiry_year=2026.
                        # F27 expiry = Dec 2026, so expiry_year=2026.
                        # They are the SAME expiry_year! Both expire in 2026.
                        #
                        # For X->Z: X has expiry in October (month 10).
                        # X26 expiry = Oct 2026, expiry_year=2026.
                        # Z26 expiry = Nov 2026, expiry_year=2026. Same year.
                        #
                        # The wrapping happens Z->F:
                        # Z26 expiry = Nov 2026. The NEXT F is F27 = Dec 2026.
                        # Both have expiry_year=2026!
                        #
                        # And F->G wrapping:
                        # F27 expiry = Dec 2026 (expiry_year=2026).
                        # G27 expiry = Jan 2027 (expiry_year=2027).
                        # So F->G: far_year = near_year + 1.
                        #
                        # General rule using expiry_month from mapping:
                        # near_expiry_month = mapping lookup for near_month
                        # far_expiry_month = mapping lookup for far_month
                        # If far_expiry_month > near_expiry_month: same expiry_year
                        # If far_expiry_month <= near_expiry_month: expiry_year + 1

                        near_expiry_month = near_row["expiry_month"] if "expiry_month" in near_row.index else None
                        if near_expiry_month is None:
                            continue

                        # Look up far month's expiry month from mapping
                        mapping = load_mapping()
                        far_em_rows = mapping[
                            (mapping["OPTIONS"] == far_month)
                            & (mapping["COMMODITY"] == commodity)
                        ]
                        if far_em_rows.empty:
                            continue
                        far_expiry_month = int(far_em_rows.iloc[0]["EXPIRY_MONTH"])

                        if far_expiry_month > near_expiry_month:
                            expected_far_year = near_year
                        else:
                            expected_far_year = near_year + 1

                        far_key = (far_month, expected_far_year)
                        if far_key not in contracts:
                            continue

                        far_row = contracts[far_key]

                        spread = near_row["atm_iv"] - far_row["atm_iv"]

                        all_spreads.append({
                            "date": dt,
                            "commodity": commodity,
                            "front_options_month": front_month,
                            "near_month": near_month,
                            "far_month": far_month,
                            "spread_pair": f"{near_month}-{far_month}",
                            "near_iv": near_row["atm_iv"],
                            "far_iv": far_row["atm_iv"],
                            "iv_spread": spread,
                            "near_label": near_row["contract_label"],
                            "far_label": far_row["contract_label"],
                        })

    if not all_spreads:
        return pd.DataFrame()

    result = pd.DataFrame(all_spreads)
    result["date"] = pd.to_datetime(result["date"])
    print(f"    {len(result):,} spread observations computed")
    return result


def compute_iv_spreads_fast(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of compute_iv_spreads for performance.

    Same logic but uses merge-based pairing instead of nested loops.
    """
    print("  Computing IV spreads (vectorized)...")

    df = vol_df.dropna(subset=["atm_iv"]).copy()
    df = df[df["atm_iv"] > 0]

    # Keep nearest expiry for each (date, commodity, options_month, expiry_year)
    df["_dte"] = (df["expiry"] - df["date"]).dt.days
    idx = df.groupby(["date", "commodity", "options_month", "expiry_year"])["_dte"].idxmin()
    nearest = df.loc[idx, ["date", "commodity", "options_month", "expiry_year",
                           "expiry_month", "atm_iv", "front_options_month",
                           "contract_label"]].copy()

    # Build expiry month lookup from mapping: (commodity, options_month) -> expiry_month
    mapping = load_mapping()
    em_lookup = mapping.set_index(["COMMODITY", "OPTIONS"])["EXPIRY_MONTH"].to_dict()

    all_spreads = []

    for commodity in COMMODITIES:
        comm_df = nearest[nearest["commodity"] == commodity]
        if comm_df.empty:
            continue

        front_months_in_data = comm_df["front_options_month"].unique()

        for front_month in front_months_in_data:
            if front_month not in MONTH_CODES:
                continue

            front_df = comm_df[comm_df["front_options_month"] == front_month]
            pairs = get_consecutive_pairs(front_month, commodity)

            for near_month, far_month in pairs:
                # Get expiry months for year-wrapping logic
                near_em = em_lookup.get((commodity, near_month))
                far_em = em_lookup.get((commodity, far_month))
                if near_em is None or far_em is None:
                    continue

                # Near leg
                near_df = front_df[front_df["options_month"] == near_month][
                    ["date", "expiry_year", "atm_iv", "contract_label"]
                ].rename(columns={
                    "atm_iv": "near_iv",
                    "expiry_year": "near_expiry_year",
                    "contract_label": "near_label",
                })

                if near_df.empty:
                    continue

                # For each near row, compute the expected far expiry year
                if far_em > near_em:
                    near_df["expected_far_year"] = near_df["near_expiry_year"]
                else:
                    near_df["expected_far_year"] = near_df["near_expiry_year"] + 1

                # Far leg
                far_df = front_df[front_df["options_month"] == far_month][
                    ["date", "expiry_year", "atm_iv", "contract_label"]
                ].rename(columns={
                    "atm_iv": "far_iv",
                    "expiry_year": "far_expiry_year",
                    "contract_label": "far_label",
                })

                if far_df.empty:
                    continue

                # Merge on date + expected year matching
                merged = near_df.merge(
                    far_df,
                    left_on=["date", "expected_far_year"],
                    right_on=["date", "far_expiry_year"],
                    how="inner",
                )

                if merged.empty:
                    continue

                merged["iv_spread"] = merged["near_iv"] - merged["far_iv"]
                merged["commodity"] = commodity
                merged["front_options_month"] = front_month
                merged["near_month"] = near_month
                merged["far_month"] = far_month
                merged["spread_pair"] = f"{near_month}-{far_month}"

                all_spreads.append(merged[[
                    "date", "commodity", "front_options_month",
                    "near_month", "far_month", "spread_pair",
                    "near_iv", "far_iv", "iv_spread",
                    "near_label", "far_label",
                ]])

    if not all_spreads:
        return pd.DataFrame()

    result = pd.concat(all_spreads, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    print(f"    {len(result):,} spread observations computed")
    return result


# ── Percentile Distribution ────────────────────────────────────────────────────

def compute_spread_percentile_dist(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full IV spread distribution at every 5th percentile for each
    (commodity, FRONT_OPTIONS, SPREAD_PAIR) combination.

    Output: one row per combo with columns:
        commodity, FRONT_OPTIONS, SPREAD_PAIR, near_month, far_month, obs_count,
        spread_p05, spread_p10, spread_p15, ..., spread_p95, spread_p100
    """
    print("Computing IV spread percentile distributions (every 5%) ...")

    if spreads_df.empty:
        return pd.DataFrame()

    grouped = spreads_df.groupby(["commodity", "front_options_month", "spread_pair",
                                  "near_month", "far_month"])

    rows = []
    for (comm, front, pair, near, far), grp in grouped:
        vals = grp["iv_spread"].dropna()
        if len(vals) < 2:
            continue
        row = {
            "commodity": comm,
            "front_options_month": front,
            "spread_pair": pair,
            "near_month": near,
            "far_month": far,
            "obs_count": len(vals),
        }
        for p in PCTILE_BREAKPOINTS:
            pct_int = int(p * 100)
            row[f"spread_p{pct_int:02d}"] = round(np.nanpercentile(vals, pct_int), 4)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Rename to match convention
    result = result.rename(columns={
        "front_options_month": "FRONT_OPTIONS",
        "spread_pair": "SPREAD_PAIR",
    })

    # Sort: commodity alpha, then front month code order, then spread pair
    # (pair order follows the consecutive sequence from front month)
    month_order = {m: i for i, m in enumerate(MONTH_CODES)}
    result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
    # Sort spread pairs by near_month position relative to front
    result["_near_pos"] = result.apply(
        lambda r: (month_order.get(r["near_month"], 0) - month_order.get(r["FRONT_OPTIONS"], 0)) % 12,
        axis=1,
    )
    result = result.sort_values(["commodity", "_fs", "_near_pos"]).drop(
        columns=["_fs", "_near_pos"]
    )

    return result


# ── Main Precompute ────────────────────────────────────────────────────────────

def precompute_all(commodity_filter: str = None, month_filter: str = None):
    """Run full pre-computation of IV spread percentile distributions."""
    print(f"Loading vol/skew data from {VOL_SKEW_PATH}...")
    vol_df = load_vol_skew()
    print(f"  {len(vol_df):,} records loaded")
    print(f"  Date range: {vol_df['date'].min().date()} to {vol_df['date'].max().date()}")
    print(f"  Commodities: {sorted(vol_df['commodity'].unique())}")
    print()

    # Apply commodity filter
    if commodity_filter:
        vol_df = vol_df[vol_df["commodity"] == commodity_filter.upper()]
        if vol_df.empty:
            print(f"No data for commodity={commodity_filter}")
            return

    # Compute all IV spreads
    spreads_df = compute_iv_spreads_fast(vol_df)

    if spreads_df.empty:
        print("No spread data computed.")
        return

    # Apply month filter
    if month_filter:
        spreads_df = spreads_df[spreads_df["front_options_month"] == month_filter.upper()]
        if spreads_df.empty:
            print(f"No spread data for front month={month_filter}")
            return

    # Compute percentile distributions
    result = compute_spread_percentile_dist(spreads_df)

    if result.empty:
        print("No percentile distributions computed.")
        return

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If filtering, merge with existing data (don't overwrite other combos)
    if commodity_filter or month_filter:
        if OUTPUT_PATH.exists():
            existing = pd.read_csv(OUTPUT_PATH)
            # Remove rows that will be replaced
            mask = pd.Series(True, index=existing.index)
            if commodity_filter:
                mask &= existing["commodity"] == commodity_filter.upper()
            if month_filter:
                mask &= existing["FRONT_OPTIONS"] == month_filter.upper()
            existing = existing[~mask]
            result = pd.concat([existing, result], ignore_index=True)
            # Re-sort
            month_order = {m: i for i, m in enumerate(MONTH_CODES)}
            result["_fs"] = result["FRONT_OPTIONS"].map(month_order)
            result["_near_pos"] = result.apply(
                lambda r: (month_order.get(r["near_month"], 0) - month_order.get(r["FRONT_OPTIONS"], 0)) % 12,
                axis=1,
            )
            result = result.sort_values(["commodity", "_fs", "_near_pos"]).drop(
                columns=["_fs", "_near_pos"]
            )

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  {len(result)} rows -> {OUTPUT_PATH}")

    # Summary
    print(f"\n  Commodities: {sorted(result['commodity'].unique())}")
    print(f"  Front months: {sorted(result['FRONT_OPTIONS'].unique())}")
    print(f"  Spread pairs: {sorted(result['SPREAD_PAIR'].unique())}")


# ── Fast Loaders for Streamlit ─────────────────────────────────────────────────

def load_iv_spread_percentile_dist(
    commodity: str = None,
    front_options_month: str = None,
) -> pd.DataFrame:
    """
    Load the pre-computed IV spread percentile distribution CSV.

    Returns DataFrame with columns:
        commodity, FRONT_OPTIONS, SPREAD_PAIR, near_month, far_month, obs_count,
        spread_p05, spread_p10, ..., spread_p95, spread_p100
    """
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(OUTPUT_PATH)

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if front_options_month:
        df = df[df["FRONT_OPTIONS"] == front_options_month.upper()]

    return df


def lookup_iv_spread_percentile(
    live_spread: float,
    commodity: str,
    front_options_month: str,
    spread_pair: str,
) -> dict:
    """
    Look up where a live IV spread value falls in the historical distribution.

    Args:
        live_spread: current IV spread (near_iv - far_iv)
        commodity: e.g. "SOY"
        front_options_month: e.g. "H"
        spread_pair: e.g. "H-J"

    Returns dict with:
        percentile: float (0.0 - 1.0) — interpolated percentile rank
        bracket_low: str — e.g. "p40"
        bracket_high: str — e.g. "p45"
        obs_count: int
    """
    dist = load_iv_spread_percentile_dist(commodity, front_options_month)
    if dist.empty:
        return {"percentile": None, "error": "No distribution data"}

    row = dist[dist["SPREAD_PAIR"] == spread_pair.upper()]
    if row.empty:
        return {"percentile": None, "error": f"No data for SPREAD_PAIR={spread_pair}"}

    row = row.iloc[0]
    obs = int(row["obs_count"])

    pctile_cols = [f"spread_p{int(p * 100):02d}" for p in PCTILE_BREAKPOINTS]

    if live_spread <= row[pctile_cols[0]]:
        return {"percentile": 0.0, "bracket_low": "min", "bracket_high": "p05",
                "obs_count": obs}

    if live_spread >= row[pctile_cols[-1]]:
        return {"percentile": 1.0, "bracket_low": "p100", "bracket_high": "max",
                "obs_count": obs}

    for i in range(len(pctile_cols) - 1):
        lo_val = row[pctile_cols[i]]
        hi_val = row[pctile_cols[i + 1]]
        if pd.isna(lo_val) or pd.isna(hi_val):
            continue
        if live_spread <= hi_val:
            lo_pct = PCTILE_BREAKPOINTS[i]
            hi_pct = PCTILE_BREAKPOINTS[i + 1]
            if hi_val == lo_val:
                pct = lo_pct
            else:
                frac = (live_spread - lo_val) / (hi_val - lo_val)
                pct = lo_pct + frac * (hi_pct - lo_pct)
            return {
                "percentile": round(pct, 4),
                "bracket_low": pctile_cols[i].replace("spread_", ""),
                "bracket_high": pctile_cols[i + 1].replace("spread_", ""),
                "obs_count": obs,
            }

    return {"percentile": None, "error": "Unexpected"}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute IV spread percentile distributions"
    )
    parser.add_argument("--commodity", type=str, help="Single commodity (e.g., SOY)")
    parser.add_argument("--month", type=str, help="Single front options month (e.g., H)")
    parser.add_argument("--status", action="store_true", help="Show cache info")
    args = parser.parse_args()

    if args.status:
        if OUTPUT_PATH.exists():
            df = pd.read_csv(OUTPUT_PATH)
            print(f"Cache: {OUTPUT_PATH}")
            print(f"  Rows: {len(df)}")
            print(f"  Commodities: {sorted(df['commodity'].unique())}")
            print(f"  Front months: {sorted(df['FRONT_OPTIONS'].unique())}")
            print(f"  Spread pairs: {sorted(df['SPREAD_PAIR'].unique())}")
            age_hours = (datetime.now().timestamp() - OUTPUT_PATH.stat().st_mtime) / 3600
            print(f"  Age: {age_hours:.1f} hours")
        else:
            print("No cached file found. Run without --status to compute.")
    else:
        precompute_all(args.commodity, args.month)
