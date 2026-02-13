"""
Recompute Median Skew by Front Options Month & Options Contract
================================================================
Reads master_vol_skew.csv, determines the front options month for each
observation date, and computes the median skew at each strike delta
for every (commodity, front_options, options_contract) combination.

Outputs:
  1. cache/median_skew_recomputed.csv  - the recomputed medians
  2. cache/median_skew_comparison.csv  - side-by-side vs user's reference file
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
VOL_SKEW_PATH = DATA_DIR / "master_vol_skew.csv"
MAPPING_PATH = DATA_DIR / "mapping.csv"
USER_REFERENCE = Path(__file__).resolve().parent.parent / "skew_medians_by_front_month_and_commodity.csv"

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
SKEW_COLS = ["skew_m1.5", "skew_m0.5", "skew_p0.5", "skew_p1.5", "skew_p3.0"]
COMMODITIES = ["SOY", "MEAL", "OIL", "CORN", "WHEAT", "KW"]


def load_mapping():
    mapping = pd.read_csv(MAPPING_PATH)
    for col in ["OPTIONS", "FUTURES", "COMMODITY"]:
        mapping[col] = mapping[col].astype(str).str.upper()
    return mapping


def load_and_enrich():
    """Load master_vol_skew and add front_options and options_month columns."""
    print(f"Loading {VOL_SKEW_PATH}...")
    df = pd.read_csv(VOL_SKEW_PATH)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["expiry"] = pd.to_datetime(df["expiry"], format="mixed")
    df["commodity"] = df["commodity"].str.upper()

    # Derive expiry month
    df["expiry_month"] = df["expiry"].dt.month

    # Build lookup: (commodity, expiry_month) -> options month code
    mapping = load_mapping()
    lookup = mapping.set_index(["COMMODITY", "EXPIRY_MONTH"])["OPTIONS"].to_dict()

    # Options month for each row (which options contract is this?)
    df["OPTIONS"] = df.apply(
        lambda r: lookup.get((r["commodity"], r["expiry_month"]), "?"), axis=1
    )

    # Front options month on the observation date
    df["obs_month"] = df["date"].dt.month
    df["FRONT_OPTIONS"] = df.apply(
        lambda r: lookup.get((r["commodity"], r["obs_month"]), "?"), axis=1
    )

    df.drop(columns=["obs_month"], inplace=True)

    print(f"  {len(df):,} rows loaded")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Commodities: {sorted(df['commodity'].unique())}")

    # Filter out rows where mapping failed
    df = df[(df["OPTIONS"] != "?") & (df["FRONT_OPTIONS"] != "?")]
    print(f"  {len(df):,} rows after mapping filter")

    return df


def compute_medians(df):
    """Compute median skew for each (commodity, front_options, options_contract) combo."""
    print("\nComputing medians...")

    grouped = df.groupby(["commodity", "FRONT_OPTIONS", "OPTIONS"])

    result = grouped[SKEW_COLS].median()
    result = result.reset_index()

    # Also add observation count
    counts = grouped.size().reset_index(name="obs_count")
    result = result.merge(counts, on=["commodity", "FRONT_OPTIONS", "OPTIONS"])

    # Sort nicely
    result["front_sort"] = result["FRONT_OPTIONS"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else 99
    )
    result["opt_sort"] = result["OPTIONS"].map(
        lambda x: MONTH_CODES.index(x) if x in MONTH_CODES else 99
    )
    result = result.sort_values(["commodity", "front_sort", "opt_sort"])
    result = result.drop(columns=["front_sort", "opt_sort"])

    print(f"  {len(result)} unique (commodity, front, options) combos")
    return result


def compare_to_reference(recomputed, ref_path):
    """Compare recomputed medians to user's reference file."""
    if not ref_path.exists():
        print(f"\nReference file not found: {ref_path}")
        return None

    print(f"\nComparing to reference: {ref_path}")
    ref = pd.read_csv(ref_path)

    # Merge on key columns
    merged = recomputed.merge(
        ref,
        on=["commodity", "FRONT_OPTIONS", "OPTIONS"],
        how="outer",
        suffixes=("_recomputed", "_reference"),
        indicator=True,
    )

    # Show merge stats
    both = (merged["_merge"] == "both").sum()
    left_only = (merged["_merge"] == "left_only").sum()
    right_only = (merged["_merge"] == "right_only").sum()
    print(f"  Matched rows: {both}")
    print(f"  Only in recomputed: {left_only}")
    print(f"  Only in reference: {right_only}")

    # Compute differences for matched rows
    matched = merged[merged["_merge"] == "both"].copy()
    for col in SKEW_COLS:
        rc = f"{col}_recomputed"
        rf = f"{col}_reference"
        if rc in matched.columns and rf in matched.columns:
            matched[f"{col}_diff"] = matched[rc] - matched[rf]

    # Summary stats on differences
    diff_cols = [f"{c}_diff" for c in SKEW_COLS if f"{c}_diff" in matched.columns]
    if diff_cols:
        print("\n  Difference stats (recomputed - reference):")
        print(matched[diff_cols].describe().round(4).to_string())

        # Find biggest discrepancies
        for dc in diff_cols:
            max_abs_diff = matched[dc].abs().max()
            if max_abs_diff > 0.01:
                worst = matched.loc[matched[dc].abs().idxmax()]
                print(f"\n  Largest diff in {dc}: {worst[dc]:.4f}")
                print(f"    -> {worst['commodity']} front={worst['FRONT_OPTIONS']} opt={worst['OPTIONS']}")

    return merged


def main():
    # 1. Load and enrich data
    df = load_and_enrich()

    # 2. Compute medians
    medians = compute_medians(df)

    # 3. Save recomputed file
    out_path = CACHE_DIR / "median_skew_recomputed.csv"
    medians.to_csv(out_path, index=False)
    print(f"\nSaved recomputed medians to: {out_path}")

    # 4. Compare to reference
    comparison = compare_to_reference(medians, USER_REFERENCE)

    if comparison is not None:
        comp_path = CACHE_DIR / "median_skew_comparison.csv"
        comparison.to_csv(comp_path, index=False)
        print(f"Saved comparison to: {comp_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
