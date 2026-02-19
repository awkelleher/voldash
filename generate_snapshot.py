"""
Vol Dashboard Snapshot Generator
=================================
Generates a condensed analytical summary of current market state
for uploading to a Claude Project as conversation context.

Mirrors the 7 key dashboard sections:
  1. Power Grid       — fwd vol vs predicted RV
  2. Vol Changes      — 1d and 5d IV/fwd changes
  3. IV Percentiles   — current IV rank vs history
  4. Realized Vol     — per-contract RV from precomputed cache
  5. Skew Analyzer    — current skew distance from median
  6. Heat Map         — current IV vs seasonal median
  7. Var Cal          — event variance calendar

Plus: cross-commodity spreads, variance ratios, notable moves.

Usage:
    - As a Streamlit button (integrated into your dashboard)
    - As a standalone script: python generate_snapshot.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\AdamKelleher\ags_book_streamlit")
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
SNAPSHOT_DIR = BASE_DIR / "snapshots"

COMMODITIES = ["SOY", "MEAL", "OIL", "CORN", "WHEAT"]
COMMODITIES_PLUS = ["SOY", "MEAL", "OIL", "CORN", "WHEAT", "KW"]

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}


def _expiry_label(expiry):
    """Convert an expiry value to a readable contract label like 'H26'."""
    try:
        ts = pd.Timestamp(expiry)
        code = MONTH_CODES.get(ts.month, "?")
        return f"{code}{ts.year % 100:02d}"
    except Exception:
        return str(expiry)


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_prices(path=None):
    """Load all_commodity_prices.csv"""
    path = path or DATA_DIR / "all_commodity_prices.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_vol_skew(path=None):
    """Load master_vol_skew.csv with derived contract_month."""
    path = path or DATA_DIR / "master_vol_skew.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    # Derive contract_month (rank within each date/commodity by expiry)
    if "contract_month" not in df.columns and "expiry" in df.columns:
        df = df.sort_values(["date", "commodity", "expiry"])
        df["contract_month"] = df.groupby(["date", "commodity"]).cumcount() + 1
    return df


def _prepare_vol_df(vol_skew_df, min_dte=5):
    """Ensure vol_skew_df has contract_month derived from expiry ordering.
    Filters out near-expiry contracts (trading_dte < min_dte) so the front
    contract is the first real tradeable month."""
    df = vol_skew_df.copy()
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Filter out near-expiry contracts before assigning contract_month
    if "trading_dte" in df.columns and min_dte > 0:
        df = df[df["trading_dte"] >= min_dte]
    # Derive contract_month from expiry ordering (M1 = nearest non-expired)
    if "expiry" in df.columns:
        df = df.sort_values(["date", "commodity", "expiry"])
        df["contract_month"] = df.groupby(["date", "commodity"]).cumcount() + 1
    # Derive fwd_vol if missing
    if "fwd_vol" not in df.columns and "dirty_vol" in df.columns:
        df["fwd_vol"] = df["dirty_vol"]
    return df


def load_precomputed_rv(path=None):
    """Load realized vol from precomputed cache."""
    path = path or CACHE_DIR / "realized_vol_precomputed.csv"
    if Path(path).exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


# ============================================================================
# SECTION 0: HEADER
# ============================================================================

def section_header(snapshot_date):
    lines = [
        f"# Vol Dashboard Snapshot",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Market Date:** {snapshot_date.strftime('%Y-%m-%d')} ({snapshot_date.strftime('%A')})",
        f"**Commodities:** {', '.join(COMMODITIES)}",
        "",
    ]
    return "\n".join(lines)


# ============================================================================
# SECTION 1: POWER GRID
# ============================================================================

def section_power_grid(vol_skew_df, snapshot_date):
    """Power Grid: (fwd_vol - predicted_rv) / predicted_rv for each contract."""
    lines = ["## 1. Power Grid\n"]
    lines.append("*IV premium over predicted realized vol. Positive = IV rich, negative = IV cheap.*\n")

    try:
        sys.path.insert(0, str(BASE_DIR / "lib"))
        import vol_analysis as va
        verdad_path = DATA_DIR / "verdad.csv"
        if not verdad_path.exists():
            verdad_path = DATA_DIR / "verdad.7.xlsx"
        if not verdad_path.exists():
            lines.append("No Verdad predictions found (data/verdad.csv).\n")
            return "\n".join(lines)

        overall, monthly = va.load_verdad_predictions(str(verdad_path))
        power_df, metadata = va.calculate_power_grid(
            vol_skew_df, snapshot_date, overall, monthly_predictions=monthly, max_months=8
        )
    except Exception as e:
        lines.append(f"Error computing power grid: {e}\n")
        return "\n".join(lines)

    if power_df.empty:
        lines.append("No power grid data available.\n")
        return "\n".join(lines)

    # Power grid table
    cols = list(power_df.columns)
    lines.append("| Commodity | " + " | ".join(cols) + " |")
    lines.append("|-----------|" + "|".join(["-------"] * len(cols)) + "|")
    for commodity in power_df.index:
        vals = []
        for c in cols:
            v = power_df.loc[commodity, c]
            vals.append(f"{v:+.1f}%" if pd.notna(v) else "—")
        lines.append(f"| {commodity} | " + " | ".join(vals) + " |")

    # Forward vol reference
    fwd_df = metadata.get("fwd_vols", pd.DataFrame())
    if not fwd_df.empty:
        lines.append("\n**Forward Vols:**")
        lines.append("| Commodity | " + " | ".join(cols) + " |")
        lines.append("|-----------|" + "|".join(["-------"] * len(cols)) + "|")
        for commodity in fwd_df.index:
            vals = []
            for c in cols:
                v = fwd_df.loc[commodity, c] if c in fwd_df.columns else None
                vals.append(f"{v:.2f}" if pd.notna(v) else "—")
            lines.append(f"| {commodity} | " + " | ".join(vals) + " |")

    # Predicted RV reference
    pred_df = metadata.get("pred_rvs", pd.DataFrame())
    if not pred_df.empty:
        lines.append("\n**Predicted RV:**")
        lines.append("| Commodity | " + " | ".join(cols) + " |")
        lines.append("|-----------|" + "|".join(["-------"] * len(cols)) + "|")
        for commodity in pred_df.index:
            vals = []
            for c in cols:
                v = pred_df.loc[commodity, c] if c in pred_df.columns else None
                vals.append(f"{v:.2f}" if pd.notna(v) else "—")
            lines.append(f"| {commodity} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# SECTION 2: VOL CHANGES
# ============================================================================

def section_vol_changes(vol_skew_df, snapshot_date, long_window=5):
    """1-day and N-day changes in IV and fwd vol."""
    lines = ["## 2. Vol Changes\n"]
    lines.append(f"*1-day and {long_window}-day changes in dirty_vol and fwd_vol.*\n")

    vol_col = "dirty_vol" if "dirty_vol" in vol_skew_df.columns else "atm_iv"
    fwd_col = "fwd_vol" if "fwd_vol" in vol_skew_df.columns else vol_col

    for commodity in COMMODITIES:
        cdf = vol_skew_df[vol_skew_df["commodity"] == commodity].copy()
        if cdf.empty:
            continue

        dates = sorted(cdf["date"].dropna().unique())
        if len(dates) < 2:
            continue

        current_date = dates[-1]
        prev_date = dates[-2] if len(dates) >= 2 else None
        long_date = dates[-long_window - 1] if len(dates) > long_window else dates[0]

        current = cdf[cdf["date"] == current_date].sort_values("expiry").head(8)
        if current.empty:
            continue

        rows = []
        for _, row in current.iterrows():
            expiry = row.get("expiry")
            label = _expiry_label(expiry)
            iv_now = row.get(vol_col)
            fwd_now = row.get(fwd_col)

            # 1d change
            iv_1d = fwd_1d = None
            if prev_date is not None:
                prev = cdf[(cdf["date"] == prev_date) & (cdf["expiry"] == expiry)]
                if not prev.empty:
                    iv_1d = iv_now - prev.iloc[0].get(vol_col) if iv_now is not None else None
                    fwd_1d = fwd_now - prev.iloc[0].get(fwd_col) if fwd_now is not None else None

            # N-day change
            iv_nd = fwd_nd = None
            long = cdf[(cdf["date"] == long_date) & (cdf["expiry"] == expiry)]
            if not long.empty:
                iv_nd = iv_now - long.iloc[0].get(vol_col) if iv_now is not None else None
                fwd_nd = fwd_now - long.iloc[0].get(fwd_col) if fwd_now is not None else None

            rows.append({
                "contract": label,
                "iv_now": f"{iv_now:.2f}" if iv_now is not None else "—",
                "fwd_now": f"{fwd_now:.2f}" if fwd_now is not None else "—",
                "iv_1d": f"{iv_1d:+.2f}" if iv_1d is not None else "—",
                "fwd_1d": f"{fwd_1d:+.2f}" if fwd_1d is not None else "—",
                "iv_nd": f"{iv_nd:+.2f}" if iv_nd is not None else "—",
                "fwd_nd": f"{fwd_nd:+.2f}" if fwd_nd is not None else "—",
            })

        if rows:
            lines.append(f"### {commodity}")
            lines.append(f"| Contract | IV Now | Fwd Now | IV 1d | Fwd 1d | IV {long_window}d | Fwd {long_window}d |")
            lines.append("|----------|--------|---------|-------|--------|--------|---------|")
            for r in rows:
                lines.append(f"| {r['contract']} | {r['iv_now']} | {r['fwd_now']} | {r['iv_1d']} | {r['fwd_1d']} | {r['iv_nd']} | {r['fwd_nd']} |")
            lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 3: IV PERCENTILES
# ============================================================================

def section_iv_percentiles(vol_skew_df, snapshot_date, master_df=None):
    """IV percentile ranks vs history using the dashboard's cached percentile grid."""
    lines = ["## 3. IV Percentiles\n"]
    lines.append("*Current IV rank vs historical distribution. >80 = rich, <20 = cheap.*\n")

    try:
        sys.path.insert(0, str(BASE_DIR / "lib"))
        import vol_analysis as va
        hist_df = master_df if master_df is not None else vol_skew_df
        grid = va.get_percentile_grid_cached(
            vol_skew_df, snapshot_date, metric="dirty_vol",
            lookback_days=252, commodities=COMMODITIES,
            max_months=8, hist_df=hist_df
        )
    except Exception as e:
        lines.append(f"Error computing percentiles: {e}\n")
        return "\n".join(lines)

    if grid.empty:
        lines.append("No percentile data available.\n")
        return "\n".join(lines)

    for commodity in COMMODITIES:
        cg = grid[grid["commodity"] == commodity].sort_values("contract_month")
        if cg.empty:
            continue
        lines.append(f"### {commodity}")
        lines.append("| Month | Current IV | Percentile | Median | Distance |")
        lines.append("|-------|-----------|------------|--------|----------|")
        for _, row in cg.iterrows():
            cm = int(row["contract_month"])
            curr = row.get("current")
            pct = row.get("percentile")
            med = row.get("median")
            dist = row.get("distance")
            curr_str = f"{curr:.2f}" if pd.notna(curr) else "—"
            pct_str = f"{pct:.0f}%" if pd.notna(pct) else "—"
            med_str = f"{med:.2f}" if pd.notna(med) else "—"
            dist_str = f"{dist:+.2f}" if pd.notna(dist) else "—"
            lines.append(f"| M{cm} | {curr_str} | {pct_str} | {med_str} | {dist_str} |")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 4: REALIZED VOL
# ============================================================================

def section_realized_vol(snapshot_date):
    """Realized vol from precomputed cache (per-contract, much more accurate)."""
    lines = ["## 4. Realized Volatility\n"]
    lines.append("*Per-contract annualized RV from close-to-close log returns.*\n")

    rv_df = load_precomputed_rv()
    if rv_df is None or rv_df.empty:
        lines.append("No precomputed RV cache found. Run `scripts/precompute_realized_vol.py`.\n")
        return "\n".join(lines)

    # Use latest date in cache
    latest_date = rv_df["date"].max()
    rv_latest = rv_df[rv_df["date"] == latest_date]

    lines.append(f"*Cache date: {latest_date.strftime('%Y-%m-%d')}*\n")

    rv_cols = [c for c in rv_latest.columns if c.startswith("rv_")]
    header_cols = ["Contract", "Close"] + [c.replace("rv_", "").upper() for c in rv_cols]

    for commodity in COMMODITIES:
        crv = rv_latest[rv_latest["commodity"] == commodity].sort_values("contract_code")
        if crv.empty:
            continue
        lines.append(f"### {commodity}")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["-------"] * len(header_cols)) + "|")
        for _, row in crv.head(8).iterrows():
            vals = [str(row.get("contract_code", "?"))]
            close = row.get("close")
            vals.append(f"{close:.2f}" if pd.notna(close) else "—")
            for rc in rv_cols:
                v = row.get(rc)
                vals.append(f"{v:.2f}" if pd.notna(v) else "—")
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 5: SKEW ANALYZER
# ============================================================================

def section_skew_analyzer(vol_skew_df, snapshot_date, master_df=None):
    """Skew distance from median using dashboard's cached skew diff grid."""
    lines = ["## 5. Skew Analyzer\n"]
    lines.append("*Current skew minus historical median. Positive = skew steeper than normal.*\n")

    try:
        sys.path.insert(0, str(BASE_DIR / "lib"))
        import vol_analysis as va
        hist_df = master_df if master_df is not None else vol_skew_df
        grid = va.get_skew_diff_grid_cached(
            vol_skew_df, snapshot_date, commodities=COMMODITIES,
            max_months=8, hist_df=hist_df
        )
    except Exception as e:
        lines.append(f"Error computing skew diffs: {e}\n")
        return "\n".join(lines)

    if grid.empty:
        lines.append("No skew data available.\n")
        return "\n".join(lines)

    skew_labels = ["P2", "P1", "C1", "C2", "C3"]
    available = [s for s in skew_labels if s in grid.columns]
    if not available:
        lines.append("No skew columns found.\n")
        return "\n".join(lines)

    for commodity in COMMODITIES:
        cg = grid[grid["commodity"] == commodity].sort_values("contract_month")
        if cg.empty:
            continue
        lines.append(f"### {commodity}")
        lines.append("| Month | " + " | ".join(available) + " |")
        lines.append("|-------|" + "|".join(["-------"] * len(available)) + "|")
        for _, row in cg.iterrows():
            ml = row.get("month_label", f"M{int(row['contract_month'])}")
            vals = []
            for s in available:
                v = row.get(s)
                vals.append(f"{v:+.2f}" if pd.notna(v) else "—")
            lines.append(f"| {ml} | " + " | ".join(vals) + " |")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 6: HEAT MAP (IV CALENDAR)
# ============================================================================

def section_heat_map(vol_skew_df, snapshot_date, master_df=None):
    """IV calendar heat map: current IV minus seasonal median."""
    lines = ["## 6. Heat Map (IV vs Seasonal)\n"]
    lines.append("*Current IV minus historical seasonal median by ~10-day period. Positive = rich vs seasonal.*\n")

    try:
        sys.path.insert(0, str(BASE_DIR / "lib"))
        import vol_analysis as va
        source_df = master_df if master_df is not None else vol_skew_df
    except Exception as e:
        lines.append(f"Error importing vol_analysis: {e}\n")
        return "\n".join(lines)

    for commodity in COMMODITIES:
        try:
            avg_grid, med_grid = va.compute_iv_calendar_grid(source_df, commodity)
        except Exception as e:
            lines.append(f"### {commodity}\nError: {e}\n")
            continue

        if med_grid.empty:
            continue

        # Get current IVs for this commodity on snapshot date
        current = vol_skew_df[
            (vol_skew_df["commodity"] == commodity) &
            (vol_skew_df["date"] == snapshot_date)
        ]
        if current.empty:
            continue

        # Determine which time period we're in
        try:
            current_period = va._date_to_time_period(snapshot_date)
        except Exception:
            continue

        # Find current period's row in median grid
        if current_period not in med_grid.index:
            continue

        med_row = med_grid.loc[current_period]

        # Get current IVs mapped to options month codes
        try:
            from lib.vol_analysis import load_options_mapping
            mapping_df = load_options_mapping()
            sub = mapping_df[mapping_df["COMMODITY"] == commodity.upper()]
            em_to_opt = {
                int(r["EXPIRY_MONTH"]): r["OPTIONS"]
                for _, r in sub.iterrows()
                if pd.notna(r.get("EXPIRY_MONTH"))
            }
        except Exception:
            em_to_opt = {}

        if not em_to_opt:
            continue

        current_ivs = {}
        for _, row in current.iterrows():
            try:
                exp_month = pd.Timestamp(row["expiry"]).month
                opt_code = em_to_opt.get(exp_month)
                if opt_code and pd.notna(row.get("dirty_vol")):
                    current_ivs[opt_code] = row["dirty_vol"]
            except Exception:
                pass

        if not current_ivs:
            continue

        # Build diff table: current - median
        lines.append(f"### {commodity} (period: {current_period})")
        codes = [c for c in med_grid.columns if c in current_ivs]
        if not codes:
            continue

        lines.append("| Month Code | Current IV | Seasonal Median | Diff |")
        lines.append("|------------|-----------|-----------------|------|")
        for code in codes:
            cur = current_ivs.get(code)
            med = med_row.get(code)
            if pd.notna(cur) and pd.notna(med):
                diff = cur - med
                lines.append(f"| {code} | {cur:.2f} | {med:.2f} | {diff:+.2f} |")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 7: VAR CAL
# ============================================================================

def _load_var_weights():
    """Load variance weights: custom JSON if available, else defaults."""
    custom_path = SNAPSHOT_DIR / "varcal_weights.json"
    if custom_path.exists():
        try:
            with open(custom_path) as f:
                return json.load(f)
        except Exception:
            pass

    # Import defaults from vol_dashboard
    try:
        # Can't import vol_dashboard directly (Streamlit dependency), so define defaults here
        pass
    except Exception:
        pass

    # Hardcoded defaults matching CME_DEFAULT_VAR in vol_dashboard.py
    return {
        "Day":                          {"SOY": 1.0,  "MEAL": 1.0,  "OIL": 1.0,  "CORN": 1.0,  "WHEAT": 1.0,  "KW": 1.0},
        "Weekend (W)":                  {"SOY": 0.15, "MEAL": 0.15, "OIL": 0.15, "CORN": 0.15, "WHEAT": 0.15, "KW": 0.15},
        "Holiday (H)":                  {"SOY": 0.15, "MEAL": 0.15, "OIL": 0.15, "CORN": 0.15, "WHEAT": 0.15, "KW": 0.15},
        "Grain Stock - Jan":            {"SOY": 4.5,  "MEAL": 5.5,  "OIL": 3.2,  "CORN": 7.5,  "WHEAT": 7.5,  "KW": 7.5},
        "Grain Stock - Mar & Planting": {"SOY": 2.5,  "MEAL": 2.5,  "OIL": 2.7,  "CORN": 3.0,  "WHEAT": 1.5,  "KW": 1.5},
        "Grain Stock - Jun":            {"SOY": 3.0,  "MEAL": 3.0,  "OIL": 3.0,  "CORN": 3.0,  "WHEAT": 3.0,  "KW": 3.0},
        "Grain Stock - Sep":            {"SOY": 2.6,  "MEAL": 3.0,  "OIL": 1.5,  "CORN": 3.6,  "WHEAT": 2.5,  "KW": 2.5},
        "Crop Production - Aug":        {"SOY": 2.5,  "MEAL": 2.5,  "OIL": 1.5,  "CORN": 3.0,  "WHEAT": 1.5,  "KW": 1.5},
        "Crop Production - Sep":        {"SOY": 2.0,  "MEAL": 2.0,  "OIL": 1.5,  "CORN": 2.0,  "WHEAT": 1.5,  "KW": 1.5},
        "Crop Production - Oct":        {"SOY": 3.0,  "MEAL": 3.0,  "OIL": 3.0,  "CORN": 2.5,  "WHEAT": 3.0,  "KW": 3.0},
        "Crop Production - Nov":        {"SOY": 4.0,  "MEAL": 4.0,  "OIL": 3.0,  "CORN": 4.0,  "WHEAT": 1.5,  "KW": 1.5},
        "Crop Production - Out":        {"SOY": 1.0,  "MEAL": 1.0,  "OIL": 1.0,  "CORN": 1.0,  "WHEAT": 1.0,  "KW": 1.0},
    }


def _load_cme_events():
    """CME event calendar — mirrors vol_dashboard.py top-level dict."""
    return {
        "2026-01-12": "Grain Stock - Jan",
        "2026-02-10": "Crop Production - Out",
        "2026-03-10": "Crop Production - Out",
        "2026-03-31": "Grain Stock - Mar & Planting",
        "2026-04-09": "Crop Production - Out",
        "2026-05-12": "Crop Production - Out",
        "2026-06-11": "Crop Production - Out",
        "2026-06-30": "Grain Stock - Jun",
        "2026-07-10": "Crop Production - Out",
        "2026-08-12": "Crop Production - Aug",
        "2026-09-11": "Crop Production - Sep",
        "2026-09-30": "Grain Stock - Sep",
        "2026-10-09": "Crop Production - Oct",
        "2026-11-10": "Crop Production - Nov",
        "2026-12-10": "Crop Production - Out",
        "2027-01-11": "Grain Stock - Jan",
        "2027-02-09": "Crop Production - Out",
        "2027-03-09": "Crop Production - Out",
        "2027-03-31": "Grain Stock - Mar & Planting",
        "2027-04-08": "Crop Production - Out",
        "2027-05-11": "Crop Production - Out",
        "2027-06-10": "Crop Production - Out",
        "2027-06-30": "Grain Stock - Jun",
        "2027-07-09": "Crop Production - Out",
        "2027-08-11": "Crop Production - Aug",
        "2027-09-10": "Crop Production - Sep",
        "2027-09-30": "Grain Stock - Sep",
        "2027-10-08": "Crop Production - Oct",
        "2027-11-09": "Crop Production - Nov",
        "2027-12-09": "Crop Production - Out",
        "2028-01-10": "Grain Stock - Jan",
        "2028-02-08": "Crop Production - Out",
        "2028-03-08": "Crop Production - Out",
        "2028-03-29": "Grain Stock - Mar & Planting",
        "2028-04-11": "Crop Production - Out",
        "2028-05-09": "Crop Production - Out",
        "2028-06-08": "Crop Production - Out",
        "2028-06-30": "Grain Stock - Jun",
        "2028-07-12": "Crop Production - Out",
        "2028-08-10": "Crop Production - Aug",
        "2028-09-12": "Crop Production - Sep",
        "2028-09-29": "Grain Stock - Sep",
        "2028-10-11": "Crop Production - Oct",
        "2028-11-09": "Crop Production - Nov",
        "2028-12-12": "Crop Production - Out",
    }


def _load_cme_holidays():
    return {
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25",
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
        "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
        "2027-11-25", "2027-12-24",
        "2028-01-01", "2028-01-17", "2028-02-21", "2028-04-14",
        "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04",
        "2028-11-23", "2028-12-25",
    }


def section_var_cal(snapshot_date, daily_lookahead=60):
    """Variance calendar: monthly totals + next N days daily detail."""
    lines = ["## 7. Variance Calendar\n"]

    var_weights = _load_var_weights()
    events = _load_cme_events()
    holidays = _load_cme_holidays()
    commodities = COMMODITIES_PLUS

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = today + timedelta(days=365 * 2 + 31)

    # Check if using custom weights
    custom_path = SNAPSHOT_DIR / "varcal_weights.json"
    if custom_path.exists():
        lines.append("*Using custom variance weights from `varcal_weights.json`.*\n")
    else:
        lines.append("*Using default CME variance weights.*\n")

    # Build full calendar
    cal_rows = []
    cur = today
    while cur <= end:
        dstr = cur.strftime("%Y-%m-%d")
        dow = cur.weekday()
        is_weekend = dow >= 5
        is_holiday = dstr in holidays
        event = events.get(dstr, "")

        row = {"date": dstr, "day": cur.strftime("%a"), "weekend": is_weekend,
               "holiday": is_holiday, "event": event}

        for c in commodities:
            if is_holiday:
                etype = "Holiday (H)"
            elif is_weekend:
                etype = "Weekend (W)"
            elif event and event in var_weights:
                etype = event
            else:
                etype = "Day"
            row[c] = var_weights.get(etype, {}).get(c, 1.0)

        cal_rows.append(row)
        cur += timedelta(days=1)

    cal_df = pd.DataFrame(cal_rows)

    # Monthly totals
    lines.append("### Monthly Variance Totals\n")
    cal_df["month"] = pd.to_datetime(cal_df["date"]).dt.to_period("M")
    monthly = cal_df.groupby("month")[commodities].sum().reset_index()
    monthly["Month"] = monthly["month"].astype(str)

    lines.append("| Month | " + " | ".join(commodities) + " |")
    lines.append("|-------|" + "|".join(["------"] * len(commodities)) + "|")
    for _, row in monthly.iterrows():
        vals = [f"{row[c]:.1f}" for c in commodities]
        lines.append(f"| {row['Month']} | " + " | ".join(vals) + " |")
    lines.append("")

    # Next N days daily detail
    lines.append(f"### Next {daily_lookahead} Days Detail\n")
    cutoff = today + timedelta(days=daily_lookahead)
    daily = cal_df[pd.to_datetime(cal_df["date"]) <= cutoff]

    # Only show rows with events, holidays, weekends, or non-standard weights
    lines.append("| Date | Day | Event | " + " | ".join(commodities) + " |")
    lines.append("|------|-----|-------|" + "|".join(["------"] * len(commodities)) + "|")
    for _, row in daily.iterrows():
        flags = []
        if row["weekend"]:
            flags.append("W")
        if row["holiday"]:
            flags.append("H")
        if row["event"]:
            flags.append(row["event"])
        event_str = ", ".join(flags) if flags else ""
        vals = [f"{row[c]:.2f}" for c in commodities]
        lines.append(f"| {row['date']} | {row['day']} | {event_str} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# BONUS SECTIONS (kept from v1)
# ============================================================================

def section_cross_commodity(vol_skew_df, snapshot_date, lookback_years=5):
    """Key cross-commodity vol spreads."""
    lines = ["## Cross-Commodity Vol Spreads\n"]

    pairs = [
        ("SOY", "MEAL", "Crush: SOY vs MEAL"),
        ("SOY", "OIL", "Crush: SOY vs OIL"),
        ("CORN", "WHEAT", "Feed: CORN vs WHEAT"),
    ]

    vol_col = "dirty_vol" if "dirty_vol" in vol_skew_df.columns else "atm_iv"
    lookback_start = snapshot_date - pd.DateOffset(years=lookback_years)

    for c1, c2, label in pairs:
        df1 = vol_skew_df[(vol_skew_df["commodity"] == c1) & (vol_skew_df["date"] >= lookback_start)]
        df2 = vol_skew_df[(vol_skew_df["commodity"] == c2) & (vol_skew_df["date"] >= lookback_start)]

        if df1.empty or df2.empty:
            continue

        latest1 = df1[df1["date"] == df1["date"].max()]
        latest2 = df2[df2["date"] == df2["date"].max()]

        if latest1.empty or latest2.empty or vol_col not in latest1.columns:
            continue

        iv1 = latest1.iloc[0].get(vol_col)
        iv2 = latest2.iloc[0].get(vol_col)

        if iv1 is not None and iv2 is not None:
            spread = iv1 - iv2
            ratio = iv1 / iv2 if iv2 != 0 else None

            hist_spreads = []
            for date in df1["date"].unique():
                d1 = df1[df1["date"] == date]
                d2 = df2[df2["date"] == date]
                if not d1.empty and not d2.empty:
                    v1 = d1.iloc[0].get(vol_col)
                    v2 = d2.iloc[0].get(vol_col)
                    if v1 is not None and v2 is not None:
                        hist_spreads.append(v1 - v2)

            if hist_spreads:
                pctile = (np.array(hist_spreads) < spread).mean() * 100
                lines.append(f"**{label}:** {iv1:.2f} - {iv2:.2f} = {spread:+.2f} "
                           f"(ratio: {ratio:.2f}, percentile: {pctile:.0f}%)")
            else:
                lines.append(f"**{label}:** {iv1:.2f} - {iv2:.2f} = {spread:+.2f}")

    lines.append("")
    return "\n".join(lines)


def section_variance_ratios(snapshot_date):
    """Load cached variance ratios."""
    lines = ["## Variance Ratios\n"]

    cache_dir = CACHE_DIR / "variance_ratios"
    if not cache_dir.exists():
        lines.append("No cached variance ratios found.\n")
        return "\n".join(lines)

    for commodity in COMMODITIES:
        parquet_files = sorted(cache_dir.glob(f"{commodity}_*.parquet"))
        csv_files = sorted(cache_dir.glob(f"{commodity}_*.csv"))
        files = parquet_files or csv_files

        if not files:
            continue

        lines.append(f"### {commodity}")
        try:
            latest_file = files[-1]
            if latest_file.suffix == ".parquet":
                df = pd.read_parquet(latest_file)
            else:
                df = pd.read_csv(latest_file)

            lines.append(f"Cached file: {latest_file.name}")
            if len(df) <= 20:
                lines.append("```")
                lines.append(df.to_string())
                lines.append("```")
            else:
                lines.append("```")
                lines.append(df.head(10).to_string())
                lines.append("... (truncated)")
                lines.append("```")
        except Exception as e:
            lines.append(f"Error reading {latest_file.name}: {e}")

        lines.append("")

    return "\n".join(lines)


def section_notable_moves(vol_skew_df, prices_df, snapshot_date, threshold_pct=2.0):
    """Flag anything that moved significantly in the last session."""
    lines = ["## Notable Moves (Last Session)\n"]

    vol_col = "dirty_vol" if "dirty_vol" in vol_skew_df.columns else "atm_iv"
    price_col = "close" if "close" in prices_df.columns else "settle"

    lines.append("### Price")
    for commodity in COMMODITIES:
        cdf = prices_df[prices_df["commodity"] == commodity].copy()
        if cdf.empty or price_col not in cdf.columns or "contract_code" not in cdf.columns:
            continue
        latest_date = cdf["date"].max()
        prev_date = cdf[cdf["date"] < latest_date]["date"].max()
        if pd.isna(prev_date):
            continue
        # Use the front month contract (alphabetically first code on latest date)
        latest_contracts = sorted(cdf[cdf["date"] == latest_date]["contract_code"].unique())
        if not latest_contracts:
            continue
        front = latest_contracts[0]
        curr_rows = cdf[(cdf["date"] == latest_date) & (cdf["contract_code"] == front)]
        prev_rows = cdf[(cdf["date"] == prev_date) & (cdf["contract_code"] == front)]
        if curr_rows.empty or prev_rows.empty:
            continue
        curr = curr_rows.iloc[0][price_col]
        prev = prev_rows.iloc[0][price_col]
        if prev != 0 and pd.notna(curr) and pd.notna(prev):
            pct_chg = ((curr - prev) / prev) * 100
            if abs(pct_chg) >= threshold_pct:
                lines.append(f"- **{commodity} ({front})**: {pct_chg:+.2f}% ({prev:.2f} → {curr:.2f})")

    lines.append("\n### Implied Vol")
    for commodity in COMMODITIES:
        cdf = vol_skew_df[vol_skew_df["commodity"] == commodity]
        if cdf.empty or vol_col not in cdf.columns:
            continue

        dates = sorted(cdf["date"].unique())
        if len(dates) < 2:
            continue

        latest = cdf[cdf["date"] == dates[-1]]
        prev = cdf[cdf["date"] == dates[-2]]

        for _, row in latest.head(3).iterrows():
            expiry = row.get("expiry")
            label = _expiry_label(expiry)
            curr_iv = row.get(vol_col)
            prev_match = prev[prev["expiry"] == expiry] if "expiry" in prev.columns else pd.DataFrame()
            if not prev_match.empty and curr_iv is not None:
                prev_iv = prev_match.iloc[0].get(vol_col)
                if prev_iv and prev_iv != 0:
                    chg = curr_iv - prev_iv
                    if abs(chg) >= 0.5:
                        lines.append(f"- **{commodity} {label}**: {chg:+.2f} ({prev_iv:.2f} → {curr_iv:.2f})")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# MAIN SNAPSHOT GENERATOR
# ============================================================================

def generate_snapshot(
    snapshot_date=None,
    lookback_years=5,
    output_dir=None,
    prices_df=None,
    vol_skew_df=None,
    master_df=None,
):
    """
    Generate a complete analytical snapshot as a markdown file.

    Parameters
    ----------
    snapshot_date : datetime, optional
        Date for the snapshot. Defaults to latest date in data.
    lookback_years : int
        Historical lookback for percentile calculations.
    output_dir : Path, optional
        Where to save the snapshot file. Defaults to SNAPSHOT_DIR.
    prices_df : DataFrame, optional
        Pre-loaded prices. If None, loads from disk.
    vol_skew_df : DataFrame, optional
        Pre-loaded vol/skew (current + live). If None, loads from disk.
    master_df : DataFrame, optional
        Full historical vol/skew for percentile/skew calcs.

    Returns
    -------
    str : Path to the generated snapshot file.
    """
    output_dir = Path(output_dir) if output_dir else SNAPSHOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if not provided
    print("Loading data...")
    if prices_df is None:
        prices_df = load_prices()
    if vol_skew_df is None:
        vol_skew_df = load_vol_skew()
    if master_df is None:
        master_df = vol_skew_df

    # Ensure contract_month is present
    vol_skew_df = _prepare_vol_df(vol_skew_df)
    master_df = _prepare_vol_df(master_df)

    # Determine snapshot date
    if snapshot_date is None:
        snapshot_date = vol_skew_df["date"].max()
    snapshot_date = pd.Timestamp(snapshot_date)

    print(f"Generating snapshot for {snapshot_date.strftime('%Y-%m-%d')}...")

    # Build snapshot — 7 main sections + bonus
    sections = [
        section_header(snapshot_date),
        section_power_grid(vol_skew_df, snapshot_date),
        section_vol_changes(vol_skew_df, snapshot_date),
        section_iv_percentiles(vol_skew_df, snapshot_date, master_df),
        section_realized_vol(snapshot_date),
        section_skew_analyzer(vol_skew_df, snapshot_date, master_df),
        section_heat_map(vol_skew_df, snapshot_date, master_df),
        section_var_cal(snapshot_date),
        section_cross_commodity(vol_skew_df, snapshot_date, lookback_years),
        section_variance_ratios(snapshot_date),
        section_notable_moves(vol_skew_df, prices_df, snapshot_date),
    ]

    snapshot_text = "\n---\n\n".join(sections)

    # Save
    filename = f"snapshot_{snapshot_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M')}.md"
    filepath = output_dir / filename
    filepath.write_text(snapshot_text, encoding="utf-8")

    latest_path = output_dir / "snapshot_latest.md"
    latest_path.write_text(snapshot_text, encoding="utf-8")

    size_kb = filepath.stat().st_size / 1024
    print(f"Snapshot saved: {filepath} ({size_kb:.1f} KB)")
    print(f"Latest copy: {latest_path}")

    return str(filepath)


# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

def add_snapshot_button(prices_df=None, vol_skew_df=None, master_df=None):
    """
    Call this from your Streamlit dashboard to add a snapshot button.

    Usage in vol_dashboard.py:
        from generate_snapshot import add_snapshot_button
        add_snapshot_button(prices_df=load_price_data(), vol_skew_df=df, master_df=master_df)
    """
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.subheader("Claude Project Snapshot")

    lookback = st.sidebar.selectbox(
        "Snapshot lookback",
        [3, 5, 7, 10],
        index=1,
        key="snapshot_lookback"
    )

    if st.sidebar.button("Generate Snapshot", key="gen_snapshot"):
        with st.sidebar.status("Generating snapshot..."):
            filepath = generate_snapshot(
                lookback_years=lookback,
                prices_df=prices_df,
                vol_skew_df=vol_skew_df,
                master_df=master_df,
            )
        st.sidebar.success(f"Saved to:\n`{filepath}`")
        st.sidebar.info(
            "Upload `snapshot_latest.md` to your Claude Project "
            "as conversation context."
        )

        with open(filepath, "r") as f:
            preview = f.read()[:3000]
        with st.sidebar.expander("Preview"):
            st.markdown(preview + "\n\n*...truncated...*")


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate vol dashboard snapshot")
    parser.add_argument("--date", type=str, help="Snapshot date (YYYY-MM-DD). Default: latest.")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback years for percentiles")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    snap_date = pd.Timestamp(args.date) if args.date else None
    generate_snapshot(
        snapshot_date=snap_date,
        lookback_years=args.lookback,
        output_dir=args.output,
    )
