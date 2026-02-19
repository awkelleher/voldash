"""
Volatility Trading Dashboard
Consolidates key metrics for options trading analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import os
from lib import vol_analysis as va
from lib import variance_ratios as vr
from lib import price_analysis as pa
from datetime import datetime, timedelta

# Shared CME holiday set — used by VAR CAL and FWD VOL CURVE
CME_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-01", "2028-01-17", "2028-02-21", "2028-04-14",
    "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04",
    "2028-11-23", "2028-12-25",
}


def count_trading_days(start_date, end_date) -> int:
    """Count trading days (Mon–Fri, excluding CME holidays) from start_date to end_date inclusive."""
    count = 0
    cur = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while cur <= end:
        if cur.weekday() < 5 and cur.strftime("%Y-%m-%d") not in CME_HOLIDAYS:
            count += 1
        cur += timedelta(days=1)
    return count


# Shared USDA/WASDE events — used by VAR CAL and FWD VOL CURVE
CME_EVENTS = {
    # 2026
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
    # 2027
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
    # 2028
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

# Shared default variance weights — used by VAR CAL and FWD VOL CURVE
CME_DEFAULT_VAR = {
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


def compute_variance_days(commodity: str, end_date, var_vals: dict) -> float:
    """
    Sum variance weights from today through end_date (inclusive) for a given commodity,
    using the VAR CAL event/holiday/weekend weighting logic.
    var_vals: dict of {event_type: {commodity: weight}} — uses session state if available.
    """
    total = 0.0
    cur = pd.Timestamp(datetime.now().date())
    end = pd.Timestamp(end_date)
    while cur <= end:
        dstr = cur.strftime("%Y-%m-%d")
        dow = cur.weekday()
        is_weekend = dow >= 5
        is_holiday = dstr in CME_HOLIDAYS
        event = CME_EVENTS.get(dstr, "")
        if is_holiday:
            etype = "Holiday (H)"
        elif is_weekend:
            etype = "Weekend (W)"
        elif event and event in var_vals:
            etype = event
        else:
            etype = "Day"
        total += var_vals.get(etype, {}).get(commodity, 1.0)
        cur += timedelta(days=1)
    return total


def compute_fvc_vols(df, commodity: str, date, var_vals: dict, used_matrix=None, vr_matrix=None, max_months: int = 12) -> dict:
    """
    Compute variance-day vols (forward vols via variance stripping) for a commodity.
    Returns {contract_month_int: var_day_vol} for all contracts up to max_months.
    M1 = clean vol, M2+ = variance-stripped forward vol.
    """
    import lib.variance_ratios as _vr
    sub = (
        df[(df['date'] == date) & (df['commodity'] == commodity)]
        .sort_values('expiry')
        .head(max_months)
        .reset_index(drop=True)
    )
    if len(sub) == 0:
        return {}

    labels = build_contract_labels_from_expiry(sub['expiry'], commodity)
    expiries = [pd.Timestamp(r.expiry) for r in sub.itertuples()]
    today_ts = pd.Timestamp(datetime.now().date())
    var_365 = compute_variance_days(commodity, today_ts + timedelta(days=365), var_vals)

    # Load variance ratio matrices if not supplied
    if vr_matrix is None or used_matrix is None:
        front_label = str(labels.iloc[0])
        front_options_month = ''.join(filter(str.isalpha, front_label))
        _vr_matrix, _ = _vr.load_cached_variance_ratios(commodity, front_options_month)
        vr_matrix = _vr_matrix
        used_matrix = None  # caller can pass session state value

    # Precompute adjacent-pair ratios
    adjacent_ratios = []
    for j in range(len(labels) - 1):
        next_label = str(labels.iloc[j + 1])
        col_display = "VarRat to FM" if j == 0 else f"VarRat to {j + 1}FM"
        col_raw = f"VarRat{j + 1}"
        ratio = None
        if used_matrix is not None and not used_matrix.empty:
            if next_label in used_matrix.index and col_display in used_matrix.columns:
                v = used_matrix.loc[next_label, col_display]
                if pd.notna(v):
                    ratio = float(v)
        if ratio is None and vr_matrix is not None and not vr_matrix.empty:
            if next_label in vr_matrix.index and col_raw in vr_matrix.columns:
                ratio = float(vr_matrix.loc[next_label, col_raw])
        adjacent_ratios.append(ratio if ratio is not None else 1.0)

    result = {}
    cumulative_mvd = 0.0
    prior_month_var_days = []
    prior_var_day_var = []

    for i, row in enumerate(sub.itertuples()):
        this_expiry = expiries[i]
        total_trd_days = count_trading_days(today_ts, this_expiry)
        variance_days = compute_variance_days(commodity, this_expiry, var_vals)
        month_var_days = variance_days - cumulative_mvd
        cumulative_mvd += month_var_days

        dirty_vol = row.dirty_vol if pd.notna(row.dirty_vol) else None
        if dirty_vol and variance_days > 0 and var_365 > 0:
            clean_vol = ((dirty_vol ** 2 / 251 * total_trd_days) / variance_days * var_365) ** 0.5
        else:
            clean_vol = None

        if i == 0:
            var_day_vol = clean_vol
            var_day_var = clean_vol ** 2 if clean_vol is not None else None
        else:
            fwd_vol = None
            if clean_vol is not None and month_var_days > 0:
                sp = sum(
                    (adjacent_ratios[j] if j < len(adjacent_ratios) else 1.0) * pmvd * pvdv
                    for j, (pmvd, pvdv) in enumerate(zip(prior_month_var_days, prior_var_day_var))
                )
                inner = (clean_vol ** 2 * variance_days - sp) / month_var_days
                if inner > 0:
                    fwd_vol = inner ** 0.5
            var_day_vol = fwd_vol
            var_day_var = fwd_vol ** 2 if fwd_vol is not None else None

        prior_month_var_days.append(month_var_days)
        prior_var_day_var.append(var_day_var if var_day_var is not None else 0.0)

        cm_int = int(row.contract_month) if hasattr(row, 'contract_month') else (i + 1)
        if var_day_vol is not None:
            result[cm_int] = var_day_vol

    return result


if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

# Page config
st.set_page_config(
    page_title="Vol Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load historical data with caching"""
    return va.load_historical_data()


def get_file_mtime(path: str) -> float:
    """Return file mtime for cache-busting (0 when unavailable)."""
    try:
        return float(os.path.getmtime(path)) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


@st.cache_data
def load_master(_cache_key: float = 0.0):
    """Load master historical vol/skew data (for percentiles/skew)"""
    return va.load_master_data()

@st.cache_data(ttl=10)
def load_live_data(_cache_key: float = 0.0):
    """Load live vol/skew CSV overlay if available."""
    try:
        if os.path.exists('data/live_vols.csv'):
            live = pd.read_csv('data/live_vols.csv')
            live['date'] = pd.to_datetime(live['date'], format='mixed')
            live['expiry'] = pd.to_datetime(live['expiry'], format='mixed')
            return live
    except Exception:
        pass
    return None

@st.cache_data
def load_price_data():
    """Load historical price data for variance ratio calculations"""
    try:
        import os
        if os.path.exists('data/all_commodity_prices.csv'):
            prices = pd.read_csv('data/all_commodity_prices.csv')
            prices['date'] = pd.to_datetime(prices['date'], format='mixed')
            return prices
    except:
        pass
    return None


@st.cache_data(ttl=30)
def load_realized_vol_data():
    """Load precomputed realized volatility cache."""
    try:
        if os.path.exists('cache/realized_vol_precomputed.csv'):
            rv = pd.read_csv('cache/realized_vol_precomputed.csv')
            rv['date'] = pd.to_datetime(rv['date'], errors='coerce', format='mixed')
            return rv
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def load_correlation_data():
    """Load precomputed correlation matrix rows."""
    try:
        if os.path.exists('cache/correlation_matrices.csv'):
            cdf = pd.read_csv('cache/correlation_matrices.csv')
            if 'window' in cdf.columns:
                cdf['window'] = pd.to_numeric(cdf['window'], errors='coerce').astype('Int64')
            if 'correlation' in cdf.columns:
                cdf['correlation'] = pd.to_numeric(cdf['correlation'], errors='coerce')
            for c in ['commodity_1', 'commodity_2']:
                if c in cdf.columns:
                    cdf[c] = cdf[c].astype(str).str.upper().str.strip()
            return cdf
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def load_median_iv_cache():
    """Load precomputed median IV table if available."""
    try:
        if os.path.exists('cache/median_iv.csv'):
            mdf = pd.read_csv('cache/median_iv.csv')
            # Normalize expected columns
            rename_map = {
                'commodity': 'commodity',
                'COMMODITY': 'commodity',
                'FRONT_OPTIONS': 'front_options',
                'front_options_month': 'front_options',
                'OPTIONS': 'options_month',
                'options_month': 'options_month',
                'median_iv': 'median_iv',
                'MEDIAN_IV': 'median_iv',
            }
            cols = {}
            for c in mdf.columns:
                if c in rename_map:
                    cols[c] = rename_map[c]
            mdf = mdf.rename(columns=cols)
            required = {'commodity', 'front_options', 'options_month', 'median_iv'}
            if not required.issubset(set(mdf.columns)):
                return pd.DataFrame()
            mdf['commodity'] = mdf['commodity'].astype(str).str.upper().str.strip()
            mdf['front_options'] = mdf['front_options'].astype(str).str.upper().str.strip()
            mdf['options_month'] = mdf['options_month'].astype(str).str.upper().str.strip()
            mdf['median_iv'] = pd.to_numeric(mdf['median_iv'], errors='coerce')
            return mdf[['commodity', 'front_options', 'options_month', 'median_iv']]
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_skew_percentile_dist_cache():
    """Load precomputed skew percentile distribution table if available."""
    try:
        p = 'cache/skew_percentile_dist.csv'
        if not os.path.exists(p):
            return pd.DataFrame()
        sdf = pd.read_csv(p)
        if len(sdf) == 0:
            return pd.DataFrame()
        for c in ['commodity', 'FRONT_OPTIONS', 'OPTIONS']:
            if c in sdf.columns:
                sdf[c] = sdf[c].astype(str).str.upper().str.strip()
        if 'commodity' in sdf.columns:
            sdf['commodity'] = sdf['commodity'].astype(str).str.upper().str.strip()
        return sdf
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_source_update_timestamps():
    """Return filesystem modified timestamps for primary source files."""
    def _fmt(path):
        try:
            if os.path.exists(path):
                ts = os.path.getmtime(path)
                return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
        return "unavailable"

    return {
        "live_data": _fmt('data/live_vols.csv'),
        "master_vol_skew": _fmt('data/master_vol_skew.csv'),
        "all_commodity_prices": _fmt('data/all_commodity_prices.csv'),
    }


def lookback_years_from_days(lookback_days: int):
    """Map trading-day lookback slider to precompute lookback years bucket."""
    yrs = int(round(float(lookback_days) / 252.0))
    return max(1, min(12, yrs))


@st.cache_data
def load_iv_snapshot_cached(commodity: str, front_options_month: str, lookback_years):
    """Load precomputed IV/skew snapshot from parquet cache."""
    try:
        from scripts import iv_percentiles_precompute as ivp
        snap, meta = ivp.load_iv_snapshot(commodity, front_options_month, lookback_years)
        return snap, meta
    except Exception as e:
        return pd.DataFrame(), {"error": str(e)}


def current_front_options_month(commodity: str, as_of_date):
    """Get front options month code for commodity/date."""
    try:
        from scripts import iv_percentiles_precompute as ivp
        return ivp.get_current_front_month(commodity, pd.to_datetime(as_of_date))
    except Exception:
        month_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
        return month_codes[pd.to_datetime(as_of_date).month - 1]


def assign_contract_month_from_snapshot(tmp: pd.DataFrame, max_months: int = 8) -> pd.DataFrame:
    """
    Build unique contract_month ordering from snapshot rows.
    Prefer trading_dte, then expiry, then curve_position.
    If multiple years of same options month are present, keep nearest-dated instance.
    """
    out = tmp.copy()

    # Keep one row per options month code when available (nearest contract on curve)
    if 'options_month' in out.columns:
        if 'trading_dte' in out.columns:
            out = out.sort_values('trading_dte', na_position='last')
        elif 'expiry' in out.columns:
            out = out.sort_values('expiry', na_position='last')
        out = out.drop_duplicates(subset=['options_month'], keep='first')

    if 'trading_dte' in out.columns:
        out = out.sort_values('trading_dte', na_position='last').reset_index(drop=True)
    elif 'expiry' in out.columns:
        out = out.sort_values('expiry', na_position='last').reset_index(drop=True)
    elif 'curve_position' in out.columns:
        out = out.sort_values('curve_position', na_position='last').reset_index(drop=True)
    if max_months is not None and max_months > 0:
        out = out.head(max_months).copy()
    out['contract_month'] = np.arange(1, len(out) + 1)
    return out


MONTH_CODE_FALLBACK = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}


def get_options_lookup_for_commodity(commodity: str) -> dict:
    """Return expiry_month -> options month code mapping for a commodity."""
    try:
        m = va.load_options_mapping('data/mapping.csv')
        sub = m[m['COMMODITY'] == str(commodity).upper()]
        if len(sub) > 0 and 'EXPIRY_MONTH' in sub.columns:
            return {
                int(r['EXPIRY_MONTH']): str(r['OPTIONS']).upper()
                for _, r in sub.iterrows()
                if pd.notna(r.get('EXPIRY_MONTH', np.nan)) and pd.notna(r.get('OPTIONS', np.nan))
            }
    except Exception:
        pass
    return {}


def build_contract_labels_from_expiry(expiry_series: pd.Series, commodity: str) -> pd.Series:
    """
    Build contract code labels from expiry + commodity mapping, applying F-year convention.
    Example: expiry 2026-12-24 -> F27.
    """
    exp = pd.to_datetime(expiry_series, errors='coerce')
    lookup = get_options_lookup_for_commodity(commodity)
    opt_code = exp.dt.month.map(
        lambda m: lookup.get(int(m), MONTH_CODE_FALLBACK.get(int(m), '?')) if pd.notna(m) else '?'
    )
    contract_year = exp.dt.year + np.where(opt_code == 'F', 1, 0)
    labels = opt_code.astype(str) + contract_year.mod(100).fillna(0).astype(int).astype(str).str.zfill(2)
    labels = labels.replace({'nan': np.nan, 'None': np.nan, '?00': np.nan})
    return labels


def build_vol_change_table(
    df: pd.DataFrame,
    commodity: str,
    selected_date,
    long_window: int,
    max_months: int = 12,
    live_df: pd.DataFrame | None = None,
    master_df: pd.DataFrame | None = None
):
    """
    Build table for 1-day and N-day changes in IV (dirty_vol) and fwd_vol.
    Uses trading-day offsets from available data dates.
    """
    cdf = df[df['commodity'] == commodity].copy()

    # Explicit live overlay for VOL CHANGE:
    # for matching (date, commodity, expiry), prefer live rows.
    if live_df is not None and len(live_df) > 0:
        live_cols = [c for c in ['date', 'commodity', 'expiry', 'dirty_vol', 'fwd_vol'] if c in live_df.columns]
        if all(c in live_cols for c in ['date', 'commodity', 'expiry', 'dirty_vol', 'fwd_vol']):
            live_sub = live_df[live_df['commodity'] == commodity][live_cols].copy()
            if len(live_sub) > 0:
                cdf = pd.concat([live_sub, cdf], ignore_index=True)
                cdf = cdf.drop_duplicates(subset=['date', 'commodity', 'expiry'], keep='first')
                cdf = cdf.sort_values(['date', 'commodity', 'expiry']).copy()
                cdf['contract_month'] = cdf.groupby(['date', 'commodity']).cumcount() + 1

    if len(cdf) == 0:
        return pd.DataFrame(), {"error": f"No data for {commodity}"}

    cdf['date'] = pd.to_datetime(cdf['date'], errors='coerce')
    cdf = cdf[cdf['date'].notna()]
    if len(cdf) == 0:
        return pd.DataFrame(), {"error": "No valid dates in data"}

    dates = pd.DatetimeIndex(cdf['date'].dropna().sort_values().unique())
    if len(dates) == 0:
        return pd.DataFrame(), {"error": "No historical dates available"}

    asof = pd.to_datetime(selected_date)
    # Current date/source rule:
    # Prefer latest live_vols snapshot (commodity-level). If unavailable, use
    # latest merged date on/before selected date.
    current_source = "merged"
    current_date = pd.NaT
    if live_df is not None and len(live_df) > 0:
        lsub = live_df[live_df['commodity'] == commodity].copy()
        if len(lsub) > 0:
            lsub['date'] = pd.to_datetime(lsub['date'], errors='coerce')
            ldates = pd.DatetimeIndex(lsub['date'].dropna().sort_values().unique())
            if len(ldates) > 0:
                current_date = ldates[-1]
                current_source = "live"
    if pd.isna(current_date):
        idx = int(dates.searchsorted(asof, side='right') - 1)
        if idx < 0:
            return pd.DataFrame(), {"error": f"No data on/before {asof.date()}"}
        current_date = dates[idx]
        current_source = "merged"

    # Reference dates for change calculations come from MASTER history only.
    # 1D = latest master date; ND = Nth-latest master date.
    if master_df is not None and len(master_df) > 0:
        mdf = master_df[master_df['commodity'] == commodity].copy()
        mdf['date'] = pd.to_datetime(mdf['date'], errors='coerce')
        master_dates = pd.DatetimeIndex(mdf['date'].dropna().sort_values().unique())
    else:
        master_dates = dates

    if len(master_dates) == 0:
        return pd.DataFrame(), {"error": f"No master history available for {commodity}"}

    # 1D baseline = absolute latest available master date.
    prev_date = master_dates[-1] if len(master_dates) >= 1 else None
    long_date = master_dates[-long_window] if len(master_dates) >= long_window else None

    if current_source == "live" and live_df is not None and len(live_df) > 0:
        current = live_df[
            (pd.to_datetime(live_df['date'], errors='coerce') == current_date) &
            (live_df['commodity'] == commodity)
        ][['expiry', 'dirty_vol', 'fwd_vol']].copy()
        current['expiry'] = pd.to_datetime(current['expiry'], errors='coerce')
        current = current.sort_values('expiry').drop_duplicates(subset=['expiry'], keep='first').head(max_months)
        current['contract_month'] = np.arange(1, len(current) + 1)
        current = current[['contract_month', 'expiry', 'dirty_vol', 'fwd_vol']]
    else:
        current = cdf[cdf['date'] == current_date][['contract_month', 'expiry', 'dirty_vol', 'fwd_vol']].copy()
        current = current.sort_values(['contract_month', 'expiry']).drop_duplicates(subset=['expiry'], keep='first').head(max_months)

    current = current.rename(columns={'dirty_vol': 'iv_now', 'fwd_vol': 'fwd_now'})

    current['contract_code'] = build_contract_labels_from_expiry(current['expiry'], commodity)
    current['contract_code'] = current['contract_code'].fillna(current['contract_month'].map(lambda x: f"M{int(x)}"))

    out = current[['contract_month', 'expiry', 'contract_code', 'iv_now', 'fwd_now']].copy()

    # Baseline values should come from MASTER history (not live-overlaid cdf)
    ref_base = cdf
    if master_df is not None and len(master_df) > 0:
        ref_base = master_df[master_df['commodity'] == commodity].copy()
        ref_base['date'] = pd.to_datetime(ref_base['date'], errors='coerce')
        if 'expiry' in ref_base.columns:
            ref_base['expiry'] = pd.to_datetime(ref_base['expiry'], errors='coerce')

    def _fill_missing_baseline_by_code(out_df, iv_col, fwd_col, cutoff_date):
        """
        If exact cutoff-date baseline is missing for a contract code, backfill from
        the latest available master row for that same code on/before cutoff_date.
        """
        if cutoff_date is None:
            return out_df
        if iv_col not in out_df.columns or fwd_col not in out_df.columns:
            return out_df
        miss_mask = out_df[iv_col].isna() | out_df[fwd_col].isna()
        if not miss_mask.any():
            return out_df

        hist = ref_base[ref_base['date'] <= cutoff_date][['date', 'expiry', 'dirty_vol', 'fwd_vol']].copy()
        if len(hist) == 0:
            return out_df
        hist['contract_code'] = build_contract_labels_from_expiry(hist['expiry'], commodity)
        hist = hist[hist['contract_code'].notna()].copy()
        if len(hist) == 0:
            return out_df
        hist = hist.sort_values(['contract_code', 'date']).drop_duplicates(subset=['contract_code'], keep='last')
        hist = hist.rename(columns={'dirty_vol': f'{iv_col}_fb', 'fwd_vol': f'{fwd_col}_fb'})
        out_df = out_df.merge(hist[['contract_code', f'{iv_col}_fb', f'{fwd_col}_fb']], on='contract_code', how='left')
        out_df[iv_col] = out_df[iv_col].fillna(out_df[f'{iv_col}_fb'])
        out_df[fwd_col] = out_df[fwd_col].fillna(out_df[f'{fwd_col}_fb'])
        out_df = out_df.drop(columns=[f'{iv_col}_fb', f'{fwd_col}_fb'])
        return out_df

    if prev_date is not None:
        prev = ref_base[ref_base['date'] == prev_date][['expiry', 'dirty_vol', 'fwd_vol']].copy()
        prev = prev.sort_values('expiry').drop_duplicates(subset=['expiry'], keep='last')
        prev['contract_code'] = build_contract_labels_from_expiry(prev['expiry'], commodity)
        prev = prev[prev['contract_code'].notna()].copy()
        prev = prev.drop_duplicates(subset=['contract_code'], keep='last')
        prev = prev.rename(columns={'dirty_vol': 'iv_prev_1d', 'fwd_vol': 'fwd_prev_1d'})
        out = out.merge(prev[['contract_code', 'iv_prev_1d', 'fwd_prev_1d']], on='contract_code', how='left')
        out = _fill_missing_baseline_by_code(out, 'iv_prev_1d', 'fwd_prev_1d', prev_date)
        out['iv_chg_1d'] = out['iv_now'] - out['iv_prev_1d']
        out['fwd_chg_1d'] = out['fwd_now'] - out['fwd_prev_1d']
    else:
        out['iv_chg_1d'] = np.nan
        out['fwd_chg_1d'] = np.nan

    if long_date is not None:
        long_df = ref_base[ref_base['date'] == long_date][['expiry', 'dirty_vol', 'fwd_vol']].copy()
        long_df = long_df.sort_values('expiry').drop_duplicates(subset=['expiry'], keep='last')
        long_df['contract_code'] = build_contract_labels_from_expiry(long_df['expiry'], commodity)
        long_df = long_df[long_df['contract_code'].notna()].copy()
        long_df = long_df.drop_duplicates(subset=['contract_code'], keep='last')
        long_df = long_df.rename(columns={'dirty_vol': 'iv_prev_long', 'fwd_vol': 'fwd_prev_long'})
        out = out.merge(long_df[['contract_code', 'iv_prev_long', 'fwd_prev_long']], on='contract_code', how='left')
        out = _fill_missing_baseline_by_code(out, 'iv_prev_long', 'fwd_prev_long', long_date)
        out['iv_chg_long'] = out['iv_now'] - out['iv_prev_long']
        out['fwd_chg_long'] = out['fwd_now'] - out['fwd_prev_long']
    else:
        out['iv_chg_long'] = np.nan
        out['fwd_chg_long'] = np.nan

    out = out.sort_values('contract_month').reset_index(drop=True)
    if 'expiry' in out.columns:
        out = out.drop(columns=['expiry'])
    return out, {
        "current_date": current_date,
        "prev_date": prev_date,
        "long_date": long_date,
        "long_window": int(long_window),
    }

# Bloomberg-style dark theme CSS
st.markdown("""
    <style>
    /* ---- Global dark overrides ---- */
    .stApp {
        background-color: #0a0e17;
        color: #c8cdd3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f1419;
        border-right: 1px solid #1e2530;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: #8b949e;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f1419;
        border-bottom: 1px solid #1e2530;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f1419;
        color: #8b949e;
        border: 1px solid #1e2530;
        font-family: 'Consolas', monospace;
        font-size: 13px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1f2e !important;
        color: #ff9500 !important;
        border-bottom: 2px solid #ff9500 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #111820;
        border: 1px solid #1e2530;
        padding: 12px;
        border-radius: 2px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-family: 'Consolas', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'Consolas', monospace !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Consolas', monospace !important;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid #1e2530;
    }

    /* Markdown text */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #c8cdd3;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #586069 !important;
        font-family: 'Consolas', monospace !important;
    }

    /* Dividers */
    hr {
        border-color: #1e2530 !important;
    }

    /* Power grid custom classes */
    .power-grid-container {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        border: 1px solid #1e2530;
        background-color: #0f1419;
    }
    .power-cell-pos {
        color: #3fb950;
        font-weight: bold;
    }
    .power-cell-neg {
        color: #f85149;
        font-weight: bold;
    }
    .power-cell-neutral {
        color: #8b949e;
    }
    .bloomberg-header {
        background-color: #111820;
        border-bottom: 2px solid #ff9500;
        padding: 8px 12px;
        margin-bottom: 4px;
    }
    .bloomberg-header span {
        color: #ff9500 !important;
        font-family: 'Consolas', monospace !important;
        font-size: 14px !important;
        font-weight: bold;
        letter-spacing: 1px;
    }

    /* Info/warning/success/error boxes */
    .stAlert {
        background-color: #111820;
        border: 1px solid #1e2530;
        color: #c8cdd3;
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background-color: #111820 !important;
        color: #e6edf3 !important;
        border-color: #1e2530 !important;
        font-family: 'Consolas', monospace !important;
    }

    /* Slider */
    .stSlider label {
        color: #8b949e !important;
    }

    /* Buttons - soft green */
    .stButton > button {
        background-color: #1f4d2e !important;
        color: #d7f5df !important;
        border: 1px solid #2f6f44 !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #275f38 !important;
        border-color: #3a8653 !important;
        color: #e8fff0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Optional light-theme overrides (applied on top of defaults)
if st.session_state.get("theme_mode", "Dark") == "Light":
    st.markdown("""
        <style>
        .stApp {
            background-color: #f6f8fa !important;
            color: #1f2328 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #d0d7de !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown span,
        section[data-testid="stSidebar"] label {
            color: #57606a !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1f2328 !important;
        }
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #1f2328 !important;
        }
        .bloomberg-header {
            background-color: #ffffff !important;
            border-bottom: 2px solid #d97706 !important;
        }
        [data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
        }
        .stAlert {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            color: #1f2328 !important;
        }
        .stDataFrame {
            border: 1px solid #d0d7de !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="bloomberg-header"><span>VOL TRADING DASHBOARD</span></div>', unsafe_allow_html=True)

# Top tab-style navigation + sidebar data controls
NAV_SECTIONS = [
    "Vol Sheet",
    "Price Sheet",
    "Skew Analyzer",
    "IV Calendar",
    "Spread Builder",
    "Trading Calendar",
    "Var Cal",
    "Settings",
]
default_section = st.session_state.get("active_section", "Vol Sheet")

# Keep main section navigation in top tab-style control.
active_section = st.segmented_control(
    "Navigation",
    NAV_SECTIONS,
    default=default_section if default_section in NAV_SECTIONS else "Vol Sheet",
    key="active_section",
    label_visibility="collapsed"
)

with st.sidebar:
    with st.expander("Data Controls", expanded=True):
        if st.button("Refresh Live", key="refresh_live_sidebar"):
            try:
                import subprocess
                result = subprocess.run(
                    ['python', 'scripts/xlsm_to_csv.py', '-o', 'data/live_vols.csv'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Refresh failed: {result.stderr[-200:]}")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

        _updates_top = get_source_update_timestamps()
        st.caption(f"Live data updated at: {_updates_top.get('live_data', 'unavailable')}")
        st.caption(f"Vol data updated at: {_updates_top.get('master_vol_skew', 'unavailable')}")
        st.caption(f"Price data updated at: {_updates_top.get('all_commodity_prices', 'unavailable')}")

if active_section is None:
    active_section = "Vol Sheet"

if active_section not in ["Vol Sheet", "Price Sheet", "Skew Analyzer", "IV Calendar", "Spread Builder", "Trading Calendar", "Var Cal", "Settings"]:
    st.markdown(f'<div class="bloomberg-header"><span>{active_section.upper()}</span></div>', unsafe_allow_html=True)
    st.info(f"{active_section} view is coming soon.")
    st.stop()

if active_section == "Settings":
    st.markdown('<div class="bloomberg-header"><span>SETTINGS</span></div>', unsafe_allow_html=True)
    st.subheader("Theme")
    current_mode = st.session_state.get("theme_mode", "Dark")
    light_mode = st.toggle("Light mode", value=(current_mode == "Light"))
    selected_mode = "Light" if light_mode else "Dark"
    if selected_mode != current_mode:
        st.session_state["theme_mode"] = selected_mode
        st.rerun()
    st.caption(f"Current mode: {selected_mode}")
    st.stop()

# Load data
try:
    df = load_data()
    master_df = load_master(get_file_mtime('data/master_vol_skew.csv'))
    master_base_df = master_df.copy()
    live_df = load_live_data(get_file_mtime('data/live_vols.csv'))

    # Ensure datetime for date columns
    for frame in [df, master_df]:
        if 'date' in frame.columns:
            frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
        if 'expiry' in frame.columns:
            frame['expiry'] = pd.to_datetime(frame['expiry'], errors='coerce')

    # Merge live data overlay — live rows replace historical for same (date, commodity, expiry)
    if live_df is not None and not live_df.empty:
        # Add compatibility columns to live data before merge
        if 'clean_vol' not in live_df.columns and 'dirty_vol' in live_df.columns:
            live_df['clean_vol'] = live_df['dirty_vol']

        df = pd.concat([live_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=['date', 'commodity', 'expiry'], keep='first')
        master_df = pd.concat([live_df, master_df], ignore_index=True)
        master_df = master_df.drop_duplicates(subset=['date', 'commodity', 'expiry'], keep='first')

        # Re-derive contract_month after merge (live rows don't have it)
        for frame in [df, master_df]:
            frame.sort_values(['date', 'commodity', 'expiry'], inplace=True)
            frame['contract_month'] = frame.groupby(['date', 'commodity']).cumcount() + 1

    latest_date = pd.to_datetime(df['date']).max()
    earliest_date = pd.to_datetime(df['date']).min()
    if pd.isna(latest_date) or pd.isna(earliest_date):
        st.error("No dates found in data.")
        st.stop()

    # Default analysis context (navigation-only sidebar; no per-view controls)
    selected_date = pd.to_datetime(latest_date)
    master_hist = master_df[master_df['date'] < selected_date].copy()
    commodity = 'SOY'
    lookback = 252
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Active section selector (compute only selected section)
if active_section == "Vol Sheet":
    tab_options = [
        "POWER GRID",
        "VOL CHANGES",
        "IV Percentiles",
        "FWD VOL CURVE",
    ]
    active_tab = st.radio("View", tab_options, horizontal=True, key="active_tab_vol")
    price_product = None
elif active_section == "Price Sheet":
    st.markdown('<div class="bloomberg-header"><span>PRICE SHEET</span></div>', unsafe_allow_html=True)
    price_tab_options = [
        "REALIZED VOL",
        "CORRELATIONS",
        "VAR RATIOS",
    ]
    active_tab = st.radio("View", price_tab_options, horizontal=True, key="active_tab_price")
    price_product = None
    if active_tab == "REALIZED VOL":
        # Narrow product selector to half its previous width (now ~10% of row)
        col_prod, _ = st.columns([1, 9])
        with col_prod:
            price_product = st.selectbox(
                "Product",
                ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW'],
                index=0,
                key="price_sheet_product"
            )
elif active_section == "Skew Analyzer":
    st.markdown('<div class="bloomberg-header"><span>SKEW ANALYZER</span></div>', unsafe_allow_html=True)
    active_tab = "SKEW"
    price_product = None
elif active_section == "IV Calendar":
    st.markdown('<div class="bloomberg-header"><span>IV CALENDAR</span></div>', unsafe_allow_html=True)
    active_tab = "IV CALENDAR"
    price_product = None
elif active_section == "Spread Builder":
    st.markdown('<div class="bloomberg-header"><span>SPREAD BUILDER</span></div>', unsafe_allow_html=True)
    active_tab = "SPREAD BUILDER"
    price_product = None
elif active_section == "Trading Calendar":
    st.markdown('<div class="bloomberg-header"><span>TRADING CALENDAR</span></div>', unsafe_allow_html=True)
    active_tab = "TRADING CALENDAR"
    price_product = None
elif active_section == "Var Cal":
    st.markdown('<div class="bloomberg-header"><span>VAR CAL</span></div>', unsafe_allow_html=True)
    active_tab = "VAR CAL"
    price_product = None

# ============================================================================
# TAB: FWD VOL CURVE
# ============================================================================
if active_tab == "FWD VOL CURVE":
    st.markdown('<div class="bloomberg-header"><span>FWD VOL CURVE</span></div>', unsafe_allow_html=True)
    st.caption("Forward volatility curve by commodity — built from market IVs. Implied Dirty Vol = current market IV.")

    # Display order requested: SOY, MEAL, CORN, WHEAT, KW, OIL
    _fvc_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    _fvc_max_months = 12

    # Row labels matching the screenshot layout
    _fvc_row_labels = [
        "Month",
        "Variance Day Variance",
        "Variance Day Vol",
        "Implied Clean Vol",
        "Implied Dirty Vol",
        "Month Variance Days",
        "Month Trading Days",
        "Variance Days",
        "Total Trading Days",
        "Expiration",
    ]

    # Get the latest date snapshot from df
    _fvc_date = df['date'].max() if df is not None and len(df) > 0 else None

    for _comm in _fvc_commodities:
        st.markdown(f"**{_comm}**")

        if _fvc_date is None:
            st.warning("No data available.")
            continue

        # Filter to this commodity on the latest date, sorted by expiry (front first)
        _fvc_sub = (
            df[(df['date'] == _fvc_date) & (df['commodity'] == _comm)]
            .sort_values('expiry')
            .head(_fvc_max_months)
            .reset_index(drop=True)
        )

        if len(_fvc_sub) == 0:
            st.caption(f"No data for {_comm}.")
            continue

        # Build contract code labels (e.g. N26, F27)
        _fvc_labels = build_contract_labels_from_expiry(_fvc_sub['expiry'], _comm)

        # Build grid: rows = metrics, columns = contract months
        _fvc_grid = {"": _fvc_row_labels}

        # Build sorted list of expiry Timestamps for prior-month boundary lookups
        _fvc_expiries = [pd.Timestamp(_r.expiry) for _r in _fvc_sub.itertuples()]

        _var_vals = st.session_state.get("varcal_var_values", CME_DEFAULT_VAR)
        _today_ts = pd.Timestamp(datetime.now().date())
        _cumulative_month_var_days = 0.0  # running sum of prior Month Variance Days

        # Total variance days over the next 365 calendar days (normalizing constant per commodity)
        _var_365 = compute_variance_days(_comm, _today_ts + timedelta(days=365), _var_vals)

        # Load variance ratio matrix for this commodity keyed to current front options month
        _front_label = str(_fvc_labels.iloc[0])
        _front_options_month = ''.join(filter(str.isalpha, _front_label))
        _vr_matrix, _ = vr.load_cached_variance_ratios(_comm, _front_options_month)

        # Prefer editable "Variance Ratios - Used" from session state if available
        _ss_key = f"varrat_used_{_comm}_{_front_options_month}"
        _used_matrix = st.session_state.get(_ss_key, None)

        # Precompute adjacent-pair ratios: _adjacent_ratios[j] = contract[j+1] / contract[j]
        # e.g. [0]=K/H, [1]=N/K, [2]=Q/N, [3]=U/Q, ...
        # Session state uses display col names ("VarRat to FM", "VarRat to 2FM", ...)
        # Raw cache uses ("VarRat1", "VarRat2", ...)
        _adjacent_ratios = []
        for _j in range(len(_fvc_labels) - 1):
            _next_label = str(_fvc_labels.iloc[_j + 1])
            _col_display = "VarRat to FM" if _j == 0 else f"VarRat to {_j + 1}FM"
            _col_raw = f"VarRat{_j + 1}"
            _ratio = None
            # Try editable session state first
            if _used_matrix is not None and not _used_matrix.empty:
                if _next_label in _used_matrix.index and _col_display in _used_matrix.columns:
                    _v = _used_matrix.loc[_next_label, _col_display]
                    if pd.notna(_v):
                        _ratio = float(_v)
            # Fallback to raw cache
            if _ratio is None and not _vr_matrix.empty:
                if _next_label in _vr_matrix.index and _col_raw in _vr_matrix.columns:
                    _ratio = float(_vr_matrix.loc[_next_label, _col_raw])
            _adjacent_ratios.append(_ratio if _ratio is not None else 1.0)

        # Accumulators for SUMPRODUCT across prior contracts
        _prior_labels = []
        _prior_month_var_days = []
        _prior_var_day_var = []

        for _i, (_label, _row) in enumerate(zip(_fvc_labels, _fvc_sub.itertuples())):
            _expiry_str = pd.Timestamp(_row.expiry).strftime("%m/%d/%Y") if pd.notna(_row.expiry) else ""
            _this_expiry = _fvc_expiries[_i]

            # Total Trading Days: trading days from today through this expiry
            _total_trd_days = count_trading_days(_today_ts, _this_expiry)

            # Month Trading Days: trading days from day after prior expiry through this expiry
            # Front month (i=0): from today through this expiry (same as total)
            if _i == 0:
                _month_trd_days = _total_trd_days
            else:
                _month_start = _fvc_expiries[_i - 1] + timedelta(days=1)
                _month_trd_days = count_trading_days(_month_start, _this_expiry)

            # Variance Days: cumulative VAR CAL variance sum from today through this expiry
            _variance_days = compute_variance_days(_comm, _this_expiry, _var_vals)

            # Month Variance Days: variance days this contract holds while front month
            # = total Variance Days minus sum of all preceding Month Variance Days
            _month_var_days = _variance_days - _cumulative_month_var_days
            _cumulative_month_var_days += _month_var_days

            # Implied Clean Vol: ((dirty_vol^2 / 251 * total_trading_days) / variance_days * var_365) ^ 0.5
            _dirty_vol = _row.dirty_vol if pd.notna(_row.dirty_vol) else None
            if _dirty_vol and _variance_days > 0 and _var_365 > 0:
                _clean_vol = ((_dirty_vol ** 2 / 251 * _total_trd_days) / _variance_days * _var_365) ** 0.5
            else:
                _clean_vol = None

            # Variance Day Vol and Variance Day Variance
            # M1: equals Clean Vol / Clean Vol^2
            # M2+: forward vol via variance stripping (replicates Ladder!F4 Excel formula)
            if _i == 0:
                _var_day_vol = _clean_vol
                _var_day_var = _clean_vol ** 2 if _clean_vol is not None else None
            else:
                _fwd_vol = None
                if _clean_vol is not None and _month_var_days > 0:
                    # SUMPRODUCT(adjacent_ratios[j], MonthVarDays[j], VarDayVariance[j])
                    # adjacent_ratios[j] = contract[j+1]/contract[j]: K/H, N/K, Q/N, ...
                    _sp = 0.0
                    for _j, (_pmvd, _pvdv) in enumerate(
                            zip(_prior_month_var_days, _prior_var_day_var)):
                        _ratio = _adjacent_ratios[_j] if _j < len(_adjacent_ratios) else 1.0
                        _sp += _ratio * _pmvd * _pvdv
                    _inner = (_clean_vol ** 2 * _variance_days - _sp) / _month_var_days
                    if _inner > 0:
                        _fwd_vol = _inner ** 0.5
                _var_day_vol = _fwd_vol
                _var_day_var = _fwd_vol ** 2 if _fwd_vol is not None else None

            # Store this contract's values for subsequent iterations' SUMPRODUCT
            _prior_labels.append(str(_label))
            _prior_month_var_days.append(_month_var_days)
            _prior_var_day_var.append(_var_day_var if _var_day_var is not None else 0.0)

            col_data = [
                str(_label),                                                    # Month (contract code)
                f"{_var_day_var:.0f}" if _var_day_var is not None else "",     # Variance Day Variance (0 decimals)
                f"{_var_day_vol:.2f}" if _var_day_vol is not None else "",     # Variance Day Vol
                f"{_clean_vol:.2f}" if _clean_vol is not None else "",         # Implied Clean Vol
                f"{_dirty_vol:.2f}" if _dirty_vol is not None else "",         # Implied Dirty Vol = market IV
                f"{_month_var_days:.2f}",                                       # Month Variance Days
                str(_month_trd_days),                                           # Month Trading Days
                f"{_variance_days:.2f}",                                        # Variance Days (cumulative)
                str(_total_trd_days),                                           # Total Trading Days (today → expiry)
                _expiry_str,                                                    # Expiration
            ]
            _fvc_grid[_expiry_str or str(_i)] = col_data

        _fvc_df = pd.DataFrame(_fvc_grid)

        # Persist computed vols to session state (also done unconditionally at Power Grid tab)
        st.session_state[f"fvc_vols_{_comm}"] = compute_fvc_vols(
            df, _comm, _fvc_date, _var_vals,
            used_matrix=st.session_state.get(f"varrat_used_{_comm}_{_front_options_month}"),
            vr_matrix=_vr_matrix,
        )

        # Style: highlight Implied Dirty Vol row (index 4) in a distinct color
        def _fvc_style(row, row_labels=_fvc_row_labels):
            label = row.iloc[0]
            if label == "Month":
                return ["background-color: #1c2333; color: #c8d8f0; font-weight: bold;"] * len(row)
            if label == "Implied Dirty Vol":
                return ["background-color: #1a2a1a; color: #a0d4a0; font-weight: bold;"] * len(row)
            if label == "Expiration":
                return ["background-color: #1c2333; color: #8b9db5;"] * len(row)
            return [""] * len(row)

        _fvc_styled = _fvc_df.style.apply(_fvc_style, axis=1)

        st.dataframe(
            _fvc_styled,
            use_container_width=True,
            hide_index=True,
            height=36 + 28 * len(_fvc_row_labels),
        )

        st.markdown("")  # spacer between commodity grids

# ============================================================================
# TAB 0: POWER GRID
# ============================================================================
if active_tab == "POWER GRID":
    st.markdown('<div class="bloomberg-header"><span>POWER GRID</span></div>', unsafe_allow_html=True)
    st.caption("(FWD VOL - PREDICTED RV) / PREDICTED RV  |  Positive = IV rich, Negative = IV cheap")

    grid_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    max_grid_months = 8

    # Load predicted RV from verdad workbook
    verdad_overall = {}
    verdad_monthly = {}
    verdad_loaded = False
    try:
        import os
        if os.path.exists('data/verdad.csv'):
            verdad_overall, verdad_monthly = va.load_verdad_predictions('data/verdad.csv')
            verdad_loaded = True
        elif os.path.exists('data/verdad.7.xlsx'):
            verdad_overall, verdad_monthly = va.load_verdad_predictions('data/verdad.7.xlsx')
            verdad_loaded = True
    except Exception as e:
        st.warning(f"Could not load verdad.7.xlsx: {e}")

    if verdad_loaded:
        st.markdown("""
        <div style="font-family:Consolas,monospace;font-size:11px;color:#56d364;margin-bottom:6px;">
            VERDAD MODEL LOADED — per-month predicted RV active
        </div>
        """, unsafe_allow_html=True)

        # Show overall levels as reference
        overall_str = " &nbsp;|&nbsp; ".join(
            [f'<span style="color:#ff9500;">{c}</span>: {verdad_overall.get(c, 0):.1f}'
             for c in grid_commodities if c in verdad_overall]
        )
        st.markdown(f"""
        <div style="font-family:Consolas,monospace;font-size:12px;color:#8b949e;margin-bottom:12px;">
            OVERALL LEVELS: {overall_str}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("**MODEL INPUTS** — Enter predicted realized vol for each commodity")

    # Fallback manual inputs when verdad is not loaded
    if not verdad_loaded:
        default_rvs = {}
        for c in grid_commodities:
            m1_data = df[
                (df['date'] == selected_date) & (df['commodity'] == c) & (df['contract_month'] == 1)
            ]
            if len(m1_data) > 0 and pd.notna(m1_data.iloc[0]['fwd_vol']):
                default_rvs[c] = float(m1_data.iloc[0]['fwd_vol'])
            else:
                default_rvs[c] = 15.0

        input_cols = st.columns(len(grid_commodities))
        manual_rv_inputs = {}
        for i, c in enumerate(grid_commodities):
            with input_cols[i]:
                manual_rv_inputs[c] = st.number_input(
                    f"{c}", min_value=0.1, max_value=100.0,
                    value=default_rvs[c], step=0.25, format="%.2f",
                    key=f"power_rv_{c}"
                )
        predicted_rv_overall = manual_rv_inputs
        predicted_rv_monthly = None
    else:
        predicted_rv_overall = verdad_overall
        predicted_rv_monthly = verdad_monthly

    st.markdown("---")

    # Always compute FWD VOL CURVE vols (variance-stripped forward vols) so Power Grid
    # uses them regardless of whether the FWD VOL CURVE tab has been visited this session.
    _pg_var_vals = st.session_state.get("varcal_var_values", CME_DEFAULT_VAR)
    _pg_fvc_date = df['date'].max() if df is not None and len(df) > 0 else selected_date
    for _pg_comm in grid_commodities:
        _pg_front_label = None
        _pg_sub = df[(df['date'] == _pg_fvc_date) & (df['commodity'] == _pg_comm)].sort_values('expiry')
        if len(_pg_sub) > 0:
            _pg_labels = build_contract_labels_from_expiry(_pg_sub['expiry'], _pg_comm)
            _pg_front_label = ''.join(filter(str.isalpha, str(_pg_labels.iloc[0])))
        _pg_used = st.session_state.get(f"varrat_used_{_pg_comm}_{_pg_front_label}", None) if _pg_front_label else None
        st.session_state[f"fvc_vols_{_pg_comm}"] = compute_fvc_vols(
            df, _pg_comm, _pg_fvc_date, _pg_var_vals, used_matrix=_pg_used
        )

    # Patch fwd_vol with FWD VOL CURVE variance-stripped forward vols
    df = df.copy()
    for _pg_comm in grid_commodities:
        _fvc_ss = st.session_state.get(f"fvc_vols_{_pg_comm}", {})
        for _pg_cm, _pg_vol in _fvc_ss.items():
            _mask = (df['date'] == selected_date) & (df['commodity'] == _pg_comm) & (df['contract_month'] == _pg_cm)
            df.loc[_mask, 'fwd_vol'] = _pg_vol

    # Calculate the power grid
    power_df, power_meta = va.calculate_power_grid(
        df, selected_date, predicted_rv_overall,
        commodities=grid_commodities, max_months=max_grid_months,
        monthly_predictions=predicted_rv_monthly
    )

    if len(power_df) > 0:
        fwd_vols_df = power_meta['fwd_vols']
        pred_rvs_df = power_meta['pred_rvs']
        month_codes_df = power_meta.get('month_codes', pd.DataFrame())

        # Display matrix as Contract x Commodity (swapped orientation), using contract codes.
        def _contract_sort_key(code):
            month_rank = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
            s = str(code) if pd.notna(code) else ""
            if len(s) < 2:
                return (9999, 99)
            try:
                yy = int(s[1:])
            except Exception:
                yy = 9999
            return (yy, month_rank.get(s[0], 99))

        def _contract_labels_for_commodity(comm):
            labels = {}
            day = df[(df['date'] == selected_date) & (df['commodity'] == comm)].sort_values('contract_month')
            if len(day) == 0:
                return labels
            for _, r in day.iterrows():
                cm = int(r['contract_month'])
                if cm > max_grid_months:
                    continue
                exp = pd.to_datetime(r.get('expiry', None), errors='coerce')
                if pd.isna(exp):
                    continue
                labels[f"M{cm}"] = build_contract_labels_from_expiry(pd.Series([exp]), comm).iloc[0]
            return labels

        contract_maps = {c: _contract_labels_for_commodity(c) for c in grid_commodities}
        contracts = set()
        for c in grid_commodities:
            if c in power_df.index:
                for mcol in power_df.columns:
                    if pd.notna(power_df.loc[c, mcol]):
                        contracts.add(contract_maps.get(c, {}).get(mcol, mcol))
        contracts = sorted(list(contracts), key=_contract_sort_key)

        display = pd.DataFrame({'Contract': contracts})
        for c in grid_commodities:
            display[c] = np.nan
            if c not in power_df.index:
                continue
            cmap = contract_maps.get(c, {})
            for mcol in power_df.columns:
                if pd.isna(power_df.loc[c, mcol]):
                    continue
                ccode = cmap.get(mcol, mcol)
                hit = display['Contract'] == ccode
                if hit.any():
                    display.loc[hit, c] = power_df.loc[c, mcol]

        def _power_cell_style(val):
            if pd.isna(val):
                return ''
            v = float(val)
            if v > 5:
                return 'background-color: #1f4d2e; color: #e6edf3; font-weight: 600;'
            if v > 0:
                return 'background-color: #214e36; color: #e6edf3;'
            if v > -5:
                return 'background-color: #5a1f2a; color: #e6edf3;'
            return 'background-color: #6b2330; color: #e6edf3; font-weight: 600;'

        value_cols = [c for c in grid_commodities if c in display.columns]
        st.dataframe(
            display.style.format({col: '{:+.1f}%' for col in value_cols}).applymap(_power_cell_style, subset=value_cols),
            use_container_width=True,
            hide_index=True,
            height=min(520, 80 + 32 * len(display))
        )

        st.markdown("")  # spacer

        # Expandable reference tables
        col_ref1, col_ref2 = st.columns(2)

        def _matrix_to_contract_grid(src_df: pd.DataFrame) -> pd.DataFrame:
            """Convert commodity x M# matrix to Contract x Commodity display."""
            if src_df is None or len(src_df) == 0:
                return pd.DataFrame()
            out = pd.DataFrame({'Contract': contracts})
            for c in grid_commodities:
                out[c] = np.nan
                if c not in src_df.index:
                    continue
                cmap = contract_maps.get(c, {})
                for mcol in src_df.columns:
                    v = src_df.loc[c, mcol] if mcol in src_df.columns else np.nan
                    if pd.isna(v):
                        continue
                    ccode = cmap.get(mcol, mcol)
                    hit = out['Contract'] == ccode
                    if hit.any():
                        out.loc[hit, c] = v
            return out

        with col_ref1:
            with st.expander("FWD VOL MATRIX"):
                if len(fwd_vols_df) > 0:
                    fwd_disp = _matrix_to_contract_grid(fwd_vols_df)
                    fwd_cols = [c for c in grid_commodities if c in fwd_disp.columns]
                    st.dataframe(
                        fwd_disp.style.format({col: '{:.2f}' for col in fwd_cols}, na_rep='—'),
                        use_container_width=True,
                        hide_index=True,
                        height=min(420, 80 + 30 * len(fwd_disp))
                    )

        with col_ref2:
            with st.expander("PREDICTED RV MATRIX"):
                if len(pred_rvs_df) > 0:
                    rv_disp = _matrix_to_contract_grid(pred_rvs_df)
                    rv_cols = [c for c in grid_commodities if c in rv_disp.columns]
                    st.dataframe(
                        rv_disp.style.format({col: '{:.2f}' for col in rv_cols}, na_rep='—'),
                        use_container_width=True,
                        hide_index=True,
                        height=min(420, 80 + 30 * len(rv_disp))
                    )

        # Legend
        st.markdown("""
        <div style="font-family:Consolas,monospace;font-size:11px;color:#586069;margin-top:8px;">
            <span style="color:#3fb950;">&#9632;</span> &gt;+5% IV RICH &nbsp;&nbsp;
            <span style="color:#56d364;">&#9632;</span> 0 to +5% &nbsp;&nbsp;
            <span style="color:#f85149;">&#9632;</span> 0 to -5% &nbsp;&nbsp;
            <span style="color:#ff7b72;">&#9632;</span> &lt;-5% IV CHEAP &nbsp;&nbsp;
            | Hover cells for FV:fwd_vol | RV:pred_rv | [month_code]
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No forward vol data available for {selected_date.date()}")

# ============================================================================
# TAB: IV CALENDAR
# ============================================================================
if active_tab == "IV CALENDAR":
    cal_col, _ = st.columns([0.5, 4.5])
    with cal_col:
        cal_commodity = st.selectbox(
            "Product",
            ['SOY', 'MEAL', 'SM-S', 'CORN', 'WHEAT', 'KW', 'OIL'],
            index=0,
            key="iv_cal_commodity"
        )

    # Use master_df (full history, not just lookback) for seasonal averages
    if cal_commodity == 'SM-S':
        avg_meal, med_meal = va.compute_iv_calendar_grid(master_df, 'MEAL')
        avg_soy, med_soy = va.compute_iv_calendar_grid(master_df, 'SOY')
        # Align columns: use intersection of both
        shared_cols = [c for c in avg_meal.columns if c in avg_soy.columns]
        avg_grid = avg_meal[shared_cols] - avg_soy[shared_cols]
        med_grid = med_meal[shared_cols] - med_soy[shared_cols]
    else:
        avg_grid, med_grid = va.compute_iv_calendar_grid(master_df, cal_commodity)

    if len(avg_grid) == 0:
        st.warning(f"No IV calendar data available for {cal_commodity}.")
    else:
        # Determine current time period for highlighting
        current_tp = va._date_to_time_period(selected_date)

        # Determine current front month for header
        if cal_commodity == 'SM-S':
            front_label = current_front_options_month('MEAL', selected_date)
            st.caption(f"MEAL minus SOY seasonal dirty vol spread  |  Front: {front_label}  |  Current period: {current_tp}")
        else:
            front_label = current_front_options_month(cal_commodity, selected_date)
            st.caption(f"Seasonal dirty vol by ~10-day period  |  Front: {front_label}  |  Current period: {current_tp}")
        value_cols = list(avg_grid.columns)

        def _iv_cal_row_highlight(row, current_period):
            """Highlight the current time period row."""
            if row.name == current_period:
                return ['background-color: #1a2332; font-weight: 600;'] * len(row)
            return [''] * len(row)

        # ---- CURRENT vs MEDIAN (top) ----
        # Helper to get current IV series for a commodity
        def _get_current_iv(commodity_name):
            cdata = df[
                (df['date'] == selected_date) & (df['commodity'] == commodity_name)
            ].copy()
            if len(cdata) == 0:
                return pd.Series(dtype=float)
            cdata['expiry'] = pd.to_datetime(cdata['expiry'], errors='coerce')
            _mapping = va.load_options_mapping()
            _em_sub = _mapping[_mapping['COMMODITY'] == commodity_name.upper()]
            _em_to_opt = {
                int(r['EXPIRY_MONTH']): r['OPTIONS']
                for _, r in _em_sub.iterrows()
                if pd.notna(r.get('EXPIRY_MONTH'))
            }
            cdata['options_code'] = cdata['expiry'].dt.month.map(_em_to_opt)
            cdata = cdata.dropna(subset=['options_code', 'dirty_vol'])
            if len(cdata) == 0:
                return pd.Series(dtype=float)
            cdata = cdata.sort_values('expiry').drop_duplicates(subset=['options_code'], keep='first')
            return cdata.set_index('options_code')['dirty_vol']

        if cal_commodity == 'SM-S':
            meal_iv = _get_current_iv('MEAL')
            soy_iv = _get_current_iv('SOY')
            shared = [c for c in meal_iv.index if c in soy_iv.index]
            current_iv = meal_iv[shared] - soy_iv[shared] if shared else pd.Series(dtype=float)
        else:
            current_iv = _get_current_iv(cal_commodity)

        if len(current_iv) > 0 and current_tp in med_grid.index:
            # Build heat map row: current IV - median for current period
            med_row = med_grid.loc[current_tp]
            heat_vals = {}
            for code in value_cols:
                if code in current_iv.index and pd.notna(med_row.get(code)):
                    heat_vals[code] = float(current_iv[code]) - float(med_row[code])
                else:
                    heat_vals[code] = np.nan

            heat_df = pd.DataFrame([heat_vals], index=[current_tp])
            heat_df.index.name = 'Period'

            def _heat_cell_style(val):
                if pd.isna(val):
                    return ''
                v = float(val)
                if v > 2:
                    return 'background-color: #1f4d2e; color: #e6edf3; font-weight: 600;'
                if v > 0:
                    return 'background-color: #214e36; color: #e6edf3;'
                if v > -2:
                    return 'background-color: #5a1f2a; color: #e6edf3;'
                return 'background-color: #6b2330; color: #e6edf3; font-weight: 600;'

            st.markdown("**CURRENT vs MEDIAN** (positive = above seasonal)")
            st.dataframe(
                heat_df.style
                    .format({col: '{:+.2f}' for col in value_cols}, na_rep='—')
                    .applymap(_heat_cell_style, subset=value_cols),
                use_container_width=True,
                hide_index=False
            )

        st.markdown("")  # spacer

        # ---- MEDIAN grid ----
        st.markdown("**MEDIAN**")
        med_display = med_grid.copy()
        med_display.index.name = 'Period'
        st.dataframe(
            med_display.style
                .format({col: '{:.2f}' for col in value_cols}, na_rep='—')
                .apply(_iv_cal_row_highlight, current_period=current_tp, axis=1),
            use_container_width=True,
            height=min(900, 80 + 28 * len(med_display))
        )

        st.markdown("")  # spacer

        # ---- AVERAGE grid ----
        st.markdown("**AVERAGE**")
        avg_display = avg_grid.copy()
        avg_display.index.name = 'Period'
        st.dataframe(
            avg_display.style
                .format({col: '{:.2f}' for col in value_cols}, na_rep='—')
                .apply(_iv_cal_row_highlight, current_period=current_tp, axis=1),
            use_container_width=True,
            height=min(900, 80 + 28 * len(avg_display))
        )

# ============================================================================
# TAB: SPREAD BUILDER
# ============================================================================
if active_tab == "SPREAD BUILDER":
    # Keep selectors narrow; place Run All on the right edge with spacer
    sb_col1, sb_col2, sb_col3, sb_col_range, sb_col_toggle, sb_col_spacer, sb_col_run = st.columns(
        [1, 1, 1, 1.35, 1.0, 3.5, 1.35],
        vertical_alignment="bottom"
    )
    with sb_col1:
        sb_commodity = st.selectbox(
            "Commodity",
            ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW'],
            index=0,
            key="sb_commodity"
        )
    _all_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    with sb_col2:
        sb_month1 = st.selectbox("Month 1", _all_codes, index=3, key="sb_month1")  # default J
    with sb_col3:
        sb_month2 = st.selectbox("Month 2", _all_codes, index=4, key="sb_month2")  # default K
    _sb_norm_enabled = st.session_state.get("sb_normalize", False)
    with sb_col_range:
        sb_norm_range = st.number_input(
            "Vol band", min_value=0.1, max_value=50.0, value=2.5, step=0.5,
            key="sb_norm_range", disabled=not _sb_norm_enabled)
    with sb_col_toggle:
        sb_normalize = st.toggle("Normalize", value=_sb_norm_enabled, key="sb_normalize")
    with sb_col_run:
        sb_run_all = st.button("Run All", key="sb_run_all", use_container_width=True)

    # ── "Run All" – show all consecutive IV spreads from live data ─────────
    if sb_run_all:
        _mapping_df = va.load_options_mapping()
        _em_sub = _mapping_df[_mapping_df['COMMODITY'] == sb_commodity.upper()]
        _em_to_opt = {
            int(r['EXPIRY_MONTH']): r['OPTIONS']
            for _, r in _em_sub.iterrows()
            if pd.notna(r.get('EXPIRY_MONTH'))
        }
        _opt_to_em = {v: k for k, v in _em_to_opt.items()}

        # Determine which source has data and gather all available IVs
        _ra_source = "live_vols"
        _ra_vols = {}   # options_month -> dirty_vol
        _ra_asof = pd.NaT

        if live_df is not None and len(live_df) > 0:
            _ra_sub = live_df[live_df['commodity'] == sb_commodity].copy()
            _ra_sub['date'] = pd.to_datetime(_ra_sub['date'], errors='coerce')
            _ra_sub = _ra_sub[_ra_sub['date'].notna()]
            if len(_ra_sub) > 0:
                _ra_le = _ra_sub[_ra_sub['date'] <= selected_date]
                _ra_asof = _ra_le['date'].max() if len(_ra_le) > 0 else _ra_sub['date'].max()
                _ra_day = _ra_sub[_ra_sub['date'] == _ra_asof].copy()
                _ra_day['expiry'] = pd.to_datetime(_ra_day['expiry'], errors='coerce')
                _ra_day = _ra_day.dropna(subset=['expiry', 'dirty_vol'])
                _ra_day['options_code'] = _ra_day['expiry'].dt.month.map(_em_to_opt)
                _ra_day = _ra_day.dropna(subset=['options_code'])
                _ra_day = _ra_day.sort_values('expiry').drop_duplicates(
                    subset=['options_code'], keep='first')
                _ra_vols = dict(zip(_ra_day['options_code'], _ra_day['dirty_vol']))

        if len(_ra_vols) < 2:
            _ra_source = "snapshot fallback"
            _ra_msub = df[df['commodity'] == sb_commodity].copy()
            _ra_msub['date'] = pd.to_datetime(_ra_msub['date'], errors='coerce')
            _ra_msub = _ra_msub[_ra_msub['date'].notna()]
            if len(_ra_msub) > 0:
                _ra_le = _ra_msub[_ra_msub['date'] <= selected_date]
                _ra_asof = _ra_le['date'].max() if len(_ra_le) > 0 else _ra_msub['date'].max()
                _ra_day = _ra_msub[_ra_msub['date'] == _ra_asof].copy()
                _ra_day['expiry'] = pd.to_datetime(_ra_day['expiry'], errors='coerce')
                _ra_day = _ra_day.dropna(subset=['expiry', 'dirty_vol'])
                _ra_day['options_code'] = _ra_day['expiry'].dt.month.map(_em_to_opt)
                _ra_day = _ra_day.dropna(subset=['options_code'])
                _ra_day = _ra_day.sort_values('expiry').drop_duplicates(
                    subset=['options_code'], keep='first')
                _ra_vols = dict(zip(_ra_day['options_code'], _ra_day['dirty_vol']))

        # Determine front month from live data (nearest DTE)
        _ra_avail = sorted(_ra_vols.keys(), key=lambda m: _all_codes.index(m) if m in _all_codes else 99)
        if len(_ra_avail) >= 2:
            # Front month = the one in live data with the smallest expiry month
            # relative to the data date.  Use the order from live DTE.
            if live_df is not None and len(live_df) > 0:
                _ra_front_sub = live_df[
                    (live_df['commodity'] == sb_commodity)
                ].copy()
                _ra_front_sub['date'] = pd.to_datetime(_ra_front_sub['date'], errors='coerce')
                _ra_front_sub['expiry'] = pd.to_datetime(_ra_front_sub['expiry'], errors='coerce')
                _ra_front_sub = _ra_front_sub[_ra_front_sub['date'] == _ra_asof]
                if len(_ra_front_sub) > 0:
                    _ra_front_row = _ra_front_sub.loc[
                        (_ra_front_sub['expiry'] - _ra_front_sub['date']).dt.days.idxmin()
                    ]
                    _ra_front_em = _ra_front_row['expiry'].month
                    _ra_front_month = _em_to_opt.get(int(_ra_front_em), _ra_avail[0])
                else:
                    _ra_front_month = _ra_avail[0]
            else:
                _ra_front_month = _ra_avail[0]

            # Build consecutive pairs from ALL available live contracts
            # ordered from front month through the 12-month cycle.
            # Serial months are included — percentile/median will show n/a
            # when no precomputed history exists for that pair.
            _ra_ordered = []
            if _ra_front_month in _all_codes:
                start_idx = _all_codes.index(_ra_front_month)
                for i in range(12):
                    code = _all_codes[(start_idx + i) % 12]
                    if code in _ra_vols:
                        _ra_ordered.append(code)

            _ra_pairs = [(_ra_ordered[i], _ra_ordered[i + 1])
                         for i in range(len(_ra_ordered) - 1)]

            if _ra_pairs:
                # Load precomputed spread percentile distribution for medians + percentile lookup
                _ra_spread_dist = pd.DataFrame()
                _ra_lookup_fn = None
                try:
                    from scripts.iv_spread_percentiles_precompute import (
                        load_iv_spread_percentile_dist, lookup_iv_spread_percentile,
                    )
                    _ra_spread_dist = load_iv_spread_percentile_dist(
                        commodity=sb_commodity, front_options_month=_ra_front_month)
                    _ra_lookup_fn = lookup_iv_spread_percentile
                except Exception:
                    pass

                _ra_asof_txt = pd.to_datetime(_ra_asof).strftime('%Y-%m-%d') if pd.notna(_ra_asof) else "n/a"
                st.markdown("---")
                st.markdown(
                    f"**ALL CONSECUTIVE SPREADS** — {sb_commodity} | Front: **{_ra_front_month}** | "
                    f"Source: `{_ra_source}` | As of: `{_ra_asof_txt}`"
                )

                # Display spreads in rows of 3 metric boxes
                _ra_n = len(_ra_pairs)
                for row_start in range(0, _ra_n, 3):
                    row_pairs = _ra_pairs[row_start:row_start + 3]
                    cols = st.columns(len(row_pairs))
                    for col_idx, (m1, m2) in enumerate(row_pairs):
                        with cols[col_idx]:
                            _v1 = _ra_vols.get(m1, np.nan)
                            _v2 = _ra_vols.get(m2, np.nan)
                            _sprd = _v1 - _v2 if pd.notna(_v1) and pd.notna(_v2) else np.nan

                            # Look up median and percentile from precomputed distribution
                            _pair_str = f"{m1}-{m2}"
                            _median = np.nan
                            _pctile = np.nan
                            if not _ra_spread_dist.empty:
                                _dist_row = _ra_spread_dist[
                                    _ra_spread_dist['SPREAD_PAIR'] == _pair_str
                                ]
                                if len(_dist_row) > 0:
                                    _median = _dist_row.iloc[0].get('spread_p50', np.nan)
                            if _ra_lookup_fn is not None and pd.notna(_sprd):
                                _lk = _ra_lookup_fn(_sprd, sb_commodity, _ra_front_month, _pair_str)
                                if _lk.get("percentile") is not None:
                                    _pctile = _lk["percentile"] * 100.0

                            # Build delta string: percentile (matching CURRENT SPREAD style)
                            if pd.notna(_pctile):
                                _delta_txt = f"{_pctile:.0f}th %ile"
                            else:
                                _delta_txt = None

                            # Display metric box
                            st.metric(
                                f"{m1}-{m2} Spread",
                                f"{_sprd:+.2f}" if pd.notna(_sprd) else "—",
                                delta=_delta_txt,
                            )
                            # Caption: median + leg IVs
                            _v1_txt = f"{_v1:.2f}" if pd.notna(_v1) else "—"
                            _v2_txt = f"{_v2:.2f}" if pd.notna(_v2) else "—"
                            _med_txt = f"Median: {_median:+.2f}" if pd.notna(_median) else "Median: n/a"
                            st.caption(f"{_med_txt}  |  {m1}: {_v1_txt}  |  {m2}: {_v2_txt}")

                st.markdown("---")
            else:
                st.warning("Not enough contracts in live data to form consecutive pairs.")
        else:
            st.warning("Not enough contracts available to compute spreads.")

    if sb_month1 == sb_month2:
        st.warning("Month 1 and Month 2 must be different.")
    else:
        sb_summary, sb_detail = va.compute_spread_builder(
            master_df, sb_commodity, sb_month1, sb_month2)

        if len(sb_summary) == 0:
            st.warning(f"No spread data for {sb_commodity} {sb_month1}-{sb_month2}.")
        else:
            # Current period
            current_tp = va._date_to_time_period(selected_date)

            # Current spread from live_vols (fallback to merged df snapshot if needed).
            def _get_current_vol(source_df, commodity_name, month_code, asof_date):
                cdata = source_df[
                    (source_df['date'] == asof_date) & (source_df['commodity'] == commodity_name)
                ].copy()
                if len(cdata) == 0:
                    return np.nan
                cdata['expiry'] = pd.to_datetime(cdata['expiry'], errors='coerce')
                _mapping = va.load_options_mapping()
                _em_sub = _mapping[_mapping['COMMODITY'] == commodity_name.upper()]
                _em_to_opt = {
                    int(r['EXPIRY_MONTH']): r['OPTIONS']
                    for _, r in _em_sub.iterrows()
                    if pd.notna(r.get('EXPIRY_MONTH'))
                }
                cdata['options_code'] = cdata['expiry'].dt.month.map(_em_to_opt)
                cdata = cdata.dropna(subset=['options_code', 'dirty_vol'])
                cdata = cdata.sort_values('expiry').drop_duplicates(
                    subset=['options_code'], keep='first')
                match = cdata[cdata['options_code'] == month_code]
                return float(match.iloc[0]['dirty_vol']) if len(match) > 0 else np.nan

            live_asof = pd.NaT
            merged_asof = pd.NaT
            current_v1 = np.nan
            current_v2 = np.nan
            spread_source = "live_vols"
            if live_df is not None and len(live_df) > 0:
                live_sub = live_df[live_df['commodity'] == sb_commodity].copy()
                live_sub['date'] = pd.to_datetime(live_sub['date'], errors='coerce')
                live_sub = live_sub[live_sub['date'].notna()]
                if len(live_sub) > 0:
                    live_le = live_sub[live_sub['date'] <= selected_date]
                    live_asof = live_le['date'].max() if len(live_le) > 0 else live_sub['date'].max()
                    current_v1 = _get_current_vol(live_sub, sb_commodity, sb_month1, live_asof)
                    current_v2 = _get_current_vol(live_sub, sb_commodity, sb_month2, live_asof)

            # Fallback if live did not have both legs.
            if not (pd.notna(current_v1) and pd.notna(current_v2)):
                spread_source = "snapshot fallback"
                merged_sub = df[df['commodity'] == sb_commodity].copy()
                merged_sub['date'] = pd.to_datetime(merged_sub['date'], errors='coerce')
                merged_sub = merged_sub[merged_sub['date'].notna()]
                if len(merged_sub) > 0:
                    merged_le = merged_sub[merged_sub['date'] <= selected_date]
                    merged_asof = merged_le['date'].max() if len(merged_le) > 0 else merged_sub['date'].max()
                    current_v1 = _get_current_vol(merged_sub, sb_commodity, sb_month1, merged_asof)
                    current_v2 = _get_current_vol(merged_sub, sb_commodity, sb_month2, merged_asof)

            current_spread = current_v1 - current_v2 if pd.notna(current_v1) and pd.notna(current_v2) else np.nan

            # Apply normalize filter: keep only obs where vol_1 is within current_v1 +/- range
            if sb_normalize and pd.notna(current_v1):
                vol_lo = current_v1 - sb_norm_range
                vol_hi = current_v1 + sb_norm_range
                sb_detail = sb_detail[
                    (sb_detail['vol_1'] >= vol_lo) & (sb_detail['vol_1'] <= vol_hi)
                ].copy()
                # Recompute summary from filtered detail
                if len(sb_detail) > 0:
                    sb_summary = sb_detail.groupby('time_period')['spread'].agg(
                        low='min', median='median', average='mean', high='max', count='count'
                    ).reset_index()
                    sb_summary['time_period'] = pd.Categorical(
                        sb_summary['time_period'],
                        categories=va._TIME_PERIOD_ORDER, ordered=True)
                    sb_summary = sb_summary.sort_values('time_period').reset_index(drop=True)
                else:
                    sb_summary = pd.DataFrame(columns=['time_period', 'low', 'median', 'average', 'high', 'count'])

            # Percentile of current spread within current period
            if pd.notna(current_spread) and len(sb_summary) > 0 and current_tp in sb_summary['time_period'].values:
                tp_detail = sb_detail[sb_detail['time_period'] == current_tp]
                if len(tp_detail) > 0:
                    pct = (tp_detail['spread'] <= current_spread).mean() * 100.0
                else:
                    pct = np.nan
            else:
                pct = np.nan

            # Top box above the grid: current spread from live_vols.
            source_asof = live_asof if spread_source == "live_vols" else merged_asof
            source_asof_txt = pd.to_datetime(source_asof).strftime('%Y-%m-%d') if pd.notna(source_asof) else "n/a"
            box_col1, box_col2, box_col3 = st.columns([1.2, 1, 1])
            with box_col1:
                st.metric(
                    f"Current Spread ({sb_month1}-{sb_month2})",
                    f"{current_spread:+.2f}" if pd.notna(current_spread) else "—",
                    delta=(f"{pct:.0f}th %ile ({current_tp})" if pd.notna(pct) else None)
                )
            with box_col2:
                st.metric(f"{sb_month1} IV", f"{current_v1:.2f}" if pd.notna(current_v1) else "—")
            with box_col3:
                st.metric(f"{sb_month2} IV", f"{current_v2:.2f}" if pd.notna(current_v2) else "—")
            caption_parts = [f"Source: `{spread_source}` | As of: `{source_asof_txt}`"]
            if sb_normalize and pd.notna(current_v1):
                caption_parts.append(
                    f"| **Normalized**: {sb_month1} IV {current_v1 - sb_norm_range:.1f} – {current_v1 + sb_norm_range:.1f} ({len(sb_detail)} obs)")
            st.caption("  ".join(caption_parts))

            # Header info
            info_parts = [f"**{sb_commodity} {sb_month1}-{sb_month2}**"]
            if pd.notna(current_v1):
                info_parts.append(f"{sb_month1}: {current_v1:.2f}")
            if pd.notna(current_v2):
                info_parts.append(f"{sb_month2}: {current_v2:.2f}")
            if pd.notna(current_spread):
                info_parts.append(f"Spread: {current_spread:+.2f}")
            if pd.notna(pct):
                info_parts.append(f"Percentile ({current_tp}): {pct:.1f}%")
            st.markdown("  |  ".join(info_parts))

            # ---- SUMMARY GRID ----
            if len(sb_detail) == 0:
                st.warning("No observations match the normalize filter. Try widening the range.")
            st.markdown("**SUMMARY BY PERIOD**")
            summary_display = sb_summary.copy()
            summary_display['time_period'] = summary_display['time_period'].astype(str)
            summary_display = summary_display.set_index('time_period')

            def _sb_row_highlight(row, current_period):
                if row.name == current_period:
                    return ['background-color: #1a2332; font-weight: 600;'] * len(row)
                return [''] * len(row)

            fmt = {'low': '{:+.2f}', 'median': '{:+.3f}', 'average': '{:+.3f}',
                   'high': '{:+.2f}', 'count': '{:.0f}'}
            st.dataframe(
                summary_display.style
                    .format(fmt, na_rep='—')
                    .apply(_sb_row_highlight, current_period=current_tp, axis=1),
                use_container_width=True,
                height=min(900, 80 + 28 * len(summary_display))
            )

            # ---- DETAIL TABLE (expandable) ----
            def _spread_cell_style(val):
                if pd.isna(val):
                    return ''
                v = float(val)
                if v > 1:
                    return 'background-color: #1f4d2e; color: #e6edf3;'
                if v > 0:
                    return 'background-color: #214e36; color: #e6edf3;'
                if v > -1:
                    return 'background-color: #5a1f2a; color: #e6edf3;'
                return 'background-color: #6b2330; color: #e6edf3;'

            # Filter to current + adjacent periods
            nearby_idx = va._TIME_PERIOD_ORDER.index(current_tp) if current_tp in va._TIME_PERIOD_ORDER else 0
            nearby_periods = set()
            for offset in [-1, 0, 1]:
                idx = nearby_idx + offset
                if 0 <= idx < len(va._TIME_PERIOD_ORDER):
                    nearby_periods.add(va._TIME_PERIOD_ORDER[idx])

            with st.expander(f"DETAIL: all observations near {current_tp} ({len(nearby_periods)} periods)"):
                detail_nearby = sb_detail[sb_detail['time_period'].isin(nearby_periods)].copy()
                if len(detail_nearby) == 0:
                    st.info("No observations in nearby periods.")
                else:
                    detail_nearby = detail_nearby.sort_values('vol_1', ascending=False)
                    detail_display = detail_nearby[['date', 'time_period', 'vol_1', 'vol_2', 'spread']].copy()
                    detail_display['date'] = pd.to_datetime(detail_display['date']).dt.strftime('%Y-%m-%d')
                    detail_display = detail_display.reset_index(drop=True)

                    st.dataframe(
                        detail_display.style
                            .format({'vol_1': '{:.2f}', 'vol_2': '{:.2f}', 'spread': '{:+.2f}'}, na_rep='—')
                            .applymap(_spread_cell_style, subset=['spread']),
                        use_container_width=True,
                        height=min(600, 80 + 28 * len(detail_display)),
                        hide_index=True
                    )
                    st.caption(f"{len(detail_display)} observations  |  Sorted by {sb_month1} vol descending")

            with st.expander("DETAIL: all observations (all periods)"):
                if len(sb_detail) == 0:
                    st.info("No observations.")
                else:
                    all_display = sb_detail[['date', 'time_period', 'vol_1', 'vol_2', 'spread']].copy()
                    all_display['date'] = pd.to_datetime(all_display['date']).dt.strftime('%Y-%m-%d')
                    all_display = all_display.reset_index(drop=True)

                    st.dataframe(
                        all_display.style
                            .format({'vol_1': '{:.2f}', 'vol_2': '{:.2f}', 'spread': '{:+.2f}'}, na_rep='—')
                            .applymap(_spread_cell_style, subset=['spread']),
                        use_container_width=True,
                        height=min(600, 80 + 28 * min(len(all_display), 20)),
                        hide_index=True
                    )
                    st.caption(f"{len(all_display)} observations  |  Sorted by {sb_month1} vol descending")

# ============================================================================
# TAB: IV PERCENTILE BATTERY
# ============================================================================
if active_tab in ["IV %ILE", "IV Percentiles"]:
    st.markdown('<div class="bloomberg-header"><span>IV PERCENTILE RANK</span></div>', unsafe_allow_html=True)

    # Display order: SOY, MEAL, CORN, WHEAT, KW, OIL
    pct_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    pct_max_months = 12

    # Calculate percentiles: always use precomputed snapshots for median and percentile.
    # When live data exists, overlay live current IV and recompute percentile from
    # the distribution CSV so that the live value is ranked correctly.
    lookback_years = lookback_years_from_days(lookback)
    pct_rows = []
    use_live = live_df is not None and not live_df.empty

    for c in pct_commodities:
        front_opt = current_front_options_month(c, selected_date)
        snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
        if isinstance(snap, pd.DataFrame) and len(snap) > 0:
            tmp = assign_contract_month_from_snapshot(snap, max_months=pct_max_months)
            tmp['commodity'] = c
            # Month labels
            if 'expiry' in tmp.columns:
                tmp['month_label'] = build_contract_labels_from_expiry(tmp['expiry'], c)
            elif 'contract_code' in tmp.columns:
                tmp['month_label'] = tmp['contract_code'].astype(str)
            elif 'options_month' in tmp.columns:
                tmp['month_label'] = tmp['options_month'].astype(str)
            else:
                tmp['month_label'] = tmp['contract_month'].map(lambda x: f"M{int(x)}")
            tmp['month_label'] = tmp['month_label'].replace({'nan': np.nan, 'None': np.nan, '?00': np.nan})
            tmp['month_label'] = tmp['month_label'].fillna(tmp['contract_month'].map(lambda x: f"M{int(x)}"))
            tmp['current'] = pd.to_numeric(tmp.get('atm_iv', np.nan), errors='coerce')
            tmp['median'] = pd.to_numeric(tmp.get('iv_hist_median', np.nan), errors='coerce')
            p = pd.to_numeric(tmp.get('iv_percentile', np.nan), errors='coerce')
            tmp['percentile'] = np.where(p <= 1, p * 100.0, p)

            # Prefer median IV from cache/median_iv.csv keyed by commodity + front month + options month.
            median_cache = load_median_iv_cache()
            if len(median_cache) > 0 and 'options_month' in tmp.columns:
                med_sub = median_cache[
                    (median_cache['commodity'] == c.upper()) &
                    (median_cache['front_options'] == str(front_opt).upper())
                ][['options_month', 'median_iv']].drop_duplicates(subset=['options_month'], keep='last')
                if len(med_sub) > 0:
                    med_map = med_sub.set_index('options_month')['median_iv']
                    tmp['median_cache'] = tmp['options_month'].astype(str).str.upper().map(med_map)
                    tmp['median'] = np.where(tmp['median_cache'].notna(), tmp['median_cache'], tmp['median'])

            # If live data present, overlay live dirty_vol as current and recompute percentile
            if use_live and 'options_month' in tmp.columns:
                from scripts import iv_percentiles_precompute as ivp
                live_sub = live_df[live_df['commodity'] == c].copy()
                if len(live_sub) > 0:
                    live_sub = live_sub.sort_values('expiry' if 'expiry' in live_sub.columns else 'date')
                    live_sub['_em'] = pd.to_datetime(live_sub['expiry'], errors='coerce').dt.month
                    mapping_df = ivp.load_mapping()
                    em_lookup = mapping_df[mapping_df['COMMODITY'] == c.upper()].set_index('EXPIRY_MONTH')['OPTIONS'].to_dict()
                    live_sub['_opt'] = live_sub['_em'].map(em_lookup)
                    for idx, row in tmp.iterrows():
                        om = row.get('options_month', None)
                        if om is None:
                            continue
                        live_match = live_sub[live_sub['_opt'] == om]
                        if len(live_match) > 0:
                            live_iv = pd.to_numeric(live_match.iloc[0].get('dirty_vol', np.nan), errors='coerce')
                            if pd.notna(live_iv) and 0 < live_iv <= 200:
                                tmp.at[idx, 'current'] = live_iv
                                # Recompute percentile from distribution CSV
                                result = ivp.lookup_iv_percentile(live_iv, c, front_opt, om)
                                if result.get('percentile') is not None:
                                    tmp.at[idx, 'percentile'] = result['percentile'] * 100.0

            keep_cols = ['commodity', 'contract_month', 'month_label', 'current', 'percentile', 'median']
            if 'options_month' in tmp.columns:
                keep_cols.append('options_month')
            pct_rows.append(tmp[keep_cols])

    if len(pct_rows) > 0:
        pct_grid = pd.concat(pct_rows, ignore_index=True)
    else:
        pct_grid = va.get_percentile_grid_cached(
            df, selected_date, metric='dirty_vol',
            lookback_days=lookback, commodities=pct_commodities,
            max_months=pct_max_months, hist_df=master_hist
        )

    # Ensure month labels exist
    if len(pct_grid) > 0:
        if 'month_label' not in pct_grid.columns or pct_grid['month_label'].isna().all():
            if 'expiry' in pct_grid.columns:
                def _mk_label_row(r):
                    try:
                        return build_contract_labels_from_expiry(pd.Series([r.get('expiry')]), str(r.get('commodity', 'SOY'))).iloc[0]
                    except Exception:
                        return np.nan
                pct_grid['month_label'] = pct_grid.apply(_mk_label_row, axis=1)
            elif 'contract_code' in pct_grid.columns:
                pct_grid['month_label'] = pct_grid['contract_code'].astype(str)
            elif 'options_month' in pct_grid.columns:
                pct_grid['month_label'] = pct_grid['options_month'].astype(str)
            else:
                pct_grid['month_label'] = pct_grid['contract_month'].map(lambda x: f"M{int(x)}")

    # Round current IV to nearest 0.05%
    def _round_to_005(val):
        try:
            return np.round(float(val) * 20.0) / 20.0
        except Exception:
            return np.nan

    if len(pct_grid) > 0:
        pct_grid['current'] = pct_grid['current'].apply(_round_to_005)

        # Build best-available labels from live/current snapshot at selected_date
        label_lookup = {}
        label_lookup_opt = {}
        snap_df = df[df['date'] == selected_date].copy()
        for c in pct_commodities:
            csub = snap_df[snap_df['commodity'] == c].copy()
            if len(csub) == 0:
                continue
            # prefer explicit identifiers
            csub = assign_contract_month_from_snapshot(csub, max_months=pct_max_months)
            for _, r in csub.iterrows():
                cm = r.get('contract_month', None)
                if pd.isna(cm):
                    continue
                label = None

                # 1) contract_code if it looks like a true code (not M#)
                cc = str(r.get('contract_code')) if 'contract_code' in r and pd.notna(r.get('contract_code')) else None
                if cc and not cc.upper().startswith('M'):
                    label = cc

                # 2) derive from expiry using mapping / fallback month codes
                if label is None and 'expiry' in r and pd.notna(r['expiry']):
                    try:
                        label = build_contract_labels_from_expiry(pd.Series([r['expiry']]), c).iloc[0]
                    except Exception:
                        label = None

                # 3) options_month text if present
                if label is None:
                    om = r.get('options_month') if 'options_month' in r else None
                    if pd.notna(om):
                        label = str(om)

                # 4) month_label from data
                if label is None:
                    ml = r.get('month_label') if 'month_label' in r else None
                    if pd.notna(ml):
                        label = str(ml)

                # 5) fallback to M#
                if label is None or str(label).lower() == 'nan':
                    label = f"M{int(cm)}"

                label_lookup[(c, int(cm))] = label
                # Also index by options month code when available for robust matching.
                om = None
                if 'options_month' in r and pd.notna(r.get('options_month', None)):
                    om = str(r.get('options_month')).upper()
                elif 'expiry' in r and pd.notna(r.get('expiry', None)):
                    try:
                        em = int(pd.to_datetime(r.get('expiry')).month)
                        om = get_options_lookup_for_commodity(c).get(em, MONTH_CODE_FALLBACK.get(em, None))
                    except Exception:
                        om = None
                if om:
                    label_lookup_opt[(c, om)] = label

    if len(pct_grid) > 0:
        # Build the battery bar HTML for each commodity
        for commodity in pct_commodities:
            cdata = pct_grid[pct_grid['commodity'] == commodity].sort_values('contract_month')
            if len(cdata) == 0:
                continue

            # Commodity header
            bars_html = textwrap.dedent(f"""
            <div style="margin-bottom:16px;">
                <div style="font-family:Consolas,monospace;font-size:13px;color:#ff9500;font-weight:bold;margin-bottom:6px;">
                    {commodity}
                </div>
            """)

            for _, row in cdata.iterrows():
                cm = int(row['contract_month'])
                pct = row['percentile']
                current = row['current']
                median = row['median']
                # Build contract label with preference map first
                label = None
                if 'label_lookup_opt' in locals() and 'options_month' in row and pd.notna(row.get('options_month', None)):
                    label = label_lookup_opt.get((commodity, str(row.get('options_month')).upper()), None)
                if (label is None or str(label).lower() == 'nan') and 'label_lookup' in locals():
                    label = label_lookup.get((commodity, cm), None)
                if label is None or str(label).lower() == 'nan':
                    for key in ['contract_code', 'options_month', 'month_label']:
                        if key in row and pd.notna(row.get(key, None)):
                            label = str(row.get(key))
                            break
                if (label is None or str(label).lower() == 'nan') and 'expiry' in row:
                    try:
                        label = pd.to_datetime(row['expiry']).strftime('%b%y')
                    except Exception:
                        label = None
                if label is None or str(label).lower() == 'nan':
                    label = f"M{cm}"

                # Color based on percentile level
                if pct >= 80:
                    fill_color = '#f85149'     # red - very high
                    text_color = '#f85149'
                elif pct >= 60:
                    fill_color = '#d29922'     # amber - high
                    text_color = '#d29922'
                elif pct <= 20:
                    fill_color = '#3fb950'     # green - very low
                    text_color = '#3fb950'
                elif pct <= 40:
                    fill_color = '#56d364'     # light green - low
                    text_color = '#56d364'
                else:
                    fill_color = '#8b949e'     # gray - neutral
                    text_color = '#8b949e'

                # Clamp width to 0-100
                bar_width = max(0, min(100, pct))

                bars_html += textwrap.dedent(f"""
                <div style="display:flex;align-items:center;margin-bottom:3px;font-family:Consolas,monospace;">
                    <div style="width:48px;font-size:11px;color:#8b949e;text-align:right;margin-right:8px;">{label}</div>
                    <div style="flex:1;max-width:400px;position:relative;">
                        <div style="background-color:#1e2530;border:1px solid #2d333b;height:18px;border-radius:2px;overflow:hidden;">
                            <div style="background-color:{fill_color};width:{bar_width}%;height:100%;border-radius:1px;transition:width 0.3s;"></div>
                        </div>
                    </div>
                    <div style="width:48px;font-size:12px;color:{text_color};text-align:right;margin-left:8px;font-weight:bold;">{pct:.0f}%</div>
                    <div style="width:80px;font-size:11px;color:#9aa4b2;text-align:right;margin-left:8px;">{current:.1f}</div>
                    <div style="width:70px;font-size:10px;color:#7f8a98;text-align:right;margin-left:4px;">med {median:.1f}</div>
                </div>
                """)

            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

        # Legend
        st.markdown(textwrap.dedent("""
        <div style="font-family:Consolas,monospace;font-size:11px;color:#586069;margin-top:12px;border-top:1px solid #1e2530;padding-top:8px;">
            <span style="color:#f85149;">&#9632;</span> &gt;80th EXPENSIVE &nbsp;&nbsp;
            <span style="color:#d29922;">&#9632;</span> 60-80th HIGH &nbsp;&nbsp;
            <span style="color:#8b949e;">&#9632;</span> 40-60th NEUTRAL &nbsp;&nbsp;
            <span style="color:#56d364;">&#9632;</span> 20-40th LOW &nbsp;&nbsp;
            <span style="color:#3fb950;">&#9632;</span> &lt;20th CHEAP &nbsp;&nbsp;
            | Values = current vol | med = median
        </div>
        """), unsafe_allow_html=True)

    else:
        st.warning(f"No percentile data available for {selected_date.date()}")

    # Debug panel: show current vols as read from CSV for the selected commodity/date
    with st.expander("Data Debug – current vols (from CSV)"):
        current_rows = df[
            (df['date'] == selected_date) &
            (df['commodity'] == commodity)
        ].sort_values('contract_month')[
            ['contract_month', 'dirty_vol', 'clean_vol', 'fwd_vol', 'expiry']
        ]
        if len(current_rows) == 0:
            st.write("No rows found for this date/commodity.")
        else:
            st.caption(f"Source date: {selected_date.date()} | Commodity: {commodity}")
            st.dataframe(
                current_rows.rename(columns={
                    'contract_month': 'CM',
                    'dirty_vol': 'Dirty Vol',
                    'clean_vol': 'Clean Vol',
                    'fwd_vol': 'Fwd Vol',
                    'expiry': 'Expiry'
                }).style.format({
                    'Dirty Vol': '{:.3f}',
                    'Clean Vol': '{:.3f}',
                    'Fwd Vol': '{:.3f}'
                }),
                use_container_width=True,
                height=240
            )

# ============================================================================
# TAB 1: MAIN DASHBOARD
# ============================================================================
if active_tab == "DASHBOARD":
    # Get front month stats
    front_vol_stats = va.calculate_percentile_rank(
        df, selected_date, 1, 'clean_vol',
        lookback_days=lookback, commodity=commodity, hist_df=master_hist
    )

    fwd_vol_stats = va.calculate_percentile_rank(
        df, selected_date, 1, 'fwd_vol',
        lookback_days=lookback, commodity=commodity, hist_df=master_hist
    )

    if front_vol_stats and fwd_vol_stats:
        if False:  # legacy live_vol_override removed — live data now merged into df
            front_vol_stats['current'] = live_vol_override
            fwd_vol_stats['current'] = live_vol_override
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Clean Vol (M1)",
                f"{front_vol_stats['current']:.2f}%",
                delta=f"{front_vol_stats['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {front_vol_stats['median']:.2f}% | Δ: {front_vol_stats['distance_from_median']:+.2f}%")
        
        with col2:
            st.metric(
                "Forward Vol (M1)",
                f"{fwd_vol_stats['current']:.2f}%",
                delta=f"{fwd_vol_stats['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {fwd_vol_stats['median']:.2f}%")
        
        # Calendar spread
        m1_m2_spread = va.calculate_calendar_spread(
            df, selected_date, 1, 2, 'fwd_vol', commodity=commodity
        )
        
        with col3:
            if m1_m2_spread:
                st.metric(
                    "M1-M2 Fwd Vol Spread",
                    f"{m1_m2_spread['current_spread']:.2f}%",
                    delta=f"{m1_m2_spread['percentile']:.0f}th %ile"
                )
                st.caption(f"Median: {m1_m2_spread['median_spread']:.2f}%")
        
        # Get current data for signal
        current_data = df[
            (df['date'] == selected_date) & 
            (df['commodity'] == commodity) & 
            (df['contract_month'] == 1)
        ]
        
        with col4:
            if len(current_data) > 0:
                row = current_data.iloc[0]
                if pd.notna(row['trading_dte']):
                    st.metric("Trading DTE", f"{int(row['trading_dte'])}")
                    if pd.notna(row['clean_dte']):
                        st.caption(f"Clean DTE: {row['clean_dte']:.1f}")
        
        st.markdown("---")
        
        # Your Forward Vol vs Predicted Realized Vol section
        st.subheader("🎯 Forward Vol vs Predicted Realized Vol")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Your Model Input:**")
            predicted_rv = st.number_input(
                "Predicted Realized Vol (%)",
                min_value=0.0,
                max_value=100.0,
                value=fwd_vol_stats['current'] if fwd_vol_stats else 15.0,
                step=0.1,
                help="Enter your model's prediction for realized vol"
            )
            
            if fwd_vol_stats:
                fwd_vol = fwd_vol_stats['current']
                diff = fwd_vol - predicted_rv
                ratio = fwd_vol / predicted_rv if predicted_rv > 0 else 0
                
                st.markdown("**Analysis:**")
                st.write(f"Forward Vol: **{fwd_vol:.2f}%**")
                st.write(f"Your Prediction: **{predicted_rv:.2f}%**")
                st.write(f"Difference: **{diff:+.2f}%**")
                st.write(f"Ratio: **{ratio:.2f}x**")
                
                if diff > 2:
                    st.success("**SIGNAL: LONG VOL** ✅")
                    st.caption("Market IV is rich vs your prediction")
                elif diff < -2:
                    st.error("**SIGNAL: SHORT VOL** ❌")
                    st.caption("Market IV is cheap vs your prediction")
                else:
                    st.info("**SIGNAL: NEUTRAL** ⚖️")
                    st.caption("Market IV is fairly valued")
        
        with col2:
            # Historical comparison
            st.markdown("**Historical Context:**")
            
            # Get historical forward vols
            hist_data = df[
                (df['commodity'] == commodity) & 
                (df['contract_month'] == 1) &
                (df['date'] < selected_date)
            ].tail(lookback)[['date', 'fwd_vol']].dropna()
            
            if len(hist_data) > 0:
                chart_data = hist_data.set_index('date')
                # Add current point
                current_point = pd.DataFrame({
                    'fwd_vol': [fwd_vol_stats['current']]
                }, index=[selected_date])
                chart_data = pd.concat([chart_data, current_point])
                
                st.line_chart(chart_data, height=200)
                st.caption(f"Forward Vol over last {lookback} days")
    
    else:
        st.warning(f"No data available for {commodity} on {selected_date.date()}")

# ============================================================================
# TAB: VOL CHANGE
# ============================================================================
if active_tab in ["VOL CHANGE", "VOL CHANGES"]:
    st.markdown('<div class="bloomberg-header"><span>VOL CHANGE</span></div>', unsafe_allow_html=True)
    st.caption("IV and Forward Vol change grids by contract and commodity")

    col_lb, _ = st.columns([0.324, 1.676])  # shrink width by an additional ~10%
    with col_lb:
        long_window = st.selectbox("Lookback (trading days)", list(range(1, 21)), index=4, key="vol_change_lookback")
    max_months = 12

    vol_change_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    def _contract_sort_key(code):
        month_rank = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
        s = str(code) if pd.notna(code) else ""
        if len(s) < 2:
            return (9999, 99)
        try:
            yy = int(s[1:])
        except Exception:
            yy = 9999
        return (yy, month_rank.get(s[0], 99))

    def _build_grid(window_days: int, metric_kind: str):
        per_comm = {}
        for c in vol_change_commodities:
            t, _meta = build_vol_change_table(
                df=df,
                commodity=c,
                selected_date=selected_date,
                long_window=window_days,
                max_months=max_months,
                live_df=live_df,
                master_df=master_base_df
            )
            if len(t) > 0:
                per_comm[c] = t[['contract_code', 'iv_chg_1d', 'fwd_chg_1d', 'iv_chg_long', 'fwd_chg_long']].copy()

        if len(per_comm) == 0:
            return pd.DataFrame()

        contracts = set()
        for t in per_comm.values():
            contracts.update([str(x) for x in t['contract_code'].dropna().tolist()])
        contracts = sorted(list(contracts), key=_contract_sort_key)

        grid = pd.DataFrame({'Contract': contracts})
        for c in vol_change_commodities:
            grid[c] = np.nan
            if c in per_comm:
                t = per_comm[c].drop_duplicates(subset=['contract_code'], keep='first').set_index('contract_code')
                if metric_kind == "IV":
                    col = 'iv_chg_1d' if window_days == 1 else 'iv_chg_long'
                else:
                    col = 'fwd_chg_1d' if window_days == 1 else 'fwd_chg_long'
                grid[c] = grid['Contract'].map(t[col]) if col in t.columns else np.nan
        return grid

    def _delta_cell_style(val):
        if pd.isna(val):
            return ''
        if isinstance(val, (int, float, np.floating, np.integer)):
            if float(val) > 0:
                return 'background-color: #1f4d2e; color: #e6edf3;'
            if float(val) < 0:
                return 'background-color: #5a1f2a; color: #e6edf3;'
        return ''

    st.markdown("**1-Day IV Change**")
    grid_1d_iv = _build_grid(window_days=1, metric_kind="IV")
    if len(grid_1d_iv) == 0:
        st.warning("No 1-day IV change data available.")
    else:
        delta_cols = [c for c in grid_1d_iv.columns if c != 'Contract']
        st.dataframe(
            grid_1d_iv.style.format({col: '{:+.2f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
            use_container_width=True,
            height=min(520, 80 + 32 * len(grid_1d_iv)),
            hide_index=True
        )

    st.markdown("**1-Day FWD Change**")
    grid_1d_fwd = _build_grid(window_days=1, metric_kind="FWD")
    if len(grid_1d_fwd) == 0:
        st.warning("No 1-day FWD change data available.")
    else:
        delta_cols = [c for c in grid_1d_fwd.columns if c != 'Contract']
        st.dataframe(
            grid_1d_fwd.style.format({col: '{:+.2f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
            use_container_width=True,
            height=min(520, 80 + 32 * len(grid_1d_fwd)),
            hide_index=True
        )

    st.markdown("---")

    st.markdown(f"**{long_window}-Day IV Change**")
    grid_nd_iv = _build_grid(window_days=long_window, metric_kind="IV")
    if len(grid_nd_iv) == 0:
        st.warning(f"No {long_window}-day IV change data available.")
    else:
        delta_cols = [c for c in grid_nd_iv.columns if c != 'Contract']
        st.dataframe(
            grid_nd_iv.style.format({col: '{:+.2f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
            use_container_width=True,
            height=min(520, 80 + 32 * len(grid_nd_iv)),
            hide_index=True
        )

    st.markdown(f"**{long_window}-Day FWD Change**")
    grid_nd_fwd = _build_grid(window_days=long_window, metric_kind="FWD")
    if len(grid_nd_fwd) == 0:
        st.warning(f"No {long_window}-day FWD change data available.")
    else:
        delta_cols = [c for c in grid_nd_fwd.columns if c != 'Contract']
        st.dataframe(
            grid_nd_fwd.style.format({col: '{:+.2f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
            use_container_width=True,
            height=min(520, 80 + 32 * len(grid_nd_fwd)),
            hide_index=True
        )

# ============================================================================
# TAB: TRADING CALENDAR
# ============================================================================
if active_tab == "TRADING CALENDAR":
    st.subheader("Trading Calendar")
    st.caption("US market holidays (2026) and available trading dates from loaded vol dataset")

    holiday_rows = [
        ("New Year's Day", "Thursday, January 1, 2026"),
        ("Martin Luther King Jr. Day", "Monday, January 19, 2026"),
        ("Washington's Birthday (Presidents' Day)", "Monday, February 16, 2026"),
        ("Memorial Day", "Monday, May 25, 2026"),
        ("Juneteenth National Independence Day", "Friday, June 19, 2026"),
        ("Independence Day", "Saturday, July 4, 2026 (banks often observe Friday, July 3)"),
        ("Labor Day", "Monday, September 7, 2026"),
        ("Veterans Day", "Wednesday, November 11, 2026"),
        ("Thanksgiving Day", "Thursday, November 26, 2026"),
        ("Christmas Day", "Friday, December 25, 2026"),
    ]
    holiday_df = pd.DataFrame(holiday_rows, columns=["Holiday", "Date"])[["Date", "Holiday"]]
    st.markdown("**US Holidays (2026)**")
    st.dataframe(holiday_df, use_container_width=True, hide_index=True, height=260)
    st.markdown("")

    usda_rows = [
        ("January 12, 2026", "WASDE"),
        ("February 10, 2026", "WASDE"),
        ("March 10, 2026", "WASDE"),
        ("April 9, 2026", "WASDE"),
        ("May 12, 2026", "WASDE"),
        ("June 11, 2026", "WASDE"),
        ("July 10, 2026", "WASDE"),
        ("August 12, 2026", "WASDE"),
        ("September 11, 2026", "WASDE"),
        ("October 9, 2026", "WASDE"),
        ("November 10, 2026", "WASDE"),
        ("December 10, 2026", "WASDE"),
        ("January 12, 2026", "Quarterly Grain Stocks"),
        ("March 31, 2026", "Quarterly Grain Stocks"),
        ("June 30, 2026", "Quarterly Grain Stocks"),
        ("September 30, 2026", "Quarterly Grain Stocks"),
    ]
    usda_df = pd.DataFrame(usda_rows, columns=["Date", "Report"])
    usda_df["__sort_date"] = pd.to_datetime(usda_df["Date"], errors="coerce")
    usda_df = usda_df.sort_values(["__sort_date", "Report"]).drop(columns=["__sort_date"]).reset_index(drop=True)
    st.markdown("**USDA Reports (WASDE 2026)**")
    st.dataframe(usda_df, use_container_width=True, hide_index=True, height=380)

# ============================================================================
# TAB: VAR CAL (Philatron-style variance calendar)
# ============================================================================
if active_tab == "VAR CAL":
    # ---- Static data: CME holidays and USDA/WASDE events ----
    # Holiday names for display — keyed from the shared CME_HOLIDAYS set at top of file
    _HOLIDAY_NAMES = {
        "2026-01-01": "NYE",   "2026-01-19": "MLK",   "2026-02-16": "Pres",
        "2026-04-03": "GF",    "2026-05-25": "Memorial", "2026-06-19": "Juneteenth",
        "2026-07-03": "Indy",  "2026-09-07": "Labor", "2026-11-26": "Thanks",
        "2026-12-25": "XMAS",
        "2027-01-01": "NYE",   "2027-01-18": "MLK",   "2027-02-15": "Pres",
        "2027-03-26": "GF",    "2027-05-31": "Memorial", "2027-06-18": "Juneteenth",
        "2027-07-05": "Indy",  "2027-09-06": "Labor", "2027-11-25": "Thanks",
        "2027-12-24": "XMAS",
        "2028-01-01": "NYE",   "2028-01-17": "MLK",   "2028-02-21": "Pres",
        "2028-04-14": "GF",    "2028-05-29": "Memorial", "2028-06-19": "Juneteenth",
        "2028-07-04": "Indy",  "2028-09-04": "Labor", "2028-11-23": "Thanks",
        "2028-12-25": "XMAS",
    }
    _VARCAL_HOLIDAYS = {d: _HOLIDAY_NAMES.get(d, "Holiday") for d in CME_HOLIDAYS}

    # Use shared module-level events dict (copy so expiry injection is local to this tab)
    _VARCAL_EVENTS = dict(CME_EVENTS)

    # ---- Inject expiry dates from all products into the events dict ----
    _expiry_dates = set()
    if df is not None and 'expiry' in df.columns:
        for _exp in df['expiry'].dropna().unique():
            try:
                _expiry_dates.add(pd.Timestamp(_exp).strftime("%Y-%m-%d"))
            except Exception:
                pass
    for _edate in sorted(_expiry_dates):
        if _edate not in _VARCAL_EVENTS:
            _VARCAL_EVENTS[_edate] = f"Expiry: {_edate}"

    # ---- Event types and default weights (reference shared module-level dicts) ----
    _COMMODITIES = ["SOY", "MEAL", "OIL", "CORN", "WHEAT", "KW"]
    _EVENT_TYPES = list(CME_DEFAULT_VAR.keys())
    _DEFAULT_VAR = CME_DEFAULT_VAR

    # ---- Session state: var values (editable) ----
    if "varcal_var_values" not in st.session_state:
        st.session_state["varcal_var_values"] = {
            evt: dict(vals) for evt, vals in CME_DEFAULT_VAR.items()
        }

    # ---- UI: Editable var values table ----
    st.subheader("Event Variance Weights")
    st.caption("Edit the variance contribution per event type per commodity. Normal day = 1.")

    # Build the editable dataframe
    _var_rows = []
    for evt in _EVENT_TYPES:
        row = {"Event": evt}
        for c in _COMMODITIES:
            row[c] = st.session_state["varcal_var_values"].get(evt, {}).get(c, 1.0)
        _var_rows.append(row)
    _var_df = pd.DataFrame(_var_rows)

    _col_config = {"Event": st.column_config.TextColumn("Event", disabled=True)}
    for c in _COMMODITIES:
        _col_config[c] = st.column_config.NumberColumn(c, min_value=0.0, max_value=20.0, step=0.1, format="%.2f")

    _edited_var = st.data_editor(
        _var_df,
        column_config=_col_config,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="varcal_var_editor",
    )

    # Persist edits back to session state
    for _, row in _edited_var.iterrows():
        evt = row["Event"]
        if evt in st.session_state["varcal_var_values"]:
            for c in _COMMODITIES:
                st.session_state["varcal_var_values"][evt][c] = float(row[c])

    # Reset button
    if st.button("Reset to defaults", key="varcal_reset"):
        st.session_state["varcal_var_values"] = {
            evt: dict(vals) for evt, vals in _DEFAULT_VAR.items()
        }
        st.rerun()

    st.markdown("---")

    # ---- Build the calendar ----
    _today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    _end = _today + timedelta(days=365 * 2 + 31)  # ~24 months

    def _varcal_classify(dt):
        """Return (weekend_flag, holiday_flag, event_name) for a given date."""
        dstr = dt.strftime("%Y-%m-%d")
        dow = dt.weekday()  # 0=Mon, 6=Sun
        is_weekend = dow >= 5  # Sat=5, Sun=6
        holiday = _VARCAL_HOLIDAYS.get(dstr, "")
        event = _VARCAL_EVENTS.get(dstr, "")
        return is_weekend, holiday, event

    def _varcal_var_for_day(is_weekend, holiday, event, commodity, var_vals):
        """Return the variance value for a day."""
        if holiday:
            etype = "Holiday (H)"
        elif is_weekend:
            etype = "Weekend (W)"
        elif event:
            # Map event name to type key
            etype = event if event in var_vals else "Day"
        else:
            etype = "Day"
        return var_vals.get(etype, {}).get(commodity, 1.0)

    # Build rows
    _var_vals = st.session_state["varcal_var_values"]
    _cal_rows = []
    _cur = _today
    while _cur <= _end:
        is_weekend, holiday, event = _varcal_classify(_cur)
        trading_day = 0 if (is_weekend or holiday) else 1
        row = {
            "Date": _cur.strftime("%Y-%m-%d"),
            "Day": _cur.strftime("%a"),
            "Wknd": "W" if is_weekend else "",
            "Holiday": holiday,
            "Event": event,
            "Trd": trading_day,
        }
        for c in _COMMODITIES:
            row[c] = _varcal_var_for_day(is_weekend, holiday, event, c, _var_vals)
        _cal_rows.append(row)
        _cur += timedelta(days=1)

    _cal_df = pd.DataFrame(_cal_rows)

    # ---- Cumulative summary by month ----
    st.subheader("Monthly Variance Totals")
    _cal_df["_month"] = pd.to_datetime(_cal_df["Date"]).dt.to_period("M")
    _monthly = _cal_df.groupby("_month")[_COMMODITIES].sum().reset_index()
    _monthly["Month"] = _monthly["_month"].astype(str)
    _monthly = _monthly[["Month"] + _COMMODITIES]
    _monthly_style = _monthly.style.format({c: "{:.2f}" for c in _COMMODITIES})
    st.dataframe(_monthly_style, use_container_width=True, hide_index=True, height=min(700, 36 + 28 * len(_monthly)))

    st.markdown("---")

    # ---- Full calendar grid ----
    st.subheader("Daily Calendar")
    st.caption(f"Today: {_today.strftime('%Y-%m-%d')} → {_end.strftime('%Y-%m-%d')}  |  Weekend=W  |  Holiday=H  |  Normal day var=1")

    # Style: highlight weekends, holidays, events
    def _varcal_row_style(row):
        styles = [""] * len(row)
        if row["Wknd"] == "W":
            return ["background-color: #1c2333; color: #8b9db5;"] * len(row)
        if row["Holiday"]:
            return ["background-color: #2a1f1f; color: #c0827a;"] * len(row)
        if row["Event"]:
            return ["background-color: #1f2a1f; color: #82c082;"] * len(row)
        return styles

    _cal_display = _cal_df.drop(columns=["_month"])
    _cal_styled = _cal_display.style.apply(_varcal_row_style, axis=1).format(
        {c: "{:.2f}" for c in _COMMODITIES}
    )
    st.dataframe(
        _cal_styled,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

# ============================================================================
# TAB 2: TERM STRUCTURE
# ============================================================================
if active_tab == "TERM STRUCT":
    st.subheader("IV Term Structure - Week Over Week Progression")
    
    term_structure = va.get_iv_term_structure(df, selected_date, commodity)
    
    if len(term_structure) > 0:
        # Filter to front 8 months
        term_structure_display = term_structure.head(8).copy()
        
        # Format for display
        term_structure_display['expiry'] = term_structure_display['expiry'].dt.strftime('%Y-%m-%d')
        term_structure_display = term_structure_display[[
            'contract_month', 'expiry', 'dirty_vol', 'clean_vol', 'fwd_vol', 'trading_dte'
        ]]
        term_structure_display.columns = [
            'Month', 'Expiry', 'Dirty Vol', 'Clean Vol', 'Fwd Vol', 'DTE'
        ]
        
        # Display table
        st.dataframe(
            term_structure_display.style.format({
                'Dirty Vol': '{:.2f}%',
                'Clean Vol': '{:.2f}%',
                'Fwd Vol': '{:.2f}%',
                'DTE': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clean Vol Curve**")
            chart_data = term_structure_display.set_index('Month')['Clean Vol']
            st.line_chart(chart_data)
        
        with col2:
            st.markdown("**Forward Vol Curve**")
            chart_data = term_structure_display.set_index('Month')['Fwd Vol']
            st.line_chart(chart_data)
    else:
        st.warning("No term structure data available")

# ============================================================================
# TAB 3: SPREADS
# ============================================================================
if active_tab == "SPREADS":
    st.subheader("Calendar & Cross-Commodity Spreads")
    
    # Calendar spreads
    st.markdown("#### Calendar Spreads (Forward Vol)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # M1-M2
        spread_12 = va.calculate_calendar_spread(
            df, selected_date, 1, 2, 'fwd_vol', commodity=commodity, hist_df=master_hist
        )
        if spread_12:
            st.metric(
                "M1 - M2",
                f"{spread_12['current_spread']:.2f}%",
                delta=f"{spread_12['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_12['median_spread']:.2f}% | Δ: {spread_12['distance_from_median']:+.2f}%")
        
        # M2-M3
        spread_23 = va.calculate_calendar_spread(
            df, selected_date, 2, 3, 'fwd_vol', commodity=commodity, hist_df=master_hist
        )
        if spread_23:
            st.metric(
                "M2 - M3",
                f"{spread_23['current_spread']:.2f}%",
                delta=f"{spread_23['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_23['median_spread']:.2f}%")
    
    with col2:
        # M3-M4
        spread_34 = va.calculate_calendar_spread(
            df, selected_date, 3, 4, 'fwd_vol', commodity=commodity, hist_df=master_hist
        )
        if spread_34:
            st.metric(
                "M3 - M4",
                f"{spread_34['current_spread']:.2f}%",
                delta=f"{spread_34['percentile']:.0f}th %ile"
            )
            st.caption(f"Median: {spread_34['median_spread']:.2f}%")
    
    st.markdown("---")
    
    # Cross-commodity spreads
    st.markdown("#### Cross-Commodity IV Spreads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SOY vs MEAL
        soy_meal = va.calculate_cross_commodity_spread(
            df, selected_date, 'SOY', 'MEAL', 1, 'clean_vol', hist_df=master_hist
        )
        if soy_meal:
            st.markdown("**SOY vs MEAL (M1 Clean Vol)**")
            st.metric(
                "Spread",
                f"{soy_meal['current_spread']:.2f}%",
                delta=f"{soy_meal['percentile']:.0f}th %ile"
            )
            st.caption(f"SOY: {soy_meal['commodity1_value']:.2f}% | MEAL: {soy_meal['commodity2_value']:.2f}%")
            st.caption(f"Median spread: {soy_meal['median_spread']:.2f}%")
    
    with col2:
        # CORN vs WHEAT
        corn_wheat = va.calculate_cross_commodity_spread(
            df, selected_date, 'CORN', 'WHEAT', 1, 'clean_vol', hist_df=master_hist
        )
        if corn_wheat:
            st.markdown("**CORN vs WHEAT (M1 Clean Vol)**")
            st.metric(
                "Spread",
                f"{corn_wheat['current_spread']:.2f}%",
                delta=f"{corn_wheat['percentile']:.0f}th %ile"
            )
            st.caption(f"CORN: {corn_wheat['commodity1_value']:.2f}% | WHEAT: {corn_wheat['commodity2_value']:.2f}%")
            st.caption(f"Median spread: {corn_wheat['median_spread']:.2f}%")

# ============================================================================
# TAB 4: SKEW ANALYSIS
# ============================================================================
if active_tab == "SKEW":
    st.subheader("Skew - Median (by Front-Month Regime)")
    st.caption("Columns: P2=-1.5, P1=-0.5, C1=+0.5, C2=+1.5, C3=+3.0 | Cell = current skew - historical median")

    month_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    try:
        default_front_opt = current_front_options_month(commodity, selected_date)
    except Exception:
        default_front_opt = 'H'
    default_idx = month_codes.index(default_front_opt) if default_front_opt in month_codes else 2

    col_front_opt, _ = st.columns([0.15, 0.85])  # shrink width by ~50% from prior
    with col_front_opt:
        front_opt_override = st.selectbox(
            "Front options month",
            month_codes,
            index=default_idx,
            help="Select which front-month options contract to use for all commodities in this view."
        )

    skew_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    lookback_years = lookback_years_from_days(lookback)
    skew_rows = []
    median_rows = []
    front_opt_by_comm = {}
    skew_source_candidates = {
        'P2': ['skew_m1.5', 'skew_neg15', 'P2'],
        'P1': ['skew_m0.5', 'skew_neg05', 'P1'],
        'C1': ['skew_p0.5', 'skew_pos05', 'C1'],
        'C2': ['skew_p1.5', 'skew_pos15', 'C2'],
        'C3': ['skew_p3.0', 'skew_pos3', 'C3'],
    }
    skew_csv_col = {
        'P2': 'skew_m1.5',
        'P1': 'skew_m0.5',
        'C1': 'skew_p0.5',
        'C2': 'skew_p1.5',
        'C3': 'skew_p3.0',
    }
    for c in skew_commodities:
        # Use user-selected front options month across commodities
        front_opt = front_opt_override or current_front_options_month(c, selected_date)
        front_opt_by_comm[c] = str(front_opt).upper()
        snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
        if isinstance(snap, pd.DataFrame) and len(snap) > 0:
            tmp = assign_contract_month_from_snapshot(snap, max_months=12)
            tmp['commodity'] = c
            # Contract labels from shared helper to keep all tabs consistent.
            tmp['contract_month_label'] = build_contract_labels_from_expiry(
                tmp.get('expiry', pd.Series(index=tmp.index)),
                c
            )
            tmp['contract_month_label'] = tmp['contract_month_label'].replace({'nan': np.nan, 'None': np.nan, '?00': np.nan})
            tmp['contract_month_label'] = tmp['contract_month_label'].fillna(
                tmp.get('contract_label', pd.Series(index=tmp.index)).astype(str).replace({'nan': np.nan, 'None': np.nan})
            )
            tmp['contract_month_label'] = tmp['contract_month_label'].fillna(tmp['contract_month'].map(lambda x: f"M{int(x)}"))

            # Capture historical medians for this commodity
            med_entry = tmp[['contract_month']].copy()
            med_entry['commodity'] = c
            label_map = (
                tmp[['contract_month', 'contract_month_label']]
                .drop_duplicates(subset=['contract_month'], keep='first')
                .set_index('contract_month')['contract_month_label']
            )
            med_entry['contract_month_label'] = med_entry['contract_month'].map(label_map)
            med_entry['contract_month_label'] = med_entry['contract_month_label'].fillna(
                med_entry['contract_month'].map(lambda x: f"M{int(x)}")
            )
            for col in ['P2', 'P1', 'C1', 'C2', 'C3']:
                med_col = f"{col}_hist_median"
                med_entry[col] = pd.to_numeric(tmp.get(med_col, np.nan), errors='coerce')
            median_rows.append(med_entry)

            # Current skew should come from live_vols when available, then subtract precomputed medians.
            live_cur = pd.DataFrame()
            if live_df is not None and len(live_df) > 0:
                lsub = live_df[live_df['commodity'] == c].copy()
                if len(lsub) > 0:
                    lsub['date'] = pd.to_datetime(lsub['date'], errors='coerce')
                    lsub = lsub[lsub['date'].notna() & (lsub['date'] <= selected_date)]
                    if len(lsub) > 0:
                        ldate = lsub['date'].max()
                        live_cur = lsub[lsub['date'] == ldate].copy()

            # Fallback to merged df current date if live is unavailable for this commodity.
            if len(live_cur) == 0:
                live_cur = df[(df['commodity'] == c) & (df['date'] == selected_date)].copy()

            if len(live_cur) > 0:
                live_cur = assign_contract_month_from_snapshot(live_cur, max_months=12)

            for col in ['P2', 'P1', 'C1', 'C2', 'C3']:
                med_col = f"{col}_hist_median"
                if med_col not in tmp.columns:
                    tmp[col] = np.nan
                    continue

                cur_vals = pd.Series(np.nan, index=tmp.index)
                if len(live_cur) > 0:
                    src_col = None
                    for cand in skew_source_candidates[col]:
                        if cand in live_cur.columns:
                            src_col = cand
                            break
                    if src_col is not None:
                        cur_map = (
                            live_cur[['contract_month', src_col]]
                            .drop_duplicates(subset=['contract_month'], keep='first')
                            .set_index('contract_month')[src_col]
                        )
                        cur_vals = tmp['contract_month'].map(cur_map)

                tmp[f"{col}_current"] = pd.to_numeric(cur_vals, errors='coerce')
                tmp[col] = pd.to_numeric(cur_vals, errors='coerce') - pd.to_numeric(tmp[med_col], errors='coerce')
            keep_cols = ['commodity', 'contract_month', 'contract_month_label', 'P2', 'P1', 'C1', 'C2', 'C3', 'P2_current', 'P1_current', 'C1_current', 'C2_current', 'C3_current']
            if 'options_month' in tmp.columns:
                keep_cols.append('options_month')
            skew_rows.append(tmp[keep_cols])

    if len(skew_rows) > 0:
        skew_grid = pd.concat(skew_rows, ignore_index=True)
    else:
        skew_grid = va.get_skew_diff_grid_cached(
            df, selected_date, commodities=skew_commodities, max_months=8, hist_df=master_hist
        )

    if len(skew_grid) == 0:
        st.warning("No skew data available")
    else:
        def skew_cell_style(val):
            if pd.isna(val):
                return ""
            # Clamp to [-3, 3] and map to red/green intensity
            v = max(-3.0, min(3.0, float(val)))
            if v >= 0:
                # red shades for positive
                t = int(35 + (v / 3.0) * 120)
                return f"background-color: rgb({80+t}, 35, 45); color: #e6edf3;"
            # green shades for negative
            t = int(35 + (abs(v) / 3.0) * 120)
            return f"background-color: rgb(30, {70+t}, 60); color: #e6edf3;"

        # Helpers for percentile calc
        def round_005(v):
            try:
                return np.round(float(v) * 20.0) / 20.0
            except Exception:
                return np.nan

        skew_pct_dist = load_skew_percentile_dist_cache()

        def skew_pct(val, comm, cm, col, opt_month=None, front_opt=None):
            # Preferred: percentile distribution conditioned on (commodity, front options month, options month)
            # from cache/skew_percentile_dist.csv, matching the same regime as current-vs-median.
            if (
                isinstance(skew_pct_dist, pd.DataFrame) and len(skew_pct_dist) > 0 and
                opt_month is not None and front_opt is not None
            ):
                csv_base = skew_csv_col.get(col, None)
                if csv_base is not None:
                    dsub = skew_pct_dist[
                        (skew_pct_dist['commodity'] == str(comm).upper()) &
                        (skew_pct_dist['FRONT_OPTIONS'] == str(front_opt).upper()) &
                        (skew_pct_dist['OPTIONS'] == str(opt_month).upper())
                    ]
                    if len(dsub) > 0 and pd.notna(val):
                        row = dsub.iloc[0]
                        pts = []
                        for p in range(5, 101, 5):
                            cname = f"{csv_base}_p{p:02d}"
                            if cname in dsub.columns and pd.notna(row.get(cname, np.nan)):
                                pts.append((float(row[cname]), float(p)))
                        if len(pts) > 0:
                            pts = sorted(pts, key=lambda x: x[0])
                            x = np.array([t[0] for t in pts], dtype=float)
                            y = np.array([t[1] for t in pts], dtype=float)
                            if len(x) == 1:
                                return float(y[0])
                            x_unique, idx = np.unique(x, return_index=True)
                            y_unique = y[idx]
                            if len(x_unique) >= 2:
                                return float(np.interp(float(val), x_unique, y_unique, left=0.0, right=100.0))

            # Fallback: historical ranks by contract_month on master_df
            if master_df is None or len(master_df) == 0:
                return np.nan
            cand_cols = skew_source_candidates.get(col, [])
            hist_vals = None
            for cand in cand_cols:
                if cand in master_df.columns:
                    h = master_df[
                        (master_df['commodity'] == comm) &
                        (master_df['contract_month'] == cm)
                    ][cand].dropna()
                    if len(h) > 0:
                        hist_vals = h
                        break
            if hist_vals is None or len(hist_vals) == 0 or pd.isna(val):
                return np.nan
            vals = np.sort(hist_vals.values.astype(float))
            v = round_005(val)
            pos = np.searchsorted(vals, v, side='right')
            return (pos / len(vals)) * 100.0

        # Precompute percentile tables per commodity
        skew_pct_tables = {}
        for c in skew_commodities:
            cdf = skew_grid[skew_grid['commodity'] == c].sort_values('contract_month')
            if len(cdf) == 0:
                continue
            label_col = 'contract_month_label' if 'contract_month_label' in cdf.columns else 'month_label'
            rows = []
            for _, r in cdf.iterrows():
                cm = int(r['contract_month'])
                row_label = r.get(label_col, f"M{cm}")
                data = {'Contract': row_label}
                for col in ['P2', 'P1', 'C1', 'C2', 'C3']:
                    cur_col = f"{col}_current"
                    val = r.get(cur_col, np.nan)
                    opt_m = r.get('options_month', None)
                    pct_val = skew_pct(val, c, cm, col, opt_month=opt_m, front_opt=front_opt_by_comm.get(c))
                    data[f"{col} %ile"] = pct_val
                rows.append(data)
            if len(rows) == 0:
                continue
            disp = pd.DataFrame(rows).set_index('Contract')
            skew_pct_tables[c] = disp

        tab_diff, tab_median = st.tabs(["Current vs Median", "Median"])

        with tab_diff:
            for c in skew_commodities:
                cdf = skew_grid[skew_grid['commodity'] == c].sort_values('contract_month')
                cdf = cdf.drop_duplicates(subset=['contract_month'], keep='last')
                if len(cdf) == 0:
                    continue
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"**{c}**")

                with col_left:
                    label_col = 'contract_month_label' if 'contract_month_label' in cdf.columns else 'month_label'
                    display = cdf[[label_col, 'P2', 'P1', 'C1', 'C2', 'C3']].set_index(label_col)
                    display.index.name = "Contract"
                    pct_table = skew_pct_tables.get(c, None)
                    right_len = len(pct_table) if pct_table is not None else len(display)
                    common_rows = max(len(display), right_len)
                    common_height = min(320, 40 + 35 * common_rows)
                    st.dataframe(
                        display.style
                        .format('{:+.2f}')
                        .applymap(skew_cell_style)
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'th.col_heading', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                            {'selector': 'th.row_heading', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                            {'selector': 'td', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                        ]),
                        use_container_width=True,
                        height=common_height
                    )

                with col_right:
                    pct_table = skew_pct_tables.get(c, None)
                    if pct_table is not None and len(pct_table) > 0:
                        st.markdown("Percentiles")
                        st.dataframe(
                            pct_table.style
                            .format('{:.0f}%', subset=[col for col in pct_table.columns])
                            .set_properties(**{'text-align': 'center'})
                            .set_table_styles([
                                {'selector': 'th.col_heading', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                                {'selector': 'th.row_heading', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                                {'selector': 'td', 'props': [('text-align', 'center'), ('width', '1%'), ('padding', '4px 6px')]},
                            ]),
                            use_container_width=True,
                            height=common_height
                        )
                    else:
                        st.caption("No percentile data.")

        with tab_median:
            if len(median_rows) == 0:
                st.warning("No median skew data available.")
            else:
                median_grid = pd.concat(median_rows, ignore_index=True)
                for c in skew_commodities:
                    cdf = median_grid[median_grid['commodity'] == c].sort_values('contract_month')
                    cdf = cdf.drop_duplicates(subset=['contract_month'], keep='last')
                    if len(cdf) == 0:
                        continue
                    st.markdown(f"**{c} — Median Skews**")
                    display = cdf[['contract_month_label', 'P2', 'P1', 'C1', 'C2', 'C3']].set_index('contract_month_label')
                    display.index.name = "Contract"
                    st.dataframe(
                        display.style.format('{:+.2f}'),
                        use_container_width=True,
                        height=min(320, 40 + 35 * len(display))
                    )


# ============================================================================
# TAB: CORRELATIONS
# ============================================================================
if active_tab == "CORRELATIONS":
    st.subheader("Cross-Commodity Correlations")

    corr_df = load_correlation_data()
    if corr_df is None or len(corr_df) == 0:
        st.warning("Correlation data not available. Expected file: cache/correlation_matrices.csv")
    else:
        windows = sorted([int(w) for w in corr_df['window'].dropna().unique().tolist()])
        if len(windows) == 0:
            st.warning("No correlation windows found in cache/correlation_matrices.csv")
        else:
            default_idx = windows.index(20) if 20 in windows else 0
            # Shrink selector width by ~60%
            col_window, _ = st.columns([0.4, 1.6])
            with col_window:
                corr_window = st.selectbox(
                    "Correlation Window (days)",
                    windows,
                    index=default_idx,
                    key="correlation_window_days"
                )

            wdf = corr_df[corr_df['window'] == corr_window].copy()
            if len(wdf) == 0:
                st.warning(f"No rows for {corr_window}-day window.")
            else:
                matrix = wdf.pivot_table(
                    index='commodity_1',
                    columns='commodity_2',
                    values='correlation',
                    aggfunc='last'
                )
                order = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW']
                matrix = matrix.reindex(index=order, columns=order)
                # Keep diagonal + upper triangle only (hide redundant mirrored values).
                mask = np.triu(np.ones(matrix.shape, dtype=bool))
                matrix = matrix.where(mask)

                def blank_cell_gray(val):
                    return 'background-color: #2d333b; color: #8b949e;' if pd.isna(val) else ''

                st.dataframe(
                    matrix.style.format('{:.3f}', na_rep='').applymap(blank_cell_gray),
                    use_container_width=True,
                    height=330
                )

# TAB: REALIZED VOL
# ============================================================================
if active_tab == "REALIZED VOL":
    st.subheader("Realized Volatility - Precomputed")

    rv_df = load_realized_vol_data()
    if rv_df is None or len(rv_df) == 0:
        st.warning("Realized vol cache not available. Run scripts/precompute_realized_vol.py first.")
    else:
        comm = rv_df[rv_df['commodity'] == price_product].copy()
        comm = comm[comm['date'].notna()]

        if len(comm) == 0:
            st.warning(f"No realized vol data available for {price_product}.")
        else:
            asof = comm[comm['date'] <= selected_date]['date'].max()
            if pd.isna(asof):
                asof = comm['date'].max()
            snap = comm[comm['date'] == asof].copy()

            month_order = {
                'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
            }

            def contract_sort_key(code):
                if pd.isna(code) or len(str(code)) < 2:
                    return (9999, 99)
                code = str(code)
                letter = code[0]
                try:
                    year = int("20" + code[1:])
                except Exception:
                    year = 9999
                return (year, month_order.get(letter, 99))

            snap = snap.sort_values('contract_code', key=lambda s: s.map(contract_sort_key)).head(12).copy()
            snap['contract_month'] = np.arange(1, len(snap) + 1)

            st.caption(f"As of: {pd.to_datetime(asof).date()} | Showing nearest {len(snap)} contracts")

            display_cols = ['contract_code', 'rv_5d', 'rv_10d', 'rv_20d', 'rv_50d']
            display = snap[display_cols].copy().reset_index(drop=True)
            display.columns = ['Contract', 'RV 5D', 'RV 10D', 'RV 20D', 'RV 50D']

            st.dataframe(
                display.style.format({
                    'RV 5D': '{:.3f}',
                    'RV 10D': '{:.3f}',
                    'RV 20D': '{:.3f}',
                    'RV 50D': '{:.3f}',
                }),
                use_container_width=True,
                height=460,
                hide_index=True
            )

            st.caption("RV is annualized from rolling log-return stdev (trading-day basis).")

            # ----------------------------------------------------------------
            # HLOC Volatility Grid (Garman-Klass style)
            # ----------------------------------------------------------------
            st.markdown("---")
            st.subheader("HLOC Volatility (High-Low-Open-Close)")

            prices_df = load_price_data()
            if prices_df is not None and len(prices_df) > 0:
                hloc_windows = [5, 10, 20, 50, 100, 200]
                hloc_result = pa.calculate_hloc_volatility(
                    prices_df, price_product,
                    windows=hloc_windows,
                    n_contracts=2,
                    as_of_date=asof
                )

                if len(hloc_result) > 0:
                    # Build HLOC display table
                    hloc_rows = []
                    for _, r in hloc_result.iterrows():
                        row_data = {'Contract': r['contract_code']}
                        for w in hloc_windows:
                            row_data[f'DR {w}D'] = r.get(f'rv_{w}d', np.nan)
                            row_data[f'HLOC {w}D'] = r.get(f'hloc_{w}d', np.nan)
                            dr_val = r.get(f'rv_{w}d', np.nan)
                            hloc_val = r.get(f'hloc_{w}d', np.nan)
                            if pd.notna(dr_val) and dr_val > 0 and pd.notna(hloc_val):
                                row_data[f'Ratio {w}D'] = hloc_val / dr_val
                            else:
                                row_data[f'Ratio {w}D'] = np.nan
                        hloc_rows.append(row_data)

                    hloc_display = pd.DataFrame(hloc_rows)

                    # Format columns: group by window
                    fmt_dict = {}
                    for w in hloc_windows:
                        fmt_dict[f'DR {w}D'] = '{:.2f}'
                        fmt_dict[f'HLOC {w}D'] = '{:.2f}'
                        fmt_dict[f'Ratio {w}D'] = '{:.2f}'

                    def ratio_color(val):
                        """Color ratios: >1 = orange (larger range), <1 = blue (smaller range)."""
                        if pd.isna(val):
                            return ''
                        if val > 1.15:
                            return 'color: #ff9500; font-weight: bold;'
                        elif val > 1.0:
                            return 'color: #ffb84d;'
                        elif val < 0.85:
                            return 'color: #4da6ff; font-weight: bold;'
                        elif val < 1.0:
                            return 'color: #80bfff;'
                        return ''

                    ratio_cols = [c for c in hloc_display.columns if c.startswith('Ratio')]

                    styled_hloc = hloc_display.style.format(fmt_dict, na_rep='')
                    for rc in ratio_cols:
                        styled_hloc = styled_hloc.applymap(ratio_color, subset=[rc])

                    st.dataframe(
                        styled_hloc,
                        use_container_width=True,
                        height=140,
                        hide_index=True
                    )

                    st.caption(
                        "HLOC uses the Garman-Klass estimator: "
                        "ln(O/C₋₁)² + 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)². "
                        "DR = Daily Return vol (RMS, not demeaned). "
                        "Ratio > 1 → range exceeds close-to-close moves."
                    )
                else:
                    st.info("No HLOC data available for this product/date.")
            else:
                st.warning("Price data not available for HLOC calculation.")

# ============================================================================
# TAB: VARIANCE RATIOS
# ============================================================================
if active_tab == "VAR RATIOS":
    st.subheader("Variance Ratios - Historical Averages")

    # Load price data
    prices_df = load_price_data()

    if prices_df is not None and len(prices_df) > 0:
        # Product, front month, and lookback selectors on one row (narrowed; lookback +10% width)
        col_prod, col_front, col_lookback, _ = st.columns([1, 1, 1.1, 6.9])

        with col_prod:
            price_product = st.selectbox(
                "Product",
                ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW'],
                index=0,
                key="price_sheet_product"
            )

        with col_front:
            front_month = st.selectbox(
                "Front Month",
                ['H', 'K', 'N', 'Q', 'U', 'X', 'F'],
                index=0,
                help="Select the front options month to view variance ratios"
            )
        # Map options month to futures month for display (aligned left under selectors)
        futures_month = vr.options_to_futures(front_month, price_product)
        st.caption(f"Underlying futures: {futures_month}")

        with col_lookback:
            lookback_years = st.selectbox(
                "Lookback (Years)",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
                index=4,  # Default to 5 years
                format_func=lambda x: "All" if x is None else str(x),
                help="Number of years of historical data to use"
            )

        # Calculate variance ratio matrix
        with st.spinner("Calculating variance ratios..."):
            matrix_df, metadata = vr.get_variance_ratio_display(
                prices_df, front_month, price_product, lookback_years=lookback_years
            )

        if len(matrix_df) > 0:
            # Display metadata
            st.markdown(f"""
            **Historical Data:**
            - Years included: {len(metadata['years_included'])} ({min(metadata['years_included'])+2000} - {max(metadata['years_included'])+2000})
            - Total trading days: {metadata['total_trading_days']}
            """)

            # --- Reshape to lower-triangle display matching screenshot ---
            n_contracts = len(matrix_df)

            # Rename VarRat1..N -> "VarRat to FM", "VarRat to 2FM", ...
            col_rename = {}
            for i in range(n_contracts):
                old = f"VarRat{i+1}"
                new = "VarRat to FM" if i == 0 else f"VarRat to {i+1}FM"
                col_rename[old] = new
            display_df = matrix_df.rename(columns=col_rename).copy()

            # Trim to n-1 columns (last row has nothing further out to reference)
            useful_cols = ["VarRat to FM"] + [f"VarRat to {i+1}FM" for i in range(1, n_contracts - 1)]
            useful_cols = [c for c in useful_cols if c in display_df.columns]
            display_df = display_df[useful_cols]

            # Blank upper triangle only — keep diagonal (should be 1.0, shown green)
            display_df = display_df.astype(float)
            for row_i in range(len(display_df)):
                for col_i in range(len(display_df.columns)):
                    if col_i > row_i:
                        display_df.iloc[row_i, col_i] = np.nan

            # Style
            def _vr_style(val):
                if pd.isna(val):
                    return ''
                if 0.99 <= val <= 1.01:
                    return 'background-color: #1f4d2e; color: #e6edf3; font-weight: 600;'
                if val < 0.7:
                    return 'background-color: #5a1f2a; color: #e6edf3;'
                elif val > 1.3:
                    return 'background-color: #1f3f5a; color: #e6edf3;'
                return ''

            st.dataframe(
                display_df.style.applymap(_vr_style).format("{:.2f}", na_rep=""),
                use_container_width=True,
                height=36 + 28 * len(display_df),
            )

            st.caption("""
            **Reading the matrix:**
            - Rows = contract being measured  |  Columns = reference contract (FM = front, 2FM = second, etc.)
            - Cell = Var(row contract) / Var(column reference)  |  Only shown where row is further out than column
            """)

            # Show the futures curve order
            with st.expander("Futures Curve Order"):
                curve = vr.get_futures_curve_for_front_month(futures_month, commodity)
                curve_display = " → ".join([f"{i+1}FM: {m}" for i, m in enumerate(curve[:8])])
                st.write(curve_display)

            # ---- Variance Ratios - Used (editable, drives forward vol) ----
            st.markdown("---")
            st.subheader("Variance Ratios Used in Fwd Vol Calc")
            st.caption("Edit these values to override what is used in Forward Vol calculation. "
                       "Click 'Load from Historical' to reset to the current historical averages above.")

            _ss_key = f"varrat_used_{price_product}_{front_month}"

            # Seed from historical display_df if not yet in session state
            if _ss_key not in st.session_state:
                st.session_state[_ss_key] = display_df.copy()

            _used_df = st.session_state[_ss_key].copy()

            # Column config — all ratio columns editable numbers
            _used_col_config = {}
            for _col in _used_df.columns:
                _used_col_config[_col] = st.column_config.NumberColumn(
                    _col, min_value=0.0, max_value=5.0, step=0.01, format="%.2f"
                )

            _edited_used = st.data_editor(
                _used_df,
                column_config=_used_col_config,
                use_container_width=True,
                hide_index=False,
                num_rows="fixed",
                key=f"varrat_used_editor_{price_product}_{front_month}",
            )

            # Persist edits back to session state
            st.session_state[_ss_key] = _edited_used.copy()

            # Reset to current historical averages
            if st.button("Load from Historical", key=f"varrat_load_hist_{price_product}_{front_month}"):
                st.session_state[_ss_key] = display_df.copy()
                st.rerun()

        else:
            if isinstance(metadata, dict) and metadata.get("error"):
                st.warning(metadata["error"])
            else:
                st.warning(f"No variance ratio data available for {front_month} options on {price_product}")

    else:
        st.warning("Price data not available. Run update_from_hertz.py to load price data.")

