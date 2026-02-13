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
from datetime import datetime, timedelta

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

@st.cache_data
def load_master():
    """Load master historical vol/skew data (for percentiles/skew)"""
    return va.load_master_data()

@st.cache_data(ttl=10)
def load_live_data():
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
    idx = int(dates.searchsorted(asof, side='right') - 1)
    if idx < 0:
        return pd.DataFrame(), {"error": f"No data on/before {asof.date()}"}

    current_date = dates[idx]

    # Reference dates for change calculations come from MASTER history only.
    # 1D = latest master date; ND = Nth-latest master date.
    if master_df is not None and len(master_df) > 0:
        mdf = master_df[master_df['commodity'] == commodity].copy()
        mdf['date'] = pd.to_datetime(mdf['date'], errors='coerce')
        master_dates = pd.DatetimeIndex(mdf['date'].dropna().sort_values().unique())
    else:
        master_dates = dates

    master_dates = master_dates[master_dates <= current_date]
    if len(master_dates) == 0:
        return pd.DataFrame(), {"error": f"No master history available for {commodity}"}

    prev_date = master_dates[-1] if len(master_dates) >= 1 else None
    long_date = master_dates[-long_window] if len(master_dates) >= long_window else None

    current = cdf[cdf['date'] == current_date][['contract_month', 'expiry', 'dirty_vol', 'fwd_vol']].copy()
    current = current.sort_values(['contract_month', 'expiry']).drop_duplicates(subset=['expiry'], keep='first').head(max_months)
    current = current.rename(columns={'dirty_vol': 'iv_now', 'fwd_vol': 'fwd_now'})

    # Build display contract code from expiry using commodity-specific OPTIONS mapping
    # (prevents invalid labels like G for commodities that don't list it).
    month_code = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    options_lookup = {}
    try:
        m = va.load_options_mapping('data/mapping.csv')
        sub = m[m['COMMODITY'] == str(commodity).upper()]
        if len(sub) > 0 and 'EXPIRY_MONTH' in sub.columns:
            options_lookup = {
                int(r['EXPIRY_MONTH']): str(r['OPTIONS']).upper()
                for _, r in sub.iterrows()
                if pd.notna(r.get('EXPIRY_MONTH', np.nan)) and pd.notna(r.get('OPTIONS', np.nan))
            }
    except Exception:
        options_lookup = {}

    exp_dt = pd.to_datetime(current['expiry'], errors='coerce')
    opt_code = exp_dt.dt.month.map(
        lambda m: options_lookup.get(int(m), month_code.get(int(m), '?')) if pd.notna(m) else '?'
    )
    # Contract-year convention: F contracts are labeled with next calendar year.
    # Example: 2026-12-24 expiry -> F27.
    contract_year = exp_dt.dt.year + np.where(opt_code == 'F', 1, 0)
    current['contract_code'] = (
        opt_code.astype(str)
        + contract_year.mod(100).fillna(0).astype(int).astype(str).str.zfill(2)
    )

    out = current[['contract_month', 'expiry', 'contract_code', 'iv_now', 'fwd_now']].copy()

    # Baseline values should come from MASTER history (not live-overlaid cdf)
    ref_base = cdf
    if master_df is not None and len(master_df) > 0:
        ref_base = master_df[master_df['commodity'] == commodity].copy()
        ref_base['date'] = pd.to_datetime(ref_base['date'], errors='coerce')
        if 'expiry' in ref_base.columns:
            ref_base['expiry'] = pd.to_datetime(ref_base['expiry'], errors='coerce')

    if prev_date is not None:
        prev = ref_base[ref_base['date'] == prev_date][['expiry', 'dirty_vol', 'fwd_vol']].copy()
        prev = prev.sort_values('expiry').drop_duplicates(subset=['expiry'], keep='last')
        prev = prev.rename(columns={'dirty_vol': 'iv_prev_1d', 'fwd_vol': 'fwd_prev_1d'})
        out = out.merge(prev, on='expiry', how='left')
        out['iv_chg_1d'] = out['iv_now'] - out['iv_prev_1d']
        out['fwd_chg_1d'] = out['fwd_now'] - out['fwd_prev_1d']
    else:
        out['iv_chg_1d'] = np.nan
        out['fwd_chg_1d'] = np.nan

    if long_date is not None:
        long_df = ref_base[ref_base['date'] == long_date][['expiry', 'dirty_vol', 'fwd_vol']].copy()
        long_df = long_df.sort_values('expiry').drop_duplicates(subset=['expiry'], keep='last')
        long_df = long_df.rename(columns={'dirty_vol': 'iv_prev_long', 'fwd_vol': 'fwd_prev_long'})
        out = out.merge(long_df, on='expiry', how='left')
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

# Top navigation + global live controls
NAV_SECTIONS = [
    "Vol Sheet",
    "Price Sheet",
    "Skew Analyzer",
    "IV Calendar",
    "Trading Calendar",
    "Settings",
]
default_section = st.session_state.get("active_section", "Vol Sheet")
# Use full-width segmented control so all tabs stay on one line.
active_section = st.segmented_control(
    "Navigation",
    NAV_SECTIONS,
    default=default_section if default_section in NAV_SECTIONS else "Vol Sheet",
    key="active_section",
    label_visibility="collapsed"
)
if active_section is None:
    active_section = "Vol Sheet"

if active_section not in ["Vol Sheet", "Price Sheet", "Skew Analyzer", "IV Calendar", "Trading Calendar", "Settings"]:
    st.markdown(f'<div class="bloomberg-header"><span>{active_section.upper()}</span></div>', unsafe_allow_html=True)
    st.info(f"{active_section} view is coming soon.")
    st.stop()

# Top-of-page data timestamps (single row)
_updates_top = get_source_update_timestamps()
ts_cols = st.columns([1, 1, 1])
with ts_cols[0]:
    st.caption(f"Live: {_updates_top.get('live_data', 'unavailable')}")
with ts_cols[1]:
    st.caption(f"Vol: {_updates_top.get('master_vol_skew', 'unavailable')}")
with ts_cols[2]:
    st.caption(f"Prices: {_updates_top.get('all_commodity_prices', 'unavailable')}")

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

    st.markdown("---")
    st.subheader("Data Timestamps")
    _updates = get_source_update_timestamps()
    st.markdown(f"- Live data: `{_updates.get('live_data', 'unavailable')}`")
    st.markdown(f"- Vol data: `{_updates.get('master_vol_skew', 'unavailable')}`")
    st.markdown(f"- Price data: `{_updates.get('all_commodity_prices', 'unavailable')}`")
    st.stop()

# Load data
try:
    df = load_data()
    master_df = load_master()
    master_base_df = master_df.copy()
    live_df = load_live_data()

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
        "IV %ILE",
        "DASHBOARD",
        "VOL CHANGE",
        "TERM STRUCT",
        "SPREADS",
        "SKEW",
    ]
    active_tab = st.radio("View", tab_options, horizontal=True, key="active_tab_vol")
    price_product = None
elif active_section == "Price Sheet":
    st.markdown('<div class="bloomberg-header"><span>PRICE SHEET</span></div>', unsafe_allow_html=True)
    price_tab_options = [
        "VAR RATIOS",
        "REALIZED VOL",
        "CORRELATIONS",
    ]
    active_tab = st.radio("View", price_tab_options, horizontal=True, key="active_tab_price")
    price_product = None
    if active_tab in ["VAR RATIOS", "REALIZED VOL"]:
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
    active_tab = "IV %ILE"
    price_product = None
elif active_section == "Trading Calendar":
    st.markdown('<div class="bloomberg-header"><span>TRADING CALENDAR</span></div>', unsafe_allow_html=True)
    active_tab = "TRADING CALENDAR"
    price_product = None

# ============================================================================
# TAB 0: POWER GRID
# ============================================================================
if active_tab == "POWER GRID":
    st.markdown('<div class="bloomberg-header"><span>POWER GRID</span></div>', unsafe_allow_html=True)
    st.caption("(FWD VOL - PREDICTED RV) / FWD VOL  |  Positive = IV rich, Negative = IV cheap")

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
            lookup = {}
            try:
                m = va.load_options_mapping('data/mapping.csv')
                sub = m[m['COMMODITY'] == str(comm).upper()]
                if len(sub) > 0 and 'EXPIRY_MONTH' in sub.columns:
                    lookup = {
                        int(r['EXPIRY_MONTH']): str(r['OPTIONS']).upper()
                        for _, r in sub.iterrows()
                        if pd.notna(r.get('EXPIRY_MONTH', np.nan)) and pd.notna(r.get('OPTIONS', np.nan))
                    }
            except Exception:
                lookup = {}
            fallback = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
            for _, r in day.iterrows():
                cm = int(r['contract_month'])
                if cm > max_grid_months:
                    continue
                exp = pd.to_datetime(r.get('expiry', None), errors='coerce')
                if pd.isna(exp):
                    continue
                code = lookup.get(int(exp.month), fallback.get(int(exp.month), '?'))
                year = int(exp.year) + (1 if code == 'F' else 0)
                labels[f"M{cm}"] = f"{code}{str(year % 100).zfill(2)}"
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

        with col_ref1:
            with st.expander("FWD VOL MATRIX"):
                if len(fwd_vols_df) > 0:
                    fwd_html_rows = []
                    for commodity in fwd_vols_df.index:
                        cells = f'<td style="color:#ff9500;font-weight:bold;padding:4px 8px;border-right:1px solid #1e2530;">{commodity}</td>'
                        for col in fwd_vols_df.columns:
                            val = fwd_vols_df.loc[commodity, col] if col in fwd_vols_df.columns else None
                            if pd.notna(val):
                                cells += f'<td style="color:#c8cdd3;padding:4px 8px;text-align:right;border-right:1px solid #1e2530;">{val:.2f}</td>'
                            else:
                                cells += f'<td style="color:#586069;padding:4px 8px;text-align:center;border-right:1px solid #1e2530;">—</td>'
                        fwd_html_rows.append(f'<tr style="border-bottom:1px solid #1e2530;">{cells}</tr>')
                    fwd_header = '<th style="color:#8b949e;padding:4px 8px;border-bottom:1px solid #ff9500;border-right:1px solid #1e2530;"></th>'
                    for col in fwd_vols_df.columns:
                        fwd_header += f'<th style="color:#8b949e;padding:4px 8px;text-align:right;border-bottom:1px solid #ff9500;border-right:1px solid #1e2530;font-size:11px;">{col}</th>'
                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;font-family:Consolas,Monaco,monospace;font-size:13px;background-color:#0f1419;border:1px solid #1e2530;">
                        <thead><tr>{fwd_header}</tr></thead>
                        <tbody>{''.join(fwd_html_rows)}</tbody>
                    </table>
                    """, unsafe_allow_html=True)

        with col_ref2:
            with st.expander("PREDICTED RV MATRIX"):
                if len(pred_rvs_df) > 0:
                    rv_html_rows = []
                    for commodity in pred_rvs_df.index:
                        cells = f'<td style="color:#ff9500;font-weight:bold;padding:4px 8px;border-right:1px solid #1e2530;">{commodity}</td>'
                        for col in pred_rvs_df.columns:
                            val = pred_rvs_df.loc[commodity, col] if col in pred_rvs_df.columns else None
                            if pd.notna(val):
                                cells += f'<td style="color:#c8cdd3;padding:4px 8px;text-align:right;border-right:1px solid #1e2530;">{val:.2f}</td>'
                            else:
                                cells += f'<td style="color:#586069;padding:4px 8px;text-align:center;border-right:1px solid #1e2530;">—</td>'
                        rv_html_rows.append(f'<tr style="border-bottom:1px solid #1e2530;">{cells}</tr>')
                    rv_header = '<th style="color:#8b949e;padding:4px 8px;border-bottom:1px solid #ff9500;border-right:1px solid #1e2530;"></th>'
                    for col in pred_rvs_df.columns:
                        rv_header += f'<th style="color:#8b949e;padding:4px 8px;text-align:right;border-bottom:1px solid #ff9500;border-right:1px solid #1e2530;font-size:11px;">{col}</th>'
                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;font-family:Consolas,Monaco,monospace;font-size:13px;background-color:#0f1419;border:1px solid #1e2530;">
                        <thead><tr>{rv_header}</tr></thead>
                        <tbody>{''.join(rv_html_rows)}</tbody>
                    </table>
                    """, unsafe_allow_html=True)

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
# TAB: IV PERCENTILE BATTERY
# ============================================================================
if active_tab == "IV %ILE":
    st.markdown('<div class="bloomberg-header"><span>IV PERCENTILE RANK</span></div>', unsafe_allow_html=True)

    # Display order: SOY, MEAL, CORN, WHEAT, KW, OIL
    pct_commodities = ['SOY', 'MEAL', 'CORN', 'WHEAT', 'KW', 'OIL']
    pct_max_months = 12

    # Calculate percentiles; if live data exists, use df/master (with live overlay) to ensure current dirty_vol drives the percentile.
    lookback_years = lookback_years_from_days(lookback)
    pct_rows = []

    use_live = live_df is not None and not live_df.empty

    if use_live:
        pct_grid = va.get_percentile_grid_cached(
            df, selected_date, metric='dirty_vol',
            lookback_days=lookback, commodities=pct_commodities,
            max_months=pct_max_months, hist_df=master_hist
        )
    else:
        for c in pct_commodities:
            front_opt = current_front_options_month(c, selected_date)
            snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
            if isinstance(snap, pd.DataFrame) and len(snap) > 0:
                tmp = assign_contract_month_from_snapshot(snap, max_months=pct_max_months)
                tmp['commodity'] = c
                # Prefer contract code, then options_month, then expiry, else M#
                if 'contract_code' in tmp.columns:
                    tmp['month_label'] = tmp['contract_code'].astype(str)
                elif 'options_month' in tmp.columns:
                    tmp['month_label'] = tmp['options_month'].astype(str)
                elif 'expiry' in tmp.columns:
                    tmp['month_label'] = pd.to_datetime(tmp['expiry'], errors='coerce').dt.strftime('%b%y')
                else:
                    tmp['month_label'] = tmp['contract_month'].map(lambda x: f"M{int(x)}")
                tmp['current'] = pd.to_numeric(tmp.get('atm_iv', np.nan), errors='coerce')
                tmp['median'] = pd.to_numeric(tmp.get('iv_hist_median', np.nan), errors='coerce')
                p = pd.to_numeric(tmp.get('iv_percentile', np.nan), errors='coerce')
                tmp['percentile'] = np.where(p <= 1, p * 100.0, p)
                pct_rows.append(tmp[['commodity', 'contract_month', 'month_label', 'current', 'percentile', 'median']])

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
            if 'contract_code' in pct_grid.columns:
                pct_grid['month_label'] = pct_grid['contract_code'].astype(str)
            elif 'options_month' in pct_grid.columns:
                pct_grid['month_label'] = pct_grid['options_month'].astype(str)
            elif 'expiry' in pct_grid.columns:
                pct_grid['month_label'] = pd.to_datetime(pct_grid['expiry'], errors='coerce').dt.strftime('%b%y')
            else:
                pct_grid['month_label'] = pct_grid['contract_month'].map(lambda x: f"M{int(x)}")

    # Recompute percentiles using live dirty_vol rounded to nearest 0.05% where possible
    def _round_to_005(val):
        try:
            return np.round(float(val) * 20.0) / 20.0
        except Exception:
            return np.nan

    def _percentile_from_hist(row):
        if master_df is None or len(master_df) == 0:
            return row.get('percentile', np.nan)
        if 'dirty_vol' not in master_df.columns:
            return row.get('percentile', np.nan)
        filt = (master_df['commodity'] == row['commodity'])
        if 'contract_month' in master_df.columns:
            filt = filt & (master_df['contract_month'] == row['contract_month'])
        vals = master_df.loc[filt, 'dirty_vol'].dropna()
        if len(vals) == 0:
            return row.get('percentile', np.nan)
        v = _round_to_005(row.get('current', np.nan))
        if pd.isna(v):
            return row.get('percentile', np.nan)
        sorted_vals = np.sort(vals.values.astype(float))
        pos = np.searchsorted(sorted_vals, v, side='right')
        pct = (pos / len(sorted_vals)) * 100.0
        return pct

    def _median_from_hist(row):
        if master_df is None or len(master_df) == 0:
            return row.get('median', np.nan)
        if 'dirty_vol' not in master_df.columns:
            return row.get('median', np.nan)
        filt = (master_df['commodity'] == row['commodity'])
        if 'contract_month' in master_df.columns:
            filt = filt & (master_df['contract_month'] == row['contract_month'])
        vals = master_df.loc[filt, 'dirty_vol'].dropna()
        if len(vals) == 0:
            return row.get('median', np.nan)
        sorted_vals = np.sort(vals.values.astype(float))
        return np.percentile(sorted_vals, 50)

    if len(pct_grid) > 0:
        pct_grid['current'] = pct_grid['current'].apply(_round_to_005)
        pct_grid['percentile'] = pct_grid.apply(_percentile_from_hist, axis=1)
        pct_grid['median'] = pct_grid.apply(_median_from_hist, axis=1)

        # Build best-available labels from live/current snapshot at selected_date
        fallback_month_codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
        options_map = {}
        try:
            opt_map_df = va.load_options_mapping('data/mapping.csv')
            if opt_map_df is not None and len(opt_map_df) > 0:
                for comm in opt_map_df['COMMODITY'].dropna().unique():
                    sub = opt_map_df[opt_map_df['COMMODITY'] == comm]
                    options_map[comm] = {
                        int(r['EXPIRY_MONTH']): str(r['OPTIONS']).upper()
                        for _, r in sub.iterrows()
                        if pd.notna(r.get('EXPIRY_MONTH', np.nan)) and pd.notna(r.get('OPTIONS', np.nan))
                    }
        except Exception:
            options_map = {}

        label_lookup = {}
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
                        exp = pd.to_datetime(r['expiry'])
                        mnum = int(exp.month)
                        code = options_map.get(c.upper(), {}).get(mnum, fallback_month_codes.get(mnum, '?'))
                        year = int(exp.year) + (1 if code == 'F' else 0)
                        label = f"{code}{str(year % 100).zfill(2)}"
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
                label = label_lookup.get((commodity, cm), None) if 'label_lookup' in locals() else None
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
                    <div style="width:80px;font-size:11px;color:#586069;text-align:right;margin-left:8px;">{current:.1f}</div>
                    <div style="width:70px;font-size:10px;color:#3d444d;text-align:right;margin-left:4px;">med {median:.1f}</div>
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
if active_tab == "VOL CHANGE":
    st.markdown('<div class="bloomberg-header"><span>VOL CHANGE</span></div>', unsafe_allow_html=True)
    st.caption("IV and Forward Vol change grids by contract and commodity")

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
            grid_1d_iv.style.format({col: '{:+.3f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
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
            grid_1d_fwd.style.format({col: '{:+.3f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
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
            grid_nd_iv.style.format({col: '{:+.3f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
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
            grid_nd_fwd.style.format({col: '{:+.3f}' for col in delta_cols}).applymap(_delta_cell_style, subset=delta_cols),
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

    front_opt_override = st.selectbox(
        "Front options month (override)",
        month_codes,
        index=default_idx,
        help="Select which front-month options contract to use for all commodities in this view."
    )

    skew_commodities = ['CORN', 'SOY', 'MEAL', 'WHEAT', 'KW', 'OIL']
    lookback_years = lookback_years_from_days(lookback)
    skew_rows = []
    median_rows = []
    for c in skew_commodities:
        # Use user-selected front options month across commodities
        front_opt = front_opt_override or current_front_options_month(c, selected_date)
        snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
        if isinstance(snap, pd.DataFrame) and len(snap) > 0:
            tmp = assign_contract_month_from_snapshot(snap, max_months=12)
            tmp['commodity'] = c
            if 'contract_label' in tmp.columns:
                tmp['contract_month_label'] = tmp['contract_label'].astype(str)
            else:
                tmp['contract_month_label'] = tmp['contract_month'].map(lambda x: f"M{int(x)}")
            tmp['contract_month_label'] = tmp['contract_month_label'].replace({'nan': np.nan, 'None': np.nan})
            tmp['contract_month_label'] = tmp['contract_month_label'].fillna(tmp['contract_month'].map(lambda x: f"M{int(x)}"))

            # Capture historical medians for this commodity
            med_entry = tmp[['contract_month']].copy()
            med_entry['commodity'] = c
            med_entry['contract_month_label'] = med_entry['contract_month'].map(lambda x: f"M{int(x)}")
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

            skew_source_candidates = {
                'P2': ['skew_m1.5', 'skew_neg15', 'P2'],
                'P1': ['skew_m0.5', 'skew_neg05', 'P1'],
                'C1': ['skew_p0.5', 'skew_pos05', 'C1'],
                'C2': ['skew_p1.5', 'skew_pos15', 'C2'],
                'C3': ['skew_p3.0', 'skew_pos3', 'C3'],
            }

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
            skew_rows.append(tmp[['commodity', 'contract_month', 'contract_month_label', 'P2', 'P1', 'C1', 'C2', 'C3', 'P2_current', 'P1_current', 'C1_current', 'C2_current', 'C3_current']])

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

        def skew_pct(val, comm, cm, col):
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
                    pct_val = skew_pct(val, c, cm, col)
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

# ============================================================================
# TAB: VARIANCE RATIOS
# ============================================================================
if active_tab == "VAR RATIOS":
    st.subheader("Variance Ratios - Historical Averages")

    # Load price data
    prices_df = load_price_data()

    if prices_df is not None and len(prices_df) > 0:
        # Front options month selector
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            front_month = st.selectbox(
                "Front Options Month",
                ['H', 'K', 'N', 'Q', 'U', 'X', 'F'],
                index=0,
                help="Select the front options month to view variance ratios"
            )

            # Map options month to futures month for display
            futures_month = vr.options_to_futures(front_month, price_product)
            st.caption(f"Underlying futures: {futures_month}")

        with col2:
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

            # Style the dataframe
            def highlight_diagonal(val):
                """Highlight cells close to 1.0 (diagonal)"""
                if pd.isna(val):
                    return ''
                if 0.99 <= val <= 1.01:
                    return 'background-color: #1f4d2e; color: #e6edf3; font-weight: 600;'  # dark green
                elif val < 0.7:
                    return 'background-color: #5a1f2a; color: #e6edf3;'  # dark rose
                elif val > 1.3:
                    return 'background-color: #1f3f5a; color: #e6edf3;'  # dark blue
                return ''

            # Display the matrix
            st.dataframe(
                matrix_df.style.applymap(highlight_diagonal).format("{:.2f}"),
                use_container_width=True,
                height=450
            )

            st.caption("""
            **Reading the matrix:**
            - Rows = contract being measured
            - Columns = contract position on curve (VarRat1 = front, VarRat2 = 2nd, etc.)
            - Cell = Variance(row) / Variance(column)
            - Green = diagonal (1.0), Red = low ratio (<0.7), Blue = high ratio (>1.3)
            """)

            # Show the futures curve order
            with st.expander("Futures Curve Order"):
                curve = vr.get_futures_curve_for_front_month(futures_month, commodity)
                curve_display = " → ".join([f"{i+1}FM: {m}" for i, m in enumerate(curve[:8])])
                st.write(curve_display)

        else:
            if isinstance(metadata, dict) and metadata.get("error"):
                st.warning(metadata["error"])
            else:
                st.warning(f"No variance ratio data available for {front_month} options on {price_product}")

    else:
        st.warning("Price data not available. Run update_from_hertz.py to load price data.")

# Footer
st.markdown("---")
footer_left = st.container()
with footer_left:
    if st.button("Refresh Live", key="refresh_live_bottom"):
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
