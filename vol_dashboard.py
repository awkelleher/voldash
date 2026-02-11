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
    live_df: pd.DataFrame | None = None
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
    prev_date = dates[idx - 1] if idx - 1 >= 0 else None
    long_date = dates[idx - long_window] if idx - long_window >= 0 else None

    current = cdf[cdf['date'] == current_date][['contract_month', 'expiry', 'dirty_vol', 'fwd_vol']].copy()
    current = current.sort_values('contract_month').head(max_months)
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
    current['contract_code'] = (
        exp_dt.dt.month.map(lambda m: options_lookup.get(int(m), month_code.get(int(m), '?')) if pd.notna(m) else '?')
        + exp_dt.dt.year.mod(100).fillna(0).astype(int).astype(str).str.zfill(2)
    )

    out = current[['contract_month', 'contract_code', 'iv_now', 'fwd_now']].copy()

    if prev_date is not None:
        prev = cdf[cdf['date'] == prev_date][['contract_month', 'dirty_vol', 'fwd_vol']].copy()
        prev = prev.rename(columns={'dirty_vol': 'iv_prev_1d', 'fwd_vol': 'fwd_prev_1d'})
        out = out.merge(prev, on='contract_month', how='left')
        out['iv_chg_1d'] = out['iv_now'] - out['iv_prev_1d']
        out['fwd_chg_1d'] = out['fwd_now'] - out['fwd_prev_1d']
    else:
        out['iv_chg_1d'] = np.nan
        out['fwd_chg_1d'] = np.nan

    if long_date is not None:
        long_df = cdf[cdf['date'] == long_date][['contract_month', 'dirty_vol', 'fwd_vol']].copy()
        long_df = long_df.rename(columns={'dirty_vol': 'iv_prev_long', 'fwd_vol': 'fwd_prev_long'})
        out = out.merge(long_df, on='contract_month', how='left')
        out['iv_chg_long'] = out['iv_now'] - out['iv_prev_long']
        out['fwd_chg_long'] = out['fwd_now'] - out['fwd_prev_long']
    else:
        out['iv_chg_long'] = np.nan
        out['fwd_chg_long'] = np.nan

    out = out.sort_values('contract_month').reset_index(drop=True)
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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="bloomberg-header"><span>VOL TRADING DASHBOARD</span></div>', unsafe_allow_html=True)

# Sidebar - Navigation
NAV_SECTIONS = [
    "Vol Sheet",
    "Price Sheet",
    "Skew Analyzer",
    "IV Calendar",
    "Trading Calendar",
    "Settings",
]
active_section = st.sidebar.radio("Navigation", NAV_SECTIONS, key="active_section")

if active_section not in ["Vol Sheet", "Price Sheet"]:
    st.markdown(f'<div class="bloomberg-header"><span>{active_section.upper()}</span></div>', unsafe_allow_html=True)
    st.info(f"{active_section} view is coming soon.")
    st.stop()

# Sidebar - Vol Sheet Settings
st.sidebar.header("Vol Sheet Settings")

# Load data
try:
    df = load_data()
    master_df = load_master()
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

    # Live data indicator + refresh button
    if live_df is not None and not live_df.empty:
        live_ts = os.path.getmtime('data/live_vols.csv')
        live_time = datetime.fromtimestamp(live_ts).strftime('%H:%M:%S')
        st.sidebar.success(f"LIVE DATA: {live_time}")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh Live"):
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
                    st.sidebar.error(f"Error: {result.stderr[-200:]}")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")
    with col2:
        if live_df is not None and not live_df.empty:
            if st.button("Clear Live"):
                os.remove('data/live_vols.csv')
                st.cache_data.clear()
                st.rerun()
    
    # Date selector
    selected_date = st.sidebar.date_input(
        "Analysis Date",
        value=latest_date.date(),
        min_value=earliest_date.date(),
        max_value=latest_date.date()
    )
    selected_date = pd.to_datetime(selected_date)
    master_hist = master_df[master_df['date'] < selected_date].copy()
    
    # Commodity selector
    commodity = st.sidebar.selectbox(
        "Commodity",
        ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT']
    )
    
    # Lookback period
    lookback = st.sidebar.slider(
        "Historical Lookback (trading days)",
        min_value=60,
        max_value=504,
        value=252
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data range: {earliest_date.date()} to {latest_date.date()}")
    st.sidebar.caption(f"Total records: {len(df):,}")
    st.sidebar.caption(f"Trading days: {df['date'].nunique():,}")
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main content
st.markdown(f'<div style="font-family:Consolas,monospace;color:#8b949e;font-size:13px;margin:4px 0 8px 0;">{commodity} &nbsp;|&nbsp; {selected_date.strftime("%Y-%m-%d")} &nbsp;|&nbsp; Lookback: {lookback}d</div>', unsafe_allow_html=True)

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
else:
    st.markdown('<div class="bloomberg-header"><span>PRICE SHEET</span></div>', unsafe_allow_html=True)
    price_product = st.selectbox(
        "Product",
        ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW'],
        index=0,
        key="price_sheet_product"
    )
    price_tab_options = [
        "VAR RATIOS",
        "REALIZED VOL",
    ]
    active_tab = st.radio("View", price_tab_options, horizontal=True, key="active_tab_price")

# ============================================================================
# TAB 0: POWER GRID
# ============================================================================
if active_tab == "POWER GRID":
    st.markdown('<div class="bloomberg-header"><span>POWER GRID</span></div>', unsafe_allow_html=True)
    st.caption("(FWD VOL - PREDICTED RV) / FWD VOL  |  Positive = IV rich, Negative = IV cheap")

    grid_commodities = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT']
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

        # Build HTML table for Bloomberg-style display
        def _power_color(val):
            """Return CSS color for a power value."""
            if pd.isna(val):
                return '#586069', ''
            if val > 5:
                return '#3fb950', 'font-weight:bold'
            elif val > 0:
                return '#56d364', ''
            elif val > -5:
                return '#f85149', ''
            else:
                return '#ff7b72', 'font-weight:bold'

        # Column headers with month codes if available
        header_cells = '<th style="color:#8b949e;padding:6px 10px;text-align:left;border-bottom:2px solid #ff9500;border-right:1px solid #1e2530;"></th>'
        for col in power_df.columns:
            # Try to get the month code for this column from the first commodity that has it
            code_label = ''
            if len(month_codes_df) > 0:
                for cidx in month_codes_df.index:
                    if col in month_codes_df.columns:
                        c_val = month_codes_df.loc[cidx, col]
                        if pd.notna(c_val) and str(c_val) != '?':
                            code_label = f'<br/><span style="color:#586069;font-size:10px;">{c_val}</span>'
                            break
            header_cells += f'<th style="color:#8b949e;padding:6px 10px;text-align:right;border-bottom:2px solid #ff9500;border-right:1px solid #1e2530;font-size:12px;">{col}{code_label}</th>'

        html_rows = []
        for commodity in power_df.index:
            cells = f'<td style="color:#ff9500;font-weight:bold;padding:6px 10px;border-right:1px solid #1e2530;white-space:nowrap;">{commodity}</td>'
            for col in power_df.columns:
                val = power_df.loc[commodity, col] if col in power_df.columns else None
                if pd.notna(val):
                    color, style = _power_color(val)
                    # Build tooltip with fwd vol, predicted RV, and month code
                    fv = fwd_vols_df.loc[commodity, col] if col in fwd_vols_df.columns and pd.notna(fwd_vols_df.loc[commodity, col]) else None
                    pr = pred_rvs_df.loc[commodity, col] if col in pred_rvs_df.columns and pd.notna(pred_rvs_df.loc[commodity, col]) else None
                    mc = ''
                    if len(month_codes_df) > 0 and col in month_codes_df.columns:
                        mc_val = month_codes_df.loc[commodity, col] if commodity in month_codes_df.index else None
                        mc = f" [{mc_val}]" if pd.notna(mc_val) else ""
                    tip_parts = []
                    if fv is not None:
                        tip_parts.append(f"FV:{fv:.1f}")
                    if pr is not None:
                        tip_parts.append(f"RV:{pr:.1f}")
                    tip_parts.append(mc.strip())
                    tooltip = " | ".join([t for t in tip_parts if t])
                    cells += f'<td style="color:{color};{style};padding:6px 10px;text-align:right;border-right:1px solid #1e2530;" title="{tooltip}">{val:+.1f}%</td>'
                else:
                    cells += f'<td style="color:#586069;padding:6px 10px;text-align:center;border-right:1px solid #1e2530;">—</td>'
            html_rows.append(f'<tr style="border-bottom:1px solid #1e2530;">{cells}</tr>')

        html_table = f"""
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-family:Consolas,Monaco,monospace;font-size:14px;background-color:#0f1419;border:1px solid #1e2530;">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{''.join(html_rows)}</tbody>
        </table>
        </div>
        """
        st.markdown(html_table, unsafe_allow_html=True)

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

    pct_commodities = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT']
    pct_max_months = 12

    # Calculate all percentiles from precomputed snapshots (dirty vol only)
    lookback_years = lookback_years_from_days(lookback)
    pct_rows = []
    for c in pct_commodities:
        front_opt = current_front_options_month(c, selected_date)
        snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
        if isinstance(snap, pd.DataFrame) and len(snap) > 0:
            tmp = assign_contract_month_from_snapshot(snap, max_months=pct_max_months)
            tmp['commodity'] = c
            tmp['current'] = pd.to_numeric(tmp.get('atm_iv', np.nan), errors='coerce')
            tmp['median'] = pd.to_numeric(tmp.get('iv_hist_median', np.nan), errors='coerce')
            p = pd.to_numeric(tmp.get('iv_percentile', np.nan), errors='coerce')
            tmp['percentile'] = np.where(p <= 1, p * 100.0, p)
            pct_rows.append(tmp[['commodity', 'contract_month', 'current', 'percentile', 'median']])

    if len(pct_rows) > 0:
        pct_grid = pd.concat(pct_rows, ignore_index=True)
    else:
        # Fallback to in-app compute/cache if precompute cache unavailable
        pct_grid = va.get_percentile_grid_cached(
            df, selected_date, metric='dirty_vol',
            lookback_days=lookback, commodities=pct_commodities,
            max_months=pct_max_months, hist_df=master_hist
        )

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
                    <div style="width:28px;font-size:11px;color:#8b949e;text-align:right;margin-right:8px;">M{cm}</div>
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
    st.caption("1-day and selectable longer-term changes for IV (dirty vol) and Forward Vol")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        long_window = st.selectbox(
            "Longer-term lookback (trading days)",
            list(range(1, 21)),
            index=4,
            key="vol_change_lookback"
        )
    with col_b:
        max_months = st.selectbox(
            "Max contract months",
            [6, 8, 10, 12],
            index=3,
            key="vol_change_max_months"
        )

    vol_change_commodities = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW']
    long_lbl = f"{long_window}D"
    long_col_lbl = long_lbl if long_window != 1 else "1D (selected)"

    for c in vol_change_commodities:
        vol_chg, meta = build_vol_change_table(
            df=df,
            commodity=c,
            selected_date=selected_date,
            long_window=long_window,
            max_months=max_months,
            live_df=live_df
        )

        st.markdown(f"**{c}**")
        if len(vol_chg) == 0:
            st.warning(meta.get("error", f"No vol change data available for {c}"))
            continue

        current_date = meta.get("current_date")
        prev_date = meta.get("prev_date")
        long_date = meta.get("long_date")

        st.caption(
            f"As of {pd.to_datetime(current_date).date()}  |  "
            f"1D ref: {pd.to_datetime(prev_date).date() if prev_date is not None else 'N/A'}  |  "
            f"{long_lbl} ref: {pd.to_datetime(long_date).date() if long_date is not None else 'N/A'}"
        )

        m1 = vol_chg[vol_chg['contract_month'] == 1]
        if len(m1) > 0:
            r = m1.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(f"{c} M1 IV 1D Δ", f"{r['iv_chg_1d']:+.2f}" if pd.notna(r['iv_chg_1d']) else "N/A")
            with c2:
                st.metric(f"{c} M1 IV {long_col_lbl} Δ", f"{r['iv_chg_long']:+.2f}" if pd.notna(r['iv_chg_long']) else "N/A")
            with c3:
                st.metric(f"{c} M1 FWD 1D Δ", f"{r['fwd_chg_1d']:+.2f}" if pd.notna(r['fwd_chg_1d']) else "N/A")
            with c4:
                st.metric(f"{c} M1 FWD {long_col_lbl} Δ", f"{r['fwd_chg_long']:+.2f}" if pd.notna(r['fwd_chg_long']) else "N/A")

        table = vol_chg.copy()
        table = table[['contract_code', 'iv_now', 'iv_chg_1d', 'iv_chg_long', 'fwd_now', 'fwd_chg_1d', 'fwd_chg_long']]
        table.columns = ['Contract', 'IV Now', 'IV 1D Δ', f'IV {long_col_lbl} Δ', 'Fwd Now', 'Fwd 1D Δ', f'Fwd {long_col_lbl} Δ']

        st.dataframe(
            table.style.format({
                'IV Now': '{:.3f}',
                'IV 1D Δ': '{:+.3f}',
                f'IV {long_col_lbl} Δ': '{:+.3f}',
                'Fwd Now': '{:.3f}',
                'Fwd 1D Δ': '{:+.3f}',
                f'Fwd {long_col_lbl} Δ': '{:+.3f}',
            }),
            use_container_width=True,
            height=min(500, 80 + 35 * len(table)),
            hide_index=True
        )
        st.markdown("---")

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

    skew_commodities = ['CORN', 'SOY', 'MEAL', 'WHEAT', 'KW', 'OIL']
    lookback_years = lookback_years_from_days(lookback)
    skew_rows = []
    for c in skew_commodities:
        front_opt = current_front_options_month(c, selected_date)
        snap, _meta = load_iv_snapshot_cached(c, front_opt, lookback_years)
        if isinstance(snap, pd.DataFrame) and len(snap) > 0:
            tmp = assign_contract_month_from_snapshot(snap, max_months=12)
            tmp['commodity'] = c
            tmp['month_label'] = tmp['contract_month'].map(lambda x: f"M{int(x)}")
            for col in ['P2', 'P1', 'C1', 'C2', 'C3']:
                med_col = f"{col}_hist_median"
                if col in tmp.columns and med_col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors='coerce') - pd.to_numeric(tmp[med_col], errors='coerce')
                else:
                    tmp[col] = np.nan
            skew_rows.append(tmp[['commodity', 'contract_month', 'month_label', 'P2', 'P1', 'C1', 'C2', 'C3']])

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

        for c in skew_commodities:
            cdf = skew_grid[skew_grid['commodity'] == c].sort_values('contract_month')
            cdf = cdf.drop_duplicates(subset=['contract_month'], keep='last')
            if len(cdf) == 0:
                continue
            st.markdown(f"**{c}**")
            display = cdf[['month_label', 'P2', 'P1', 'C1', 'C2', 'C3']].set_index('month_label')
            st.dataframe(
                display.style
                .format('{:+.2f}')
                .applymap(skew_cell_style),
                use_container_width=True,
                height=min(320, 40 + 35 * len(display))
            )

# ============================================================================
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

            display_cols = ['contract_code', 'close', 'rv_5d', 'rv_10d', 'rv_20d', 'rv_50d']
            display = snap[display_cols].copy().reset_index(drop=True)
            display.columns = ['Contract', 'Close', 'RV 5D', 'RV 10D', 'RV 20D', 'RV 50D']

            st.dataframe(
                display.style.format({
                    'Close': '{:.2f}',
                    'RV 5D': '{:.3f}',
                    'RV 10D': '{:.3f}',
                    'RV 20D': '{:.3f}',
                    'RV 50D': '{:.3f}',
                }),
                use_container_width=True,
                height=460,
                hide_index=True
            )

            m1_code = snap.iloc[0]['contract_code'] if len(snap) > 0 else None
            if m1_code is not None:
                hist = comm[comm['contract_code'] == m1_code].sort_values('date')
                if len(hist) > 0:
                    chart = hist.set_index('date')[['rv_5d', 'rv_10d', 'rv_20d', 'rv_50d']].dropna(how='all')
                    if len(chart) > 0:
                        st.markdown(f"**M1 ({m1_code}) RV History**")
                        st.line_chart(chart.tail(252), height=220)

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
_updates = get_source_update_timestamps()
st.markdown(
    f"""
    <div style="font-family:Consolas,monospace;font-size:10px;color:#586069;text-align:right;line-height:1.5;">
        vol data updated at: {_updates['master_vol_skew']}<br/>
        price data updated at: {_updates['all_commodity_prices']}
    </div>
    """,
    unsafe_allow_html=True
)
