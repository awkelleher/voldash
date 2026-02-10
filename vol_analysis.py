"""
Historical Vol & Skew Analysis Module

This module provides functions to work with your cleaned historical volatility and skew data.
Replace the messy Excel formulas with clean Python functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import functools
import os


def load_historical_data(filepath='master_vol_skew.csv'):
    """
    Load daily vol/skew data for current display.
    Uses master_vol_skew.csv as the single source of vol/skew history.
    Adds clean_vol/fwd_vol columns as aliases of dirty_vol for compatibility.
    Derives contract_month if not present by ordering expiry per date/commodity.
    """
    df = pd.read_csv(filepath)

    # Robust date parsing (mixed formats)
    for col in ['date', 'expiry']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format=None)

    # Derive contract_month if missing
    if 'contract_month' not in df.columns:
        df = df.sort_values(['date', 'commodity', 'expiry'])
        df['contract_month'] = df.groupby(['date', 'commodity']).cumcount() + 1

    # Compatibility columns
    if 'clean_vol' not in df.columns:
        df['clean_vol'] = df['dirty_vol']
    if 'fwd_vol' not in df.columns:
        df['fwd_vol'] = df['dirty_vol']

    return df


def load_master_data(filepath='master_vol_skew.csv'):
    """
    Load master historical vol/skew dataset and derive contract_month by
    ordering expiries for each date/commodity.
    """
    df = pd.read_csv(filepath, parse_dates=['date', 'expiry'])
    # Derive contract_month by sorting expiries for each date/commodity
    df = df.sort_values(['date', 'commodity', 'expiry'])
    df['contract_month'] = (
        df.groupby(['date', 'commodity'])['expiry']
        .rank(method='first')
        .astype(int)
    )
    return df


def calculate_percentile_rank(df, date, contract_month, metric, lookback_days=252,
                              commodity=None, hist_df=None):
    """
    Calculate where current value sits in historical distribution
    
    Args:
        df: Historical dataframe
        date: Date to analyze
        contract_month: Contract month (1 = front month, 2 = second month, etc.)
        metric: Column name (e.g., 'clean_vol', 'fwd_vol', 'skew_pos15')
        lookback_days: Number of trading days to include in distribution
        commodity: Commodity name (e.g., 'SOY', 'CORN'). If None, uses all data.
        
    Returns:
        dict with current value, percentile, median, and distance from median
    """
    # Filter by commodity if specified
    if commodity:
        df = df[df['commodity'] == commodity]
        if hist_df is not None:
            hist_df = hist_df[hist_df['commodity'] == commodity]
    if hist_df is None:
        hist_df = df
    
    # Get current value
    current_data = df[(df['date'] == date) & (df['contract_month'] == contract_month)]
    
    if len(current_data) == 0:
        return None
        
    # Choose metric, allow fallback for dirty_vol if data looks like prices
    calc_metric = metric
    if metric not in current_data.columns:
        return None

    current = current_data[metric].values[0]
    display_current = current  # what we show in UI
    percentile_current = current  # what we use for percentile math

    if metric == 'dirty_vol' and (pd.isna(current) or current <= 0 or current > 200):
        if 'clean_vol' in current_data.columns:
            alt = current_data['clean_vol'].values[0]
            if not pd.isna(alt) and 0 < alt <= 200:
                calc_metric = 'clean_vol'
                current = alt
                display_current = alt
                percentile_current = alt

    # Discard obviously bad values (e.g., prices mistakenly in vol column)
    if pd.isna(current) or current <= 0 or current > 200:
        return None
    
    # Determine the current contract's month code (e.g., H, J, K)
    # Determine the current contract's month code (e.g., H, J, K)
    current_code = None
    cm_map = None
    try:
        cm_map = get_contract_month_codes(df, date, commodity if commodity else current_data.iloc[0]['commodity'])
        current_code = cm_map.get(contract_month)
    except Exception:
        current_code = None

    # Build historical distribution sorted by date, then filter to same month code, THEN apply lookback
    hist_base = hist_df[
        (hist_df['contract_month'] == contract_month) &
        (hist_df['date'] < date)
    ].sort_values('date')

    if current_code:
        code_cache = {}
        filtered_frames = []
        for d, sub in hist_base.groupby('date'):
            key = (commodity if commodity else sub.iloc[0]['commodity'], pd.to_datetime(d).normalize())
            if key not in code_cache:
                code_cache[key] = get_contract_month_codes(hist_df, d, commodity if commodity else sub.iloc[0]['commodity'])
            hist_code = code_cache[key].get(contract_month)
            if hist_code == current_code or hist_code is None:
                filtered_frames.append(sub)
        hist_base = pd.concat(filtered_frames) if filtered_frames else pd.DataFrame(columns=hist_base.columns)

    # For matched month code, use full history of that code; otherwise apply lookback window
    if current_code:
        hist = hist_base
    else:
        hist = hist_base.tail(lookback_days)
    
    if len(hist) == 0:
        return None
    
    # Calculate stats
    effective_metric = calc_metric if calc_metric in hist.columns else metric if metric in hist.columns else None
    if effective_metric is None:
        return None
    hist_values = hist[effective_metric].dropna()
    hist_values = hist_values[(hist_values > 0) & (hist_values <= 200)]
    # Optional: IV calendar distribution for dirty_vol when available
    if metric == 'dirty_vol':
        iv_hist = get_iv_calendar_hist(commodity if commodity else current_data.iloc[0]['commodity'], date, current_code)
        if iv_hist is not None and len(iv_hist) > 0:
            hist_values = iv_hist
    # Fallback: if dirty_vol history is contaminated (e.g., prices), retry with clean_vol
    if len(hist_values) == 0 and metric == 'dirty_vol' and 'clean_vol' in hist.columns:
        alt_vals = hist['clean_vol'].dropna()
        alt_vals = alt_vals[(alt_vals > 0) & (alt_vals <= 200)]
        if len(alt_vals) > 0:
            hist_values = alt_vals
            calc_metric = 'clean_vol'
    percentile = (hist_values < percentile_current).sum() / len(hist_values) * 100
    median = hist_values.median()
    distance = percentile_current - median
    
    return {
        'current': display_current,
        'percentile': percentile,
        'median': median,
        'distance_from_median': distance,
        'lookback_count': len(hist_values)
    }


def get_iv_term_structure(df, date, commodity='SOY'):
    """
    Get the full IV term structure for a given date
    
    Returns:
        DataFrame with contract months and their vols
    """
    term_structure = df[
        (df['date'] == date) & 
        (df['commodity'] == commodity)
    ].sort_values('contract_month')[
        ['contract_month', 'expiry', 'dirty_vol', 'clean_vol', 'fwd_vol', 'trading_dte']
    ].copy()
    
    return term_structure


def calculate_calendar_spread(df, date, near_month=1, far_month=2, metric='fwd_vol', commodity=None, hist_df=None):
    """
    Calculate the spread between two contract months
    
    Args:
        df: Historical dataframe
        date: Date to analyze
        near_month: Near contract month
        far_month: Far contract month
        metric: What to spread ('fwd_vol', 'clean_vol', etc.)
        commodity: Commodity name (e.g., 'SOY', 'CORN'). If None, uses all data.
        
    Returns:
        Spread value and historical percentile
    """
    # Filter by commodity if specified
    if commodity:
        df = df[df['commodity'] == commodity]
        if hist_df is not None:
            hist_df = hist_df[hist_df['commodity'] == commodity]
    if hist_df is None:
        hist_df = df

    # Guard: if the history source is missing the metric, fall back to df
    if metric not in hist_df.columns:
        if metric not in df.columns:
            return None
        hist_df = df
    
    near = df[(df['date'] == date) & (df['contract_month'] == near_month)][metric].values
    far = df[(df['date'] == date) & (df['contract_month'] == far_month)][metric].values
    
    if len(near) == 0 or len(far) == 0:
        return None
    
    current_spread = far[0] - near[0]
    
    # Get historical spreads
    dates = hist_df['date'].unique()
    spreads = []
    
    for d in dates:
        if d >= date:
            continue
        n = hist_df[(hist_df['date'] == d) & (hist_df['contract_month'] == near_month)][metric].values
        f = hist_df[(hist_df['date'] == d) & (hist_df['contract_month'] == far_month)][metric].values
        if len(n) > 0 and len(f) > 0 and not pd.isna(n[0]) and not pd.isna(f[0]):
            spreads.append(f[0] - n[0])
    
    if len(spreads) == 0:
        return None
    
    spreads = np.array(spreads)
    percentile = (spreads < current_spread).sum() / len(spreads) * 100
    median_spread = np.median(spreads)
    
    return {
        'current_spread': current_spread,
        'percentile': percentile,
        'median_spread': median_spread,
        'distance_from_median': current_spread - median_spread
    }


def get_skew_summary(df, date, contract_month, commodity=None, hist_df=None, skew_columns=None):
    """
    Get all skew points for a contract with historical context
    
    Args:
        df: Historical dataframe
        date: Date to analyze
        contract_month: Contract month
        commodity: Commodity name (e.g., 'SOY', 'CORN'). If None, uses all data.
    
    Returns:
        DataFrame with current vs historical for all skew points
    """
    # Default skew columns: prefer master schema, fall back to legacy names.
    if skew_columns is None:
        candidates = [
            ['skew_m1.5', 'skew_m0.5', 'skew_p0.5', 'skew_p1.5', 'skew_p3.0'],
            ['skew_neg15', 'skew_neg05', 'skew_pos05', 'skew_pos15', 'skew_pos3'],
        ]
        cols_available = set((hist_df.columns if hist_df is not None else df.columns))
        for cand in candidates:
            if all(c in cols_available for c in cand):
                skew_columns = cand
                break
    if skew_columns is None:
        return pd.DataFrame()
    
    results = []
    for skew in skew_columns:
        stats = calculate_percentile_rank(df, date, contract_month, skew, commodity=commodity, hist_df=hist_df)
        if stats:
            stats['strike'] = skew
            results.append(stats)
    
    return pd.DataFrame(results)


def forward_vol_vs_predicted(df, date, contract_month, predicted_rv):
    """
    Compare your forward vol calculation to your model's predicted realized vol
    
    Args:
        df: Historical dataframe
        date: Date to analyze
        contract_month: Contract month
        predicted_rv: Your model's prediction for realized vol
        
    Returns:
        dict with comparison metrics
    """
    data = df[(df['date'] == date) & (df['contract_month'] == contract_month)]
    
    if len(data) == 0:
        return None
    
    fwd_vol = data['fwd_vol'].values[0]
    
    if pd.isna(fwd_vol):
        return None
    
    difference = fwd_vol - predicted_rv
    ratio = fwd_vol / predicted_rv if predicted_rv != 0 else None
    
    return {
        'fwd_vol': fwd_vol,
        'predicted_rv': predicted_rv,
        'difference': difference,
        'ratio': ratio,
        'signal': 'LONG VOL' if difference > 0 else 'SHORT VOL'
    }


def append_daily_snapshot(df, new_data):
    """
    Append new daily data to your historical dataset
    
    Args:
        df: Existing historical dataframe
        new_data: Dict or DataFrame with today's data
        
    Returns:
        Updated dataframe
    """
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Ensure date columns are datetime
    new_data['date'] = pd.to_datetime(new_data['date'])
    if 'expiry' in new_data.columns:
        new_data['expiry'] = pd.to_datetime(new_data['expiry'])
    
    # Concatenate and remove duplicates (keep most recent)
    updated = pd.concat([df, new_data], ignore_index=True)
    updated = updated.sort_values('date').groupby(['date', 'contract_month']).last().reset_index()
    
    return updated


# Example usage functions for Streamlit dashboard
def get_dashboard_summary(df, date, commodity='SOY'):
    """
    Get all the key metrics for the main dashboard view
    
    Returns:
        dict with all dashboard metrics
    """
    summary = {
        'date': date,
        'commodity': commodity,
        'term_structure': get_iv_term_structure(df, date, commodity),
        'front_month_vol_stats': calculate_percentile_rank(df, date, 1, 'clean_vol', commodity=commodity),
        'front_month_skew': get_skew_summary(df, date, 1, commodity=commodity),
        'm1_m2_spread': calculate_calendar_spread(df, date, 1, 2, 'fwd_vol', commodity=commodity),
    }
    
    return summary


def calculate_cross_commodity_spread(df, date, commodity1, commodity2, contract_month=1, metric='clean_vol', hist_df=None):
    """
    Calculate IV spread between two commodities (e.g., SOY vs MEAL)
    
    Args:
        df: Historical dataframe
        date: Date to analyze
        commodity1: First commodity (e.g., 'SOY')
        commodity2: Second commodity (e.g., 'MEAL')
        contract_month: Which contract month to compare
        metric: What to compare ('clean_vol', 'dirty_vol', etc.)
        
    Returns:
        dict with current spread and historical percentile
    """
    c1_data = df[(df['date'] == date) & (df['commodity'] == commodity1) & (df['contract_month'] == contract_month)][metric].values
    c2_data = df[(df['date'] == date) & (df['commodity'] == commodity2) & (df['contract_month'] == contract_month)][metric].values
    
    if len(c1_data) == 0 or len(c2_data) == 0:
        return None
    
    current_spread = c2_data[0] - c1_data[0]
    
    # Get historical spreads
    if hist_df is None:
        hist_df = df
    if metric not in hist_df.columns:
        if metric not in df.columns:
            return None
        hist_df = df
    dates = hist_df['date'].unique()
    spreads = []
    
    for d in dates:
        if d >= date:
            continue
        c1 = hist_df[(hist_df['date'] == d) & (hist_df['commodity'] == commodity1) & (hist_df['contract_month'] == contract_month)][metric].values
        c2 = hist_df[(hist_df['date'] == d) & (hist_df['commodity'] == commodity2) & (hist_df['contract_month'] == contract_month)][metric].values
        
        if len(c1) > 0 and len(c2) > 0 and not pd.isna(c1[0]) and not pd.isna(c2[0]):
            spreads.append(c2[0] - c1[0])
    
    if len(spreads) == 0:
        return None
    
    spreads = np.array(spreads)
    percentile = (spreads < current_spread).sum() / len(spreads) * 100
    median_spread = np.median(spreads)
    
    return {
        'commodity1': commodity1,
        'commodity1_value': c1_data[0],
        'commodity2': commodity2,
        'commodity2_value': c2_data[0],
        'current_spread': current_spread,
        'percentile': percentile,
        'median_spread': median_spread,
        'distance_from_median': current_spread - median_spread
    }


def calculate_percentile_grid(df, date, metric='clean_vol', lookback_days=252,
                              commodities=None, max_months=8, hist_df=None):
    """
    Calculate percentile ranks for every commodity × contract month in one pass.

    Args:
        df: Historical dataframe
        date: Analysis date
        metric: Column to rank (e.g. 'clean_vol', 'dirty_vol', 'fwd_vol')
        lookback_days: Trading days for historical distribution
        commodities: List of commodities. None = all in data.
        max_months: Max contract months to include

    Returns:
        DataFrame with columns: commodity, contract_month, current, percentile, median
    """
    if commodities is None:
        commodities = sorted(df['commodity'].unique())
    if hist_df is None:
        hist_df = df

    results = []

    # Cache month-code mappings per (commodity, date) to avoid recomputing
    code_cache = {}

    for commodity in commodities:
        cdf = df[df['commodity'] == commodity]
        hist_cdf = hist_df[hist_df['commodity'] == commodity]

        for cm in range(1, max_months + 1):
            # Current value
            current_row = cdf[(cdf['date'] == date) & (cdf['contract_month'] == cm)]
            if len(current_row) == 0:
                continue
            calc_metric = metric
            current = current_row[metric].values[0]
            display_current = current
            percentile_current = current
            if (pd.isna(current) or current <= 0 or current > 200) and metric == 'dirty_vol':
                if 'clean_vol' in current_row.columns:
                    alt = current_row['clean_vol'].values[0]
                    if not pd.isna(alt) and 0 < alt <= 200:
                        calc_metric = 'clean_vol'
                        current = alt
                        display_current = alt
                        percentile_current = alt
            if pd.isna(current) or current <= 0 or current > 200:
                continue

            # Only enforce month-code matching for the front contract; others use lookback
            current_code = None
            hist_df_local = hist_cdf[
                (hist_cdf['contract_month'] == cm) & (hist_cdf['date'] < date)
            ].sort_values('date')

            if cm == 1:
                try:
                    today_map = get_contract_month_codes(cdf, date, commodity, max_months)
                    current_code = today_map.get(cm)
                except Exception:
                    current_code = None

                if current_code:
                    filtered = []
                    for d, sub in hist_df_local.groupby('date'):
                        key = (commodity, pd.to_datetime(d).normalize())
                        if key not in code_cache:
                            code_cache[key] = get_contract_month_codes(hist_cdf, d, commodity, max_months)
                        hist_code = code_cache[key].get(cm)
                        if hist_code == current_code or hist_code is None:
                            filtered.append(sub)
                    hist_df_local = pd.concat(filtered) if filtered else pd.DataFrame(columns=hist_df_local.columns)

            # If we have a month code match (front), keep full history for that code; others use lookback
            hist_source = hist_df_local if current_code else hist_df_local.tail(lookback_days)

            # Guard: if clean_vol not present in hist source, fall back to dirty_vol
            effective_metric = calc_metric if calc_metric in hist_source.columns else metric
            if effective_metric not in hist_source.columns:
                continue

            hist = hist_source[effective_metric].dropna()
            hist = hist[(hist > 0) & (hist <= 200)]
            if len(hist) == 0 and metric == 'dirty_vol' and 'clean_vol' in hist_source.columns:
                alt_hist = hist_source['clean_vol'].dropna()
                alt_hist = alt_hist[(alt_hist > 0) & (alt_hist <= 200)]
                if len(alt_hist) > 0:
                    hist = alt_hist
                    calc_metric = 'clean_vol'

            # Optional: use IV calendar history for dirty_vol if available (by month code + front code)
            if metric == 'dirty_vol':
                cm_code = None
                try:
                    cm_code = get_contract_month_codes(hist_cdf, date, commodity, max_months).get(cm, None)
                except Exception:
                    cm_code = None
                iv_hist = get_iv_calendar_hist(commodity, date, cm_code)
                if iv_hist is not None and len(iv_hist) > 0:
                    hist = iv_hist

            if len(hist) == 0:
                continue

            percentile = (hist < percentile_current).sum() / len(hist) * 100
            median = hist.median()
            
            results.append({
                'commodity': commodity,
                'contract_month': cm,
                'current': display_current,
                'percentile': percentile,
                'median': median,
                'distance': percentile_current - median,
                'lookback_count': len(hist),
            })

    return pd.DataFrame(results)


def load_verdad_predictions(filepath='verdad.7.xlsx', sheet='dashboard'):
    """
    Load predicted realized vol from the verdad workbook.

    The dashboard sheet has:
      Row 0: 'overall level', S, SM, BO, C, W, KW  (single RV per commodity)
      Row 1: values
      Rows 4-15: monthly breakdown by futures month code (F through Z)
              columns: month_code, S, SM, BO, C, W, KW

    Returns:
        overall: dict {commodity: predicted_rv}
        monthly: dict {commodity: {month_code: predicted_rv}}
    """
    dash = pd.read_excel(filepath, sheet_name=sheet, header=None)

    # Column mapping: verdad column names -> our commodity names
    col_map = {'S': 'SOY', 'SM': 'MEAL', 'BO': 'OIL', 'C': 'CORN', 'W': 'WHEAT', 'KW': 'KW'}

    # Read column headers from row 0 (skip first col which is label)
    headers = [str(dash.iloc[0, i]).strip() for i in range(1, dash.shape[1])]

    # Overall level (row 1)
    overall = {}
    for i, h in enumerate(headers):
        if h in col_map:
            val = dash.iloc[1, i + 1]
            if pd.notna(val):
                overall[col_map[h]] = float(val)

    # Monthly breakdown (rows 4-15)
    monthly = {c: {} for c in col_map.values()}
    for row_idx in range(4, 16):
        month_code = str(dash.iloc[row_idx, 0]).strip()
        if len(month_code) != 1:
            continue
        for i, h in enumerate(headers):
            if h in col_map:
                val = dash.iloc[row_idx, i + 1]
                if pd.notna(val):
                    monthly[col_map[h]][month_code] = float(val)

    return overall, monthly


@functools.lru_cache(maxsize=1)
def load_iv_calendar(filepath='IV calendar.xlsm'):
    """Load IV calendar workbook."""
    if not os.path.exists(filepath):
        return None
    try:
        xl = pd.ExcelFile(filepath)
        return xl
    except Exception:
        return None


def get_iv_calendar_hist(commodity, date, cm_code):
    """
    Get historical IV values from IV calendar (val column) for a commodity/month code,
    filtered to rows where front_contract matches the current front code for that date.
    """
    if cm_code is None:
        return None
    sheet_map = {
        'CORN': 'c data',
        'SOY': 's data',
        'MEAL': 'sm data',
        'WHEAT': 'w data',
        'OIL': None,  # no sheet provided
        'KW': None,   # no sheet provided
    }
    sheet = sheet_map.get(commodity)
    xl = load_iv_calendar()
    if xl is None or sheet is None or sheet not in xl.sheet_names:
        return None
    try:
        df = xl.parse(sheet_name=sheet)
        df['date_trade'] = pd.to_datetime(df['date_trade'])
    except Exception:
        return None

    # Determine current front code from historical df if available
    front_code = None
    try:
        # Approximate using the latest row in IV calendar
        front_code = df.sort_values('date_trade').iloc[-1]['front_contract']
    except Exception:
        front_code = None

    hist = df[
        (df['code_month'].astype(str).str.upper() == str(cm_code).upper()) &
        (df['val'].notna())
    ]
    if front_code:
        hist = hist[hist['front_contract'].astype(str).str.upper() == str(front_code).upper()]

    series = pd.to_numeric(hist['val'], errors='coerce')
    series = series.dropna()
    series = series[(series > 0) & (series <= 200)]
    return series if len(series) > 0 else None


def get_contract_month_codes(df, date, commodity, max_months=8):
    """
    Determine the futures month code for each contract_month ordinal position.

    Uses expiry dates when available: delivery_month = expiry_month + 1.
    Falls back to a default curve order when expiry is missing.

    Args:
        df: Historical dataframe with 'expiry' column
        date: Analysis date
        commodity: Commodity name
        max_months: Max ordinal positions to map

    Returns:
        dict {contract_month_int: month_code_str} e.g. {1: 'H', 2: 'J', 3: 'K', ...}
    """
    num_to_code = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }

    data = df[
        (df['date'] == date) & (df['commodity'] == commodity)
    ].sort_values('contract_month')

    mapping = {}
    for _, r in data.iterrows():
        cm = int(r['contract_month'])
        if cm > max_months:
            continue
        exp = r.get('expiry', None)
        if pd.notna(exp):
            try:
                exp_dt = pd.to_datetime(exp)
                delivery_month = (exp_dt.month % 12) + 1
                mapping[cm] = num_to_code[delivery_month]
            except:
                pass

    # If we got mappings from expiry, return them
    if len(mapping) >= max_months // 2:
        return mapping

    # Fallback: build a default curve from the current month
    # Standard ag options months by commodity
    default_months = {
        'SOY':   ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
        'MEAL':  ['F', 'H', 'K', 'N', 'Q', 'U', 'V', 'Z'],
        'OIL':   ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
        'CORN':  ['H', 'K', 'N', 'U', 'Z'],
        'WHEAT': ['H', 'K', 'N', 'U', 'Z'],
        'KW':    ['H', 'K', 'N', 'U', 'Z'],
    }

    code_to_num = {v: k for k, v in num_to_code.items()}

    months_list = default_months.get(commodity, list(num_to_code.values()))
    current_month = pd.to_datetime(date).month

    # Find the first active month code at or after current month
    # Build an ordered list starting from the next available month
    month_nums = sorted([code_to_num[c] for c in months_list])
    # Double the list to handle wrap-around
    extended = month_nums + [m + 12 for m in month_nums]

    start_idx = 0
    for i, m in enumerate(extended):
        actual_m = m if m <= 12 else m - 12
        if actual_m >= current_month:
            start_idx = i
            break

    for cm in range(1, max_months + 1):
        if cm not in mapping:
            idx = start_idx + cm - 1
            if idx < len(extended):
                actual_m = extended[idx] if extended[idx] <= 12 else extended[idx] - 12
                mapping[cm] = num_to_code[actual_m]

    return mapping


def calculate_power_grid(df, date, predicted_rv_dict, commodities=None, max_months=8,
                         monthly_predictions=None):
    """
    Calculate the Power Grid: (fwd_vol - predicted_rv) / fwd_vol for each
    commodity and contract month.

    Args:
        df: Historical dataframe
        date: Date to analyze
        predicted_rv_dict: Dict of {commodity: predicted_rv_value}
                          Used as fallback when monthly_predictions is not available.
                          e.g. {'SOY': 16.3, 'CORN': 19.0}
        commodities: List of commodities to include. If None, uses all in predicted_rv_dict.
        max_months: Maximum contract months to display
        monthly_predictions: Dict of {commodity: {month_code: predicted_rv}}
                            e.g. {'SOY': {'H': 13.8, 'K': 14.7, ...}}
                            When provided, maps each contract_month to its month code
                            and uses the per-month predicted RV.

    Returns:
        power_df: DataFrame with commodities as rows, contract months as columns,
                  values = (fwd_vol - predicted_rv) / fwd_vol as percentage
        metadata: dict with 'fwd_vols', 'pred_rvs', 'month_codes' DataFrames
    """
    if commodities is None:
        commodities = list(predicted_rv_dict.keys())

    results = {}
    fwd_vols = {}
    pred_rvs = {}
    month_codes = {}

    for commodity in commodities:
        commodity_data = df[
            (df['date'] == date) & (df['commodity'] == commodity)
        ].sort_values('contract_month')

        # Get month code mapping if we have monthly predictions
        cm_to_code = {}
        if monthly_predictions and commodity in monthly_predictions:
            cm_to_code = get_contract_month_codes(df, date, commodity, max_months)

        row = {}
        fv_row = {}
        pr_row = {}
        mc_row = {}

        for _, r in commodity_data.iterrows():
            cm = int(r['contract_month'])
            if cm > max_months:
                continue

            fwd_vol = r['fwd_vol']
            if pd.isna(fwd_vol) or fwd_vol == 0:
                continue

            # Determine predicted RV for this position
            pred_rv = None
            code = cm_to_code.get(cm, None)

            if code and monthly_predictions and commodity in monthly_predictions:
                pred_rv = monthly_predictions[commodity].get(code, None)

            # Fallback to overall level
            if pred_rv is None and commodity in predicted_rv_dict:
                pred_rv = predicted_rv_dict[commodity]

            if pred_rv is not None and pred_rv > 0:
                power = (fwd_vol - pred_rv) / fwd_vol * 100  # as percentage
                row[f'M{cm}'] = power
                fv_row[f'M{cm}'] = fwd_vol
                pr_row[f'M{cm}'] = pred_rv
                mc_row[f'M{cm}'] = code if code else '?'

        if row:
            results[commodity] = row
            fwd_vols[commodity] = fv_row
            pred_rvs[commodity] = pr_row
            month_codes[commodity] = mc_row

    if not results:
        return pd.DataFrame(), {'fwd_vols': pd.DataFrame(), 'pred_rvs': pd.DataFrame(), 'month_codes': pd.DataFrame()}

    power_df = pd.DataFrame(results).T
    # Sort columns by contract month number
    month_cols = sorted(power_df.columns, key=lambda x: int(x[1:]))
    power_df = power_df[month_cols]

    metadata = {
        'fwd_vols': pd.DataFrame(fwd_vols).T.reindex(power_df.index),
        'pred_rvs': pd.DataFrame(pred_rvs).T.reindex(power_df.index),
        'month_codes': pd.DataFrame(month_codes).T.reindex(power_df.index),
    }
    # Align metadata columns
    for key in metadata:
        if len(metadata[key].columns) > 0:
            metadata[key] = metadata[key].reindex(columns=month_cols)

    return power_df, metadata


if __name__ == "__main__":
    # Test the functions with all commodities
    df = load_historical_data()
    latest_date = df['date'].max()
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Date range: {df['date'].min().date()} to {latest_date.date()}")
    print(f"Commodities: {', '.join(df['commodity'].unique())}")
    print("\n" + "="*60)
    
    # Test with CORN
    print("Testing with CORN:")
    stats = calculate_percentile_rank(df, latest_date, 1, 'clean_vol', commodity='CORN')
    if stats:
        print(f"Front month clean vol: {stats['current']:.2f}%")
        print(f"Percentile rank: {stats['percentile']:.1f}%")
        print(f"Distance from median: {stats['distance_from_median']:.2f}%")
    
    print("\n" + "="*60)
    
    # Test calendar spread
    spread = calculate_calendar_spread(df, latest_date, 1, 2, 'fwd_vol', commodity='SOY')
    if spread:
        print(f"SOY M1-M2 fwd vol spread: {spread['current_spread']:.2f}%")
        print(f"Historical percentile: {spread['percentile']:.1f}%")
    
    print("\n" + "="*60)
    
    # Test cross-commodity spread
    soy_iv = df[(df['date'] == latest_date) & (df['commodity'] == 'SOY') & (df['contract_month'] == 1)]['clean_vol'].values
    meal_iv = df[(df['date'] == latest_date) & (df['commodity'] == 'MEAL') & (df['contract_month'] == 1)]['clean_vol'].values
    
    if len(soy_iv) > 0 and len(meal_iv) > 0:
        print(f"SOY vs MEAL IV spread: {soy_iv[0]:.2f}% vs {meal_iv[0]:.2f}% = {meal_iv[0] - soy_iv[0]:.2f}%")
    
    print("\n" + "="*60)
    
    # Test dashboard summary
    summary = get_dashboard_summary(df, latest_date, 'WHEAT')
    print("\nWHEAT term structure:")
    print(summary['term_structure'][['contract_month', 'clean_vol', 'fwd_vol']].head())
