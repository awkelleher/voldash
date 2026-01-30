"""
Historical Vol & Skew Analysis Module

This module provides functions to work with your cleaned historical volatility and skew data.
Replace the messy Excel formulas with clean Python functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_historical_data(filepath='historical_vol_skew_all_commodities.csv'):
    """Load and prepare historical vol/skew data"""
    df = pd.read_csv(filepath, parse_dates=['date', 'expiry'])
    return df


def calculate_percentile_rank(df, date, contract_month, metric, lookback_days=252, commodity=None):
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
    
    # Get current value
    current_data = df[(df['date'] == date) & (df['contract_month'] == contract_month)]
    
    if len(current_data) == 0:
        return None
        
    current = current_data[metric].values[0]
    
    if pd.isna(current):
        return None
    
    # Get historical distribution
    hist = df[
        (df['contract_month'] == contract_month) & 
        (df['date'] < date)
    ].tail(lookback_days)
    
    if len(hist) == 0:
        return None
    
    # Calculate stats
    hist_values = hist[metric].dropna()
    percentile = (hist_values < current).sum() / len(hist_values) * 100
    median = hist_values.median()
    distance = current - median
    
    return {
        'current': current,
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


def calculate_calendar_spread(df, date, near_month=1, far_month=2, metric='fwd_vol', commodity=None):
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
    
    near = df[(df['date'] == date) & (df['contract_month'] == near_month)][metric].values
    far = df[(df['date'] == date) & (df['contract_month'] == far_month)][metric].values
    
    if len(near) == 0 or len(far) == 0:
        return None
    
    current_spread = far[0] - near[0]
    
    # Get historical spreads
    dates = df['date'].unique()
    spreads = []
    
    for d in dates:
        if d >= date:
            continue
        n = df[(df['date'] == d) & (df['contract_month'] == near_month)][metric].values
        f = df[(df['date'] == d) & (df['contract_month'] == far_month)][metric].values
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


def get_skew_summary(df, date, contract_month, commodity=None):
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
    skew_cols = ['skew_neg15', 'skew_neg05', 'skew_pos05', 'skew_pos15', 'skew_pos3']
    
    results = []
    for skew in skew_cols:
        stats = calculate_percentile_rank(df, date, contract_month, skew, commodity=commodity)
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


def calculate_cross_commodity_spread(df, date, commodity1, commodity2, contract_month=1, metric='clean_vol'):
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
    dates = df['date'].unique()
    spreads = []
    
    for d in dates:
        if d >= date:
            continue
        c1 = df[(df['date'] == d) & (df['commodity'] == commodity1) & (df['contract_month'] == contract_month)][metric].values
        c2 = df[(df['date'] == d) & (df['commodity'] == commodity2) & (df['contract_month'] == contract_month)][metric].values
        
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
