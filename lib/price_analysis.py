"""
Price Analysis Module

Calculates realized volatility and other metrics from futures prices.
Handles contract rolls and continuous pricing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_price_data(filepath='data/all_commodity_prices.csv'):
    """Load historical price data"""
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values(['commodity', 'contract_code', 'date'])
    return df


def create_continuous_series(df, commodity, method='front_month'):
    """
    Create a continuous price series for a commodity
    
    Args:
        df: Price dataframe
        commodity: Commodity name (e.g., 'SOY')
        method: 'front_month' or 'volume_weighted' (for now just front month)
        
    Returns:
        DataFrame with continuous prices
    """
    # Filter to this commodity
    comm_df = df[df['commodity'] == commodity].copy()
    
    # Get all unique dates
    dates = sorted(comm_df['date'].unique())
    
    continuous_prices = []
    
    for date in dates:
        # Get all contracts for this date
        day_data = comm_df[comm_df['date'] == date]
        
        if len(day_data) == 0:
            continue
        
        # Sort by contract code to get front month
        # Contract codes: F, H, K, N, Q, U, X
        # Month order: F(Jan), G(Feb), H(Mar), K(May), N(Jul), Q(Aug), U(Sep), X(Nov)
        contract_order = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 
                         'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
        
        # Extract month and year from contract codes
        def parse_contract(code):
            if pd.isna(code) or len(code) < 2:
                return (9999, 99)  # Sort to end if invalid
            month_code = code[0]
            year = int('20' + code[1:]) if len(code) >= 2 else 2099
            month_rank = contract_order.get(month_code, 99)
            return (year, month_rank)
        
        day_data['contract_sort'] = day_data['contract_code'].apply(parse_contract)
        day_data = day_data.sort_values('contract_sort')
        
        # Take front month (first valid contract)
        front_month = day_data.iloc[0]
        
        continuous_prices.append({
            'date': date,
            'commodity': commodity,
            'contract_code': front_month['contract_code'],
            'close': front_month['close'],
            'high': front_month['high'],
            'low': front_month['low'],
            'open': front_month['open']
        })
    
    continuous_df = pd.DataFrame(continuous_prices)
    continuous_df = continuous_df.sort_values('date').reset_index(drop=True)
    
    # Mark contract rolls
    continuous_df['contract_rolled'] = continuous_df['contract_code'] != continuous_df['contract_code'].shift(1)
    
    return continuous_df


def calculate_log_returns(prices):
    """
    Calculate log returns from price series
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of log returns
    """
    return np.log(prices / prices.shift(1))


def calculate_realized_volatility(continuous_df, windows=[5, 10, 15, 20]):
    """
    Calculate realized volatility for multiple windows
    
    Args:
        continuous_df: Continuous price series
        windows: List of window sizes in days
        
    Returns:
        DataFrame with realized vol columns added
    """
    df = continuous_df.copy()
    
    # Calculate log returns
    df['log_return'] = calculate_log_returns(df['close'])
    
    # For each window, calculate realized vol
    for window in windows:
        # Rolling standard deviation of log returns
        df[f'rv_{window}d'] = df['log_return'].rolling(window=window).std()
        
        # Annualize (multiply by sqrt(252) for trading days)
        df[f'rv_{window}d_annualized'] = df[f'rv_{window}d'] * np.sqrt(252) * 100  # Convert to percentage
    
    return df


def get_realized_vol_summary(df, commodity, date=None, windows=[5, 10, 15, 20]):
    """
    Get realized vol summary for a commodity on a specific date
    
    Args:
        df: Price dataframe
        commodity: Commodity name
        date: Date to analyze (if None, uses latest)
        windows: Window sizes
        
    Returns:
        Dict with realized vol values
    """
    # Create continuous series
    continuous = create_continuous_series(df, commodity)
    
    # Calculate realized vol
    with_rv = calculate_realized_volatility(continuous, windows)
    
    # Get data for specific date
    if date is None:
        date = with_rv['date'].max()
    
    date_data = with_rv[with_rv['date'] == date]
    
    if len(date_data) == 0:
        return None
    
    row = date_data.iloc[0]
    
    result = {
        'date': row['date'],
        'commodity': row['commodity'],
        'contract_code': row['contract_code'],
        'close': row['close']
    }
    
    for window in windows:
        result[f'rv_{window}d'] = row[f'rv_{window}d_annualized']
    
    return result


def calculate_all_commodities_rv(df, commodities=['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT', 'KW'], 
                                  windows=[5, 10, 15, 20]):
    """
    Calculate realized volatility for all commodities
    
    Args:
        df: Price dataframe
        commodities: List of commodities
        windows: Window sizes
        
    Returns:
        DataFrame with all continuous prices and realized vols
    """
    all_results = []
    
    for commodity in commodities:
        print(f"Processing {commodity}...")
        
        # Create continuous series
        continuous = create_continuous_series(df, commodity)
        
        # Calculate realized vol
        with_rv = calculate_realized_volatility(continuous, windows)
        
        all_results.append(with_rv)
    
    # Combine all
    combined = pd.concat(all_results, ignore_index=True)
    
    return combined


def analyze_contract_rolls(continuous_df):
    """
    Analyze when contracts roll and the price impact
    
    Args:
        continuous_df: Continuous price series with 'contract_rolled' flag
        
    Returns:
        DataFrame with roll analysis
    """
    rolls = continuous_df[continuous_df['contract_rolled']].copy()
    
    if len(rolls) == 0:
        return pd.DataFrame()
    
    # Calculate price change at roll
    rolls['price_change'] = rolls['close'] - rolls['close'].shift(1)
    rolls['price_change_pct'] = rolls['price_change'] / rolls['close'].shift(1) * 100
    
    return rolls[['date', 'commodity', 'contract_code', 'close', 'price_change', 'price_change_pct']]


def detect_roll_schedule(df, commodity):
    """
    Detect the typical contract roll schedule for a commodity
    
    Args:
        df: Price dataframe
        commodity: Commodity name
        
    Returns:
        Summary of when contracts typically roll
    """
    continuous = create_continuous_series(df, commodity)
    rolls = continuous[continuous['contract_rolled']]
    
    # Group by month to see pattern
    rolls['month'] = rolls['date'].dt.month
    roll_counts = rolls.groupby('month').size()
    
    print(f"\n{commodity} Roll Schedule:")
    print(f"Typical roll months: {roll_counts.index.tolist()}")
    print(f"Total rolls: {len(rolls)}")
    
    return roll_counts


if __name__ == "__main__":
    print("="*70)
    print("PRICE ANALYSIS - REALIZED VOLATILITY")
    print("="*70)
    
    # Load data
    print("\nLoading price data...")
    df = load_price_data()
    print(f"✓ Loaded {len(df):,} records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Commodities: {', '.join(sorted(df['commodity'].unique()))}")
    
    # Test with SOY
    print("\n" + "="*70)
    print("CALCULATING REALIZED VOL FOR ALL COMMODITIES")
    print("="*70)
    
    all_rv = calculate_all_commodities_rv(df)
    
    print(f"\n✓ Calculated realized vol for {len(all_rv):,} observations")
    
    # Save results
    all_rv.to_csv('data/continuous_prices_with_rv.csv', index=False)
    print(f"✓ Saved to data/continuous_prices_with_rv.csv")
    
    # Show latest values
    print("\n" + "="*70)
    print("LATEST REALIZED VOLATILITY (Annualized %)")
    print("="*70)
    
    latest_date = all_rv['date'].max()
    latest = all_rv[all_rv['date'] == latest_date]
    
    print(f"\nDate: {latest_date.date()}\n")
    print(f"{'Commodity':<10} {'Contract':<8} {'Price':<10} {'5D RV':<8} {'10D RV':<8} {'15D RV':<8} {'20D RV':<8}")
    print("-"*70)
    
    for _, row in latest.iterrows():
        print(f"{row['commodity']:<10} {row['contract_code']:<8} {row['close']:<10.2f} "
              f"{row['rv_5d_annualized']:<8.2f} {row['rv_10d_annualized']:<8.2f} "
              f"{row['rv_15d_annualized']:<8.2f} {row['rv_20d_annualized']:<8.2f}")
    
    # Analyze contract rolls
    print("\n" + "="*70)
    print("CONTRACT ROLL ANALYSIS")
    print("="*70)
    
    for commodity in ['SOY', 'CORN', 'WHEAT']:
        detect_roll_schedule(df, commodity)
