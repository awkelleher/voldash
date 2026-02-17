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


def calculate_hloc_volatility(df, commodity, windows=[5, 10, 20, 50, 100, 200],
                               n_contracts=2, as_of_date=None):
    """
    Calculate HLOC (High-Low-Open-Close) realized volatility per contract.

    Uses the modified Garman-Klass estimator:
        HLOC_var = ln(O/C_prev)^2 + 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2

    Annualized HLOC vol = sqrt(mean(HLOC_var over N days)) * sqrt(252)

    Args:
        df: Price dataframe with date, commodity, contract_code, high, low, open, close
        commodity: Commodity code (e.g. 'SOY')
        windows: List of lookback windows in trading days
        n_contracts: Number of nearest contracts to return (default 2)
        as_of_date: Date to compute as of (default: latest available)

    Returns:
        DataFrame with columns: contract_code, hloc_5d, hloc_10d, ... (annualized %)
        Also includes daily_ret columns for ratio calculation.
    """
    comm_df = df[df['commodity'] == commodity].copy()
    if len(comm_df) == 0:
        return pd.DataFrame()

    comm_df = comm_df.sort_values(['contract_code', 'date'])

    if as_of_date is None:
        as_of_date = comm_df['date'].max()
    else:
        as_of_date = pd.to_datetime(as_of_date)

    # Determine the contract sort order
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

    # Find active contracts as of the date
    active_on_date = comm_df[comm_df['date'] == as_of_date]['contract_code'].unique()
    if len(active_on_date) == 0:
        # Fall back to nearest prior date
        prior = comm_df[comm_df['date'] <= as_of_date]['date'].max()
        if pd.isna(prior):
            return pd.DataFrame()
        active_on_date = comm_df[comm_df['date'] == prior]['contract_code'].unique()

    # Sort contracts and take nearest n
    sorted_contracts = sorted(active_on_date, key=contract_sort_key)[:n_contracts]

    max_window = max(windows)
    results = []

    for contract in sorted_contracts:
        cdf = comm_df[comm_df['contract_code'] == contract].copy()
        cdf = cdf.sort_values('date').reset_index(drop=True)

        # Need at least 2 rows
        if len(cdf) < 2:
            continue

        # Only keep data up to as_of_date
        cdf = cdf[cdf['date'] <= as_of_date]
        if len(cdf) < 2:
            continue

        # Calculate HLOC variance per day
        # HLOC_var = ln(O/C_prev)^2 + 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2
        close_prev = cdf['close'].shift(1)
        ln2_coeff = 2 * np.log(2) - 1  # ~0.3863

        overnight = np.log(cdf['open'] / close_prev)
        hl_range = np.log(cdf['high'] / cdf['low'])
        co_return = np.log(cdf['close'] / cdf['open'])

        cdf['hloc_var'] = overnight**2 + 0.5 * hl_range**2 - ln2_coeff * co_return**2

        # Also calculate daily log return (close-to-close) for ratio
        cdf['daily_ret'] = np.log(cdf['close'] / close_prev)

        # Take the most recent max_window+1 rows (need shift, so +1)
        cdf = cdf.dropna(subset=['hloc_var']).tail(max_window)

        rec = {'contract_code': contract}

        for w in windows:
            recent = cdf.tail(w)
            if len(recent) >= min(w, 3):  # Need at least 3 obs or full window
                # HLOC vol: sqrt(mean(hloc_var)) * sqrt(252) * 100
                hloc_mean = recent['hloc_var'].mean()
                if hloc_mean > 0:
                    rec[f'hloc_{w}d'] = np.sqrt(hloc_mean) * np.sqrt(252) * 100
                else:
                    rec[f'hloc_{w}d'] = np.nan

                # Daily ret vol: RMS (root mean square, not demeaned) * sqrt(252) * 100
                # This matches the Excel hvol() function: sqrt(mean(ret^2)) * sqrt(252)
                rms = np.sqrt(np.mean(recent['daily_ret']**2))
                if rms > 0:
                    rec[f'rv_{w}d'] = rms * np.sqrt(252) * 100
                else:
                    rec[f'rv_{w}d'] = np.nan
            else:
                rec[f'hloc_{w}d'] = np.nan
                rec[f'rv_{w}d'] = np.nan

        results.append(rec)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


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
