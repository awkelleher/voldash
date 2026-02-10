"""
Variance Ratio Calculations

Calculates historical variance ratios between futures contracts.
Variance Ratio = Variance(Contract A) / Variance(Contract B)

Used for relative value analysis across the futures curve.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import functools
import os


# Futures month codes in calendar order
MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

# Soybean futures months (F, H, K, N, Q, U, X)
SOY_MONTHS = ['F', 'H', 'K', 'N', 'Q', 'U', 'X']

@functools.lru_cache(maxsize=1)
def load_month_mapping(path: str = 'mapping.csv') -> pd.DataFrame:
    """Load the options->futures mapping per commodity."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['OPTIONS', 'FUTURES', 'COMMODITY', 'EXPIRY_MONTH'])
    df = pd.read_csv(path)
    df['OPTIONS'] = df['OPTIONS'].astype(str).str.upper()
    df['FUTURES'] = df['FUTURES'].astype(str).str.upper()
    df['COMMODITY'] = df['COMMODITY'].astype(str).str.upper()
    return df


def options_to_futures(option_code: str, commodity: str, mapping_path: str = 'mapping.csv') -> str:
    mapping = load_month_mapping(mapping_path)
    subset = mapping[(mapping['OPTIONS'] == option_code.upper()) & (mapping['COMMODITY'] == commodity.upper())]
    if len(subset) == 0:
        return option_code.upper()  # fallback: same code
    return subset.iloc[0]['FUTURES']


def futures_month_sequence(commodity: str, mapping_path: str = 'mapping.csv') -> list:
    """
    Return ordered futures month codes for a commodity based on mapping expiry order.
    """
    mapping = load_month_mapping(mapping_path)
    subset = mapping[mapping['COMMODITY'] == commodity.upper()].copy()
    if 'EXPIRY_MONTH' in subset.columns:
        subset = subset.sort_values('EXPIRY_MONTH')
    months = subset['FUTURES'].unique().tolist()
    # Fallback to full list if empty
    return months if months else MONTH_CODES


def get_futures_curve_for_front_month(front_month: str, commodity: str = 'SOY') -> list:
    """
    Get the ordered list of futures months on the curve when a given month is front.

    Args:
        front_month: The front futures month code (e.g., 'H' for March)
        commodity: Commodity name (default 'SOY')

    Returns:
        List of month codes in curve order starting from front month
    """
    months = futures_month_sequence(commodity)

    # Find starting position
    if front_month not in months:
        raise ValueError(f"Month {front_month} not in {commodity} futures months")

    start_idx = months.index(front_month)

    # Create curve: from front month through the cycle, then wrap around
    curve = months[start_idx:] + months[:start_idx]

    # Extend to get multiple years on the curve (typically need ~12 contracts)
    extended_curve = curve + curve + curve

    return extended_curve[:12]  # Return first 12 contracts


def get_contract_year(date: pd.Timestamp, month_code: str, position_on_curve: int) -> int:
    """
    Determine the year for a contract given the date and its position on the curve.

    Args:
        date: Current date
        month_code: Contract month code (e.g., 'H', 'K')
        position_on_curve: 0 = front month, 1 = second month, etc.

    Returns:
        Year (2-digit) for the contract
    """
    current_year = date.year % 100  # 2-digit year
    current_month = date.month

    # Month number for the contract
    month_num = MONTH_CODES.index(month_code) + 1

    # If contract month is before current month, it's next year
    # But we also need to account for position on curve
    if position_on_curve == 0:
        # Front month
        if month_num < current_month:
            return current_year + 1
        else:
            return current_year
    else:
        # For deferred months, we need to track year rollovers
        # This is simplified - in reality need to track through the curve
        year = current_year
        prev_month_num = MONTH_CODES.index(get_futures_curve_for_front_month('H')[0]) + 1

        for i in range(position_on_curve + 1):
            month = get_futures_curve_for_front_month('H')[i]
            m_num = MONTH_CODES.index(month) + 1
            if i > 0 and m_num < prev_month_num:
                year += 1
            prev_month_num = m_num

        return year % 100


def identify_front_month_periods(prices_df: pd.DataFrame,
                                  front_options_month: str,
                                  commodity: str = 'SOY') -> pd.DataFrame:
    """
    Identify date ranges when a given options month was the front month.

    For H options (on H futures), front month period is roughly mid-Jan to late Feb.

    Args:
        prices_df: DataFrame with columns ['date', 'contract_code', 'close', 'commodity']
        front_options_month: Options month code (e.g., 'H')
        commodity: Commodity name

    Returns:
        DataFrame with date ranges and year for each front month period
    """
    # Get the underlying futures month from mapping
    front_futures_month = options_to_futures(front_options_month, commodity)

    # Filter to commodity
    df = prices_df[prices_df['commodity'] == commodity].copy()

    # Extract month and year from contract code
    df['contract_month'] = df['contract_code'].str[0]
    df['contract_year'] = df['contract_code'].str[1:].astype(int)

    # Find all instances of the front futures month
    front_contracts = df[df['contract_month'] == front_futures_month]['contract_code'].unique()

    periods = []

    for contract in sorted(front_contracts):
        year = int(contract[1:])

        # Get date range for this contract
        contract_dates = df[df['contract_code'] == contract]['date'].sort_values()

        if len(contract_dates) == 0:
            continue

        # Front month period: when this contract has highest volume / is nearest expiry
        # Approximate: 2 months before expiry month
        # For H (March), front month period is roughly Jan-Feb
        # For K (May), front month period is roughly Mar-Apr
        # etc.

        expiry_month = MONTH_CODES.index(front_futures_month) + 1  # 1-12

        # Front month starts ~2 months before expiry
        start_month = expiry_month - 2
        start_year = 2000 + year
        if start_month <= 0:
            start_month += 12
            start_year -= 1

        # Front month ends at expiry (roughly 3rd Friday of expiry month)
        end_month = expiry_month
        end_year = 2000 + year

        period_start = pd.Timestamp(year=start_year, month=start_month, day=1)
        period_end = pd.Timestamp(year=end_year, month=end_month, day=28)

        # Filter to actual trading dates in our data
        mask = (contract_dates >= period_start) & (contract_dates <= period_end)
        actual_dates = contract_dates[mask]

        if len(actual_dates) > 0:
            periods.append({
                'contract': contract,
                'year': year,
                'start_date': actual_dates.min(),
                'end_date': actual_dates.max(),
                'trading_days': len(actual_dates)
            })

    return pd.DataFrame(periods)


@functools.lru_cache(maxsize=32)
def _load_prices(path: str = 'all_commodity_prices.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df


@functools.lru_cache(maxsize=64)
def _filtered_prices(path: str, commodity: str) -> pd.DataFrame:
    prices = _load_prices(path)
    return prices[prices['commodity'] == commodity].copy()


@functools.lru_cache(maxsize=128)
def _cached_daily_returns(path: str, commodity: str) -> pd.DataFrame:
    prices_df = _filtered_prices(path, commodity)
    returns_df = calculate_daily_returns(prices_df, commodity)
    return returns_df


def calculate_daily_returns(prices_df: pd.DataFrame, commodity: str = 'SOY') -> pd.DataFrame:
    """
    Calculate daily log returns for each contract.

    Args:
        prices_df: DataFrame with columns ['date', 'contract_code', 'close', 'commodity']
        commodity: Commodity name

    Returns:
        DataFrame with daily returns added
    """
    df = prices_df[prices_df['commodity'] == commodity].copy()
    df = df.sort_values(['contract_code', 'date'])

    # Calculate log returns
    df['return'] = df.groupby('contract_code')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    return df


def calculate_realized_variance(returns_df: pd.DataFrame,
                                 contract_code: str,
                                 start_date: pd.Timestamp,
                                 end_date: pd.Timestamp) -> float:
    """
    Calculate realized variance for a contract over a date range.

    Realized Variance = mean of squared returns (annualized)

    Args:
        returns_df: DataFrame with returns
        contract_code: Contract to calculate variance for
        start_date: Start of period
        end_date: End of period

    Returns:
        Annualized realized variance (as a decimal, not percentage)
    """
    mask = (
        (returns_df['contract_code'] == contract_code) &
        (returns_df['date'] >= start_date) &
        (returns_df['date'] <= end_date)
    )

    returns = returns_df.loc[mask, 'return'].dropna()

    if len(returns) < 5:  # Need minimum observations
        return np.nan

    # Realized variance = sum of squared returns
    # Annualize by multiplying by 252 (trading days)
    realized_var = (returns ** 2).mean() * 252

    return realized_var


def build_variance_ratio_matrix(prices_df: pd.DataFrame,
                                 front_options_month: str,
                                 commodity: str = 'SOY',
                                 num_contracts: int = 12,
                                 lookback_years: int = None) -> pd.DataFrame:
    """
    Build the variance ratio matrix for when a given options month is front.

    Matrix structure:
    - Rows: denominator contract (the contract you're dividing BY)
    - Columns: position on curve (1FM, 2FM, 3FM, etc.)
    - Cell value: Variance(column contract) / Variance(row contract)

    Args:
        prices_df: DataFrame with price data
        front_options_month: Options month code (e.g., 'H')
        commodity: Commodity name
        num_contracts: Number of contracts on the curve to include
        lookback_years: Number of years to include (None = all available)

    Returns:
        DataFrame with variance ratio matrix
    """
    # Get the futures curve order using mapping
    front_futures_month = options_to_futures(front_options_month, commodity)
    curve = get_futures_curve_for_front_month(front_futures_month, commodity)[:num_contracts]

    # Identify historical front month periods
    periods = identify_front_month_periods(prices_df, front_options_month, commodity)

    if len(periods) == 0:
        print(f"No historical periods found for {front_options_month} options")
        return pd.DataFrame()

    # Apply lookback filter if specified
    if lookback_years is not None and lookback_years > 0:
        # Keep only the most recent N years
        periods = periods.sort_values('year', ascending=False).head(lookback_years)

    # Calculate returns (cached)
    returns_df = _cached_daily_returns('all_commodity_prices.csv', commodity)

    # For each historical period, calculate variance for each contract on the curve
    all_variances = []

    for _, period in periods.iterrows():
        front_contract = period['contract']
        front_year = period['year']
        start_date = period['start_date']
        end_date = period['end_date']

        period_variances = {'period': front_contract}

        # Build contract codes for each position on the curve
        year = front_year
        prev_month_idx = MONTH_CODES.index(curve[0])

        for pos, month in enumerate(curve):
            month_idx = MONTH_CODES.index(month)

            # Year rolls over when month index decreases
            if pos > 0 and month_idx < prev_month_idx:
                year += 1
            prev_month_idx = month_idx

            contract_code = f"{month}{year:02d}"

            # Calculate variance for this contract during the period
            var = calculate_realized_variance(returns_df, contract_code, start_date, end_date)
            period_variances[f'{pos+1}FM'] = var
            period_variances[f'{pos+1}FM_code'] = contract_code

        all_variances.append(period_variances)

    variances_df = pd.DataFrame(all_variances)

    # Reduce to positions that actually have variance data (avoid empty/missing months)
    valid_positions = [p for p in range(num_contracts) if variances_df.get(f'{p+1}FM', pd.Series(dtype=float)).notna().any()]
    if len(valid_positions) == 0:
        return pd.DataFrame()

    # Remap curve and variances_df to only valid positions
    curve = [curve[p] for p in valid_positions]
    num_contracts = len(valid_positions)

    def remap_row(row):
        new_row = {'period': row['period']}
        for new_idx, old_pos in enumerate(valid_positions):
            new_row[f'{new_idx+1}FM'] = row.get(f'{old_pos+1}FM', np.nan)
            new_row[f'{new_idx+1}FM_code'] = row.get(f'{old_pos+1}FM_code', None)
        return new_row

    variances_df = pd.DataFrame([remap_row(r) for _, r in variances_df.iterrows()])

    # Calculate variance ratios for each period, then average the ratios
    # This is more accurate than averaging variances then computing ratio
    #
    # Matrix structure:
    # - Row = option month (maps to a futures contract)
    # - Column = denominator position on curve (VarRat1 = 1FM = H, VarRat2 = 2FM = K, etc.)
    # - Cell = row's futures variance / column position's futures variance
    #
    # Example when H is front:
    # - H26 VarRat1 = H/H = 1.00
    # - H26 VarRat4 = Q/H (row is H=1FM, but we want 4FM variance / 1FM variance)
    # - Q26 VarRat1 = Q/H (row is Q=4FM variance / 1FM variance)
    # - Q26 VarRat4 = Q/Q = 1.00
    #
    # So the cell is: row_position's variance / column_position's variance
    all_ratios = []

    for _, row in variances_df.iterrows():
        period_ratios = {'period': row['period']}

        for row_pos in range(num_contracts):
            row_var = row.get(f'{row_pos+1}FM', np.nan)

            for col_pos in range(num_contracts):
                col_var = row.get(f'{col_pos+1}FM', np.nan)

                if pd.notna(row_var) and pd.notna(col_var) and col_var > 0 and row_var > 0:
                    # Cell = deferred contract variance / front contract variance
                    # The "deferred" is whichever position is further out on the curve
                    # The "front" is whichever position is nearer
                    #
                    # Example: H26 row (pos 1), VarRat4 col (pos 4) = 4FM_var / 1FM_var = Q/H
                    # Example: K26 row (pos 2), VarRat1 col (pos 1) = 2FM_var / 1FM_var = K/H
                    # Example: Q26 row (pos 4), VarRat5 col (pos 5) = 5FM_var / 4FM_var = U/Q
                    #
                    # So: numerator = max position, denominator = min position
                    if row_pos >= col_pos:
                        # Row is further out (or equal), so row/col
                        ratio = row_var / col_var
                    else:
                        # Column is further out, so col/row
                        ratio = col_var / row_var
                else:
                    ratio = np.nan

                period_ratios[f'row{row_pos+1}_col{col_pos+1}'] = ratio

        all_ratios.append(period_ratios)

    ratios_df = pd.DataFrame(all_ratios)

    # Average the ratios across all periods
    avg_ratios = {}
    for row_pos in range(num_contracts):
        for col_pos in range(num_contracts):
            key = f'row{row_pos+1}_col{col_pos+1}'
            avg_ratios[key] = ratios_df[key].mean()

    # Build the ratio matrix
    # Rows = denominator (each contract on the curve)
    # Columns = numerator position (1FM, 2FM, etc.)

    row_labels = []
    matrix_data = []

    # Use the most recent year for row labels
    latest_year = periods['year'].max()
    year = latest_year
    prev_month_idx = MONTH_CODES.index(curve[0])

    for row_pos, row_month in enumerate(curve):
        month_idx = MONTH_CODES.index(row_month)
        if row_pos > 0 and month_idx < prev_month_idx:
            year += 1
        prev_month_idx = month_idx

        row_code = f"{row_month}{year:02d}"
        row_labels.append(row_code)

        row_data = {}
        for col_pos in range(num_contracts):
            key = f'row{row_pos+1}_col{col_pos+1}'
            ratio = avg_ratios.get(key, np.nan)
            row_data[f'VarRat{col_pos+1}'] = ratio

        matrix_data.append(row_data)

    result_df = pd.DataFrame(matrix_data, index=row_labels)
    result_df.index.name = 'Contract'

    return result_df


@functools.lru_cache(maxsize=64)
def _cached_variance_ratio_display(front_options_month: str,
                                   commodity: str = 'SOY',
                                   lookback_years: int = None,
                                   num_contracts: int = 12,
                                   prices_path: str = 'all_commodity_prices.csv') -> tuple:
    prices_df = _filtered_prices(prices_path, commodity)
    matrix = build_variance_ratio_matrix(
        prices_df, front_options_month, commodity,
        num_contracts=num_contracts,
        lookback_years=lookback_years
    )
    if len(matrix) == 0:
        return pd.DataFrame(), {}

    display_df = matrix.round(2)

    periods = identify_front_month_periods(prices_df, front_options_month, commodity)
    if lookback_years is not None and lookback_years > 0:
        periods = periods.sort_values('year', ascending=False).head(lookback_years)

    metadata = {
        'front_options_month': front_options_month,
        'front_futures_month': OPTIONS_TO_FUTURES.get(front_options_month, front_options_month),
        'commodity': commodity,
        'num_historical_periods': len(periods),
        'years_included': sorted(periods['year'].unique().tolist()) if len(periods) > 0 else [],
        'total_trading_days': periods['trading_days'].sum() if len(periods) > 0 else 0
    }

    return display_df, metadata


def get_variance_ratio_display(prices_df: pd.DataFrame,
                                front_options_month: str,
                                commodity: str = 'SOY',
                                lookback_years: int = None) -> tuple:
    """
    Get variance ratio matrix formatted for display in Streamlit.

    Args:
        prices_df: DataFrame with price data
        front_options_month: Options month code (e.g., 'H')
        commodity: Commodity name
        lookback_years: Number of years to include (None = all available)

    Returns:
        Tuple of (matrix_df, metadata_dict)
    """
    # Use cached builder ignoring incoming prices_df to avoid recompute on rerun
    return _cached_variance_ratio_display(front_options_month, commodity, lookback_years)


# Test function
if __name__ == "__main__":
    print("Testing Variance Ratio Calculations")
    print("=" * 60)

    # Load price data
    try:
        prices = pd.read_csv('all_commodity_prices.csv')
        prices['date'] = pd.to_datetime(prices['date'], format='mixed')
        print(f"Loaded {len(prices):,} price records")
        print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
        print(f"Commodities: {prices['commodity'].unique()}")
    except FileNotFoundError:
        print("Price file not found - run update_from_hertz.py first")
        exit(1)

    print("\n" + "=" * 60)
    print("Testing H options (H futures) for SOY")
    print("=" * 60)

    # Get curve
    curve = get_futures_curve_for_front_month('H', 'SOY')
    print(f"Futures curve when H is front: {curve}")

    # Get historical periods
    periods = identify_front_month_periods(prices, 'H', 'SOY')
    print(f"\nHistorical H-front periods:")
    print(periods.to_string())

    # Build variance ratio matrix
    print("\n" + "=" * 60)
    print("Variance Ratio Matrix")
    print("=" * 60)

    matrix, metadata = get_variance_ratio_display(prices, 'H', 'SOY')

    if len(matrix) > 0:
        print(f"\nMetadata: {metadata}")
        print(f"\nMatrix:")
        print(matrix.to_string())
    else:
        print("No data available")
