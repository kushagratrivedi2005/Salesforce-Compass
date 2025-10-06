"""
Data loading and preprocessing utilities for the vehicle sales forecasting pipeline.

This module handles:
- CSV data loading with proper date parsing
- Numeric data cleaning (removing commas, handling missing values)
- Time series index setup and validation
- Train/test splitting for time series data
"""

import os
import numpy as np
import pandas as pd
from config import DATE_COL, TEST_MONTHS


def ensure_dir(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def parse_and_clean_numeric(value):
    """
    Remove commas and convert to numeric (safe conversion).
    
    Args:
        value: Input value to clean and convert
        
    Returns:
        float: Cleaned numeric value or NaN if conversion fails
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    # Remove commas and non-breaking spaces
    return pd.to_numeric(str(value).replace(',', '').replace('\xa0', ''), errors='coerce')


def load_and_prepare(csv_path):
    """
    Load CSV file, parse dates, clean numeric columns and set datetime index.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and prepared DataFrame with datetime index
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If DATE_COL is not found in the dataset
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    # Read CSV with more specific NA values to avoid treating empty strings as NaN
    df = pd.read_csv(csv_path, na_values=['NA', 'NULL', 'NaN', '#N/A', 'N/A', 'nan'], keep_default_na=False)
    
    if DATE_COL not in df.columns:
        raise KeyError(f"Date column '{DATE_COL}' not found in dataset")
    
    # Parse date column - handle various formats
    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format='%Y-%m')
    except Exception:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    
    # Clean numeric columns: strip commas from values imported as strings
    numeric_columns = []
    for col in df.columns:
        if col == DATE_COL:
            continue
        
        # Skip string columns like Month_Name and month_name
        if col in ['Month_Name', 'month_name']:
            continue
        
        # Clean and convert object columns or numeric-like strings
        if df[col].dtype == 'object':
            df[col] = df[col].apply(parse_and_clean_numeric)
            
        # Ensure dtype is numeric for all non-date columns except month names
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_columns.append(col)
        except Exception:
            pass
    
    print(f"Processed {len(numeric_columns)} numeric columns")
    
    # Sort by date and set as index
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    df = df.set_index(DATE_COL)
    
    # Handle duplicate indexes by aggregating with sum
    if df.index.duplicated().any():
        print("Warning: Found duplicate dates. Aggregating by sum.")
        df = df.groupby(df.index).sum()
    
    # Get the actual data range
    min_date = df.index.min()
    max_date = df.index.max()
    
    # Apply cutoff date limit (August 2025)
    cutoff_date = pd.Timestamp('2025-09-01')
    if max_date > cutoff_date:
        print(f"Limiting data to cutoff date: {cutoff_date.strftime('%Y-%m')}")
        df = df.loc[:cutoff_date]
        max_date = cutoff_date
    
    print(f"✓ Detected actual data range: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
    
    # Check for truly empty cells that should be considered NaN
    empty_cells = (df == '') | (df.astype(str) == 'nan')
    if empty_cells.values.any():
        print("Found empty cells that should be converted to NaN")
        df = df.replace('', np.nan).replace('nan', np.nan)
    
    # Check for NaN values before ensuring monthly frequency
    nan_count_before = df.isna().sum().sum()
    if nan_count_before > 0:
        print(f"Found {nan_count_before} NaN values in original data.")
        # Show columns with most NaNs, but exclude month name columns
        cols_to_check = [col for col in df.columns if col not in ['Month_Name', 'month_name']]
        nan_cols = df[cols_to_check].isna().sum().sort_values(ascending=False)
        top_nan_cols = nan_cols[nan_cols > 0].head(5)
        if len(top_nan_cols) > 0:
            print("Top columns with NaNs (excluding month name columns):")
            for col, count in top_nan_cols.items():
                print(f"  - {col}: {count} NaNs")
        else:
            print("All NaN values are in month name columns, which is expected and not problematic")
    else:
        print("No NaN values detected in the dataset")
    
    # Ensure monthly frequency but only within the actual data range
    df = ensure_monthly_frequency(df, min_date, max_date)
    
    # Check for NaN values after ensuring monthly frequency
    nan_count_after = df.isna().sum().sum()
    if nan_count_after > nan_count_before:
        print(f"Additional {nan_count_after - nan_count_before} NaNs introduced when ensuring monthly frequency")
    
    # Handle any remaining NaN values to prevent model errors, but only for numeric columns
    # We'll leave string columns like month names alone
    print("Checking for NaN values in numeric columns...")
    nan_count_numeric = df[numeric_columns].isna().sum().sum()
    if nan_count_numeric > 0:
        print(f"Found {nan_count_numeric} NaN values in numeric columns. Applying forward-fill and backward-fill.")
        
        # Use ffill and bfill methods only on numeric columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
    else:
        print("No NaN values in numeric columns")
    
    print(f"Final dataset shape: {df.shape}")
    return df


def ensure_monthly_frequency(df, start_date=None, end_date=None):
    """
    Ensure the DataFrame has continuous monthly frequency within a date range.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        start_date: Start date to use (defaults to df index min)
        end_date: End date to use (defaults to df index max)
        
    Returns:
        pd.DataFrame: DataFrame with continuous monthly index
    """
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    
    # Create a continuous monthly date range
    full_idx = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    if not all(full_idx.isin(df.index)):
        print('Reindexing to monthly frequency within actual data range')
        df = df.reindex(full_idx)
        # Forward-fill missing values using ffill method
        df = df.ffill()
    
    return df


def train_test_split_ts(data, test_months=TEST_MONTHS):
    """
    Split time series data into train/test chronologically.
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data to split
        test_months (int): Number of months to reserve for testing
        
    Returns:
        tuple: (train_data, test_data)
        
    Raises:
        ValueError: If input is not pandas Series or DataFrame
    """
    if isinstance(data, pd.Series):
        n = len(data)
        train = data.iloc[: n - test_months]
        test = data.iloc[n - test_months:]
        return train, test
    elif isinstance(data, pd.DataFrame):
        n = len(data)
        train = data.iloc[: n - test_months, :]
        test = data.iloc[n - test_months:, :]
        return train, test
    else:
        raise ValueError('Input must be pandas Series or DataFrame')


def train_test_split_by_date(data, train_end_date='2025-01-01', test_end_date='2025-09-01'):
    """
    Split time series data into train/test by specific dates.
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data to split
        train_end_date (str): End date for training data (exclusive)
        test_end_date (str): End date for test data (inclusive)
        
    Returns:
        tuple: (train_data, test_data)
        
    Raises:
        ValueError: If input is not pandas Series or DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('Input must have DatetimeIndex')
    
    train_end = pd.Timestamp(train_end_date)
    test_end = pd.Timestamp(test_end_date)
    
    train_data = data.loc[:train_end].copy()
    test_data = data.loc[train_end+pd.Timedelta(days=1):test_end].copy()
    
    print(f"Split data by date: train ends {train_end.strftime('%Y-%m-%d')}, test ends {test_end.strftime('%Y-%m-%d')}")
    
    return train_data, test_data


def prepare_exogenous_variables(df, candidate_exogs, target_col, top_k=5):
    """
    Prepare and select exogenous variables based on correlation with target.

    Args:
        df (pd.DataFrame): Input DataFrame
        candidate_exogs (list): List of candidate exogenous variable names
        target_col (str): Target column name
        top_k (int): Number of top correlated variables to select

    Returns:
        tuple: (selected_exogs_list, exog_dataframe, correlations_series)
    """
    # Keep only exogenous variables present in dataset
    exogs_present = [c for c in candidate_exogs if c in df.columns]
    print(f"Found {len(exogs_present)} exogenous candidates in dataset")

    # Filter dataset for these exogs
    exog_df = df[exogs_present].copy()

    # Drop all-NaN columns
    nan_only_cols = [c for c in exog_df.columns if exog_df[c].isna().all()]
    if nan_only_cols:
        print(f"[DEBUG] Dropping all-NaN columns: {nan_only_cols}")
        exog_df = exog_df.drop(columns=nan_only_cols)

    # Drop constant columns (std = 0)
    constant_cols = [c for c in exog_df.columns if exog_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"[DEBUG] Dropping constant columns: {constant_cols}")
        exog_df = exog_df.drop(columns=constant_cols)

    # Align with target index
    exog_df = exog_df.loc[df[target_col].index]

    # Replace inf/-inf with NaN
    exog_df = exog_df.replace([np.inf, -np.inf], np.nan)

    # Interpolate and fill missing values
    if exog_df.isna().sum().sum() > 0:
        print(f"[DEBUG] Found {exog_df.isna().sum().sum()} NaN values. Interpolating...")
        exog_df = exog_df.interpolate(method='linear', limit_direction='both')
        exog_df = exog_df.fillna(method='ffill').fillna(method='bfill')

    # Calculate correlations safely
    correlations = {}
    for col in exog_df.columns:
        try:
            correlations[col] = df[target_col].corr(exog_df[col])
        except Exception as e:
            print(f"[DEBUG] Could not compute correlation for {col}: {e}")
            correlations[col] = np.nan

    # Sort by absolute correlation
    corrs_series = pd.Series(correlations).dropna().sort_values(
        key=lambda x: x.abs(), ascending=False
    )

    # Select top-k variables
    top_exogs = corrs_series.head(top_k).index.tolist()
    print(f"Selected top {len(top_exogs)} exogenous variables: {top_exogs}")

    return top_exogs, exog_df[top_exogs], corrs_series


def detect_and_handle_outliers(series, method='iqr', replace_with='median'):
    """
    Detect and handle outliers in time series data.
    
    Args:
        series (pd.Series): Input time series
        method (str): Method for outlier detection ('iqr' or 'zscore')
        replace_with (str): How to replace outliers ('median', 'mean', or 'interpolate')
        
    Returns:
        pd.Series: Series with outliers handled
    """
    series_clean = series.copy()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > 3
    else:
        return series
    
    n_outliers = outliers.sum()
    if n_outliers > 0:
        print(f"[INFO] Detected {n_outliers} outliers ({n_outliers/len(series)*100:.2f}%)")
        
        if replace_with == 'median':
            series_clean[outliers] = series.median()
        elif replace_with == 'mean':
            series_clean[outliers] = series.mean()
        elif replace_with == 'interpolate':
            series_clean[outliers] = np.nan
            series_clean = series_clean.interpolate(method='linear')
    
    return series_clean



def standardize_exogenous_variables(train_exog, test_exog):
    """
    Standardize exogenous variables using training set statistics.
    
    Args:
        train_exog (pd.DataFrame): Training exogenous variables
        test_exog (pd.DataFrame): Test exogenous variables
        
    Returns:
        tuple: (train_scaled, test_scaled, means, stds)
    """
    means = train_exog.mean()
    stds = train_exog.std().replace(0, 1)  # Avoid division by zero
    
    train_scaled = (train_exog - means) / stds
    test_scaled = (test_exog - means) / stds
    
    return train_scaled, test_scaled, means, stds
