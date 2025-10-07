"""
Configuration file for the vehicle sales forecasting pipeline.

This file contains all user-configurable parameters including file paths,
model settings, and pipeline configurations.
"""

import numpy as np

# ------------------------- File Paths -------------------------
CSV_PATH = 'final_merged_dataset.csv'      # Path to merged dataset
RESULTS_DIR = 'forecast_results'           # Directory to save outputs
DATASET_DIR = '/Users/evanb/OneDrive/Documents/GitHub/Salesforce-Compass/dataset'

# ------------------------- Model Configuration -------------------------
TARGET = 'category_LIGHT PASSENGER VEHICLE'                      # Primary forecasting target
DATE_COL = 'Month'                         # Monthly date column name
TEST_MONTHS = 6                            # Holdout months for testing
RANDOM_SEED = 42                           # Random seed for reproducibility
FUTURE_FORECAST_MONTHS = 12                # Number of months to forecast into the future

# Set random seed
np.random.seed(RANDOM_SEED)

# ------------------------- Exogenous Variables Configuration -------------------------
# Set to True to use top-k correlated variables, False to use MANUAL_EXOGS
USE_TOP_K_EXOGS = True

# Candidate exogenous features for causal modeling (used if USE_TOP_K_EXOGS is True)
CANDIDATE_EXOGS = [
    'interest_rate', 'repo_rate', 'holiday_count', 
    'major_national_holiday', 'major_religious_holiday',
]

# Manual list of exogenous variables (used if USE_TOP_K_EXOGS is False)
MANUAL_EXOGS = ['interest_rate', 'repo_rate']

# Number of top correlated exogenous variables to select (if USE_TOP_K_EXOGS is True)
TOP_K_EXOGS = 5

# ------------------------- Model Parameters -------------------------
# ARIMA parameters for auto_arima
ARIMA_PARAMS = {
    'seasonal': True,
    'm': 12,
    'stepwise': True,
    'suppress_warnings': True,
    'error_action': 'ignore',
    'trace': False,
    'max_p': 8,
    'max_q': 8,
    'max_P': 3,
    'max_Q': 3
}

# Exponential Smoothing parameters
ETS_PARAMS = {
    'seasonal': 'add',
    'trend': 'add',
    'seasonal_periods': 12
}

# SARIMAX parameters
SARIMAX_PARAMS = {
    'enforce_stationarity': False,
    'enforce_invertibility': False,
    'max_iter': 500
}

# ------------------------- Segment Analysis Configuration -------------------------
# Optional segment for additional analysis
SEGMENT_TARGET = 'fuel_PURE EV'

# ------------------------- Visualization Configuration -------------------------
PLOT_CONFIG = {
    'figsize': (12, 5),
    'marker': 'o',
    'alpha': 0.3,
    'bbox_inches': 'tight'
}
