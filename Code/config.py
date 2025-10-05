"""
Configuration file for the vehicle sales forecasting pipeline.

This file contains all user-configurable parameters including file paths,
model settings, and pipeline configurations.
"""

import numpy as np

# ------------------------- File Paths -------------------------
CSV_PATH = 'final_merged_dataset.csv'      # Path to merged dataset
RESULTS_DIR = 'forecast_results'           # Directory to save outputs
DATASET_DIR = '/Users/hellgamerhell/Downloads/salesforce/dataset'

# ------------------------- Model Configuration -------------------------
TARGET = 'fuel_Total'                      # Primary forecasting target
DATE_COL = 'Month'                         # Monthly date column name
TEST_MONTHS = 6                            # Holdout months for testing
RANDOM_SEED = 42                           # Random seed for reproducibility
FUTURE_FORECAST_DATE = "2026-01-01"        # Date until which to forecast future values

# Set random seed
np.random.seed(RANDOM_SEED)

# ------------------------- Exogenous Variables Configuration -------------------------
# Candidate exogenous features for causal modeling
CANDIDATE_EXOGS = [
    'interest_rate', 'repo_rate', 'holiday_count', 
    'major_national_holiday', 'major_religious_holiday',
    'fuel_DIESEL', 'fuel_PETROL', 'fuel_PURE EV', 'fuel_PLUG IN HYBRID EV',
    'fame_ii', 'fame_iii', 'pm_edrive', 'bs7_norms', 
    'scrappage_policy', 'pli_scheme',
    "GasPrices", 
    "UnemploymentRate", 
    "ConsumerConfidence", 
    "InterestRates", 
    "AdvertisingSpend", 
    "CompetitorSales", 
    "SeasonalIndex"
]

# Number of top correlated exogenous variables to select
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
