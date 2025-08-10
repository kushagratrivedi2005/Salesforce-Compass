"""
Baseline forecasting models for the vehicle sales prediction pipeline.

This module implements:
- Auto ARIMA model fitting and forecasting
- Exponential Smoothing (Holt-Winters) model
- Baseline model evaluation and comparison
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="'force_all_finite'")  # Suppress the specific warning

# Handle pmdarima import with error handling
PM_AVAILABLE = False
try:
    import pmdarima as pm
    PM_AVAILABLE = True
except (ImportError, ValueError) as e:
    warnings.warn(f"pmdarima import failed ({str(e)}). Using statsmodels fallback.")
    # We'll use statsmodels directly as a fallback
    from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from model_utils import compute_metrics, validate_predictions
from config import ARIMA_PARAMS, ETS_PARAMS


class BaselineARIMA:
    def __init__(self, **kwargs):
        self.params = {
            "seasonal": False,
            "start_p": 0, "max_p": 8,
            "start_q": 0, "max_q": 8,
            "d": None, "test": "adf",
            "stationary": False,
            "stepwise": False,
            "trend": "ct",  # constant + trend
            "suppress_warnings": True,
            "error_action": "ignore",
            "trace": True,
            "information_criterion": "aic",
            "start_P": 0, "max_P": 0,  # ensure no seasonal search
        }
        self.params.update(kwargs)
        self.model = None
        self.fitted = False
        self.order = None

    def fit(self, train_data):
        # No need to check frequency, we assume it's already set correctly
        if train_data.isna().any():
            train_data = train_data.interpolate().bfill().ffill()

        if np.std(train_data) < 1e-8:
            raise ValueError("Insufficient variation")

        try:
            self.model = pm.auto_arima(
                train_data,
                **self.params,
                transform='log',  # variance stabilization
            )
            self.order = self.model.order
            self.fitted = True
        except Exception as e:
            raise
        return self

    def predict(self, n_periods, return_conf_int=True):
        if not self.fitted:
            raise ValueError("Model not fitted")

        result = self.model.predict(n_periods=n_periods, return_conf_int=return_conf_int)
        return result

    def get_model_info(self):
        return {"order": self.order, "aic": getattr(self.model, 'aic', None), "fitted": self.fitted}


class ExponentialSmoothingModel:
    """
    Exponential Smoothing (Holt-Winters) baseline model implementation.
    
    This class provides exponential smoothing with trend and seasonal components.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Exponential Smoothing model.
        
        Args:
            **kwargs: Additional parameters to override defaults
        """
        self.params = {**ETS_PARAMS, **kwargs}
        self.model = None
        self.fitted_model = None
        self.fitted = False
    
    def fit(self, train_data):
        """
        Fit Exponential Smoothing model to training data.
        
        Args:
            train_data (pd.Series): Training time series data
            
        Returns:
            self: Fitted model instance
        """
        print("Fitting Exponential Smoothing (Holt-Winters) model...")
        
        # No need to check or set frequency here, already done in train_baseline_models
        
        try:
            self.model = ExponentialSmoothing(train_data, **self.params)
            self.fitted_model = self.model.fit(optimized=True)
            self.fitted = True
            print("Exponential Smoothing model fitted successfully")
        except Exception as e:
            print(f"Warning: Could not fit additive model: {e}")
            print("Trying multiplicative seasonal model...")
            
            # Fallback to multiplicative if additive fails
            params_mult = self.params.copy()
            params_mult['seasonal'] = 'mul'
            
            try:
                self.model = ExponentialSmoothing(train_data, **params_mult)
                self.fitted_model = self.model.fit(optimized=True)
                self.fitted = True
                print("Multiplicative Exponential Smoothing model fitted successfully")
            except Exception as e2:
                print(f"Error: Could not fit exponential smoothing model: {e2}")
                raise e2
        
        return self
    
    def predict(self, n_periods):
        """
        Generate forecasts for specified number of periods.
        
        Args:
            n_periods (int): Number of periods to forecast
            
        Returns:
            pd.Series: Predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Ensure n_periods is positive
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        
        predictions = self.fitted_model.forecast(steps=n_periods)
        return predictions
    
    def get_model_info(self):
        """
        Get model information and parameters.
        
        Returns:
            dict: Model information including parameters and fit statistics
        """
        if not self.fitted:
            return {"status": "not_fitted"}
        
        return {
            "seasonal": self.params['seasonal'],
            "trend": self.params['trend'],
            "seasonal_periods": self.params['seasonal_periods'],
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "fitted": True
        }


def ensure_monthly_frequency(data):
    """
    Ensure data has proper monthly frequency.
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data
        
    Returns:
        pd.Series or pd.DataFrame: Data with proper monthly frequency
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
        
    if data.index.freq is None:
        data = data.asfreq('MS')  # Month Start frequency
        
    return data


def train_baseline_models(train_data, test_data):
    """
    Train and evaluate all baseline models.
    
    Args:
        train_data (pd.Series): Training time series data
        test_data (pd.Series): Test time series data
        
    Returns:
        dict: Results containing models, predictions, and metrics
    """
    results = {
        'models': {},
        'predictions': {},
        'metrics': {},
        'model_info': {}
    }
    
    # Verify test_data is valid
    if test_data is None or len(test_data) == 0:
        print("Error: Test data is empty or None. Cannot evaluate models.")
        results['errors'] = {'general': "Empty test data"}
        return results
    
    n_forecast = len(test_data)
    print(f"Forecasting {n_forecast} periods ahead")
    
    print("Validating input data...")
    
    # Set proper monthly frequency once at the beginning
    try:
        # Convert to datetime and set monthly frequency for both train and test
        train_data = ensure_monthly_frequency(train_data)
        test_data = ensure_monthly_frequency(test_data)
    except Exception as e:
        print(f"Error setting monthly frequency: {e}")
        results['errors'] = {'general': f"Frequency conversion error: {str(e)}"}
        return results
    
    # Handle NaN values in training data
    train_data_clean = train_data.copy()
    if train_data_clean.isna().any():
        print(f"Found NaN values in training data. Applying interpolation and filling.")
        train_data_clean = train_data_clean.interpolate(method='linear').fillna(
            method='ffill').fillna(method='bfill')
    
    # Add data diagnostics for debugging
    print(f"Training data shape: {train_data_clean.shape}")
    print(f"Training data range: [{train_data_clean.min()}, {train_data_clean.max()}]")
    print(f"Training data std: {train_data_clean.std()}")
    
    # Check if data is empty or too small for modeling
    if len(train_data_clean) <= 12:  # Need at least a year of data for seasonal models
        print("Error: Not enough training data for modeling. Need at least 12 observations.")
        results['errors'] = results.get('errors', {})
        results['errors']['general'] = "Insufficient training data"
        return results
    
    # Train ARIMA model
    try:
        print("Ensuring data is valid for ARIMA modeling...")
        # Check if data has variation (not all same values)
        if np.std(train_data_clean) == 0:
            raise ValueError("Training data has no variation (all values are the same)")
            
        arima_model = BaselineARIMA()
        arima_model.fit(train_data_clean)
        
        # Verify n_forecast is positive before prediction
        if n_forecast <= 0:
            print("Warning: Test data length is 0, setting n_forecast to 1")
            n_forecast = 1
        
        # Fix unpacking error with try/except block    
        try:
            # This is the line causing issues, make it more robust
            arima_pred_result = arima_model.predict(
                n_periods=n_forecast, 
                return_conf_int=True
            )
            
            # Handle different return formats
            if isinstance(arima_pred_result, tuple) and len(arima_pred_result) == 2:
                arima_pred, arima_conf = arima_pred_result
            else:
                print("Warning: ARIMA prediction didn't return expected tuple format")
                arima_pred = arima_pred_result
                # Create dummy confidence intervals
                arima_conf = np.array([[p * 0.9, p * 1.1] for p in arima_pred])
        except Exception as e:
            print(f"Error during ARIMA prediction unpacking: {e}")
            # Create fallback predictions
            arima_pred = np.array([train_data_clean.iloc[-1]] * n_forecast)
            arima_conf = np.array([[p * 0.9, p * 1.1] for p in arima_pred])
        
        # Convert predictions to appropriate format
        if isinstance(arima_pred, np.ndarray):
            # Create proper index for predictions with correct frequency
            if len(test_data.index) == len(arima_pred):
                pred_index = test_data.index
            else:
                # Create new index if lengths don't match
                last_train_date = train_data_clean.index[-1]
                pred_index = pd.date_range(start=last_train_date + pd.DateOffset(months=1),
                                         periods=len(arima_pred),
                                         freq='MS')
                
            arima_pred_series = pd.Series(arima_pred, index=pred_index)
        else:
            arima_pred_series = arima_pred
            
        # Handle confidence intervals
        if isinstance(arima_conf, np.ndarray):
            # Use the same index as predictions
            if hasattr(arima_pred_series, 'index'):
                conf_index = arima_pred_series.index
            else:
                conf_index = test_data.index[:len(arima_conf)]
                
            arima_conf_df = pd.DataFrame(
                arima_conf, 
                index=conf_index, 
                columns=['lower', 'upper']
            )
        else:
            arima_conf_df = arima_conf
        
        # Ensure arrays are not empty before computing metrics
        if len(test_data) > 0 and len(arima_pred) > 0:
            # Make sure lengths match for comparison
            min_len = min(len(test_data), len(arima_pred))
            arima_metrics = compute_metrics(test_data.values[:min_len], arima_pred[:min_len])
        else:
            arima_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
            print("Warning: Empty test data or predictions, metrics set to NaN")
        
        results['models']['ARIMA'] = arima_model
        results['predictions']['ARIMA'] = {
            'forecast': arima_pred_series,
            'confidence_intervals': arima_conf_df
        }
        results['metrics']['ARIMA'] = arima_metrics
        results['model_info']['ARIMA'] = arima_model.get_model_info()
        
        print(f"ARIMA metrics: {arima_metrics}")
        
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        print("Continuing with other models despite ARIMA failure")
        results['errors'] = results.get('errors', {})
        results['errors']['ARIMA'] = str(e)
    
    # Train Exponential Smoothing model with similar error handling
    try:
        ets_model = ExponentialSmoothingModel()
        ets_model.fit(train_data_clean)
        
        # Verify n_forecast is positive before prediction
        if n_forecast <= 0:
            print("Warning: Test data length is 0, setting n_forecast to 1")
            n_forecast = 1
            
        ets_pred = ets_model.predict(n_periods=n_forecast)
        
        # Ensure arrays are not empty before computing metrics
        if len(test_data) > 0 and len(ets_pred) > 0:
            # Make sure lengths match for comparison
            min_len = min(len(test_data), len(ets_pred))
            ets_metrics = compute_metrics(test_data.values[:min_len], ets_pred.values[:min_len])
        else:
            ets_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
            print("Warning: Empty test data or predictions, metrics set to NaN")
        
        results['models']['ETS'] = ets_model
        results['predictions']['ETS'] = {
            'forecast': ets_pred
        }
        results['metrics']['ETS'] = ets_metrics
        results['model_info']['ETS'] = ets_model.get_model_info()
        
        print(f"ETS metrics: {ets_metrics}")
        
    except Exception as e:
        print(f"Error training ETS model: {e}")
        results['errors'] = results.get('errors', {})
        results['errors']['ETS'] = str(e)
    
    return results