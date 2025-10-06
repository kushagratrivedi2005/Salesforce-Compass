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

# ============================================================================
# IMPORT STATEMENTS
# ============================================================================
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from model_utils import compute_metrics, validate_predictions
from config import ARIMA_PARAMS, ETS_PARAMS

print("[INFO] Using statsmodels ARIMA/SARIMA for forecasting")


# ============================================================================
# BASELINE MODEL CLASSES
# ============================================================================


class BaselineARIMA:
    """
    ARIMA/SARIMA model implementation using statsmodels with seasonal support.
    
    Automatically selects the best order based on grid search and AIC.
    """
    
    def __init__(self, seasonal=True, m=12, **kwargs):
        """
        Initialize ARIMA model with default parameters.
        
        Args:
            seasonal (bool): Whether to use seasonal ARIMA (SARIMA)
            m (int): Seasonal period (12 for monthly data)
        """
        self.params = kwargs
        self.model = None
        self.fitted = False
        self.order = None
        self.seasonal_order = None
        self.seasonal = seasonal
        self.m = m
        self.transform_method = None
        self.transform_lambda = None

    def _apply_transform(self, data):
        """Apply log or Box-Cox transformation for variance stabilization."""
        if data.min() > 0:
            # Use log transformation for positive data
            self.transform_method = 'log'
            return np.log(data), None
        else:
            # No transformation if data contains zeros/negatives
            self.transform_method = None
            return data, None
    
    def _inverse_transform(self, data):
        """Inverse transformation to get back to original scale."""
        if self.transform_method == 'log':
            return np.exp(data)
        else:
            return data

    def fit(self, train_data):
        """
        Fit SARIMA model to training data with automatic order selection.
        
        Args:
            train_data (pd.Series): Training time series data
            
        Returns:
            self: Fitted model instance
        """
        # Handle missing values
        if train_data.isna().any():
            train_data = train_data.interpolate().bfill().ffill()

        # Check for sufficient variation
        if np.std(train_data) < 1e-8:
            raise ValueError("Insufficient variation in training data")

        print("[INFO] Fitting SARIMA model with grid search for optimal parameters...")
        
        # Apply transformation for variance stabilization
        train_transformed, self.transform_lambda = self._apply_transform(train_data)
        if self.transform_method:
            print(f"[INFO] Applied {self.transform_method} transformation")
        
        # Test for stationarity using ADF test
        adf_result = adfuller(train_transformed)
        d = 0 if adf_result[1] < 0.05 else 1
        print(f"[INFO] ADF test p-value: {adf_result[1]:.4f}, differencing order d={d}")
        
        # Determine trend parameter based on differencing order
        if d == 0:
            trend_param = 'ct'
        elif d == 1:
            trend_param = 't'
        else:
            trend_param = None
        
        # Optimized grid search - fewer combinations for speed
        p_values = [0, 1, 2, 3]
        q_values = [0, 1, 2]
        
        if self.seasonal and len(train_data) >= 24:
            # Test for seasonal differencing
            seasonal_d = 0
            if len(train_data) >= self.m * 2:
                seasonal_adf = adfuller(train_transformed.diff(self.m).dropna())
                seasonal_d = 0 if seasonal_adf[1] < 0.05 else 1
            
            P_values = [0, 1]
            Q_values = [0, 1]
            seasonal_orders = [(P, seasonal_d, Q, self.m) for P in P_values for Q in Q_values]
            print(f"[INFO] Using SARIMA with seasonal period m={self.m}")
        else:
            seasonal_orders = [(0, 0, 0, 0)]
            print("[INFO] Using non-seasonal ARIMA (insufficient data for seasonality)")
        
        best_aic = np.inf
        best_model = None
        best_order = None
        best_seasonal_order = None
        
        # Grid search with early stopping
        total_combinations = len(p_values) * len(q_values) * len(seasonal_orders)
        print(f"[INFO] Testing up to {total_combinations} model combinations...")
        
        tested = 0
        no_improvement_count = 0
        
        for p in p_values:
            for q in q_values:
                order = (p, d, q)
                for seasonal_order in seasonal_orders:
                    tested += 1
                    try:
                        # Skip if no parameters
                        if p == 0 and q == 0 and seasonal_order[0] == 0 and seasonal_order[2] == 0:
                            continue
                        
                        temp_model = SARIMAX(
                            train_transformed,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend=trend_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        temp_fitted = temp_model.fit(disp=False, maxiter=50, method='lbfgs')
                        
                        if temp_fitted.aic < best_aic:
                            best_aic = temp_fitted.aic
                            best_model = temp_fitted
                            best_order = order
                            best_seasonal_order = seasonal_order
                            no_improvement_count = 0
                            print(f"[INFO] ✓ Best: SARIMA{order}x{seasonal_order}, AIC={best_aic:.2f}")
                        else:
                            no_improvement_count += 1
                        
                        # Early stopping if no improvement in last 10 models
                        if no_improvement_count >= 10:
                            print(f"[INFO] Early stopping after {tested} models")
                            break
                            
                    except Exception:
                        continue
                        
                if no_improvement_count >= 10:
                    break
            if no_improvement_count >= 10:
                break
        
        if best_model is None:
            raise ValueError("Could not fit SARIMA model with any configuration")
        
        self.model = best_model
        self.order = best_order
        self.seasonal_order = best_seasonal_order
        self.fitted = True
        print(f"[INFO] ✓ Final SARIMA model: order={self.order}, seasonal_order={self.seasonal_order}, AIC={best_aic:.2f}")
        
        return self

    def predict(self, n_periods, return_conf_int=True):
        """
        Generate forecasts for specified number of periods.
        
        Args:
            n_periods (int): Number of periods to forecast
            return_conf_int (bool): Whether to return confidence intervals
            
        Returns:
            tuple: (predictions, confidence_intervals) if return_conf_int=True
            pd.Series: predictions if return_conf_int=False
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Generate forecast in transformed space
        forecast_result = self.model.forecast(steps=n_periods)
        
        # Inverse transform back to original scale
        forecast_result = self._inverse_transform(forecast_result)
        
        if return_conf_int:
            # Get prediction intervals in transformed space
            pred_summary = self.model.get_forecast(steps=n_periods)
            conf_int = pred_summary.conf_int()
            
            # Inverse transform confidence intervals
            conf_int_transformed = pd.DataFrame({
                'lower': self._inverse_transform(conf_int.iloc[:, 0]),
                'upper': self._inverse_transform(conf_int.iloc[:, 1])
            }, index=conf_int.index)
            
            return forecast_result, conf_int_transformed
        else:
            return forecast_result

    def get_model_info(self):
        """
        Get model information and parameters.
        
        Returns:
            dict: Model information including order and AIC
        """
        if not self.fitted:
            return {"status": "not_fitted"}
            
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.model.aic, 
            "bic": self.model.bic,
            "transformation": self.transform_method,
            "fitted": True
        }


class ExponentialSmoothingModel:
    """Exponential Smoothing (Holt-Winters) model with automatic fallback."""
    
    def __init__(self, **kwargs):
        self.params = {**ETS_PARAMS, **kwargs}
        self.fitted_model = None
        self.fitted = False
    
    def fit(self, train_data):
        """Fit model with automatic fallback to multiplicative if additive fails."""
        print("→ Fitting Exponential Smoothing...")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                model = ExponentialSmoothing(train_data, **self.params)
                self.fitted_model = model.fit(optimized=True)
                self.fitted = True
                print("✓ Additive model fitted")
            except Exception:
                # Fallback to multiplicative
                params_mult = self.params.copy()
                params_mult['seasonal'] = 'mul'
                model = ExponentialSmoothing(train_data, **params_mult)
                self.fitted_model = model.fit(optimized=True)
                self.fitted = True
                print("✓ Multiplicative model fitted")
        
        return self
    
    def predict(self, n_periods):
        """Generate forecasts."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.fitted_model.forecast(steps=n_periods)
    
    def get_model_info(self):
        """Get model information."""
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
    results = {'models': {}, 'predictions': {}, 'metrics': {}, 'model_info': {}, 'errors': {}}
    
    # Validate inputs
    if test_data is None or len(test_data) == 0:
        print("Error: Test data is empty")
        results['errors'] = {'general': "Empty test data"}
        return results
    
    if len(train_data) <= 12:
        print("Error: Need at least 12 observations for modeling")
        results['errors'] = {'general': "Insufficient training data"}
        return results
    
    n_forecast = len(test_data)
    print(f"Forecasting {n_forecast} periods ahead")
    
    # Ensure monthly frequency
    train_data = ensure_monthly_frequency(train_data)
    test_data = ensure_monthly_frequency(test_data)
    
    # Clean training data
    train_clean = train_data.copy()
    if train_clean.isna().any():
        train_clean = train_clean.interpolate(method='linear').ffill().bfill()
    
    print(f"Training data: {len(train_clean)} obs, range=[{train_clean.min():.0f}, {train_clean.max():.0f}], std={train_clean.std():.0f}")
    
    # Train ARIMA
    _train_arima_model(train_clean, test_data, n_forecast, results)
    
    # Train ETS
    _train_ets_model(train_clean, test_data, n_forecast, results)
    
    return results


def _train_arima_model(train_data, test_data, n_forecast, results):
    """Helper function to train ARIMA model."""
    try:
        print("\n→ Training ARIMA/SARIMA model...")
        arima_model = BaselineARIMA()
        arima_model.fit(train_data)
        
        arima_pred, arima_conf = arima_model.predict(n_periods=n_forecast, return_conf_int=True)
        
        # Create series with proper index
        pred_index = test_data.index if len(test_data.index) == len(arima_pred) else \
                     pd.date_range(start=train_data.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
        
        arima_pred_series = pd.Series(arima_pred, index=pred_index)
        arima_conf.index = pred_index
        
        # Compute metrics
        min_len = min(len(test_data), len(arima_pred))
        arima_metrics = compute_metrics(test_data.values[:min_len], arima_pred[:min_len])
        
        results['models']['ARIMA'] = arima_model
        results['predictions']['ARIMA'] = {'forecast': arima_pred_series, 'confidence_intervals': arima_conf}
        results['metrics']['ARIMA'] = arima_metrics
        results['model_info']['ARIMA'] = arima_model.get_model_info()
        
        print(f"✓ ARIMA: MAE={arima_metrics['MAE']:.0f}, RMSE={arima_metrics['RMSE']:.0f}, MAPE={arima_metrics['MAPE']:.2f}%")
        
    except Exception as e:
        print(f"✗ ARIMA failed: {e}")
        results['errors'] = results.get('errors', {})
        results['errors']['ARIMA'] = str(e)


def _train_ets_model(train_data, test_data, n_forecast, results):
    """Helper function to train ETS model."""
    try:
        print("\n→ Training Exponential Smoothing model...")
        ets_model = ExponentialSmoothingModel()
        ets_model.fit(train_data)
        ets_pred = ets_model.predict(n_periods=n_forecast)
        
        # Compute metrics
        min_len = min(len(test_data), len(ets_pred))
        ets_metrics = compute_metrics(test_data.values[:min_len], ets_pred.values[:min_len])
        
        results['models']['ETS'] = ets_model
        results['predictions']['ETS'] = {'forecast': ets_pred}
        results['metrics']['ETS'] = ets_metrics
        results['model_info']['ETS'] = ets_model.get_model_info()
        
        print(f"✓ ETS: MAE={ets_metrics['MAE']:.0f}, RMSE={ets_metrics['RMSE']:.0f}, MAPE={ets_metrics['MAPE']:.2f}%")
        
    except Exception as e:
        print(f"✗ ETS failed: {e}")
        results['errors'] = results.get('errors', {})
        results['errors']['ETS'] = str(e)