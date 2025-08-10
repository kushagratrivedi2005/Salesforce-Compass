"""
Causal forecasting models with exogenous variables for vehicle sales prediction.

This module implements:
- SARIMAX model with safer order selection and exogenous preprocessing
- Robust data cleaning, frequency enforcement, and diagnostics
"""

import numpy as np
import pandas as pd
import traceback
from statsmodels.tsa.statespace.sarimax import SARIMAX
from model_utils import compute_metrics, extract_model_parameters
import warnings
warnings.filterwarnings("ignore", message="'force_all_finite'")  # sklearn warning suppression

# Default SARIMAX params used only for model creation (not fit kwargs)
SARIMAX_DEFAULTS = {
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}

# Fit-time kwargs (passed explicitly to .fit())
SARIMAX_FIT_KWARGS = {
    "maxiter": 200,  # will be passed as maxiter to .fit()
    "disp": False
}


class SARIMAXModel:
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.fitted = False
        self.exog_names = None

    def _ensure_monthly_freq(self, series_or_df):
        """Ensure MS frequency on index (in-place safe)."""
        if not isinstance(series_or_df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a pandas DatetimeIndex")
        if series_or_df.index.freq is None:
            # try to infer; if fails set to MS
            inferred = pd.infer_freq(series_or_df.index)
            if inferred is None:
                print("[DEBUG] No inferred freq — forcing MS (month start)")
                series_or_df = series_or_df.asfreq("MS")
            else:
                series_or_df = series_or_df.asfreq(inferred)
        return series_or_df

    def fit(self, train_data, exog_data=None, order=None, seasonal_order=None):
        # override if provided
        if order is not None:
            self.order = order
        if seasonal_order is not None:
            self.seasonal_order = seasonal_order

        # basic validation
        if self.order is None:
            raise ValueError("ARIMA order (p,d,q) must be provided")

        # ensure datetime freq
        train_data = self._ensure_monthly_freq(train_data)
        if exog_data is not None:
            exog_data = exog_data.copy()
            exog_data = self._ensure_monthly_freq(exog_data)

        # store exog names
        self.exog_names = list(exog_data.columns) if exog_data is not None else []

        print(f"[DEBUG] Fitting SARIMAX(order={self.order}, seasonal_order={self.seasonal_order})")
        print(f"[DEBUG] Train length: {len(train_data)}, exog columns: {self.exog_names}")

        # build model using only safe kwargs
        try:
            self.model = SARIMAX(
                endog=train_data,
                exog=exog_data if exog_data is not None else None,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **SARIMAX_DEFAULTS
            )

            # fit using explicit kwargs (avoid passing unknown fit kwargs into SARIMAX constructor)
            self.fitted_model = self.model.fit(**SARIMAX_FIT_KWARGS)
            self.fitted = True
            print("[DEBUG] SARIMAX model fitted successfully")
        except Exception as e:
            print("[ERROR] SARIMAX fitting failed:")
            traceback.print_exc()
            raise e

        return self

    def predict(self, n_periods, exog_data=None, return_conf_int=True):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # ensure lengths / index
        if exog_data is not None:
            if len(exog_data) != n_periods:
                raise ValueError(f"exog_data length ({len(exog_data)}) != n_periods ({n_periods})")
            exog_data = self._ensure_monthly_freq(exog_data)

        # forecast
        forecast_obj = self.fitted_model.get_forecast(steps=n_periods, exog=exog_data)
        preds = forecast_obj.predicted_mean
        if return_conf_int:
            conf = forecast_obj.conf_int()
            return preds, conf
        else:
            return preds

    def get_model_summary(self):
        if not self.fitted:
            return {"status": "not_fitted"}
        summary = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_variables": self.exog_names,
            "aic": getattr(self.fitted_model, "aic", None),
            "bic": getattr(self.fitted_model, "bic", None),
            "log_likelihood": getattr(self.fitted_model, "llf", None),
            "fitted": True,
        }
        if self.exog_names:
            try:
                exog_params = extract_model_parameters(self.fitted_model, self.exog_names)
                summary["exog_coefficients"] = exog_params.to_dict()
            except Exception:
                summary["exog_coefficients"] = None
        return summary

    def get_diagnostics(self):
        if not self.fitted:
            return {"status": "not_fitted"}
        res = self.fitted_model.resid
        return {
            "residuals_mean": float(res.mean()),
            "residuals_std": float(res.std()),
            # leave placeholders for advanced tests
            "ljung_box_pvalue": None,
            "jarque_bera_pvalue": None
        }


def _clean_and_align_exog(exog_train, exog_test, target_index):
    """
    - Drops all-NaN and constant columns
    - Replaces infs, interpolates and fills remaining NaNs
    - Aligns indexes to target_index and ensures same columns between train/test
    """
    exog_train = exog_train.copy()
    exog_test = exog_test.copy()

    # replace infs
    exog_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    exog_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # drop all-NaN columns (based on train)
    allnan_cols = [c for c in exog_train.columns if exog_train[c].isna().all()]
    if allnan_cols:
        print(f"[DEBUG] Dropping all-NaN exog columns (train): {allnan_cols}")
        exog_train.drop(columns=allnan_cols, inplace=True)
        exog_test.drop(columns=allnan_cols, inplace=True, errors='ignore')

    # drop constant columns (train-based)
    const_cols = [c for c in exog_train.columns if exog_train[c].nunique(dropna=True) <= 1]
    if const_cols:
        print(f"[DEBUG] Dropping constant exog columns (train): {const_cols}")
        exog_train.drop(columns=const_cols, inplace=True)
        exog_test.drop(columns=const_cols, inplace=True, errors='ignore')

    # Align column sets (take intersection)
    common_cols = [c for c in exog_train.columns if c in exog_test.columns]
    exog_train = exog_train[common_cols]
    exog_test = exog_test[common_cols]

    # Align with target index (reindex and fill if needed)
    exog_train = exog_train.reindex(target_index)
    exog_test = exog_test.reindex(pd.date_range(start=exog_test.index.min(), periods=len(exog_test), freq=exog_test.index.freq or "MS"))

    # Interpolate and forward/backfill on both
    for df in (exog_train, exog_test):
        if df.isna().sum().sum() > 0:
            print(f"[DEBUG] Interpolating/filling NaNs in exog (shape={df.shape})")
            df.interpolate(method='linear', limit_direction='both', inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

    return exog_train, exog_test


def select_exogenous_variables(df, target_col, candidate_exogs, method='correlation', top_k=5):
    """
    Safe selection of exogenous variables (drops constant/all-NaN candidates).
    Returns selected names and full scores dict.
    """
    available = [c for c in candidate_exogs if c in df.columns]
    if not available:
        return [], {}

    # Prepare temporary exog frame
    tmp = df[available].copy()
    # drop all-NaN and constants
    tmp.dropna(axis=1, how='all', inplace=True)
    consts = [c for c in tmp.columns if tmp[c].nunique(dropna=True) <= 1]
    if consts:
        print(f"[DEBUG] Dropping constant candidates: {consts}")
        tmp.drop(columns=consts, inplace=True)

    # align with target index and fill temporarily
    tmp = tmp.reindex(df[target_col].index)
    tmp = tmp.fillna(0)

    # compute correlations
    scores = {}
    for c in tmp.columns:
        try:
            corr = df[target_col].corr(tmp[c])
            scores[c] = abs(corr) if not np.isnan(corr) else 0.0
        except Exception:
            scores[c] = 0.0

    sorted_vars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [v for v, s in sorted_vars[:top_k]]
    return selected, dict(sorted_vars)


def train_causal_models(train_data, test_data, exog_train, exog_test, baseline_orders=None):
    results = {'models': {}, 'predictions': {}, 'metrics': {}, 'model_info': {}, 'errors': {}}
    n_forecast = len(test_data)

    # Ensure datetime freq on train/test
    train_data = train_data.copy()
    test_data = test_data.copy()
    if not isinstance(train_data.index, pd.DatetimeIndex):
        raise ValueError("train_data index must be a DatetimeIndex")
    if not isinstance(test_data.index, pd.DatetimeIndex):
        raise ValueError("test_data index must be a DatetimeIndex")

    train_data = train_data.asfreq(train_data.index.freq or "MS")
    test_data = test_data.asfreq(test_data.index.freq or "MS")

    # Decide seasonal_order based on data length (automatic heuristic)
    if baseline_orders and 'ARIMA' in baseline_orders:
        info = baseline_orders['ARIMA']
        order = info.get('order', (1, 1, 1))
        seasonal_order = info.get('seasonal_order', None)
    else:
        order = (1, 1, 1)
        # heuristic: if enough history use annual seasonality, else smaller or none
        if len(train_data) >= 36:
            seasonal_order = (1, 1, 1, 12)
        elif len(train_data) >= 18:
            seasonal_order = (1, 1, 1, 3)
        else:
            seasonal_order = None
        print(f"[DEBUG] Using default heuristic orders: order={order}, seasonal_order={seasonal_order}")

    # Clean training series
    if train_data.isna().any():
        print("[DEBUG] Interpolating/filling NaNs in target")
        train_data.interpolate(method='linear', inplace=True)
        train_data.fillna(method='ffill', inplace=True)
        train_data.fillna(method='bfill', inplace=True)

    # Clean and align exog
    exog_train_clean, exog_test_clean = _clean_and_align_exog(exog_train, exog_test, train_data.index)

    # Check data sufficiency
    if len(train_data) < 8:
        results['errors']['SARIMAX'] = "Insufficient training length (<8)"
        print("[ERROR] Insufficient training length for SARIMAX")
        return results

    # If seasonal_order is None, set to (0,0,0,0) for fitting plain ARIMA via SARIMAX (statsmodels tolerates it)
    seasonal_order_for_fit = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)

    try:
        model = SARIMAXModel(order=order, seasonal_order=seasonal_order_for_fit)
        model.fit(train_data, exog_data=exog_train_clean, order=order, seasonal_order=seasonal_order_for_fit)

        preds, conf = model.predict(n_periods=n_forecast, exog_data=exog_test_clean, return_conf_int=True)

        # compute metrics
        sarimax_metrics = compute_metrics(test_data.values, preds.values)
        results['models']['SARIMAX'] = model
        results['predictions']['SARIMAX'] = {'forecast': preds, 'confidence_intervals': conf}
        results['metrics']['SARIMAX'] = sarimax_metrics
        results['model_info']['SARIMAX'] = model.get_model_summary()

        print(f"[DEBUG] SARIMAX metrics: {sarimax_metrics}")

        # print coefficients if available
        summary = model.get_model_summary()
        if summary.get("exog_coefficients"):
            print("\nExogenous variable coefficients:")
            for k, v in summary["exog_coefficients"].items():
                try:
                    print(f"  {k}: {v:.4f}")
                except Exception:
                    print(f"  {k}: {v}")

    except Exception as e:
        print("[ERROR] SARIMAX training/prediction failed:")
        traceback.print_exc()
        results['errors']['SARIMAX'] = str(e)

    return results


def analyze_causal_impact(sarimax_model, exog_names):
    """
    Analyze the causal impact of exogenous variables.
    
    Args:
        sarimax_model (SARIMAXModel): Fitted SARIMAX model
        exog_names (list): Names of exogenous variables
        
    Returns:
        pd.DataFrame: Analysis of variable impacts
    """
    if not sarimax_model.fitted:
        return pd.DataFrame()
    
    try:
        # Get parameter estimates and standard errors
        params = sarimax_model.fitted_model.params
        std_errors = sarimax_model.fitted_model.bse
        
        # Filter for exogenous variables
        exog_data = []
        for name in exog_names:
            for param_name in params.index:
                if name.lower() in param_name.lower():
                    exog_data.append({
                        'variable': name,
                        'parameter': param_name,
                        'coefficient': params[param_name],
                        'std_error': std_errors[param_name],
                        'abs_coefficient': abs(params[param_name])
                    })
        
        if exog_data:
            impact_df = pd.DataFrame(exog_data)
            impact_df = impact_df.sort_values('abs_coefficient', ascending=False)
            return impact_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error analyzing causal impact: {e}")
        return pd.DataFrame()
