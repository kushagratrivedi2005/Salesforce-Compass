"""
Causal forecasting models with exogenous variables for vehicle sales prediction.
"""

import numpy as np
import pandas as pd
import traceback
from statsmodels.tsa.statespace.sarimax import SARIMAX
from model_utils import compute_metrics, extract_model_parameters
import warnings
warnings.filterwarnings("ignore")


class SARIMAXModel:
    """SARIMAX model with exogenous variables."""
    
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.fitted = False
        self.exog_names = None

    def _ensure_freq(self, data):
        """Ensure monthly frequency."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        if data.index.freq is None:
            data = data.asfreq("MS")
        return data

    def fit(self, train_data, exog_data=None, order=None, seasonal_order=None):
        """Fit SARIMAX model."""
        if order:
            self.order = order
        if seasonal_order:
            self.seasonal_order = seasonal_order
        if not self.order:
            raise ValueError("ARIMA order required")

        train_data = self._ensure_freq(train_data)
        if exog_data is not None:
            exog_data = self._ensure_freq(exog_data.copy())
            self.exog_names = list(exog_data.columns)

        try:
            model = SARIMAX(
                endog=train_data,
                exog=exog_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = model.fit(maxiter=100, disp=False, method='lbfgs')
            self.fitted = True
        except Exception as e:
            print(f"[ERROR] SARIMAX fitting failed: {e}")
            raise
        return self

    def predict(self, n_periods, exog_data=None, return_conf_int=True):
        """Generate predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        if exog_data is not None:
            if len(exog_data) != n_periods:
                raise ValueError(f"exog_data length mismatch")
            exog_data = self._ensure_freq(exog_data)

        forecast_obj = self.fitted_model.get_forecast(steps=n_periods, exog=exog_data)
        preds = forecast_obj.predicted_mean
        
        if return_conf_int:
            return preds, forecast_obj.conf_int()
        return preds

    def get_model_summary(self):
        """Get model summary."""
        if not self.fitted:
            return {"status": "not_fitted"}
        
        summary = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_variables": self.exog_names,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "fitted": True
        }
        
        if self.exog_names:
            try:
                exog_params = extract_model_parameters(self.fitted_model, self.exog_names)
                summary["exog_coefficients"] = exog_params.to_dict()
            except:
                pass
        return summary


def _clean_and_align_exog(exog_train, exog_test, target_index):
    """Clean and align exogenous variables."""
    exog_train = exog_train.copy()
    exog_test = exog_test.copy()

    # Replace infinities
    exog_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    exog_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop all-NaN and constant columns
    allnan_cols = [c for c in exog_train.columns if exog_train[c].isna().all()]
    const_cols = [c for c in exog_train.columns if exog_train[c].nunique(dropna=True) <= 1]
    drop_cols = set(allnan_cols + const_cols)
    
    if drop_cols:
        print(f"[DEBUG] Dropping {len(drop_cols)} problematic columns")
        exog_train.drop(columns=drop_cols, inplace=True)
        exog_test.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Keep common columns only
    common_cols = [c for c in exog_train.columns if c in exog_test.columns]
    exog_train = exog_train[common_cols]
    exog_test = exog_test[common_cols]

    # Align with target index
    exog_train = exog_train.reindex(target_index)
    exog_test = exog_test.reindex(pd.date_range(
        start=exog_test.index.min(), 
        periods=len(exog_test), 
        freq=exog_test.index.freq or "MS"
    ))

    # Fill missing values
    for df in (exog_train, exog_test):
        if df.isna().sum().sum() > 0:
            df.interpolate(method='linear', limit_direction='both', inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

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

    # Use the best order from baseline ARIMA if available
    if baseline_orders and 'ARIMA' in baseline_orders:
        info = baseline_orders['ARIMA']
        order = info.get('order', (1, 1, 1))
        seasonal_order = info.get('seasonal_order', (1, 1, 1, 12))
        print(f"[INFO] Using ARIMA baseline orders: order={order}, seasonal_order={seasonal_order}")
    else:
        # More robust default orders
        order = (2, 1, 2)
        seasonal_order = (1, 1, 1, 12) if len(train_data) >= 36 else (0, 0, 0, 0)
        print(f"[DEBUG] Using default orders: order={order}, seasonal_order={seasonal_order}")

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

    try:
        model = SARIMAXModel(order=order, seasonal_order=seasonal_order)
        model.fit(train_data, exog_data=exog_train_clean, order=order, seasonal_order=seasonal_order)

        preds, conf = model.predict(n_periods=n_forecast, exog_data=exog_test_clean, return_conf_int=True)

        # compute metrics
        sarimax_metrics = compute_metrics(test_data.values, preds.values)
        results['models']['SARIMAX'] = model
        results['predictions']['SARIMAX'] = {'forecast': preds, 'confidence_intervals': conf}
        results['metrics']['SARIMAX'] = sarimax_metrics
        results['model_info']['SARIMAX'] = model.get_model_summary()

        print(f"✓ SARIMAX metrics: MAE={sarimax_metrics['MAE']:.4f}, RMSE={sarimax_metrics['RMSE']:.4f}, MAPE={sarimax_metrics['MAPE']:.2f}%")

        # print coefficients if available
        summary = model.get_model_summary()
        if summary.get("exog_coefficients"):
            print("\n→ Exogenous variable coefficients:")
            for k, v in summary["exog_coefficients"].items():
                try:
                    print(f"  • {k}: {v:.4f}")
                except Exception:
                    print(f"  • {k}: {v}")

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
