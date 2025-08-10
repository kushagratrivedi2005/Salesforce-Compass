"""
Model training and evaluation utilities for the forecasting pipeline.

This module provides:
- Performance metrics calculation (MAE, RMSE, MAPE)
- Model comparison functions
- Results saving utilities
- Common model fitting patterns
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", message="'force_all_finite'")  # Suppress the specific warning


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive evaluation metrics for forecasting models.
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing MAE, RMSE, and MAPE metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    # Fix: Calculate RMSE manually instead of using squared=False parameter
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Handle zeros in y_true for MAPE calculation
    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def compare_models(results_dict, test_actual):
    """
    Compare multiple models and create metrics summary.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and predictions as values
        test_actual (array-like): Actual test values
        
    Returns:
        pd.DataFrame: Comparison table with metrics for each model
    """
    metrics_list = []
    model_names = []
    
    for model_name, predictions in results_dict.items():
        metrics = compute_metrics(test_actual, predictions)
        metrics_list.append(metrics)
        model_names.append(model_name)
    
    metrics_df = pd.DataFrame(metrics_list, index=model_names).T
    metrics_df = metrics_df.round(4)
    
    return metrics_df


def save_model_results(results_df, metrics_df, results_dir, file_prefix="model"):
    """
    Save model results and metrics to CSV files.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing predictions and actual values
        metrics_df (pd.DataFrame): DataFrame containing model metrics
        results_dir (str): Directory to save results
        file_prefix (str): Prefix for output filenames
    """
    # Save predictions
    predictions_file = os.path.join(results_dir, f'{file_prefix}_predictions.csv')
    results_df.to_csv(predictions_file)
    print(f'Saved predictions to {predictions_file}')
    
    # Save metrics
    metrics_file = os.path.join(results_dir, f'{file_prefix}_metrics.csv')
    metrics_df.to_csv(metrics_file)
    print(f'Saved metrics to {metrics_file}')
    
    return predictions_file, metrics_file


def extract_model_parameters(model_result, exog_names=None):
    """
    Extract and format model parameters for analysis.
    
    Args:
        model_result: Fitted model result object
        exog_names (list, optional): Names of exogenous variables
        
    Returns:
        pd.Series: Formatted parameter estimates
    """
    if not hasattr(model_result, 'params'):
        return pd.Series(dtype=float)
    
    params = model_result.params
    
    if exog_names:
        # Filter for exogenous parameters only
        exog_params = {k: v for k, v in params.items() 
                      if any(name in k for name in exog_names)}
        params_series = pd.Series(exog_params)
    else:
        params_series = pd.Series(params)
    
    # Sort by absolute value
    params_series = params_series.sort_values(key=lambda x: x.abs(), ascending=False)
    
    return params_series


def validate_predictions(predictions, test_data):
    """
    Validate prediction results and check for common issues.
    
    Args:
        predictions (array-like): Model predictions
        test_data (array-like): Test data for comparison
        
    Returns:
        dict: Validation results with warnings and statistics
    """
    validation_results = {
        'has_nan': np.isnan(predictions).any(),
        'has_negative': (predictions < 0).any() if not np.isnan(predictions).any() else False,
        'mean_prediction': np.nanmean(predictions),
        'mean_actual': np.nanmean(test_data),
        'prediction_range': (np.nanmin(predictions), np.nanmax(predictions)),
        'actual_range': (np.nanmin(test_data), np.nanmax(test_data))
    }
    
    # Generate warnings
    warnings = []
    if validation_results['has_nan']:
        warnings.append("Predictions contain NaN values")
    if validation_results['has_negative']:
        warnings.append("Predictions contain negative values")
    
    validation_results['warnings'] = warnings
    
    return validation_results


def create_results_summary(model_results, target_name):
    """
    Create a comprehensive summary of model results.
    
    Args:
        model_results (dict): Dictionary containing model results
        target_name (str): Name of the target variable
        
    Returns:
        dict: Comprehensive results summary
    """
    summary = {
        'target_variable': target_name,
        'models_trained': list(model_results.get('predictions', {}).keys()),
        'best_model_mae': None,
        'best_model_rmse': None,
        'best_model_mape': None,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Find best models by each metric
    if 'metrics' in model_results:
        metrics_dict = model_results['metrics']
        
        # Convert metrics dict to DataFrame if it's not already one
        if isinstance(metrics_dict, dict):
            metrics_df = pd.DataFrame(metrics_dict).T
        else:
            metrics_df = metrics_dict
        
        if 'MAE' in metrics_df.index:
            best_mae = metrics_df.loc['MAE'].idxmin()
            summary['best_model_mae'] = best_mae
        
        if 'RMSE' in metrics_df.index:
            best_rmse = metrics_df.loc['RMSE'].idxmin()
            summary['best_model_rmse'] = best_rmse
            
        if 'MAPE' in metrics_df.index:
            best_mape = metrics_df.loc['MAPE'].idxmin()
            summary['best_model_mape'] = best_mape
    
    return summary
