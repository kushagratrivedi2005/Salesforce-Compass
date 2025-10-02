"""
Flask API Server for Vehicle Sales Forecasting System

This Flask application serves the output of the vehicle sales forecasting models
through REST API endpoints. It provides access to model predictions, performance
metrics, and comparative analysis.

API Endpoints:
- GET /health - Health check
- GET /api/v1/predictions - All model predictions
- GET /api/v1/metrics - Model performance metrics
- GET /api/v1/models - Available models information
- GET /api/v1/forecast/{model_name} - Specific model forecast
- GET /api/v1/comparison - Model comparison data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'forecast_results')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'forecast_predictions.csv')
METRICS_FILE = os.path.join(RESULTS_DIR, 'forecast_metrics.csv')

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder

def load_forecast_data() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load forecast predictions and metrics from CSV files.
    
    Returns:
        tuple: (predictions_df, metrics_df) or (None, None) if files not found
    """
    try:
        if os.path.exists(PREDICTIONS_FILE) and os.path.exists(METRICS_FILE):
            predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col=0, parse_dates=True)
            metrics_df = pd.read_csv(METRICS_FILE, index_col=0)
            logger.info("Successfully loaded forecast data")
            return predictions_df, metrics_df
        else:
            logger.error(f"Forecast files not found: {PREDICTIONS_FILE}, {METRICS_FILE}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading forecast data: {str(e)}")
        return None, None

def format_datetime_index(df: pd.DataFrame) -> Dict:
    """
    Format DataFrame with datetime index for JSON response.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        dict: Formatted data with ISO datetime strings
    """
    result = {}
    for col in df.columns:
        result[col] = []
        for idx, val in df[col].items():
            result[col].append({
                'date': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'value': float(val) if pd.notna(val) else None
            })
    return result

def create_error_response(message: str, status_code: int = 500) -> tuple:
    """
    Create standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        tuple: (response_dict, status_code)
    """
    return {
        'error': True,
        'message': message,
        'timestamp': datetime.utcnow().isoformat(),
        'status_code': status_code
    }, status_code

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'Vehicle Sales Forecasting API',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }

@app.route('/api/v1/predictions', methods=['GET'])
def get_all_predictions():
    """
    Get all model predictions with actual values.
    
    Returns:
        JSON response with predictions from all models
    """
    try:
        predictions_df, _ = load_forecast_data()
        if predictions_df is None:
            return create_error_response("Forecast data not available", 503)
        
        # Format the response
        response_data = {
            'models': list(predictions_df.columns),
            'time_series': format_datetime_index(predictions_df),
            'summary': {
                'total_periods': len(predictions_df),
                'date_range': {
                    'start': predictions_df.index.min().isoformat(),
                    'end': predictions_df.index.max().isoformat()
                }
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'data_source': 'Vehicle Sales Forecasting System'
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_all_predictions: {str(e)}")
        return create_error_response(f"Internal server error: {str(e)}")

@app.route('/api/v1/metrics', methods=['GET'])
def get_model_metrics():
    """
    Get performance metrics for all models.
    
    Returns:
        JSON response with MAE, RMSE, and MAPE for each model
    """
    try:
        _, metrics_df = load_forecast_data()
        if metrics_df is None:
            return create_error_response("Metrics data not available", 503)
        
        # Format metrics data
        models_metrics = {}
        for model_name in metrics_df.index:
            models_metrics[model_name] = {
                'MAE': float(metrics_df.loc[model_name, 'MAE']) if pd.notna(metrics_df.loc[model_name, 'MAE']) else None,
                'RMSE': float(metrics_df.loc[model_name, 'RMSE']) if pd.notna(metrics_df.loc[model_name, 'RMSE']) else None,
                'MAPE': float(metrics_df.loc[model_name, 'MAPE']) if pd.notna(metrics_df.loc[model_name, 'MAPE']) else None
            }
        
        # Find best performing models
        best_models = {}
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in metrics_df.columns:
                best_model = metrics_df[metric].idxmin()
                best_value = float(metrics_df[metric].min())
                best_models[metric] = {
                    'model': best_model,
                    'value': best_value
                }
        
        response_data = {
            'models': models_metrics,
            'best_performing': best_models,
            'model_count': len(models_metrics),
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics_description': {
                    'MAE': 'Mean Absolute Error',
                    'RMSE': 'Root Mean Square Error',
                    'MAPE': 'Mean Absolute Percentage Error (%)'
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_model_metrics: {str(e)}")
        return create_error_response(f"Internal server error: {str(e)}")

@app.route('/api/v1/models', methods=['GET'])
def get_models_info():
    """
    Get information about available models.
    
    Returns:
        JSON response with model details and descriptions
    """
    try:
        predictions_df, metrics_df = load_forecast_data()
        if predictions_df is None or metrics_df is None:
            return create_error_response("Model data not available", 503)
        
        models_info = {}
        model_columns = [col for col in predictions_df.columns if col != 'Actual']
        
        for model_name in model_columns:
            models_info[model_name] = {
                'name': model_name,
                'type': get_model_type(model_name),
                'description': get_model_description(model_name),
                'available': True,
                'metrics': {
                    'MAE': float(metrics_df.loc[model_name, 'MAE']) if model_name in metrics_df.index and pd.notna(metrics_df.loc[model_name, 'MAE']) else None,
                    'RMSE': float(metrics_df.loc[model_name, 'RMSE']) if model_name in metrics_df.index and pd.notna(metrics_df.loc[model_name, 'RMSE']) else None,
                    'MAPE': float(metrics_df.loc[model_name, 'MAPE']) if model_name in metrics_df.index and pd.notna(metrics_df.loc[model_name, 'MAPE']) else None
                }
            }
        
        response_data = {
            'models': models_info,
            'total_models': len(models_info),
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'forecast_period': {
                    'start': predictions_df.index.min().isoformat(),
                    'end': predictions_df.index.max().isoformat(),
                    'periods': len(predictions_df)
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_models_info: {str(e)}")
        return create_error_response(f"Internal server error: {str(e)}")

@app.route('/api/v1/forecast/<model_name>', methods=['GET'])
def get_model_forecast(model_name: str):
    """
    Get forecast for a specific model.
    
    Args:
        model_name: Name of the model (ARIMA, ETS, SARIMAX)
        
    Returns:
        JSON response with forecast data for the specified model
    """
    try:
        predictions_df, metrics_df = load_forecast_data()
        if predictions_df is None:
            return create_error_response("Forecast data not available", 503)
        
        # Check if model exists
        if model_name not in predictions_df.columns:
            available_models = [col for col in predictions_df.columns if col != 'Actual']
            return create_error_response(
                f"Model '{model_name}' not found. Available models: {available_models}", 
                404
            )
        
        # Get model predictions
        model_predictions = []
        for idx, val in predictions_df[model_name].items():
            actual_val = predictions_df.loc[idx, 'Actual'] if 'Actual' in predictions_df.columns else None
            model_predictions.append({
                'date': idx.isoformat(),
                'forecast': float(val) if pd.notna(val) else None,
                'actual': float(actual_val) if pd.notna(actual_val) else None
            })
        
        # Get model metrics
        model_metrics = {}
        if metrics_df is not None and model_name in metrics_df.index:
            model_metrics = {
                'MAE': float(metrics_df.loc[model_name, 'MAE']) if pd.notna(metrics_df.loc[model_name, 'MAE']) else None,
                'RMSE': float(metrics_df.loc[model_name, 'RMSE']) if pd.notna(metrics_df.loc[model_name, 'RMSE']) else None,
                'MAPE': float(metrics_df.loc[model_name, 'MAPE']) if pd.notna(metrics_df.loc[model_name, 'MAPE']) else None
            }
        
        response_data = {
            'model': {
                'name': model_name,
                'type': get_model_type(model_name),
                'description': get_model_description(model_name)
            },
            'forecast': model_predictions,
            'metrics': model_metrics,
            'summary': {
                'total_periods': len(model_predictions),
                'date_range': {
                    'start': predictions_df.index.min().isoformat(),
                    'end': predictions_df.index.max().isoformat()
                }
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_model_forecast for {model_name}: {str(e)}")
        return create_error_response(f"Internal server error: {str(e)}")

@app.route('/api/v1/comparison', methods=['GET'])
def get_model_comparison():
    """
    Get comparative analysis of all models.
    
    Returns:
        JSON response with side-by-side comparison of all models
    """
    try:
        predictions_df, metrics_df = load_forecast_data()
        if predictions_df is None or metrics_df is None:
            return create_error_response("Comparison data not available", 503)
        
        model_columns = [col for col in predictions_df.columns if col != 'Actual']
        
        # Create comparison data
        comparison_data = []
        for idx in predictions_df.index:
            period_data = {
                'date': idx.isoformat(),
                'actual': float(predictions_df.loc[idx, 'Actual']) if 'Actual' in predictions_df.columns and pd.notna(predictions_df.loc[idx, 'Actual']) else None
            }
            
            for model in model_columns:
                period_data[model] = float(predictions_df.loc[idx, model]) if pd.notna(predictions_df.loc[idx, model]) else None
            
            comparison_data.append(period_data)
        
        # Calculate accuracy rankings
        rankings = {}
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in metrics_df.columns:
                sorted_models = metrics_df[metric].sort_values()
                rankings[metric] = [
                    {
                        'rank': idx + 1,
                        'model': model,
                        'value': float(value)
                    }
                    for idx, (model, value) in enumerate(sorted_models.items())
                ]
        
        response_data = {
            'comparison': comparison_data,
            'model_rankings': rankings,
            'summary': {
                'models_compared': len(model_columns),
                'time_periods': len(comparison_data),
                'best_overall': metrics_df['MAPE'].idxmin() if 'MAPE' in metrics_df.columns else None
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'ranking_criteria': 'Lower values indicate better performance'
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_model_comparison: {str(e)}")
        return create_error_response(f"Internal server error: {str(e)}")

def get_model_type(model_name: str) -> str:
    """Get model type description"""
    model_types = {
        'ARIMA': 'Time Series',
        'ETS': 'Exponential Smoothing',
        'SARIMAX': 'Causal Time Series'
    }
    return model_types.get(model_name, 'Unknown')

def get_model_description(model_name: str) -> str:
    """Get model description"""
    descriptions = {
        'ARIMA': 'AutoRegressive Integrated Moving Average model for time series forecasting',
        'ETS': 'Exponential Smoothing (Holt-Winters) model with trend and seasonality',
        'SARIMAX': 'Seasonal ARIMA with eXogenous variables including economic indicators and policy factors'
    }
    return descriptions.get(model_name, 'Forecasting model')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_error_response("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return create_error_response("Internal server error", 500)

if __name__ == '__main__':
    # Check if forecast data exists
    if not os.path.exists(RESULTS_DIR):
        logger.error(f"Results directory not found: {RESULTS_DIR}")
        logger.info("Please run the forecasting pipeline first: python main_pipeline.py")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
