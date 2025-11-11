"""
Visualization utilities for the vehicle sales forecasting pipeline.

This module provides:
- Forecast plotting with confidence intervals
- Model comparison visualizations
- Diagnostic plots for model evaluation
- Results summary visualizations
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import PLOT_CONFIG


def plot_forecast(train_data, test_data, predictions, confidence_intervals=None,
                 title='Forecast', ylabel='Value', save_path=None, show_plot=True):
    """
    Plot time series forecast with actual vs predicted values.
    
    Args:
        train_data (pd.Series): Training data
        test_data (pd.Series): Test/actual data
        predictions (pd.Series): Model predictions
        confidence_intervals (pd.DataFrame): Confidence intervals (optional)
        title (str): Plot title
        ylabel (str): Y-axis label
        save_path (str): Path to save plot (optional)
        show_plot (bool): Whether to display plot
    """
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    
    # Plot actual test data
    plt.plot(test_data.index, test_data.values, 
            label='Actual (Test)', marker=PLOT_CONFIG['marker'], 
            color='green', linewidth=2)
    
    # Plot predictions
    plt.plot(test_data.index, predictions.values, 
            label='Forecast', marker=PLOT_CONFIG['marker'], 
            color='red', linewidth=2)
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        plt.fill_between(test_data.index, 
                        confidence_intervals.iloc[:, 0], 
                        confidence_intervals.iloc[:, 1], 
                        color='gray', alpha=PLOT_CONFIG['alpha'], 
                        label='Confidence Interval')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches=PLOT_CONFIG['bbox_inches'], 
                   dpi=300, facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(test_data, predictions_dict, title='Model Comparison',
                         ylabel='Value', save_path=None, show_plot=True):
    """
    Compare multiple model predictions on the same plot.
    
    Args:
        test_data (pd.Series): Actual test data
        predictions_dict (dict): Dictionary of model predictions
        title (str): Plot title
        ylabel (str): Y-axis label
        save_path (str): Path to save plot
        show_plot (bool): Whether to display plot
    """
    plt.figure(figsize=(14, 6))
    
    # Plot actual data
    plt.plot(test_data.index, test_data.values, 
            label='Actual', marker='o', color='black', 
            linewidth=3, markersize=6)
    
    # Plot predictions from different models
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(test_data.index, predictions.values,
                label=f'{model_name}', marker='s', 
                color=color, linewidth=2, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches=PLOT_CONFIG['bbox_inches'], 
                   dpi=300, facecolor='white')
        print(f"Comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(metrics_df, title='Model Performance Comparison',
                           save_path=None, show_plot=True):
    """
    Create bar plots comparing model performance metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics as columns and models as index
        title (str): Plot title
        save_path (str): Path to save plot
        show_plot (bool): Whether to display plot
    """
    # Transpose the DataFrame if metrics are in columns (to make metrics the index)
    if 'MAE' in metrics_df.columns:
        metrics_df = metrics_df.transpose()
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    for i, metric in enumerate(metrics):
        if metric in metrics_df.index:
            ax = axes[i]
            metric_data = metrics_df.loc[metric]
            
            bars = ax.bar(range(len(metric_data)), metric_data.values, 
                         color=['skyblue', 'lightcoral', 'lightgreen'][:len(metric_data)])
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_xlabel('Models')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(metric_data.index, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches=PLOT_CONFIG['bbox_inches'], 
                   dpi=300, facecolor='white')
        print(f"Metrics comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_residuals_analysis(model_result, title='Residuals Analysis',
                           save_path=None, show_plot=True):
    """
    Plot residuals analysis for model diagnostics.
    
    Args:
        model_result: Fitted model result with residuals
        title (str): Plot title
        save_path (str): Path to save plot
        show_plot (bool): Whether to display plot
    """
    if not hasattr(model_result, 'resid'):
        print("Model result does not have residuals attribute")
        return
    
    residuals = model_result.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, marker='o', alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals.dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Residuals Histogram')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot (approximate)
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ACF of residuals (simplified)
    axes[1, 1].acorr(residuals.dropna(), maxlags=20, alpha=0.05)
    axes[1, 1].set_title('Residuals Autocorrelation')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches=PLOT_CONFIG['bbox_inches'], 
                   dpi=300, facecolor='white')
        print(f"Residuals analysis plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_exog_coefficients(coefficients_dict, title='Exogenous Variable Coefficients',
                          save_path=None, show_plot=True):
    """
    Plot exogenous variable coefficients from SARIMAX model.
    
    Args:
        coefficients_dict (dict): Dictionary of variable coefficients
        title (str): Plot title
        save_path (str): Path to save plot
        show_plot (bool): Whether to display plot
    """
    if not coefficients_dict:
        print("No coefficients to plot")
        return
    
    # Convert to series and sort by absolute value
    coef_series = pd.Series(coefficients_dict)
    coef_series = coef_series.reindex(
        coef_series.abs().sort_values(ascending=True).index
    )
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar plot
    colors = ['red' if x < 0 else 'blue' for x in coef_series.values]
    bars = plt.barh(range(len(coef_series)), coef_series.values, color=colors, alpha=0.7)
    
    plt.yticks(range(len(coef_series)), coef_series.index)
    plt.xlabel('Coefficient Value')
    plt.title(title, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, coef_series.values)):
        plt.text(value + (0.01 if value >= 0 else -0.01), i,
                f'{value:.4f}', va='center', 
                ha='left' if value >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches=PLOT_CONFIG['bbox_inches'], 
                   dpi=300, facecolor='white')
        print(f"Coefficients plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_forecast_dashboard(results_dict, target_name, save_dir=None):
    """
    Create a comprehensive dashboard with all forecast visualizations.
    
    Args:
        results_dict (dict): Complete results from model training
        target_name (str): Name of target variable
        save_dir (str): Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("Creating forecast dashboard...")
    
    # Individual model plots
    if 'predictions' in results_dict:
        for model_name, pred_data in results_dict['predictions'].items():
            if 'train_data' in results_dict and 'test_data' in results_dict:
                conf_int = pred_data.get('confidence_intervals')
                save_path = os.path.join(save_dir, f'{model_name}_forecast.png') if save_dir else None
                
                plot_forecast(
                    results_dict['train_data'],
                    results_dict['test_data'],
                    pred_data['forecast'],
                    confidence_intervals=conf_int,
                    title=f'{model_name} Forecast - {target_name}',
                    ylabel=target_name,
                    save_path=save_path,
                    show_plot=False
                )
    
    # Model comparison plot
    if 'predictions' in results_dict and 'test_data' in results_dict:
        predictions_dict = {name: data['forecast'] 
                          for name, data in results_dict['predictions'].items()}
        save_path = os.path.join(save_dir, 'model_comparison.png') if save_dir else None
        
        plot_model_comparison(
            results_dict['test_data'],
            predictions_dict,
            title=f'Model Comparison - {target_name}',
            ylabel=target_name,
            save_path=save_path,
            show_plot=False
        )
    
    # Metrics comparison
    if 'metrics' in results_dict:
        metrics_df = pd.DataFrame(results_dict['metrics']).T
        save_path = os.path.join(save_dir, 'metrics_comparison.png') if save_dir else None
        
        plot_metrics_comparison(
            metrics_df,
            title='Model Performance Comparison',
            save_path=save_path,
            show_plot=False
        )
    
    # SARIMAX coefficients plot
    if 'model_info' in results_dict and 'SARIMAX' in results_dict['model_info']:
        sarimax_info = results_dict['model_info']['SARIMAX']
        if 'exog_coefficients' in sarimax_info:
            save_path = os.path.join(save_dir, 'sarimax_coefficients.png') if save_dir else None
            
            plot_exog_coefficients(
                sarimax_info['exog_coefficients'],
                title='SARIMAX Exogenous Variable Coefficients',
                save_path=save_path,
                show_plot=False
            )
    
    print(f"Dashboard created. Plots saved to: {save_dir}")
