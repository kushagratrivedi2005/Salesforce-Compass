"""
Main pipeline for vehicle sales forecasting.

This script orchestrates the entire forecasting pipeline:
1. Data loading and preprocessing
2. Train/test splitting
3. Baseline model training and evaluation
4. Causal model training with exogenous variables
5. Model comparison and visualization
6. Results export
"""

import os
import pandas as pd
import numpy as np
import sys
import warnings

# Import modules from the project
from data_loader import load_and_prepare, train_test_split_by_date, prepare_exogenous_variables, ensure_dir
from baseline_models import train_baseline_models, PM_AVAILABLE
from causal_models import train_causal_models, analyze_causal_impact
from model_utils import create_results_summary
from visualization import plot_forecast, plot_model_comparison, plot_metrics_comparison, create_forecast_dashboard

# Import configuration
from config import (
    CSV_PATH, RESULTS_DIR, TARGET, TEST_MONTHS, 
    CANDIDATE_EXOGS, TOP_K_EXOGS, DATASET_DIR
)

def run_forecasting_pipeline(csv_path=CSV_PATH, target=TARGET, 
                            test_months=TEST_MONTHS, results_dir=RESULTS_DIR):
    """
    Run the complete forecasting pipeline.
    
    Args:
        csv_path (str): Path to the input CSV file
        target (str): Target variable for forecasting
        test_months (int): Number of months to use for testing
        results_dir (str): Directory to save results
    
    Returns:
        dict: Complete results from the pipeline
    """
    # Check for pmdarima availability
    if not PM_AVAILABLE:
        print("\nWARNING: pmdarima not available. Using statsmodels fallback.")
        print("This will use predefined ARIMA orders instead of automatic order selection.")
        print("For better results, fix the pmdarima installation.")
        print("Suggestion: Try 'pip install --upgrade --force-reinstall pmdarima numpy'")
        print("or create a new environment with compatible package versions.\n")

    # Step 1: Create necessary directories
    print("\n" + "="*80)
    print("Step 1: Preparing environment")
    print("="*80)
    ensure_dir(results_dir)
    
    # Check if data exists at the specified path
    full_csv_path = os.path.join(DATASET_DIR, csv_path)
    if not os.path.exists(full_csv_path):
        print(f"Error: CSV file not found at {full_csv_path}")
        return None
    
    # Step 2: Load and preprocess data
    print("\n" + "="*80)
    print("Step 2: Loading and preprocessing data")
    print("="*80)
    df = load_and_prepare(full_csv_path)
    
    if target not in df.columns:
        print(f"Error: Target variable '{target}' not found in the dataset")
        print(f"Available columns: {', '.join(df.columns)}")
        return None
    
    print(f"Target variable: {target}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 3: Split data into train and test sets
    print("\n" + "="*80)
    print(f"Step 3: Splitting data into train and test sets")
    print("="*80)
    target_series = df[target]
    
    # Use specific date ranges as requested
    train_end_date = '2025-01-01'  # Train data ends on 2025-01-01
    test_end_date = '2025-07-01'   # Test data ends on 2025-07-01
    
    train_data, test_data = train_test_split_by_date(
        target_series,
        train_end_date=train_end_date,
        test_end_date=test_end_date
    )
    
    print(f"Training data: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Test data: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")
    
    # Step 4: Train baseline models
    print("\n" + "="*80)
    print("Step 4: Training baseline forecasting models")
    print("="*80)
    baseline_results = train_baseline_models(train_data, test_data)
    
    # Step 5: Prepare exogenous variables for causal models
    print("\n" + "="*80)
    print("Step 5: Preparing exogenous variables for causal models")
    print("="*80)
    top_exogs, exog_df, correlations = prepare_exogenous_variables(
        df, CANDIDATE_EXOGS, target, top_k=TOP_K_EXOGS
    )
    
    print(f"Selected top {len(top_exogs)} exogenous variables:")
    for exog in top_exogs:
        print(f"  - {exog} (correlation: {correlations[exog]:.4f})")
    
    # Split exogenous variables into train and test sets using the same date ranges
    exog_train, exog_test = train_test_split_by_date(
        exog_df,
        train_end_date=train_end_date,
        test_end_date=test_end_date
    )
    
    # Step 6: Train causal models
    print("\n" + "="*80)
    print("Step 6: Training causal models with exogenous variables")
    print("="*80)
    baseline_orders = baseline_results.get('model_info', {})
    causal_results = train_causal_models(train_data, test_data, exog_train, exog_test, baseline_orders)
    
    # Step 7: Combine results from all models
    print("\n" + "="*80)
    print("Step 7: Combining and comparing all models")
    print("="*80)
    
    # Merge baseline and causal model results
    all_results = {
        'train_data': train_data,
        'test_data': test_data,
        'predictions': {**baseline_results['predictions'], **causal_results['predictions']},
        'metrics': {**baseline_results['metrics'], **causal_results['metrics']},
        'model_info': {**baseline_results['model_info'], **causal_results['model_info']},
        'exogenous_variables': top_exogs
    }
    
    # Step 8: Create visualizations
    print("\n" + "="*80)
    print("Step 8: Creating visualization dashboard")
    print("="*80)
    viz_dir = os.path.join(results_dir, 'visualizations')
    ensure_dir(viz_dir)
    
    create_forecast_dashboard(all_results, target, save_dir=viz_dir)
    
    # Step 9: Generate summary and save results
    print("\n" + "="*80)
    print("Step 9: Generating summary and saving results")
    print("="*80)
    
    # Create overall summary
    summary = create_results_summary(all_results, target)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_results['metrics']).T
    metrics_file = os.path.join(results_dir, 'forecast_metrics.csv')
    metrics_df.to_csv(metrics_file)
    print(f"Metrics saved to {metrics_file}")
    
    # Save predictions to CSV
    predictions_dict = {name: data['forecast'] for name, data in all_results['predictions'].items()}
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df['Actual'] = test_data
    
    predictions_file = os.path.join(results_dir, 'forecast_predictions.csv')
    predictions_df.to_csv(predictions_file)
    print(f"Predictions saved to {predictions_file}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    
    # Print best model according to different metrics
    print("\nBest models:")
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in metrics_df.columns:
            best_model = metrics_df.idxmin()[metric]
            best_value = metrics_df.min()[metric]
            print(f"  By {metric}: {best_model} ({best_value:.4f})")
    
    return all_results


if __name__ == "__main__":
    # Run the complete forecasting pipeline
    run_forecasting_pipeline()
