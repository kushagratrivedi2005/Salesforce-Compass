"""
Main pipeline for vehicle sales forecasting.

This script orchestrates the entire forecasting pipeline:
1. Data loading and preprocessing
2. Train/test splitting
3. Baseline model training and evaluation
4. Causal model training with exogenous variables
5. Model comparison and visualization
6. Results export
7. Future forecasting up to a specified number of months
"""

import os
import pandas as pd
import numpy as np
import warnings
from data_loader import (
    load_and_prepare, train_test_split_by_date, 
    prepare_exogenous_variables, ensure_dir, detect_and_handle_outliers
)
from baseline_models import train_baseline_models
from causal_models import train_causal_models
from model_utils import create_results_summary
from visualization import create_forecast_dashboard
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="No frequency information")
warnings.filterwarnings("ignore", message="Optimization failed to converge")

# Import configuration
import sys
import importlib.util

if len(sys.argv) > 2 and sys.argv[1] == '--config':
    # Load the specified config file
    config_path = sys.argv[2]
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Get configuration from the loaded module
    CSV_PATH = config.CSV_PATH
    RESULTS_DIR = config.RESULTS_DIR
    TARGET = config.TARGET
    DATASET_DIR = config.DATASET_DIR
    FUTURE_FORECAST_MONTHS = config.FUTURE_FORECAST_MONTHS
    USE_TOP_K_EXOGS = config.USE_TOP_K_EXOGS
    CANDIDATE_EXOGS = config.CANDIDATE_EXOGS
    MANUAL_EXOGS = config.MANUAL_EXOGS
    TOP_K_EXOGS = config.TOP_K_EXOGS
    TEST_MONTHS = config.TEST_MONTHS  # Add TEST_MONTHS
    VISUALIZATION_YEARS = getattr(config, 'VISUALIZATION_YEARS', None)
else:
    # Fall back to default config if no config file specified
    from config import (
        CSV_PATH, RESULTS_DIR, TARGET,
        DATASET_DIR, FUTURE_FORECAST_MONTHS, USE_TOP_K_EXOGS,
        CANDIDATE_EXOGS, MANUAL_EXOGS, TOP_K_EXOGS
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section_header(title, char="="):
    """Print a formatted section header."""
    print("\n" + char*80)
    print(title)
    print(char*80)

# ============================================================================
# FUTURE FORECASTING FUNCTION
# ============================================================================

def generate_future_forecast(model_results, full_history, future_periods, exog_data=None):
    """
    Generate future forecasts beyond the historical data.

    Args:
        model_results (dict): Dictionary containing trained models.
        full_history (pd.Series): Complete historical data (train + test).
        future_periods (int): Number of months to forecast into the future.
        exog_data (pd.DataFrame, optional): Exogenous variables data for future dates.

    Returns:
        dict: Future forecasts for each model.
    """
    print_section_header(f"Generating Future Forecasts for {future_periods} Months")
    
    # Start future forecasts from the current month
    current_date = pd.Timestamp.now().normalize()
    forecast_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
    
    future_dates = pd.date_range(
        start=forecast_start,
        periods=future_periods,
        freq='MS'
    )
    
    print(f"Future forecast period: {future_dates[0].strftime('%Y-%m')} to {future_dates[-1].strftime('%Y-%m')}")

    future_forecasts = {}
    models = model_results.get('models', {})

    # Only process ARIMA, ETS, and SARIMAX models
    for model_name in ['ARIMA', 'ETS', 'SARIMAX']:
        if model_name not in models:
            print(f"(!) Skipping future forecast for {model_name} (model not found).")
            continue

        print(f"\n-> Generating future forecast for {model_name}...")
        model = models[model_name]
        
        try:
            if model_name == 'SARIMAX':
                # Refit SARIMAX on full history with exogenous variables
                full_history_exog = None
                if exog_data is not None and not exog_data.empty:
                    # Align exog data with the full history
                    full_history_exog = exog_data.reindex(full_history.index).ffill().bfill()
                
                # Refit the model on full history
                print(f"  Refitting SARIMAX on full historical data...")
                model.fit(full_history, exog_data=full_history_exog)
                
                # Prepare future exogenous variables
                future_exog = None
                if exog_data is not None and not exog_data.empty:
                    # Use last known values for future exogenous variables
                    # In practice, you would forecast these variables or use known future values
                    future_exog = pd.DataFrame(index=future_dates, columns=exog_data.columns)
                    for col in exog_data.columns:
                        last_known_value = exog_data[col].dropna().iloc[-1]
                        future_exog[col] = last_known_value
                
                preds, conf_int = model.predict(n_periods=future_periods, exog_data=future_exog, return_conf_int=True)
                
                # Ensure preds is a Series with correct index
                if not isinstance(preds, pd.Series):
                    preds = pd.Series(preds, index=future_dates)
                else:
                    preds.index = future_dates
                    
                future_forecasts[model_name] = {
                    'forecast': preds,
                    'lower_ci': pd.Series(conf_int.iloc[:, 0].values, index=future_dates),
                    'upper_ci': pd.Series(conf_int.iloc[:, 1].values, index=future_dates)
                }
                
            else:  # ARIMA or ETS
                # Refit the model on the full history before forecasting
                print(f"  Refitting {model_name} on full historical data...")
                model.fit(full_history)
                
                if model_name == 'ARIMA':
                    preds_result, conf_int_result = model.predict(n_periods=future_periods, return_conf_int=True)
                    
                    # Convert to Series with proper index
                    if isinstance(preds_result, np.ndarray):
                        preds = pd.Series(preds_result, index=future_dates)
                    else:
                        preds = pd.Series(preds_result, index=future_dates)
                    
                    # Handle confidence intervals
                    if isinstance(conf_int_result, np.ndarray):
                        conf_int = pd.DataFrame(conf_int_result, index=future_dates, columns=['lower', 'upper'])
                    else:
                        conf_int = conf_int_result
                        conf_int.index = future_dates
                    
                    future_forecasts[model_name] = {
                        'forecast': preds,
                        'lower_ci': conf_int.iloc[:, 0],
                        'upper_ci': conf_int.iloc[:, 1]
                    }
                    
                elif model_name == 'ETS':
                    preds = model.predict(n_periods=future_periods)
                    
                    # Ensure it's a Series with proper index
                    if not isinstance(preds, pd.Series):
                        preds = pd.Series(preds, index=future_dates)
                    else:
                        preds.index = future_dates
                        
                    future_forecasts[model_name] = {'forecast': preds}
                    
            print(f"  (+) {model_name} future forecast generated successfully")
            
        except Exception as e:
            print(f"  ✗ Error generating future forecast for {model_name}: {e}")
            continue

    return future_forecasts


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_forecasting_pipeline(csv_path=CSV_PATH, target=TARGET,
                            results_dir=RESULTS_DIR,
                            future_forecast_months=FUTURE_FORECAST_MONTHS):
    """
    Run the complete forecasting pipeline.
    
    Args:
        csv_path (str): Path to the input CSV file
        target (str): Target variable for forecasting
        results_dir (str): Directory to save results
        future_forecast_months (int): Number of months to forecast into future
    
    Returns:
        dict: Complete results from the pipeline
    """
    
    # ========================================================================
    # STEP 1: Environment Preparation
    # ========================================================================
    print_section_header("STEP 1: Preparing Environment")
    ensure_dir(results_dir)

    full_csv_path = os.path.join(DATASET_DIR, csv_path)
    if not os.path.exists(full_csv_path):
        print(f"(x) Error: CSV file not found at {full_csv_path}")
        return None
    print(f"(+) Environment prepared successfully")
    
    # ========================================================================
    # STEP 2: Data Loading and Preprocessing
    # ========================================================================
    print_section_header("STEP 2: Loading and Preprocessing Data")
    df = load_and_prepare(full_csv_path)

    if target not in df.columns:
        print(f"(x) Error: Target variable '{target}' not found in the dataset")
        return None

    print(f"(+) Target variable: {target}")
    print(f"(+) Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    print(f"(+) Total observations: {len(df)}")
    
    # ========================================================================
    # STEP 3: Train/Test Splitting
    # ========================================================================
    print_section_header("STEP 3: Splitting Data into Train and Test Sets")
    target_series = df[target]
    
    # Handle outliers in target series before splitting
    print("-> Detecting and handling outliers in target variable...")
    target_series_clean = detect_and_handle_outliers(target_series, method='iqr', replace_with='interpolate')
    n_outliers = (target_series != target_series_clean).sum()
    if n_outliers > 0:
        print(f"(+) Handled {n_outliers} outliers in target variable")
        target_series = target_series_clean
    
    # Fixed test end date at September 2025
    test_end_date = '2025-09-01'
    
    # Calculate train end date based on test_months parameter
    test_start = pd.Timestamp(test_end_date) - pd.DateOffset(months=TEST_MONTHS)
    train_end_date = test_start.strftime('%Y-%m-%d')
    
    print(f"-> Training period ends at: {train_end_date}")
    print(f"-> Testing period: {train_end_date} to {test_end_date} ({TEST_MONTHS} months)")
    
    train_data, test_data = train_test_split_by_date(
        target_series,
        train_end_date=train_end_date,
        test_end_date=test_end_date
    )
    
    print(f"(+) Training data: {len(train_data)} observations ({train_data.index.min().strftime('%Y-%m')} to {train_data.index.max().strftime('%Y-%m')})")
    print(f"(+) Test data: {len(test_data)} observations ({test_data.index.min().strftime('%Y-%m')} to {test_data.index.max().strftime('%Y-%m')})")
    
    # ========================================================================
    # STEP 4: Baseline Model Training (ARIMA, ETS)
    # ========================================================================
    print_section_header("STEP 4: Training Baseline Forecasting Models")
    print("Training ARIMA and Exponential Smoothing models...")
    baseline_results = train_baseline_models(train_data, test_data)
    
    if baseline_results.get('models'):
        print(f"(+) Successfully trained {len(baseline_results['models'])} baseline model(s): {', '.join(baseline_results['models'].keys())}")
    else:
        print("(!) Warning: No baseline models were successfully trained")
    
    # ========================================================================
    # STEP 5: Exogenous Variables Preparation
    # ========================================================================
    print_section_header("STEP 5: Preparing Exogenous Variables for Causal Models")
    
    selected_exogs = []
    exog_df = pd.DataFrame()

    if USE_TOP_K_EXOGS:
        print(f"-> Using top-{TOP_K_EXOGS} correlated exogenous variables")
        selected_exogs, exog_df, correlations = prepare_exogenous_variables(
            df, CANDIDATE_EXOGS, target, top_k=TOP_K_EXOGS
        )
        if selected_exogs:
            print(f"(+) Selected exogenous variables:")
            for exog in selected_exogs:
                print(f"  * {exog} (correlation: {correlations[exog]:.4f})")
    else:
        print(f"-> Using manually specified exogenous variables: {MANUAL_EXOGS}")
        selected_exogs = [exog for exog in MANUAL_EXOGS if exog in df.columns]
        if selected_exogs:
            exog_df = df[selected_exogs]
            print(f"(+) Selected exogenous variables: {', '.join(selected_exogs)}")

    if not selected_exogs:
        print("(!) Warning: No exogenous variables selected or available")
    
    exog_train, exog_test = train_test_split_by_date(
        exog_df,
        train_end_date=train_end_date,
        test_end_date=test_end_date
    )
    
    # ========================================================================
    # STEP 6: Causal Model Training (SARIMAX)
    # ========================================================================
    print_section_header("STEP 6: Training Causal Models with Exogenous Variables")
    print("Training SARIMAX model with exogenous predictors...")
    causal_results = train_causal_models(
        train_data, test_data, exog_train, exog_test, 
        baseline_results.get('model_info', {})
    )
    
    if causal_results.get('models'):
        print(f"(+) Successfully trained {len(causal_results['models'])} causal model(s): {', '.join(causal_results['models'].keys())}")
    else:
        print("(!) Warning: No causal models were successfully trained")
    
    # ========================================================================
    # STEP 7: Results Combination
    # ========================================================================
    print_section_header("STEP 7: Combining and Comparing All Models")
    
    all_results = {
        'train_data': train_data,
        'test_data': test_data,
        'predictions': {**baseline_results.get('predictions', {}), **causal_results.get('predictions', {})},
        'metrics': {**baseline_results.get('metrics', {}), **causal_results.get('metrics', {})},
        'models': {**baseline_results.get('models', {}), **causal_results.get('models', {})},
        'model_info': {**baseline_results.get('model_info', {}), **causal_results.get('model_info', {})},
        'exogenous_variables': selected_exogs
    }
    
    total_models = len(all_results['models'])
    print(f"(+) Combined {total_models} models for comparison")
    
    # ========================================================================
    # STEP 8: Visualization Dashboard Creation
    # ========================================================================
    print_section_header("STEP 8: Creating Visualization Dashboard")
    
    viz_dir = os.path.join(results_dir, 'visualizations')
    ensure_dir(viz_dir)
    
    if all_results['predictions']:
        try:
            create_forecast_dashboard(all_results, target, save_dir=viz_dir, visualization_years=VISUALIZATION_YEARS)
            print(f"(+) Visualizations saved to: {viz_dir}")
            # List created visualizations
            if os.path.exists(viz_dir):
                viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
                print(f"(+) Created visualizations: {', '.join(viz_files)}")
        except Exception as e:
            print(f"(!) Warning: Error creating visualizations: {str(e)}")
    else:
        print("(!) Warning: No predictions available for visualization")
    
    # ========================================================================
    # STEP 9: Summary and Results Export
    # ========================================================================
    print_section_header("STEP 9: Generating Summary and Saving Results")
    
    summary = create_results_summary(all_results, target)
    
    # Save metrics
    if all_results['metrics']:
        metrics_df = pd.DataFrame(all_results['metrics']).T
        metrics_file = os.path.join(results_dir, 'forecast_metrics.csv')
        metrics_df.to_csv(metrics_file)
        print(f"(+) Metrics saved to: {metrics_file}")

    # Save predictions
    if all_results['predictions']:
        predictions_df = pd.DataFrame({
            name: data['forecast'] 
            for name, data in all_results['predictions'].items()
        })
        predictions_df['Actual'] = test_data
        predictions_file = os.path.join(results_dir, 'forecast_predictions.csv')
        predictions_df.to_csv(predictions_file)
        print(f"(+) Predictions saved to: {predictions_file}")
    
    # ========================================================================
    # STEP 10: Future Forecasting
    # ========================================================================
    if future_forecast_months > 0:
        print_section_header("STEP 10: Generating Future Forecasts")
        
        full_history = pd.concat([train_data, test_data])
        print(f"-> Generating {future_forecast_months}-month forecasts beyond {full_history.index.max().strftime('%Y-%m')}")
        
        future_forecasts = generate_future_forecast(
            all_results,
            full_history,
            future_forecast_months,
            exog_data=exog_df
        )

        if future_forecasts:
            # Save future forecasts
            future_df = pd.DataFrame({
                name: data['forecast'] 
                for name, data in future_forecasts.items()
            })
            future_file = os.path.join(results_dir, 'future_forecasts.csv')
            future_df.to_csv(future_file)
            print(f"\n(+) Future forecasts saved to: {future_file}")

            all_results['future_forecasts'] = future_forecasts
            
            # Create visualization
            future_viz_file = os.path.join(viz_dir, 'future_forecasts_plot.png')
            
            plt.figure(figsize=(14, 7))
            
            # Optionally show historical data based on VISUALIZATION_YEARS
            if VISUALIZATION_YEARS:
                months_to_show = VISUALIZATION_YEARS * 12
                recent_history = full_history.iloc[-months_to_show:] if len(full_history) > months_to_show else full_history
                plt.plot(recent_history.index, recent_history.values, 
                        label='Historical Data', color='black', linewidth=2, alpha=0.7)
            
            # Plot future forecasts for each model
            colors = {'ARIMA': 'blue', 'ETS': 'green', 'SARIMAX': 'red'}
            for model_name, forecast_data in future_forecasts.items():
                forecast = forecast_data['forecast']
                plt.plot(forecast.index, forecast.values, 
                        label=f'{model_name} Forecast', 
                        color=colors.get(model_name, 'gray'), 
                        linestyle='--', marker='o', linewidth=2)
                
                # Plot confidence intervals if available
                if 'lower_ci' in forecast_data and 'upper_ci' in forecast_data:
                    plt.fill_between(forecast.index, 
                                    forecast_data['lower_ci'].values,
                                    forecast_data['upper_ci'].values,
                                    alpha=0.2, color=colors.get(model_name, 'gray'))
            
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.title(f'Future Forecasts for {future_forecast_months} Months', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(future_viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"(+) Future forecast visualization saved to: {future_viz_file}")
        else:
            print("(!) Warning: No future forecasts were generated")
    
    # ========================================================================
    # PIPELINE COMPLETION
    # ========================================================================
    print_section_header("PIPELINE COMPLETED SUCCESSFULLY!", "=")

    # Display best models
    if 'metrics_df' in locals() and not metrics_df.empty:
        print("\nBest Performing Models:")
        print("-" * 80)
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in metrics_df.columns:
                best_model = metrics_df[metric].idxmin()
                best_value = metrics_df[metric].min()
                print(f"  * By {metric:4s}: {best_model:10s} = {best_value:.4f}")
        print("=" * 80)

    return all_results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  VEHICLE SALES FORECASTING PIPELINE")
    print("="*80)
    run_forecasting_pipeline()