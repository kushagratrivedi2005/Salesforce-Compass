from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import pandas as pd
import os
import tempfile
import json
import base64

app = Flask(__name__)
CORS(app)

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # For JMeter, a simple 200 OK is usually sufficient.
    # The response can be extended with more details if needed.
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Run the forecasting pipeline with parameters from the request body.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    # --- Extract parameters from the request ---
    target = data.get('TARGET', 'category_LIGHT PASSENGER VEHICLE')
    test_months = data.get('TEST_MONTHS', 6)
    future_forecast_months = data.get('FUTURE_FORECAST_MONTHS', 12)
    use_top_k_exogs = data.get('USE_TOP_K_EXOGS', True)
    candidate_exogs = data.get('CANDIDATE_EXOGS', [
        'interest_rate', 'repo_rate', 'holiday_count',
        'major_national_holiday', 'major_religious_holiday',
    ])
    manual_exogs = data.get('MANUAL_EXOGS', ['interest_rate', 'repo_rate'])
    top_k_exogs = data.get('TOP_K_EXOGS', 5)
    start_date = data.get('START_DATE', None)  # Get start date from request

    # --- Create a temporary config file ---
    config_content = f"""
# Auto-generated config from API request
import numpy as np

# ------------------------- File Paths -------------------------
CSV_PATH = r'{os.path.join(project_root, 'dataset', 'final_merged_dataset.csv').replace('\\', '/')}'
RESULTS_DIR = r'{os.path.join(project_root, 'forecast_results').replace('\\', '/')}'
DATASET_DIR = r'{os.path.join(project_root, 'dataset').replace('\\', '/')}'

# ------------------------- Model Configuration -------------------------
TARGET = '{target}'
DATE_COL = 'Month'
TEST_MONTHS = {test_months}
RANDOM_SEED = 42
FUTURE_FORECAST_MONTHS = {future_forecast_months}

np.random.seed(RANDOM_SEED)

# ------------------------- Exogenous Variables Configuration -------------------------
USE_TOP_K_EXOGS = {use_top_k_exogs}
CANDIDATE_EXOGS = {json.dumps(candidate_exogs)}
MANUAL_EXOGS = {json.dumps(manual_exogs)}
TOP_K_EXOGS = {top_k_exogs}

# ------------------------- Model Parameters (from original config) -------------------------
ARIMA_PARAMS = {{
    'seasonal': True, 'm': 12, 'stepwise': True, 'suppress_warnings': True,
    'error_action': 'ignore', 'trace': False, 'max_p': 8, 'max_q': 8,
    'max_P': 3, 'max_Q': 3
}}
ETS_PARAMS = {{
    'seasonal': 'add', 'trend': 'add', 'seasonal_periods': 12
}}
SARIMAX_PARAMS = {{
    'enforce_stationarity': False, 'enforce_invertibility': False, 'max_iter': 500
}}
SEGMENT_TARGET = 'fuel_PURE EV'
PLOT_CONFIG = {{
    'figsize': (12, 5), 'marker': 'o', 'alpha': 0.3, 'bbox_inches': 'tight'
}}
"""
    
    temp_config_file = None
    try:
        # Use a temporary file for the config
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', dir=os.path.dirname(__file__)) as fp:
            temp_config_file = fp.name
            fp.write(config_content)
        
        # --- Run the main pipeline script ---
        # Determine the correct python executable from the virtual environment
        if os.name == 'nt': # Windows
            python_executable = os.path.join(project_root, 'env', 'Scripts', 'python.exe')
        else: # Linux/macOS
            python_executable = os.path.join(project_root, 'env', 'bin', 'python')

        main_pipeline_path = os.path.join(os.path.dirname(__file__), 'main_pipeline.py')
        cmd = [python_executable, main_pipeline_path, '--config', os.path.basename(temp_config_file)]
        
        # The working directory should be the 'Code' directory
        cwd = os.path.dirname(__file__)
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)

        # --- Read and return the results ---
        results_dir = os.path.join(project_root, 'forecast_results')
        metrics_path = os.path.join(results_dir, 'forecast_metrics.csv')
        predictions_path = os.path.join(results_dir, 'future_forecasts.csv')

        if not os.path.exists(metrics_path) or not os.path.exists(predictions_path):
            return jsonify({
                "error": "Pipeline ran, but result files were not generated.",
                "stdout": process.stdout,
                "stderr": process.stderr
            }), 500

        metrics_df = pd.read_csv(metrics_path)
        predictions_df = pd.read_csv(predictions_path)

        # Get all visualization files
        viz_dir = os.path.join(results_dir, 'visualizations')
        viz_data = {}
        
        if os.path.exists(viz_dir):
            for viz_file in os.listdir(viz_dir):
                if viz_file.endswith('.png'):
                    viz_path = os.path.join(viz_dir, viz_file)
                    try:
                        with open(viz_path, 'rb') as img_file:
                            viz_data[viz_file] = base64.b64encode(img_file.read()).decode('utf-8')
                    except Exception as e:
                        print(f"Error reading visualization file {viz_file}: {str(e)}")
        
        if not viz_data:
            print("No visualization files found in:", viz_dir)

        # Format the response as requested
        response = {
            "error": metrics_df.rename(columns={'Unnamed: 0': 'Model'}).to_dict(orient='records'),
            "prediction": predictions_df.rename(columns={'Unnamed: 0': 'Date'}).set_index('Date').to_dict(),
            "visualization": viz_data
        }
        
        return jsonify(response)

    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Pipeline execution failed.",
            "command": e.cmd,
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except FileNotFoundError as e:
        return jsonify({"error": f"Result file not found: {e.filename}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # Clean up the temporary config file
        if temp_config_file and os.path.exists(temp_config_file):
            os.remove(temp_config_file)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
