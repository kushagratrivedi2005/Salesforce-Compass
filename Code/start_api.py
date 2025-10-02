"""
Startup script for the Vehicle Sales Forecasting API server.

This script:
1. Checks if forecast data exists
2. Optionally runs the forecasting pipeline if data is missing
3. Starts the Flask API server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_forecast_data():
    """
    Check if forecast data files exist.
    
    Returns:
        bool: True if data exists, False otherwise
    """
    current_dir = Path(__file__).parent
    predictions_file = current_dir / 'forecast_results' / 'forecast_predictions.csv'
    metrics_file = current_dir / 'forecast_results' / 'forecast_metrics.csv'
    
    return predictions_file.exists() and metrics_file.exists()

def run_forecasting_pipeline():
    """
    Run the forecasting pipeline to generate data.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("🔄 Running forecasting pipeline to generate data...")
        print("This may take a few minutes...")
        
        result = subprocess.run([sys.executable, 'main_pipeline.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Forecasting pipeline completed successfully!")
            return True
        else:
            print(f"❌ Forecasting pipeline failed!")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Forecasting pipeline timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running forecasting pipeline: {str(e)}")
        return False

def check_dependencies():
    """
    Check if required packages are installed.
    
    Returns:
        list: List of missing packages
    """
    required_packages = ['flask', 'flask_cors', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def start_api_server():
    """Start the Flask API server."""
    try:
        print("🚀 Starting Vehicle Sales Forecasting API server...")
        print("Server will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_api.txt")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")

def main():
    """Main startup function."""
    print("🏭 Vehicle Sales Forecasting API Startup")
    print("="*50)
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements_api.txt")
        return
    
    print("✅ All required packages are installed")
    
    # Check if forecast data exists
    if not check_forecast_data():
        print("📊 Forecast data not found")
        
        response = input("Would you like to run the forecasting pipeline to generate data? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not run_forecasting_pipeline():
                print("❌ Failed to generate forecast data. Cannot start API server.")
                return
        else:
            print("❌ Forecast data is required to run the API server.")
            print("Please run: python main_pipeline.py")
            return
    else:
        print("✅ Forecast data found")
    
    # Start the API server
    start_api_server()

if __name__ == "__main__":
    main()
