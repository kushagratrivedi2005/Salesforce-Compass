"""
Example script demonstrating how to use the Vehicle Sales Forecasting API.

This script shows how to make requests to all available endpoints and
process the responses.
"""

import requests
import json
from datetime import datetime
import pandas as pd

# API base URL
BASE_URL = "http://localhost:5000"

def make_request(endpoint, method="GET"):
    """
    Make a request to the API endpoint.
    
    Args:
        endpoint (str): API endpoint
        method (str): HTTP method
        
    Returns:
        dict: JSON response or error message
    """
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.request(method, url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": True,
                "status_code": response.status_code,
                "message": response.text
            }
    except requests.exceptions.ConnectionError:
        return {
            "error": True,
            "message": "Could not connect to API server. Make sure it's running on localhost:5000"
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Request failed: {str(e)}"
        }

def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if response.get("error"):
        print(f"❌ Error: {response.get('message')}")
        if response.get("status_code"):
            print(f"Status Code: {response.get('status_code')}")
    else:
        print(json.dumps(response, indent=2))

def demo_health_check():
    """Test the health check endpoint"""
    response = make_request("/health")
    print_response("Health Check", response)
    return not response.get("error")

def demo_predictions():
    """Test the predictions endpoint"""
    response = make_request("/api/v1/predictions")
    print_response("All Predictions", response)
    
    if not response.get("error"):
        print(f"\n📊 Summary:")
        print(f"Models available: {', '.join(response.get('models', []))}")
        print(f"Forecast periods: {response.get('summary', {}).get('total_periods', 'N/A')}")
        print(f"Date range: {response.get('summary', {}).get('date_range', {}).get('start', 'N/A')} to {response.get('summary', {}).get('date_range', {}).get('end', 'N/A')}")

def demo_metrics():
    """Test the metrics endpoint"""
    response = make_request("/api/v1/metrics")
    print_response("Model Metrics", response)
    
    if not response.get("error"):
        print(f"\n📈 Best Performing Models:")
        best_models = response.get("best_performing", {})
        for metric, info in best_models.items():
            print(f"  {metric}: {info.get('model')} ({info.get('value'):.2f})")

def demo_models():
    """Test the models information endpoint"""
    response = make_request("/api/v1/models")
    print_response("Models Information", response)
    
    if not response.get("error"):
        print(f"\n🔧 Available Models:")
        models = response.get("models", {})
        for model_name, info in models.items():
            print(f"  {model_name}: {info.get('type')} - {info.get('description')}")

def demo_specific_model(model_name="SARIMAX"):
    """Test the specific model forecast endpoint"""
    response = make_request(f"/api/v1/forecast/{model_name}")
    print_response(f"Forecast for {model_name}", response)
    
    if not response.get("error"):
        forecast_data = response.get("forecast", [])
        if forecast_data:
            print(f"\n📅 Sample Forecast Data:")
            for i, period in enumerate(forecast_data[:3]):  # Show first 3 periods
                date = period.get("date", "N/A")
                forecast_val = period.get("forecast", "N/A")
                actual_val = period.get("actual", "N/A")
                print(f"  {date}: Forecast={forecast_val:.0f}, Actual={actual_val}")
                if i == 2 and len(forecast_data) > 3:
                    print(f"  ... and {len(forecast_data) - 3} more periods")

def demo_comparison():
    """Test the model comparison endpoint"""
    response = make_request("/api/v1/comparison")
    print_response("Model Comparison", response)
    
    if not response.get("error"):
        rankings = response.get("model_rankings", {})
        if "MAPE" in rankings:
            print(f"\n🏆 MAPE Rankings (lower is better):")
            for rank_info in rankings["MAPE"]:
                print(f"  {rank_info.get('rank')}. {rank_info.get('model')}: {rank_info.get('value'):.2f}%")

def demo_error_handling():
    """Test error handling with invalid endpoints"""
    print_response("Invalid Model Test", make_request("/api/v1/forecast/INVALID_MODEL"))
    print_response("Invalid Endpoint Test", make_request("/api/v1/nonexistent"))

def create_summary_report(responses):
    """Create a summary report from all API responses"""
    print(f"\n{'='*60}")
    print("📋 SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Health status
    health_ok = not responses["health"].get("error")
    print(f"🏥 API Health: {'✅ Healthy' if health_ok else '❌ Unhealthy'}")
    
    if not health_ok:
        print("❌ API is not healthy. Cannot generate full report.")
        return
    
    # Data availability
    predictions_ok = not responses["predictions"].get("error")
    metrics_ok = not responses["metrics"].get("error")
    
    print(f"📊 Predictions Data: {'✅ Available' if predictions_ok else '❌ Not Available'}")
    print(f"📈 Metrics Data: {'✅ Available' if metrics_ok else '❌ Not Available'}")
    
    if predictions_ok and metrics_ok:
        # Extract key information
        models = responses["predictions"].get("models", [])
        best_model = responses["metrics"].get("best_performing", {}).get("MAPE", {}).get("model", "Unknown")
        
        print(f"🤖 Available Models: {', '.join([m for m in models if m != 'Actual'])}")
        print(f"🏆 Best Performing Model (by MAPE): {best_model}")
        
        # Forecast period
        summary = responses["predictions"].get("summary", {})
        print(f"📅 Forecast Period: {summary.get('total_periods', 'N/A')} periods")
        
        date_range = summary.get("date_range", {})
        if date_range:
            start_date = date_range.get("start", "N/A")[:10]  # Just the date part
            end_date = date_range.get("end", "N/A")[:10]
            print(f"📅 Date Range: {start_date} to {end_date}")

def main():
    """Run all API endpoint demos"""
    print("🚀 Vehicle Sales Forecasting API Demo")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Testing API at: {BASE_URL}")
    
    # Store responses for summary
    responses = {}
    
    # Test all endpoints
    responses["health"] = make_request("/health")
    demo_health_check()
    
    responses["predictions"] = make_request("/api/v1/predictions")
    demo_predictions()
    
    responses["metrics"] = make_request("/api/v1/metrics")
    demo_metrics()
    
    demo_models()
    demo_specific_model("SARIMAX")
    demo_specific_model("ARIMA")
    demo_comparison()
    
    # Test error handling
    print(f"\n{'='*60}")
    print("🔧 Testing Error Handling")
    print(f"{'='*60}")
    demo_error_handling()
    
    # Generate summary report
    create_summary_report(responses)
    
    print(f"\n✅ Demo completed! Check the API documentation for more details.")

if __name__ == "__main__":
    main()
