# Vehicle Sales Forecasting API Documentation

## Overview

This Flask API server provides REST endpoints to access the output of the Vehicle Sales Forecasting System. The API serves predictions from multiple models (ARIMA, ETS, SARIMAX) along with performance metrics and comparative analysis.

## Base URL

```
http://localhost:5000
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Vehicle Sales Forecasting API",
  "timestamp": "2024-09-07T12:00:00.000Z",
  "version": "1.0.0"
}
```

### 2. Get All Predictions

**GET** `/api/v1/predictions`

Retrieve predictions from all models including actual values.

**Response:**
```json
{
  "models": ["ARIMA", "ETS", "SARIMAX", "Actual"],
  "time_series": {
    "ARIMA": [
      {
        "date": "2025-02-01T00:00:00.000Z",
        "value": 2493234.06
      }
    ],
    "ETS": [...],
    "SARIMAX": [...],
    "Actual": [...]
  },
  "summary": {
    "total_periods": 6,
    "date_range": {
      "start": "2025-02-01T00:00:00.000Z",
      "end": "2025-07-01T00:00:00.000Z"
    }
  },
  "metadata": {
    "timestamp": "2024-09-07T12:00:00.000Z",
    "data_source": "Vehicle Sales Forecasting System"
  }
}
```

### 3. Get Model Metrics

**GET** `/api/v1/metrics`

Retrieve performance metrics for all models.

**Response:**
```json
{
  "models": {
    "ARIMA": {
      "MAE": 566629.68,
      "RMSE": 598919.48,
      "MAPE": 27.48
    },
    "ETS": {
      "MAE": 239096.41,
      "RMSE": 256084.04,
      "MAPE": 11.65
    },
    "SARIMAX": {
      "MAE": 191619.02,
      "RMSE": 240478.82,
      "MAPE": 9.16
    }
  },
  "best_performing": {
    "MAE": {
      "model": "SARIMAX",
      "value": 191619.02
    },
    "RMSE": {
      "model": "SARIMAX",
      "value": 240478.82
    },
    "MAPE": {
      "model": "SARIMAX",
      "value": 9.16
    }
  },
  "model_count": 3,
  "metadata": {
    "timestamp": "2024-09-07T12:00:00.000Z",
    "metrics_description": {
      "MAE": "Mean Absolute Error",
      "RMSE": "Root Mean Square Error",
      "MAPE": "Mean Absolute Percentage Error (%)"
    }
  }
}
```

### 4. Get Models Information

**GET** `/api/v1/models`

Retrieve information about available models.

**Response:**
```json
{
  "models": {
    "ARIMA": {
      "name": "ARIMA",
      "type": "Time Series",
      "description": "AutoRegressive Integrated Moving Average model for time series forecasting",
      "available": true,
      "metrics": {
        "MAE": 566629.68,
        "RMSE": 598919.48,
        "MAPE": 27.48
      }
    },
    "ETS": {
      "name": "ETS",
      "type": "Exponential Smoothing",
      "description": "Exponential Smoothing (Holt-Winters) model with trend and seasonality",
      "available": true,
      "metrics": {
        "MAE": 239096.41,
        "RMSE": 256084.04,
        "MAPE": 11.65
      }
    },
    "SARIMAX": {
      "name": "SARIMAX",
      "type": "Causal Time Series",
      "description": "Seasonal ARIMA with eXogenous variables including economic indicators and policy factors",
      "available": true,
      "metrics": {
        "MAE": 191619.02,
        "RMSE": 240478.82,
        "MAPE": 9.16
      }
    }
  },
  "total_models": 3,
  "metadata": {
    "timestamp": "2024-09-07T12:00:00.000Z",
    "forecast_period": {
      "start": "2025-02-01T00:00:00.000Z",
      "end": "2025-07-01T00:00:00.000Z",
      "periods": 6
    }
  }
}
```

### 5. Get Specific Model Forecast

**GET** `/api/v1/forecast/{model_name}`

Retrieve forecast data for a specific model.

**Parameters:**
- `model_name` (string): Model name (ARIMA, ETS, or SARIMAX)

**Example:** `/api/v1/forecast/SARIMAX`

**Response:**
```json
{
  "model": {
    "name": "SARIMAX",
    "type": "Causal Time Series",
    "description": "Seasonal ARIMA with eXogenous variables including economic indicators and policy factors"
  },
  "forecast": [
    {
      "date": "2025-02-01T00:00:00.000Z",
      "forecast": 1914565.81,
      "actual": 1923300.0
    },
    {
      "date": "2025-03-01T00:00:00.000Z",
      "forecast": 2179116.62,
      "actual": 2155581.0
    }
  ],
  "metrics": {
    "MAE": 191619.02,
    "RMSE": 240478.82,
    "MAPE": 9.16
  },
  "summary": {
    "total_periods": 6,
    "date_range": {
      "start": "2025-02-01T00:00:00.000Z",
      "end": "2025-07-01T00:00:00.000Z"
    }
  },
  "metadata": {
    "timestamp": "2024-09-07T12:00:00.000Z"
  }
}
```

### 6. Get Model Comparison

**GET** `/api/v1/comparison`

Retrieve comparative analysis of all models.

**Response:**
```json
{
  "comparison": [
    {
      "date": "2025-02-01T00:00:00.000Z",
      "actual": 1923300.0,
      "ARIMA": 2493234.06,
      "ETS": 2275123.78,
      "SARIMAX": 1914565.81
    }
  ],
  "model_rankings": {
    "MAE": [
      {
        "rank": 1,
        "model": "SARIMAX",
        "value": 191619.02
      },
      {
        "rank": 2,
        "model": "ETS",
        "value": 239096.41
      },
      {
        "rank": 3,
        "model": "ARIMA",
        "value": 566629.68
      }
    ],
    "RMSE": [...],
    "MAPE": [...]
  },
  "summary": {
    "models_compared": 3,
    "time_periods": 6,
    "best_overall": "SARIMAX"
  },
  "metadata": {
    "timestamp": "2024-09-07T12:00:00.000Z",
    "ranking_criteria": "Lower values indicate better performance"
  }
}
```

## Error Responses

All endpoints return standardized error responses in case of failures:

```json
{
  "error": true,
  "message": "Error description",
  "timestamp": "2024-09-07T12:00:00.000Z",
  "status_code": 500
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `404` - Not Found (invalid endpoint or model name)
- `500` - Internal Server Error
- `503` - Service Unavailable (forecast data not available)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Generate Forecast Data

Before starting the API server, run the forecasting pipeline to generate the required data:

```bash
cd Code
python main_pipeline.py
```

This will create the following files:
- `forecast_results/forecast_predictions.csv`
- `forecast_results/forecast_metrics.csv`

### 3. Start the API Server

```bash
cd Code
python app.py
```

The server will start on `http://localhost:5000`

### 4. Test the API

You can test the API using curl, Postman, or any HTTP client:

```bash
# Health check
curl http://localhost:5000/health

# Get all predictions
curl http://localhost:5000/api/v1/predictions

# Get metrics
curl http://localhost:5000/api/v1/metrics

# Get specific model forecast
curl http://localhost:5000/api/v1/forecast/SARIMAX
```

## Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## CORS Support

The API includes CORS (Cross-Origin Resource Sharing) support, allowing web applications from different domains to access the API.

## Data Format

- All dates are returned in ISO 8601 format (UTC)
- Numerical values are returned as floats
- Missing values are represented as `null`
- Timestamps indicate when the response was generated

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider adding rate limiting middleware.

## Security Considerations

For production deployment:
1. Add authentication/authorization
2. Implement rate limiting
3. Add input validation
4. Use HTTPS
5. Add request logging
6. Implement proper error handling without exposing internal details
