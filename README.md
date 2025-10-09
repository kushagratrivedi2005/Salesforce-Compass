# Vehicle Sales Forecasting System

A full-stack web application for forecasting automotive sales using advanced time series models (ARIMA, ETS, SARIMAX) and exogenous variables. The project features a modern React frontend and a Flask backend API.

---

## Features
- Interactive web interface for forecasting vehicle sales
- Dynamic selection of target, test months, forecast horizon, and exogenous variables
- Visualizations and tabular results for model performance and forecasts
- Professional, minimal dark-themed UI
- REST API for programmatic access

---

## Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/kushagratrivedi2005/Salesforce-Compass.git
cd Salesforce-Compass
```

### 2. Create and Activate a Python Virtual Environment
**Linux/macOS:**
```bash
python3 -m venv env
source env/bin/activate
```
**Windows:**
```bash
python -m venv env
env\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements_api.txt
```

### 4. Install Frontend Dependencies
```bash
cd Code/frontend
npm install
```

---

## How to Start the Application

### Start the Backend (Flask API)
```bash
cd /path/to/Salesforce-Compass/Code
python app.py
```
The backend will be available at [http://localhost:5000](http://localhost:5000)

### Start the Frontend (React)
```bash
cd /path/to/Salesforce-Compass/Code/frontend
npm run dev
```
The frontend will be available at the local address shown in the terminal (usually [http://localhost:5173](http://localhost:5173)).

---

## Sample Output

### API Response Example
```
POST /predict
```
**Sample Response:**
```json
{
  "error": [
    { "Model": "ARIMA", "MAE": 3675.02, "RMSE": 4661.11, "MAPE": 12.05 },
    { "Model": "ETS", "MAE": 2877.94, "RMSE": 3289.48, "MAPE": 9.19 },
    { "Model": "SARIMAX", "MAE": 2824.57, "RMSE": 3307.71, "MAPE": 9.14 }
  ],
  "prediction": {
    "ARIMA": { "2025-10-01": 34915.30, "2025-11-01": 32603.18 },
    "ETS": { "2025-10-01": 34825.98, "2025-11-01": 33303.93 },
    "SARIMAX": { "2025-10-01": 33677.72, "2025-11-01": 32454.36 }
  },
  "visualization": {
    "forecast_comparison.png": "<base64-encoded-image>"
  }
}
```

### Web UI Example
- Model metrics and forecasts are shown in styled tables
- Forecast graphs are displayed below the tables
- All results are shown in a modern, minimal dark theme

---

## Project Structure
```
Salesforce-Compass/
├── Code/
│   ├── app.py
│   ├── main_pipeline.py
│   ├── ...
│   └── frontend/
│       ├── src/
│       └── ...
├── dataset/
├── requirements_api.txt
├── README.md
└── ...
```

---

## License
This project is for academic and demonstration purposes.
