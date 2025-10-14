# Server Implementation Summary

## Overview

This document captures the work completed to expose the vehicle sales forecasting pipeline through a lightweight Flask server and explains how JMeter will be used to monitor the service health.

## Implemented Server Features

- **Fresh Flask Application (`Code/app.py`)**
  - Replaced the previous REST design with a simplified server that trains all forecasting models on every request to guarantee up-to-date output.
  - Centralised shared paths (dataset, results directory) and ensures the pipeline runs inside the project's virtual environment so that all dependencies resolve correctly.

- **`/predict` Endpoint**
  - Accepts a JSON payload mirroring the configurable parameters in `config.py` (target column, horizon length, exogenous variable options, etc.).
  - Creates a temporary configuration module from the request, invokes `main_pipeline.py`, and tears down the temporary file afterwards.
  - Returns two structured blocks:
    - `error`: model performance metrics (MAE, RMSE, MAPE) for ARIMA, ETS, and SARIMAX.
    - `prediction`: future forecasts for each model covering the requested horizon.
  - Includes guarded error handling so pipeline or filesystem issues surface as informative 4xx/5xx responses rather than silent failures.

- **`/health` Endpoint**
  - Quick JSON heartbeat (`{"status": "healthy"}`) designed to be inexpensive and easily monitored.
  - Intended for automated probes, with JMeter as the primary tool for exercising the endpoint alongside other uptime monitors.

- **Support Artifacts**
  - Added `Procfile` for platform-as-a-service deployments that expect a `web: gunicorn Code.app:app` entry point.
  - Introduced `requirements_api.txt` and `setup_api.ps1` to capture Python dependencies and provide a reproducible setup path on Windows.
  - Updated API documentation to describe the new behaviour, request payload, and response schema.

## JMeter for Health Monitoring

### What is JMeter?

Apache JMeter is an open-source load-testing tool that can execute HTTP requests in bulk, record response times, and assert expected behaviour. It is well suited for validating API availability and performance characteristics without building a custom client.

### Planned Usage

In this project, JMeter will be configured to exercise the Flask server's health and prediction endpoints:

1. **Health Sampler**
   - Create an HTTP Request sampler that targets `GET http://localhost:5000/health`.
   - Add a Response Assertion expecting the JSON fragment `"status":"healthy"` to ensure the endpoint returns the expected payload.
   - Schedule the sampler inside a Thread Group that runs at a modest interval (e.g., one request every 10 seconds) to simulate continuous health polling.

2. **Prediction Workflow (Optional Extension)**
   - Build an HTTP Request sampler configured for `POST http://localhost:5000/predict`.
   - Supply the JSON body used in production (matching the configuration keys in `config.py`).
   - Add assertions for HTTP 200 responses and optionally verify that metric keys such as `MAE` and `RMSE` appear in the returned JSON to confirm a full pipeline run.

3. **Monitoring & Reporting**
   - Attach Listeners such as Summary Report or View Results Tree to capture latency, success rate, and failure details.
   - Export JTL logs for longer-term monitoring or integrate the test plan into CI/CD pipelines so regressions are caught automatically.

By adopting JMeter for these checks, we obtain repeatable verification that the Flask API remains responsive and produces the expected outputs under load or across deployments.
