# Causal and Time Series Forecasting of Regional Vehicle Sales

## Background / Business Context
Automotive OEMs rely on their regional dealership networks to sell vehicles, track leads, and monitor demand. On Salesforce Manufacturing Cloud, sales executives and operations planners need monthly vehicle sales forecasts across different regions and vehicle types to plan production, manage inventory allocation, and create sales incentives. However, sales are influenced by both time-driven patterns (e.g., seasonality, festivals, model launches) and external economic indicators (e.g., fuel prices, interest rates, policy changes). A statistically rigorous forecasting model can improve planning accuracy and reduce inventory holding costs.

## Goal / Problem Statement
Build and compare statistical forecasting models that accurately predict monthly vehicle sales at a regional dealership level. Evaluate the incremental lift provided by incorporating external variables (fuel prices, interest rates, seasonality) over baseline time-series models. Use model diagnostics to recommend region-specific strategies or risk indicators.

If not addressed, OEMs may face poor production alignment, inventory imbalances, and missed market opportunities—especially in volatile regions.

## Scope Definition

### In-Scope
List the specific areas, deliverables, functionalities, or activities that are part of the project:
- Cleaning and structuring simulated or public regional sales data over 3–5 years
- Building and evaluating ARIMA, SARIMAX, and Exponential Smoothing models
- Feature engineering and testing causal impact of external variables (e.g., oil prices, policy index, holidays)
- Visualizing forecasts and error bands across regions and vehicle segments

### Out-of-Scope
List items, functionalities, or areas that are explicitly excluded from the project:
- Real-time forecasting deployment on Salesforce
- Use of deep learning models like LSTM (can be a mention in future work)

## Deliverables
List the key deliverables or outcomes expected from the project:
- Preprocessed dataset with regional monthly sales and external features
- Time series forecast models with baseline vs causal comparison
- Tableau CRM dashboard or Jupyter-based visual analytics summary
- Final report with model evaluations, error metrics, and business recommendations

## Success Criteria
Define how success will be measured and what constitutes project completion:
- Forecasting model achieves MAPE < 10% on holdout test set across the majority of regions
- Clear causal insights—e.g., how much fuel price change impacts SUV sales in specific zones
- Acceptance of model findings by mock stakeholders (e.g., Salesforce architects, mentors)

Thanks,  
Dhaval
