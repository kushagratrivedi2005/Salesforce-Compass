# Dataset Documentation Report

## Overview
This document provides details on the data collection, processing, and integration process used to create the final merged dataset for vehicle sales analysis in India. The dataset combines vehicle sales data with economic indicators, holidays, and automotive policy information.

## Data Sources

### Vehicle Sales Data
After exploring multiple potential sources with various limitations:
- **SIAM** (Society of Indian Automobile Manufacturers): Required paid subscription (INR 48,500+ GST)
- **FADA** (Federation of Automotive Dealers): Website was inaccessible
- **Trading Economics**: Requires paid access
- **KAPSARC**: Data only available until 2022 (https://datasource.kapsarc.org/explore/dataset/india-vehicle-sales-trends)
- **data.gov.in**: Last updated in 2019
- **dataful.in**: Requires paid subscription
- **India Data Portal**: Limited to January data only

The final source used was the government portal:
- **Source**: https://analytics.parivahan.gov.in/analytics/vahanpublicreport?lang=en
- **Timeframe**: 2023 to 2025
- **Categorization**: Data sorted by fuel type, vehicle category, and vehicle class

### Fuel Price Data
- **Source**: Government international prices of crude oil, Indian basket
- **URL**: https://ppac.gov.in/prices/international-prices-of-crude-oil

### Holidays Data
Due to challenges in classifying holidays and limited government recognition of national holidays, this dataset was generated using the Python 'holidays' library.

### Interest Rates Data
- **Source**: Long-Term Government Bond Yields: 10-Year Benchmark for India (INDIRLTLT01STM)
- **URL**: https://fred.stlouisfed.org/series/INDIRLTLT01STM

### Automotive Policies Data
Policy implementation flags were generated using an automated Python script, tracking:
- FAME II scheme
- FAME III scheme
- PM e-Drive initiative
- BS7 norms
- Bharat NCAP
- Vehicle scrappage policy
- PLI scheme
- Repo rate information
- Expected repo rate cuts for 2025

## Data Processing Workflow

### Step 1: Joining Vehicle Sales Datasets
The script `join_datasets.py` was used to combine sales data categorized by:
- Fuel type
- Vehicle category
- Vehicle class

This process resulted in a dataset with 36 rows and 129 columns, saved as `joined_data.csv`.

### Step 2: Creating Supporting Datasets
The script `combine_datasets.py` integrated:
- Interest rates data
- Holidays data (including counts of major religious and national holidays)
- Automotive policies data

This resulted in a dataset with 36 rows and 16 columns, saved as `combined_data.csv`. The columns included:
```
year, month, month_name, interest_rate, holiday_count, major_religious_holiday, 
major_national_holiday, fame_ii, fame_iii, pm_edrive, bs7_norms, bharat_ncap, 
scrappage_policy, pli_scheme, repo_rate_cut_2025, repo_rate
```

### Step 3: Final Dataset Creation
A final script merged the joined vehicle sales data with the combined supporting datasets based on year and month keys, resulting in `final_merged_dataset.csv` with:
- 36 rows (representing monthly data)
- 143 columns (combining all metrics and indicators)

## Dataset Structure
The final dataset contains monthly data points with extensive information about:
- Vehicle sales segmented by various categories
- Economic indicators (interest rates, repo rates)
- Holiday information that may impact sales
- Automotive policy implementation status

This comprehensive dataset enables robust analysis of factors influencing vehicle sales in India across the 2023-2025 timeframe.
