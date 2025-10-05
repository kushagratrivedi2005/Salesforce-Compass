import pandas as pd
from datetime import datetime

# Create monthly date range from Jan 2023 to Dec 2025
dates = pd.date_range(start='2005-01-01', end='2025-12-31', freq='MS')

# Initialize DataFrame
policy_df = pd.DataFrame({'date': dates})

# Define policy activation periods (format: (start_date, end_date))
policies = {
    'sub_4m_rule': ('2006-03-01', None),  # Excise duty reduction for sub-4m cars, FY 06 Budget[web:1][web:2] (no sunset date)
    'bs3_norms': ('2005-04-01', '2010-03-31'),  # Bharat Stage III, major cities, then nationwide[web:10][web:16]
    'bs4_norms': ('2010-04-01', '2017-03-31'),  # Bharat Stage IV, major cities and then nationwide[web:10][web:16]
    'bs6_norms': ('2020-04-01', None),         # Bharat Stage VI national rollout[web:10][web:16] (ongoing)
    'fame_i': ('2015-04-01', '2019-03-31'),    # FAME India Phase I[web:6][web:12]
    'fame_ii': ('2019-04-01', '2024-03-31'),   # FAME India Phase II[web:6]
    'fame_iii': ('2025-04-01', None),          # Hypothetical/draft Phase III (not officially launched yet)
    'pli_scheme': ('2021-04-01', '2026-03-31'),# Automotive & Telecom PLI, effective April 2021 (most applications until 2025)[web:7][web:5][web:13]
    'vehicle_scrappage_policy': ('2021-03-15', None), # National Vehicle Scrappage Policy announced March 2021, major rollout from April 2023[web:9][web:15]
    'bharat_ncap': ('2023-10-01', None),       # Bharat NCAP crash test program effective October 2023[web:14][web:8]
}


# Create policy flags
for policy, (start_date, end_date) in policies.items():
    start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
    end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
    
    policy_df[policy] = 0
    mask = (policy_df['date'] >= pd.to_datetime(start_date)) if start_date else False
    if end_date:
        mask &= (policy_df['date'] <= pd.to_datetime(end_date))
    policy_df.loc[mask, policy] = 1

# Save to CSV
policy_df.to_csv('india_auto_policies_2023_2025.csv', index=False)
print("Dataset generated successfully!")