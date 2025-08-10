import pandas as pd
from datetime import datetime

# Create monthly date range from Jan 2023 to Dec 2025
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='MS')

# Initialize DataFrame
policy_df = pd.DataFrame({'date': dates})

# Define policy activation periods (format: (start_date, end_date))
policies = {
    'fame_ii': ('2019-04-01', '2024-03-31'),  # Extended to March 2024
    'fame_iii': ('2025-04-01', '2025-12-31'),  # Hypothetical start (draft)
    'pm_edrive': ('2024-10-01', '2026-03-31'),
    'bs7_norms': ('2030-01-01', '2030-12-31'),  # Expected future
    'bharat_ncap': ('2023-10-01', None),  # Ongoing
    'scrappage_policy': ('2023-01-01', None),  # Ongoing
    'pli_scheme': ('2021-01-01', '2030-12-31'),
    'repo_rate_cut_2025': ('2025-06-01', '2025-12-31')  # Rate cut period
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

# Add repo rate values (example)
repo_rates = {
    '2023': 6.50,
    '2024': 6.25,
    '2025': 5.50  # Reduced from June 2025
}

def get_repo_rate(row):
    year = row.year
    month = row.month
    if year == 2025 and month >= 6:  # Rate cut in June 2025
        return 5.50
    return repo_rates.get(str(year), 6.50)

policy_df['repo_rate'] = policy_df['date'].apply(get_repo_rate)

# Save to CSV
policy_df.to_csv('india_auto_policies_2023_2025.csv', index=False)
print("Dataset generated successfully!")