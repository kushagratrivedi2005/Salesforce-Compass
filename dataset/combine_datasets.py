import pandas as pd
import os
from datetime import datetime

# Define file paths
data_dir = '/home/alookaladdoo/Salesforce-Compass/dataset'
interest_rates_file = os.path.join(data_dir, 'rbi_repo_rate.csv')
holidays_file = os.path.join(data_dir, 'india_holidays_2005_2025.csv')
auto_policies_file = os.path.join(data_dir, 'india_auto_policies_2005_2025.csv')
output_file = os.path.join(data_dir, 'combined_data.csv')

# Check if files exist
for file_path in [interest_rates_file, holidays_file, auto_policies_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Process Interest Rates data
print("Processing Interest Rates data...")
interest_df = pd.read_csv(interest_rates_file)
# Parse the Month column to extract year and month
interest_df[['year', 'month']] = interest_df['Month'].str.split('-', expand=True)
interest_df['year'] = interest_df['year'].astype(int)
interest_df['month'] = interest_df['month'].astype(int)
# Rename to avoid confusion
interest_df = interest_df.rename(columns={'Repo Rate': 'repo_rate'})
# Create monthly interest rates dataframe
interest_monthly = interest_df[['year', 'month', 'repo_rate']]

# Process Holidays data
print("Processing Holidays data...")
holidays_df = pd.read_csv(holidays_file)
# Use Year and Month columns directly from the CSV
holidays_df['year'] = holidays_df['Year'].astype(int)
# Convert month name to month number
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
holidays_df['month'] = holidays_df['Month'].map(month_to_num)
holidays_df = holidays_df[(holidays_df['year'] >= 2005) & (holidays_df['year'] <= 2025)]

# Add flag for major religious holidays (Dussehra and Diwali)
holidays_df['major_religious_holiday'] = holidays_df['Holiday Name'].str.contains('Dussehra|Diwali', case=False, na=False).astype(int)

# Add flag for major national holidays (Independence Day and Republic Day)
holidays_df['major_national_holiday'] = holidays_df['Holiday Name'].str.contains('Independence Day|Republic Day', case=False, na=False).astype(int)

# Create monthly holiday counts - using max for flag columns to ensure they're 1 if any holiday of that type exists in the month
holiday_monthly = holidays_df.groupby(['year', 'month']).agg({
    'Holiday Name': 'count',
    'major_religious_holiday': 'max',
    'major_national_holiday': 'max'
}).reset_index()
holiday_monthly = holiday_monthly.rename(columns={'Holiday Name': 'holiday_count'})

# Process Auto Policies data
print("Processing Auto Policies data...")
policies_df = pd.read_csv(auto_policies_file)
policies_df['date'] = pd.to_datetime(policies_df['date'])
policies_df['year'] = policies_df['date'].dt.year
policies_df['month'] = policies_df['date'].dt.month
policies_df = policies_df[(policies_df['year'] >= 2005) & (policies_df['year'] <= 2025)]
# Drop the original date column after extraction
policies_df = policies_df.drop(columns=['date'])

# Generate all month-year combinations from 2005-01 to 2025-08
start_date = pd.Timestamp('2005-01-01')
end_date = pd.Timestamp('2025-09-30')
all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
date_df = pd.DataFrame({
    'year': all_dates.year,
    'month': all_dates.month
})

# Join the datasets on month and year
print("Joining datasets...")
# Start with all possible month-year combinations
combined_df = date_df.copy()
# Join with interest rates
combined_df = pd.merge(combined_df, interest_monthly, on=['year', 'month'], how='left')
# Then join with holidays
combined_df = pd.merge(combined_df, holiday_monthly, on=['year', 'month'], how='left')
# Finally join with auto policies
combined_df = pd.merge(combined_df, policies_df, on=['year', 'month'], how='left')

# Add month name for better readability
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
combined_df['month_name'] = combined_df['month'].map(month_names)

# Fill NaN values with 0 for numeric columns (especially for holiday counts that might be missing)
numeric_cols = combined_df.select_dtypes(include=['number']).columns
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)

# Reorder columns to have year, month, and month_name first
cols = ['year', 'month', 'month_name'] + [col for col in combined_df.columns 
                                         if col not in ['year', 'month', 'month_name']]
combined_df = combined_df[cols]

# Sort by year and month to ensure chronological order
combined_df = combined_df.sort_values(['year', 'month'])

# Save the combined dataframe
print(f"Saving combined data to {output_file}...")
combined_df.to_csv(output_file, index=False)

print(f"Data processing complete. Shape of the combined data: {combined_df.shape}")
print(f"Columns in the combined data: {list(combined_df.columns)}")