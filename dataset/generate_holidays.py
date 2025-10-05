import holidays
import pandas as pd
from datetime import datetime

# Initialize Indian holidays for 2023-2025
india_holidays = holidays.India(years=range(2005, 2026))  # range(2023, 2026) gives 2023, 2024, 2025

# Convert to DataFrame and extract month/year
holiday_list = []
for date, name in india_holidays.items():
    holiday_list.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Day": date.strftime("%A"),
        "Month": date.strftime("%B"),
        "Year": date.year,
        "Holiday Name": name,
        "Type": "National"  # Default (modify for regional holidays)
    })

# Create DataFrame, sort by date, and save as CSV
df = pd.DataFrame(holiday_list)
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime for proper sorting
df = df.sort_values('Date')  # Sort chronologically
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # Convert back to string format

df.to_csv("india_holidays_2005_2024.csv", index=False)

print("CSV generated successfully!")
print(f"Total holidays generated: {len(df)}")
print("\nPreview:")
print(df.head(10))
print(f"\nHolidays by year:")
print(df.groupby('Year').size())
