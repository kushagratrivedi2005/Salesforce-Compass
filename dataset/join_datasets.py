import pandas as pd
import os

# Define file paths
data_dir = '/Users/hellgamerhell/Downloads/salesforce/dataset'
fuel_file = os.path.join(data_dir, 'fuel.csv')
vehicle_class_file = os.path.join(data_dir, 'vehicle_class.csv')
vehicle_category_file = os.path.join(data_dir, 'vehicle_category.csv')
output_file = os.path.join(data_dir, 'joined_data.csv')

# Check if files exist
for file_path in [fuel_file, vehicle_class_file, vehicle_category_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Function to read and preprocess a CSV file
def read_and_preprocess(file_path, prefix):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter out the 'Total' row
    df = df[df['Month'] != 'Total']
    
    # Clean numeric columns by removing commas
    for col in df.columns:
        if col != 'Month':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Add prefix to all columns except 'Month' to avoid conflicts during join
    if prefix:
        df.columns = [col if col == 'Month' else f"{prefix}_{col}" for col in df.columns]
    
    return df

# Read and preprocess each dataset
print("Reading and preprocessing datasets...")
fuel_df = read_and_preprocess(fuel_file, 'fuel')
vehicle_class_df = read_and_preprocess(vehicle_class_file, 'class')
vehicle_category_df = read_and_preprocess(vehicle_category_file, 'category')

# Join the datasets on the 'Month' column
print("Joining datasets...")
merged_df = pd.merge(fuel_df, vehicle_class_df, on='Month', how='outer')
merged_df = pd.merge(merged_df, vehicle_category_df, on='Month', how='outer')

# Filter out any remaining problematic rows (just to be safe)
merged_df = merged_df[merged_df['Month'].str.contains('-', na=False)]

# Extract year and month from the 'Month' column
print("Extracting year and month components...")
try:
    merged_df['Year'] = merged_df['Month'].str.split('-').str[0].astype(int)
    merged_df['Month_Number'] = merged_df['Month'].str.split('-').str[1].astype(int)
    
    # Create month name column
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    merged_df['Month_Name'] = merged_df['Month_Number'].map(month_names)
    
    # Reorder columns to have Month, Year, Month_Number, and Month_Name at the beginning
    first_cols = ['Month', 'Year', 'Month_Number', 'Month_Name']
    other_cols = [col for col in merged_df.columns if col not in first_cols]
    merged_df = merged_df[first_cols + other_cols]
    
except Exception as e:
    print(f"Error during date processing: {e}")
    print("Checking for problematic rows:")
    problematic = merged_df[~merged_df['Month'].str.contains('-', na=True)]
    if not problematic.empty:
        print(f"Found {len(problematic)} problematic rows:")
        print(problematic['Month'].unique())

# Save the joined dataframe
print(f"Saving joined data to {output_file}...")
merged_df.to_csv(output_file, index=False)

print(f"Data processing complete. Shape of the joined data: {merged_df.shape}")
print(f"Columns in the joined data: {len(merged_df.columns)}")
