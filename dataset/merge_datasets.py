import pandas as pd
import os

def merge_datasets():
    # File paths
    base_dir = '/Users/hellgamerhell/Downloads/salesforce/dataset/'
    joined_data_path = os.path.join(base_dir, 'joined_data.csv')
    combined_data_path = os.path.join(base_dir, 'combined_data.csv')
    output_path = os.path.join(base_dir, 'final_merged_dataset.csv')
    
    # Load the datasets
    print(f"Loading datasets...")
    joined_data = pd.read_csv(joined_data_path)
    combined_data = pd.read_csv(combined_data_path)
    
    # Display basic info about the datasets
    print(f"Joined data shape: {joined_data.shape}")
    print(f"Combined data shape: {combined_data.shape}")
    
    # Extract month and year from joined_data for merging
    joined_data['Year'] = joined_data['Year'].astype(int)
    joined_data['Month_Number'] = joined_data['Month_Number'].astype(int)
    
    # Prepare combined_data for merging
    combined_data['year'] = combined_data['year'].astype(int)
    combined_data['month'] = combined_data['month'].astype(int)
    
    # Merge the datasets on year and month
    print("Merging datasets based on year and month...")
    merged_data = pd.merge(
        joined_data,
        combined_data,
        left_on=['Year', 'Month_Number'],
        right_on=['year', 'month'],
        how='inner'  # Only keep rows that have matching data in both datasets
    )
    
    # Check for any missing matches
    joined_count = joined_data.shape[0]
    combined_count = combined_data.shape[0]
    merged_count = merged_data.shape[0]
    print(f"Joined data rows: {joined_count}")
    print(f"Combined data rows: {combined_count}")
    print(f"Merged data rows: {merged_count}")
    
    if merged_count < min(joined_count, combined_count):
        print("Warning: Some rows were lost in the merge. Check for mismatched dates.")
        
        # Find missing dates
        joined_dates = set(zip(joined_data['Year'], joined_data['Month_Number']))
        combined_dates = set(zip(combined_data['year'], combined_data['month']))
        
        missing_in_joined = combined_dates - joined_dates
        missing_in_combined = joined_dates - combined_dates
        
        if missing_in_joined:
            print("\nDates in combined_data but missing in joined_data:")
            for year, month in sorted(missing_in_joined):
                print(f"  {year}-{month:02d}")
        
        if missing_in_combined:
            print("\nDates in joined_data but missing in combined_data:")
            for year, month in sorted(missing_in_combined):
                print(f"  {year}-{month:02d}")
    
    # Clean up redundant columns (keep original columns from joined_data)
    columns_to_drop = ['year', 'month']  # These are duplicated
    merged_data = merged_data.drop(columns=columns_to_drop)
    
    # Save the merged dataset
    print(f"Saving merged dataset...")
    merged_data.to_csv(output_path, index=False)
    print(f"Final merged dataset shape: {merged_data.shape}")
    print(f"Datasets successfully merged and saved to {output_path}")
    
    return merged_data

if __name__ == "__main__":
    merge_datasets()
