'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/data_processing/db_builder.py
'''

import pandas as pd
import os

# Base path to the folder containing the year by year data folders
base_path = '../TFG/data/'

# Initialize an empty list to store DataFrames
all_data = []

# Iterate through each year folder and process all files
for root, dirs, files in os.walk(base_path):
    for file in files:
        # Check if the file starts with 'marginalpdbc_' (ignorering the rest, including the extension)
        if file.startswith('marginalpdbc_'):
            # Extract year and file date from the filename
            file_path = os.path.join(root, file)
            base_name = os.path.basename(file)
            file_date = base_name.split('_')[1].split('.')[0] # Second split to remove the extension
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Remove the first line (MARGINALPDBC;) and the last line (*)
            data_lines = lines[1:-1]
            
            # Convert the list of strings into a DataFrame
            daily_data = pd.DataFrame([line.strip().split(';') for line in data_lines])
            
            # Remove trailing ';' from the last column
            if daily_data.shape[1] > 6:
                # Drop empty columns
                daily_data = daily_data.iloc[:, :6]
            
            # Assign column names
            daily_data.columns = ['Year', 'Month', 'Day', 'Hour', 'MarginalPT', 'MarginalES']
            
            # Remove rows with invalid data
            daily_data = daily_data[daily_data['Year'].str.isdigit()]  # Only keep rows where 'Year' is a number
            
            # Convert the columns to the appropriate data types
            daily_data = daily_data.astype({
                'Year': int, 'Month': int, 'Day': int, 'Hour': int, 'MarginalPT': float, 'MarginalES': float
                })
            
            # Append daily data to the list
            all_data.append(daily_data)

# Concatenate all daily DataFrames into one big DataFrame
full_data = pd.concat(all_data, ignore_index=True)

# Create a datetime column from Year, Month, Day, and Hour
# Adjust the 'Hour' column to deal with the potential 25th hour issue ?????
full_data['Hour'] = full_data['Hour'].apply(lambda x: 0 if x == 25 else x)
full_data['Datetime'] = pd.to_datetime(full_data[['Year', 'Month', 'Day', 'Hour']], errors='coerce')

# Set the 'Datetime' column as the index
full_data.set_index('Datetime', inplace=True)

# Drop unnecessary columns
# Our target variable is 'MarginalES', so 'MarginalPT' will be dropped
full_data.drop(['Year', 'Month', 'Day', 'Hour', 'MarginalPT'], axis=1, inplace=True)

# Save the full dataset to a CSV file
full_data.to_csv('../../data/raw_data.csv')


# Database cleanup
# Load the dataset
df = pd.read_csv('../../data/raw_data.csv')

# Sort the data by the 'Datetime' column
df = df.sort_values(by='Datetime')

# Save the sorted data to a new file if needed
df.to_csv('../../data/processed_data.csv', index=False)

# Read the CSV file
df = pd.read_csv('../../data/processed_data.csv')

# Convert 'Datetime' column to datetime objects
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Generate a complete range of hourly timestamps
full_range = pd.date_range(start=df['Datetime'].min(), end=df['Datetime'].max(), freq='h')

# Find missing timestamps
missing_timestamps = full_range.difference(df['Datetime'])

# If successful, print a message
if missing_timestamps.empty:
    print("Data processing completed successfully.")

# Else, print an error message
else:
    print("Error: Missing timestamps found. Check the data.")

    # Print missing timestamps
    print("Missing timestamps:")
    print(missing_timestamps)

    print("Data processing completed with errors.")
