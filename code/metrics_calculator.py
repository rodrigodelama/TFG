'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/metrics_calculator.py
'''

import pandas as pd
import regression
import regression_more_metrics
import os # To check if the file exists

# Iterate through the whole database to calculate the regression metrics
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
csv_hour_file = 'data/processed_data_no_weekends.csv'

# Define the file where to save the results
# file_path = f'data/metrics/results_more_metrics.csv'
file_path = f'data/metrics/results_no_weekends.csv'

# Check if the file exists and is not empty
write_header = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0

window_sizes = [3, 5, 7]  # Test different window sizes
days_back_options = [10, 15, 20, 30]  # Test different number of days back


for target_date in pd.read_csv(csv_hour_file, parse_dates=['Datetime'])['Datetime']:

    # Only write header if the file doesn’t exist
    write_header = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0

    print(f"Target date: {target_date}")

    # results_df = regression.test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)
    results_df = regression_more_metrics.test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)
    
    # Review appending results about the headers - headers may be added later when the file is parsed to a DataFrame
    # Append the results to the results file
    # Write to CSV, adding headers only if the file is new or empty
    results_df.to_csv(file_path, mode='a', header=write_header, index=False)
