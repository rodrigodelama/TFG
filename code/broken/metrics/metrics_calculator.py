'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/metrics/metrics_calculator.py
'''

# We want to iterate through the whole database to calculate the regression metrics

import pandas as pd
import regression
import regression_more_metrics
import os # To check if the file exists

# File definitions: where to grab the data from, where to save the results to
csv_hour_file = 'data/hour_14_metrics.csv'
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# csv_hour_file = 'data/processed_data_no_weekends.csv'
# csv_hour_file = 'data/return_data.csv'
# results_file = f'data/metrics/results_more_metrics.csv'
# results_file = f'data/metrics/results_no_weekends.csv'
# results_file = f'data/metrics/results_return_data.csv'
results_file = f'data/metrics/results.csv'

# Check if the file exists and is not empty
write_header = not os.path.isfile(results_file) or os.path.getsize(results_file) == 0

window_sizes = [3, 5, 7]  # Test different window sizes
days_back_options = [10, 15, 20, 30]  # Test different number of days back

# Load the target dates
target_dates = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])['Datetime']

# Iterate through each target date and calculate metrics
for target_date in target_dates:
    try:
        print(f"Target date: {target_date}")

        # Calculate the results
        results_df = regression.test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)
        # results_df = regression_more_metrics.test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)
        
        # Append the results to the results file
        results_df.to_csv(results_file, mode='a', header=write_header, index=False)
        
        # After writing once, reset write_header to False
        write_header = False
    except Exception as e:
        print(f"Failed for date {target_date} with error: {e}")
