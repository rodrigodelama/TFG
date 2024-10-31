'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado 
'''

import pandas as pd
import regression
import os

# Iterate through the whole database to calculate the regression metrics
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'

# Define the file to save the results
results_file = 'data/metrics/results.csv'
file_path = f'data/metrics/results.csv'
# write_header = not os.path.exists(file_path) # Only write header if the file doesn’t exist

# Check if the file is empty or doesn't exist
# Check if the file exists and is not empty
write_header = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0

# if not os.path.isfile(file_path):
#     print("File does not exist; will create new file.")
# elif os.path.getsize(file_path) == 0:
#     print("File exists but is empty; will write headers.")
# else:
#     print("File exists and is not empty; will not write headers.")

# print(f"Writing headers: {write_header}")

window_sizes = [3, 5, 7]  # Test different window sizes
days_back_options = [10, 15, 20, 30]  # Test different number of days back

# counter = 0

for target_date in pd.read_csv(csv_hour_file, parse_dates=['Datetime'])['Datetime']:

    # write_header = not os.path.exists(file_path) # Only write header if the file doesn’t exist
    write_header = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0

    print(f"Target date: {target_date}")

    # Check if the target date was already evaluated
    # if os.path.exists(file_path):
    #     # If empty file continue
    #     if os.stat(file_path).st_size == 0:
    #         print("Empty file")
    #         continue
    #     if pd.read_csv(file_path)['target_date'].str.contains(str(target_date)).any():
    #         print("Already evaluated")
    #         continue

    results_df = regression.test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)
    
    # # Append the results to the results file
    # if counter == 0:
    #     results_df.to_csv(file_path, mode='a', header=True, index=False)
    #     counter += 1

    # results_df.to_csv(file_path, mode='a', header=False, index=False)

    # When saving, check if the file exists to control the header
    # results_df.to_csv(file_path, mode='a', header=(not os.path.exists(file_path)), index=False)

    # Write to CSV, adding headers only if the file is new or empty
    results_df.to_csv(file_path, mode='a', header=write_header, index=False)

