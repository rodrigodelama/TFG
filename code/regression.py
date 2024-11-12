'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/regression.py
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    max_error
)

# Import custom functions
import matrix_builder

# import os # To check if the file exists

# Debug flag
debug = False

def debug_print(message):
    if debug:
        print(message)


# Function to evaluate a linear regression model
def evaluate_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

# Function to test different window sizes and days back
def test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options):
    results = []

    data = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    data = data[['Datetime', 'MarginalES']]
    
    for num_days_back in days_back_options:
        debug_print(f"\nTesting num_days_back: {num_days_back}")
        
        # Select data for the specified number of days back
        filtered_data = matrix_builder.select_data_for_window(data, target_date, num_days_back, max(window_sizes))
        
        prices = filtered_data['MarginalES'].values

        for window_size in window_sizes:
            debug_print(f"Testing window size: {window_size}")

            try:
                # Generate sliding window matrix
                X, y = matrix_builder.sliding_window_matrix(prices, window_size, num_days_back)
                
                # Evaluate model performance
                mse, r2 = evaluate_model(X, y)
                
                # Store the results for analysis
                results.append({
                    'target_date': target_date,
                    'window_size': window_size,
                    'num_days_back': num_days_back,
                    'mse': mse,
                    'r2': r2
                })
                debug_print(f"Window size: {window_size}, Days back: {num_days_back}, MSE: {mse}, R²: {r2}")

            except Exception as e:
                print(f"Failed for date {target_date} with dimensions width={window_size}, depth={num_days_back}: {e}")
            

    return pd.DataFrame(results)

'''
# Example test run
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-03-27 14:00:00'
window_sizes = [3, 5, 7]  # Test different window sizes
days_back_options = [10, 15, 20, 30]  # Test different number of days back

# Run the tests
results_df = test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)

# Display all results
print("\nResults:")
print(results_df)

# Append the results to the results file
results_df.to_csv(f'data/metrics/results_{target_date}.csv', index=False)

file_path = f'data/metrics/results.csv'
write_header = not os.path.exists(file_path) # Only write header if the file doesn’t exist

# Append the results to the results file
results_df.to_csv(file_path, mode='a', header=write_header, index=False)
'''
