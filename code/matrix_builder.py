'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/matrix_builder.py
'''

import pandas as pd

# New Version to handle weekends and other gaps in the dataset 
def select_data_for_window(data, target_date, num_data_points, max_window_size):
    """
    Selects the specified number of data points before the target date,
    ignoring weekends or other gaps in the dataset.
    """
    # Filter data up to the target_date
    data = data[data['Datetime'] < target_date]
    
    # Sort by date in descending order to get the most recent points first
    data = data.sort_values(by='Datetime', ascending=False)
    
    # Select the specified number of data points (e.g., num_days_back)
    selected_data = data.head(num_data_points + max_window_size)
    
    # Reverse to maintain chronological order after filtering
    return selected_data.sort_values(by='Datetime', ascending=True)

# Build the matrix and vector (m x n) for the sliding window approach
# WIDTH --> window_size: Number of columns in the matrix
# DEPTH --> num_days_back: Number of rows in the matrix
def sliding_window_matrix(prices, window_size, num_days_back):
    X, y = [], []
    
    # Loop through the prices to create the matrix and vector
    for i in range(num_days_back):
        X.append(prices[i : i+window_size]) # Create a row of the matrix of width window_size days
        # print(f"X: {X}")
        y.append(prices[i + window_size]) # Add the next price after window_size days as the output value
    
    X = pd.DataFrame(X, columns=[f"day-{j}" for j in range(window_size, 0, -1)])
    y = pd.Series(y)

    return X, y

'''
# Example usage
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-01-27 14:00:00'
window_size = 6  # Number of previous days to include in the window
num_days_back = 6  # Total days to look back for the matrix, inclusive of the window

# Load the CSV data once to avoid reloading
data = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

# Ensure the data is in DataFrame format
# print("Type of 'data' loaded from CSV:", type(data))

# Reduce to the necessary columns for the example
data = data[['Datetime', 'MarginalES']]

# Convert target_date to a Timestamp for use in filtering with the DataFrame
# target_date = pd.to_datetime(target_date)

# Select the data from the DataFrame, target date and number of days back to collect
filered_data = select_data_for_window(data, target_date, num_days_back, window_size) # window_size is used to calculate the date range

# Extract the prices from the filtered DataFrame
prices = filered_data['MarginalES'].values
print("\nPrices array:", prices)

# Create the sliding window matrix
X, y = sliding_window_matrix(prices, window_size, num_days_back)

# Output the matrix and vector for verification
print("\nInput matrix (X):")
print(X)
print("\nOutput vector (y):")
print(y)
'''
