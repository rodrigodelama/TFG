'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado 
'''

import pandas as pd

def select_data_for_window(csv_hour_file, target_date, window_size, num_days_back):
    # Read the CSV file
    df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

    # Filter the DataFrame to include only the same time of day
    target_datetime = pd.to_datetime(target_date)
    target_time = target_datetime.time()
    
    # Get the starting date for filtering
    start_date = target_datetime - pd.Timedelta(days = num_days_back)

    # Filter based on the target time and the date range
    filtered_df = df[(df['Datetime'].dt.time == target_time) & 
                     (df['Datetime'] >= start_date) & 
                     (df['Datetime'] <= target_datetime)]

    # Print full DataFrame for debugging
    # print("Full DataFrame:")
    # print(df)

    # Print filtered DataFrame
    # print("\nFiltered DataFrame (same time of day within date range):")
    # print(filtered_df)

    # Filter to only date and price columns
    filtered_df = filtered_df[['Datetime', 'MarginalES']]
    print("\nFiltered DataFrame (only date and price columns):")
    print(filtered_df)

    # Extract prices for the selected time
    prices = filtered_df['MarginalES'].values
    print("\nPrices:", prices)

    return prices

def sliding_window_matrix(prices, window_size):
    X, y = [], []
    
    for i in range(len(prices) - window_size):
        # Create the sliding window matrix
        X.append(prices[i:i + window_size])
        # Append the next value as target
        y.append(prices[i + window_size])
    
    return pd.DataFrame(X), pd.Series(y)

# Example usage
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-01-27 14:00:00'
window_size = 3  # Number of previous days to include in the window
num_days_back = 10 + window_size # Number of days back from the target date to consider
# Include additional days for the window size so we have enough data for the matrix

# Select the data from the CSV
selected_data = select_data_for_window(csv_hour_file, target_date, window_size, num_days_back)

# Create the sliding window matrix
X, y = sliding_window_matrix(selected_data, window_size)

print("\nInput matrix (X):")
print(X)

print("\nOutput vector (y):")
print(y)
