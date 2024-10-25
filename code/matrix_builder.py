import pandas as pd

def select_data_for_window(data, target_date, window_size, num_days_back):
    # Confirm type of 'data' to verify it’s a DataFrame
    print("Type of 'data' received in select_data_for_window:", type(data))
    print("First few rows of 'data':\n", data.head())  # This should now work if 'data' is a DataFrame

    # Proceed with filtering logic based on the target date and date range
    filtered_data = data[
        (data['Datetime'] <= target_date) &
        (data['Datetime'] >= (target_date - pd.Timedelta(days=num_days_back)))
    ]
    
    # Print filtered data to verify
    print("\nFiltered DataFrame (within date range):")
    print(filtered_data)
    
    return filtered_data.tail(window_size)

def sliding_window_matrix(prices, window_size):
    X, y = [], []
    
    # Create the sliding window matrix
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])  # Next value as target
    
    # Return as DataFrame and Series for compatibility
    return pd.DataFrame(X), pd.Series(y)

# Example usage
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-01-27 14:00:00'
window_size = 3  # Number of previous days to include in the window
num_days_back = 10 + window_size  # Total days to look back for the matrix, inclusive of the window

# Load the CSV data once to avoid reloading
data = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

# Convert target_date to a Timestamp for consistency
target_date = pd.to_datetime(target_date)

# Select the data from the CSV using the DataFrame and target date
selected_data = select_data_for_window(data, target_date, window_size, num_days_back)

# Ensure the filtered DataFrame is limited to the necessary columns
prices = selected_data['MarginalES'].values
print("\nPrices array:", prices)

# Create the sliding window matrix
X, y = sliding_window_matrix(prices, window_size)

# Output the matrix and vector for verification
print("\nInput matrix (X):")
print(X)
print("\nOutput vector (y):")
print(y)