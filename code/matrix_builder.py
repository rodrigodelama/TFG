#### Version 1

# import pandas as pd
# import numpy as np

# def matrix_builder_from_csv(csv_file, n, m):
#     """
#     Builds a matrix of size n x m using the prices from the CSV file.
    
#     Args:
#         csv_file (str): Path to the CSV file.
#         n (int): Number of rows in the output matrix.
#         m (int): Number of columns in the output matrix.
    
#     Returns:
#         np.ndarray: A matrix of size n x m filled with the prices from the CSV file.
#     """
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file)
    
#     # Extract the 'MarginalES' column (prices)
#     prices = df['MarginalES'].values
    
#     # Ensure there are enough values to reshape
#     if len(prices) < n * m:
#         raise ValueError(f"Not enough data to fill a {n}x{m} matrix.")
    
#     # Reshape the prices into a matrix of size n x m
#     matrix = prices[:n*m].reshape(n, m)
    
#     return matrix

# # Example usage:
# csv_file = 'path_to_your_file.csv'
# n = 2  # Number of rows
# m = 3  # Number of columns
# matrix = matrix_builder_from_csv(csv_file, n, m)
# print(matrix)

#### Version 2

# import pandas as pd
# import numpy as np

# def select_previous_data(csv_file, target_date, hours):
#     """
#     Selects a block of data from the CSV based on the target date and the number of previous hours.
    
#     Args:
#         csv_file (str): Path to the CSV file.
#         target_date (str): The target date in the format 'YYYY-MM-DD HH:MM:SS'.
#         hours (int): Number of previous hours to include.
    
#     Returns:
#         np.ndarray: Array of selected prices from the previous hours.
#     """
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file, parse_dates=['Datetime'])
    
#     # Convert the target date to a datetime object
#     target_date = pd.to_datetime(target_date)
    
#     # Filter the data for the specified time range (previous 'hours' before the target date)
#     start_date = target_date - pd.Timedelta(hours=hours)
#     filtered_df = df[(df['Datetime'] > start_date) & (df['Datetime'] <= target_date)]
    
#     # Extract the 'MarginalES' (prices) column as a NumPy array
#     prices = filtered_df['MarginalES'].values
    
#     if len(prices) < hours:
#         raise ValueError(f"Not enough data for the specified range. Only found {len(prices)} prices.")
    
#     return prices

# def matrix_builder_from_data(data, n, m):
#     """
#     Builds a matrix from the selected data.
    
#     Args:
#         data (np.ndarray): Array of selected data (prices).
#         n (int): Number of rows in the matrix.
#         m (int): Number of columns in the matrix.
    
#     Returns:
#         np.ndarray: Matrix of size n x m.
#     """
#     if len(data) < n * m:
#         raise ValueError(f"Not enough data to fill a {n}x{m} matrix.")
    
#     # Reshape the data into an n x m matrix
#     matrix = data[:n*m].reshape(n, m)
    
#     return matrix

# # Example usage:
# csv_file = 'path_to_your_file.csv'
# target_date = '2018-01-01 06:00:00'
# hours = 6  # Number of previous hours you want to include

# # Select the previous 6 hours of data
# selected_data = select_previous_data(csv_file, target_date, hours)

# # Build a matrix from the selected data
# n = 2  # Number of rows
# m = 3  # Number of columns
# matrix = matrix_builder_from_data(selected_data, n, m)

# print(matrix)

#### Version 3

# import pandas as pd
# import numpy as np

# def select_data_for_window(csv_hour_file, target_date, days_backwards):
#     """
#     Selects data for building a sliding window matrix using prices from previous days.

#     Args:
#         csv_hour_file (str): Path to the CSV file.
#         target_date (str): The target date in the format 'YYYY-MM-DD HH:MM:SS'.
#         days_backwards (int): Number of previous days to include in the window.

#     Returns:
#         np.ndarray: Array of selected MarginalES (MWh prices).
#     """
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    
#     # Convert the target date to a datetime object
#     target_date = pd.to_datetime(target_date)
    
#     # Filter data for only the rows that match the same time of day as the target date
#     target_hour = target_date.time()
#     filtered_df = df[df['Datetime'].dt.time == target_hour]
    
#     # Sort the data by date to ensure it's ordered correctly
#     filtered_df = filtered_df.sort_values(by='Datetime')

#     # Extract the 'MarginalES' (prices) column as a NumPy array
#     prices = filtered_df['MarginalES'].values
    
#     # Find the index of the target date
#     target_index = filtered_df.index[filtered_df['Datetime'] == target_date].tolist()
    
#     if not target_index:
#         raise ValueError(f"Target date {target_date} not found in the dataset.")
    
#     # Get the index of the target date
#     target_index = target_index[0]
    
#     # Ensure there are enough days of data for the window
#     if target_index < days_backwards:
#         raise ValueError(f"Not enough previous data for the specified window size. Only {target_index} days available.")
    
#     # Select the previous `days_backwards` days of data
#     selected_prices = prices[target_index - days_backwards:target_index]
    
#     return selected_prices

# def sliding_window_matrix(data, window_size):
#     """
#     Builds a sliding window matrix from the selected data.

#     Args:
#         data (np.ndarray): Array of prices.
#         window_size (int): Size of the window (how many previous values per row).

#     Returns:
#         np.ndarray, np.ndarray: Input matrix (X) and output vector (y).
#     """
#     X = []
#     y = []
    
#     for i in range(len(data) - window_size):
#         # Create a window of 'window_size' elements
#         X.append(data[i:i + window_size])
#         # The corresponding target value is the next one after the window
#         y.append(data[i + window_size])
    
#     return np.array(X), np.array(y)

# # Example usage:
# csv_hour_file = 'path_to_your_file.csv'
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# target_date = '2018-01-15 14:00:00'
# window_size = 3  # Number of previous days to include in the window

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size)

# print("Input matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)

#### Version 4

# import pandas as pd
# import numpy as np

# def select_data_for_window(csv_hour_file, target_date, days_backwards):
#     """
#     Selects data for building a sliding window matrix using prices from previous days.

#     Args:
#         csv_hour_file (str): Path to the CSV file.
#         target_date (str): The target date in the format 'YYYY-MM-DD HH:MM:SS'.
#         days_backwards (int): Number of previous days to include in the window.

#     Returns:
#         np.ndarray: Array of selected MarginalES (MWh prices).
#     """
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    
#     # Print the DataFrame for debugging
#     print("Full DataFrame:")
#     print(df.head())
    
#     # Convert the target date to a datetime object
#     target_date = pd.to_datetime(target_date)
#     target_hour = target_date.time()

#     # Print the target date and time for debugging
#     print(f"Target date: {target_date}, Target time: {target_hour}")
    
#     # Filter data for only the rows that match the same time of day as the target date
#     filtered_df = df[df['Datetime'].dt.time == target_hour]
    
#     # Print the filtered DataFrame to check the filtering step
#     print("Filtered DataFrame (same time of day):")
#     print(filtered_df)
    
#     # Sort the data by date to ensure it's ordered correctly
#     filtered_df = filtered_df.sort_values(by='Datetime')

#     # Extract the 'MarginalES' (prices) column as a NumPy array
#     prices = filtered_df['MarginalES'].values
    
#     # Print prices for debugging
#     print(f"Prices: {prices}")
    
#     # Find the index of the target date
#     target_index = filtered_df.index[filtered_df['Datetime'] == target_date].tolist()
    
#     if not target_index:
#         raise ValueError(f"Target date {target_date} not found in the dataset.")
    
#     # Get the index of the target date
#     target_index = target_index[0]
    
#     # Ensure there are enough days of data for the window
#     if target_index < days_backwards:
#         raise ValueError(f"Not enough previous data for the specified window size. Only {target_index} days available.")
    
#     # Select the previous `days_backwards` days of data
#     selected_prices = prices[target_index - days_backwards:target_index]
    
#     return selected_prices

# def sliding_window_matrix(data, window_size):
#     """
#     Builds a sliding window matrix from the selected data.

#     Args:
#         data (np.ndarray): Array of prices.
#         window_size (int): Size of the window (how many previous values per row).

#     Returns:
#         np.ndarray, np.ndarray: Input matrix (X) and output vector (y).
#     """
#     X = []
#     y = []
    
#     for i in range(len(data) - window_size):
#         # Create a window of 'window_size' elements
#         X.append(data[i:i + window_size])
#         # The corresponding target value is the next one after the window
#         y.append(data[i + window_size])
    
#     return np.array(X), np.array(y)

# # Example usage:
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# target_date = '2018-01-15 14:00:00'
# window_size = 3  # Number of previous days to include in the window

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size)

# print("Input matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)

#### Version 5

# import pandas as pd
# import numpy as np

# def select_data_for_window(csv_hour_file, target_date, days_backwards):
#     """
#     Selects data for building a sliding window matrix using prices from previous days.

#     Args:
#         csv_hour_file (str): Path to the CSV file.
#         target_date (str): The target date in the format 'YYYY-MM-DD HH:MM:SS'.
#         days_backwards (int): Number of previous days to include in the window.

#     Returns:
#         np.ndarray: Array of selected MarginalES (MWh prices).
#     """
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    
#     # Print the DataFrame for debugging
#     print("Full DataFrame:")
#     print(df.head())
    
#     # Convert the target date to a datetime object
#     target_date = pd.to_datetime(target_date)
#     target_hour = target_date.time()

#     # Print the target date and time for debugging
#     print(f"Target date: {target_date}, Target time: {target_hour}")
    
#     # Filter data for only the rows that match the same time of day as the target date
#     filtered_df = df[df['Datetime'].dt.time == target_hour]
    
#     # Print the filtered DataFrame to check the filtering step
#     print("Filtered DataFrame (same time of day):")
#     print(filtered_df)
    
#     # Sort the data by date to ensure it's ordered correctly
#     filtered_df = filtered_df.sort_values(by='Datetime')

#     # Extract the 'MarginalES' (prices) column as a NumPy array
#     prices = filtered_df['MarginalES'].values
    
#     # Print prices for debugging
#     print(f"Prices: {prices}")
    
#     # Find the index of the target date
#     target_index = filtered_df.index[filtered_df['Datetime'] == target_date].tolist()
    
#     if not target_index:
#         raise ValueError(f"Target date {target_date} not found in the dataset.")
    
#     # Get the index of the target date
#     target_index = target_index[0]
    
#     # Ensure there are enough days of data for the window
#     if target_index < days_backwards:
#         raise ValueError(f"Not enough previous data for the specified window size. Only {target_index} days available.")
    
#     # Select the previous `days_backwards` days of data
#     selected_prices = prices[target_index - days_backwards:target_index+1]  # Adjust the range to include the target value
    
#     return selected_prices

# def sliding_window_matrix(data, window_size):
#     """
#     Builds a sliding window matrix from the selected data.

#     Args:
#         data (np.ndarray): Array of prices.
#         window_size (int): Size of the window (how many previous values per row).

#     Returns:
#         np.ndarray, np.ndarray: Input matrix (X) and output vector (y).
#     """
#     X = []
#     y = []
    
#     if len(data) < window_size + 1:
#         raise ValueError("Not enough data to build the sliding window matrix.")
    
#     for i in range(len(data) - window_size):
#         # Create a window of 'window_size' elements
#         X.append(data[i:i + window_size])
#         # The corresponding target value is the next one after the window
#         y.append(data[i + window_size])
    
#     return np.array(X), np.array(y)

# # Example usage:
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# target_date = '2018-01-15 14:00:00'
# window_size = 5  # Number of previous days to include in the window

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size)

# print("Input matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)

#### Version 6

# import pandas as pd

# def select_data_for_window(csv_hour_file, target_date, window_size):
#     # Load data from CSV
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    
#     # Ensure data is sorted by date
#     df = df.sort_values(by='Datetime')
    
#     # Extract rows corresponding to the target time each day
#     target_time = pd.to_datetime(target_date).time()
#     df_filtered = df[df['Datetime'].dt.time == target_time]
    
#     return df_filtered

# def sliding_window_matrix(selected_data, window_size):
#     prices = selected_data['MarginalES'].values
    
#     # Create the input matrix X and output vector y
#     X, y = [], []
    
#     for i in range(len(prices) - window_size):
#         X.append(prices[i:i + window_size])  # Window of 'window_size' days
#         y.append(prices[i + window_size])    # The next day (the target)
    
#     X = pd.DataFrame(X)  # Optional: Convert to DataFrame for readability
#     y = pd.Series(y)     # Optional: Convert to Series for readability
    
#     return X, y

# # Example usage:

# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# target_date = '2018-01-15 14:00:00'
# window_size = 3  # Number of previous days to include in the window

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size)

# print("Input matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)

#### Version 7

# import pandas as pd

# def select_data_for_window(csv_hour_file, target_date, window_size):
#     # Read the CSV file
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

#     # Filter the DataFrame to include only the same time of day
#     target_datetime = pd.to_datetime(target_date)
#     target_time = target_datetime.time()
#     filtered_df = df[df['Datetime'].dt.time == target_time]

#     # Print full DataFrame for debugging
#     print("Full DataFrame:")
#     print(df)

#     # Print filtered DataFrame for debugging
#     print("\nFiltered DataFrame (same time of day):")
#     print(filtered_df)

#     # Extract prices for the selected time
#     prices = filtered_df['MarginalES'].values
#     print("\nPrices:", prices)

#     return prices

# def sliding_window_matrix(prices, window_size, num_days):
#     X, y = [], []
    
#     for i in range(len(prices) - window_size):
#         # Stop if we reach the limit of days
#         if len(X) >= num_days:
#             break
#         # Create the sliding window matrix
#         X.append(prices[i:i + window_size])
#         # Append the next value as target
#         y.append(prices[i + window_size])
    
#     return pd.DataFrame(X), pd.Series(y)

# # Example usage
# # csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# csv_hour_file = 'data/processed_data.csv'
# target_date = '2018-01-08 14:00:00'
# window_size = 3  # Number of previous days to include in the window
# num_days = 6    # Limit the number of rows in the input matrix

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size, num_days)

# print("\nInput matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)

#### Version 8

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
    print("Full DataFrame:")
    print(df)

    # Print filtered DataFrame
    print("\nFiltered DataFrame (same time of day within date range):")
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
target_date = '2018-01-26 14:00:00'
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

#### Version 9 - BREAKS error was NOT in sliding_window_matrix

# import pandas as pd

# def select_data_for_window(csv_hour_file, target_date, window_size, num_days_back):
#     # Read the CSV file
#     df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

#     # Filter the DataFrame to include only the same time of day
#     target_datetime = pd.to_datetime(target_date)
#     target_time = target_datetime.time()
    
#     # Get the starting date for filtering
#     start_date = target_datetime - pd.Timedelta(days=num_days_back)

#     # Filter based on the target time and the date range
#     filtered_df = df[(df['Datetime'].dt.time == target_time) & 
#                      (df['Datetime'] >= start_date) & 
#                      (df['Datetime'] <= target_datetime)]

#     # Print full DataFrame for debugging
#     print("Full DataFrame:")
#     print(df)

#     # Print filtered DataFrame
#     print("\nFiltered DataFrame (same time of day within date range):")
#     print(filtered_df)

#     # Extract prices for the selected time
#     prices = filtered_df['MarginalES'].values
#     print("\nPrices:", prices)

#     return prices

# def sliding_window_matrix(prices, window_size):
#     X, y = [], []
    
#     # Adjust the range to ensure we do not exceed bounds
#     for i in range(len(prices) - window_size):  # Corrected range
#         # Create the sliding window matrix
#         X.append(prices[i:i + window_size])
#         # Append the next value as target
#         y.append(prices[i + window_size])
    
#     return pd.DataFrame(X), pd.Series(y)

# # Example usage
# csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
# target_date = '2018-01-26 14:00:00'
# window_size = 3  # Number of previous values to include in the window
# num_days_back = 10  # Number of days back from the target date to consider

# # Select the data from the CSV
# selected_data = select_data_for_window(csv_hour_file, target_date, window_size, num_days_back)

# # Create the sliding window matrix
# X, y = sliding_window_matrix(selected_data, window_size)

# print("\nInput matrix (X):")
# print(X)

# print("\nOutput vector (y):")
# print(y)