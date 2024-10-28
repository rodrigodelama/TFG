# matrix_builder.py

import pandas as pd

def sliding_window_matrix(data, window_size):
    prices = data['MarginalES'].values  # Make sure this is the correct column name
    X = []
    y = []

    # Check if there's enough data
    if len(prices) < window_size + 1:
        print("Not enough data to form the matrix.")
        return pd.DataFrame(X), pd.Series(y)

    # Building the sliding window
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    
    # Convert X to DataFrame and y to Series
    return pd.DataFrame(X), pd.Series(y)

# Include this in your data selection and processing functions
def select_data_for_window(data, start_date, end_date):
    # Filter the DataFrame based on date range
    filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]
    print(f"Filtered DataFrame (within date range):\n{filtered_data.head()}")  # Debug statement

    # Check if filtered_data is empty
    if filtered_data.empty:
        print("No data available for the selected date range.")
        return pd.DataFrame(), pd.Series()

    return filtered_data

# Your main testing function
def test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options):
    # Load your data here
    data = pd.read_csv(csv_hour_file)  # Assuming you are reading CSV in this manner
    data['Datetime'] = pd.to_datetime(data['Datetime'])  # Ensure Datetime is in correct format

    # Example of how you would loop through window sizes and days back
    for num_days_back in days_back_options:
        selected_data = select_data_for_window(data, target_date - pd.Timedelta(days=num_days_back), target_date)

        for window_size in window_sizes:
            X, y = sliding_window_matrix(selected_data, window_size)

            # Debug statements to check shapes
            print(f"Testing window size: {window_size}")
            print(f"Input matrix (X):\n{X}")
            print(f"Output vector (y):\n{y}")

            if X.empty or y.empty:
                print(f"Skipping due to empty input or output for window size {window_size}.")
                continue
            
            # Proceed with your regression or analysis using X and y
            # ...

# Your main execution logic
if __name__ == "__main__":
    csv_hour_file = "path_to_your_data.csv"  # Update this with your actual file path
    target_date = pd.to_datetime("2018-01-17")  # Example target date
    window_sizes = [3, 5, 10]  # Example window sizes
    days_back_options = [1, 3, 5]  # Example days back options

    results_df = test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)