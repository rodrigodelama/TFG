import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from matrix_builder import *

# Assuming your DataFrame is df and contains columns 'Datetime' and 'MarginalES'

def create_input_output_matrix(prices, window_size):
    # Create the input matrix X and output vector y
    X = []
    y = []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    return pd.DataFrame(X), pd.Series(y)

def evaluate_window_and_days_back(df, target_date, num_days_back, window_sizes):
    # Filter data for the last `num_days_back` days before the target date
    filtered_df = df[df['Datetime'] < target_date].tail(num_days_back)
    
    if len(filtered_df) < max(window_sizes):
        # Not enough data points for the max window size, skip this case
        return None, None, None, None

    prices = filtered_df['MarginalES'].values

    results = []
    for window_size in window_sizes:
        X, y = create_input_output_matrix(prices, window_size)
        if len(X) == 0:  # If the matrix is empty, skip
            continue
        
        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions and metrics
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        results.append((window_size, num_days_back, mse, r2))
    
    return results

# Iterate over each target date in the data
results = []
window_sizes = [3, 5, 7]  # Define the window sizes to test
num_days_back_list = [10, 15, 20, 30]  # Define the num_days_back to test

for idx, target_date in enumerate(df['Datetime']):
    for num_days_back in num_days_back_list:
        # Evaluate for each date and days_back combination
        res = evaluate_window_and_days_back(df, target_date, num_days_back, window_sizes)
        if res:
            for result in res:
                window_size, num_days_back, mse, r2 = result
                results.append({
                    'target_date': target_date,
                    'window_size': window_size,
                    'num_days_back': num_days_back,
                    'mse': mse,
                    'r2': r2
                })

# Convert the results to a DataFrame for analysis
results_df = pd.DataFrame(results)

# Find the best configuration for each date
best_results = results_df.loc[results_df.groupby('target_date')['mse'].idxmin()]

print(best_results)