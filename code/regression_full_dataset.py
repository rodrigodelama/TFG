import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/processed_data.csv')


# Assuming your DataFrame is df and contains columns 'Datetime' and 'MarginalES'

# Function to create input-output matrix
def create_input_output_matrix(prices, window_size):
    # Create the input matrix X and output vector y
    X = []
    y = []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    return pd.DataFrame(X), pd.Series(y)

# Function to evaluate window size and num_days_back
def evaluate_window_and_days_back(df, target_date, num_days_back, window_sizes):
    # Filter data for the last `num_days_back` days before the target date
    filtered_df = df[df['Datetime'] < target_date].tail(num_days_back)
    
    if len(filtered_df) < max(window_sizes):
        return []  # Not enough data
    
    prices = filtered_df['MarginalES'].values

    results = []
    for window_size in window_sizes:
        X, y = create_input_output_matrix(prices, window_size)
        if len(X) < 2:  # Need at least 2 samples to fit the model
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

# Ensure your DataFrame is sorted by Datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.sort_values('Datetime', inplace=True)

# Iterate over each target date, starting from the minimum required date
results = []
window_sizes = [3, 5, 7]  # Define window sizes
num_days_back_list = [10, 15, 20, 30]  # Define num_days_back options

# Define the earliest date you can process based on your data and num_days_back
earliest_date = df['Datetime'].min() + pd.Timedelta(days=max(num_days_back_list))

for target_date in df['Datetime']:
    if target_date < earliest_date:
        continue  # Skip dates that don't have enough historical data

    for num_days_back in num_days_back_list:
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
        else:
            print(f"Not enough data for {target_date} with num_days_back={num_days_back}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find the best configuration for each date
if not results_df.empty:
    best_results = results_df.loc[results_df.groupby('target_date')['mse'].idxmin()]
    print(best_results)
else:
    print("No valid results to display.")