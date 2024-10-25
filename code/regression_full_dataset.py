import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import custom functions from matrix_builder
from matrix_builder import select_data_for_window, sliding_window_matrix

def evaluate_model(X, y):
    # Check if we have enough data for train-test split
    if X.shape[0] < 2 or y.shape[0] < 2:
        print("Not enough samples to perform train-test split. Skipping this evaluation.")
        return None, None  # Or use some default values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Assuming you have a model defined
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def test_all_dates_and_dimensions(csv_hour_file, window_sizes, days_back_options):
    # Step 1: Load data as a DataFrame here
    data = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    print("Type of 'data' after loading:", type(data))  # Should print <class 'pandas.core.frame.DataFrame'>
    print("First few rows of 'data':\n", data.head())   # Show some initial data to confirm structure

    results = []

    for date_idx, target_date in enumerate(data['Datetime']):
        print(f"\nProcessing date index: {date_idx} - Target date: {target_date}")
        
        for num_days_back in days_back_options:
            print(f"Testing num_days_back: {num_days_back}")
            
            # Select data for the specified number of days back
            selected_data = select_data_for_window(data, target_date, max(window_sizes), num_days_back)
            
            if selected_data.empty:
                print("Selected data is empty after filtering. Skipping to next iteration.")
                continue  # Skip to the next iteration if no data is available
            
        for window_size in window_sizes:
            print(f"Testing window size: {window_size}")
            
            # Generate sliding window matrix
            X, y = sliding_window_matrix(selected_data['MarginalES'].values, window_size)
            
            print(f"Input matrix (X) shape: {X.shape}")
            print(f"Output vector (y) shape: {y.shape}")
            
            # Check if the matrices are empty before proceeding
            if X.size == 0 or y.size == 0:
                print(f"Input matrix (X) or output vector (y) is empty. Skipping this configuration.")
                continue
            
            # Check if we have enough data for model evaluation
            if X.shape[0] < 2 or y.shape[0] < 2:
                print(f"Not enough samples to perform train-test split for window size {window_size}. Skipping this configuration.")
                continue

            # Evaluate model performance
            mse, r2 = evaluate_model(X, y)

            # Ensure that the results are valid before storing
            if mse is not None and r2 is not None:
                # Store the results for analysis
                results.append({
                    'window_size': window_size,
                    'num_days_back': num_days_back,
                    'mse': mse,
                    'r2': r2
                })
                
                print(f"Window size: {window_size}, Days back: {num_days_back}, MSE: {mse}, R²: {r2}")
            else:
                print(f"Evaluation for window size {window_size} returned None values. Skipping storage.")
    
    return pd.DataFrame(results)

# Example test run
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-01-27 14:00:00'
window_sizes = [3, 5, 7]  # Test different window sizes
days_back_options = [10, 15, 20, 30]  # Test different number of days back

# Run the tests
results_df = test_all_dates_and_dimensions(csv_hour_file, window_sizes, days_back_options)
print("\nResults:")
print(results_df)