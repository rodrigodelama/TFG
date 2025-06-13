'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/regression_tests/regression_more_metrics.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    max_error
)
import matrix_builder

# Debug flag
debug = False

def debug_print(message):
    if debug:
        print(message)

#! To be reviewed and deprecated
#! The idea is to combine it all into regression.py and remove this file
#! In this way, we can have a single file for the regression model and metrics
#! And we can use the same functions for both the basic and advanced metrics
#! If we only need some metrics we can use DataFrame filtering to get those columns

# Function to evaluate a linear regression model
def evaluate_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    # DEPRECATED # rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root Mean Squared Error
    rmse = root_mean_squared_error(y_test, y_pred)           # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)                # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)                            # R² score
    explained_variance = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    # Calculate adjusted R²
    n = len(y_test)  # Number of samples
    p = X.shape[1]   # Number of predictors
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return mse, rmse, mae, r2, adjusted_r2, explained_variance, max_err, mape

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
                
                # Evaluate model performance with multiple metrics
                mse, rmse, mae, r2, adjusted_r2, explained_variance, max_err, mape = evaluate_model(X, y)
                
                # Store all metrics in results for analysis
                results.append({
                    'target_date': target_date,
                    'window_size': window_size,
                    'num_days_back': num_days_back,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'adjusted_r2': adjusted_r2,
                    'explained_variance': explained_variance,
                    'max_error': max_err,
                    'mape': mape
                })
                debug_print(
                    f"Window size: {window_size}, Days back: {num_days_back}, "
                    f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}, "
                    f"Adjusted R²: {adjusted_r2}, Explained Variance: {explained_variance}, "
                    f"Max Error: {max_err}, MAPE: {mape}%"
                )

            except Exception as e:
                print(f"Failed for date {target_date} with dimensions width={window_size}, depth={num_days_back}: {e}")
    
    # Convert results to a DataFrame for analysis
    return pd.DataFrame(results)
