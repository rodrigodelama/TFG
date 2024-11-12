'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: static_regression.py
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import custom functions
from matrix_builder import *
# select_data_for_window
# sliding_window_matrix

# Function to evaluate a linear regression model
def evaluate_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Mean Squared Error of the model (0 is perfect)
    mse = mean_squared_error(y_test, y_pred)

    # R² score of the model (1 is perfect, 0 is random)
    r2 = r2_score(y_test, y_pred)
    
    # METRICAS HABLADAS CON EMILIO
    # mse relativo dividir por lo que tienes que predecir
    # estadisticas de casos peores
    # percentil 95
    # expectation short fall (ESF)

    # Relative MSE (divided by the mean of the target variable)
    mse = mse / y_test.mean()  # Relative MSE
    
    # 95th percentile 
    percentile_95 = np.percentile(y_test, 95)

    # Expectation Short Fall (ESF)
    # Average difference between the predicted value and the real value for the 5% worst cases
    esf = np.mean(y_test[y_test < percentile_95] - y_pred[y_test < percentile_95])
    
    return mse, r2, percentile_95, esf

# Function to test different window sizes and days back
def test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options):
    results = []
    
    data = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])
    data = data[['Datetime', 'MarginalES']]

    for num_days_back in days_back_options:
        print(f"\nTesting num_days_back: {num_days_back}")
        
        # # Select data for the specified number of days back
        # selected_data = select_data_for_window(data, target_date, max(window_sizes), num_days_back)
        
        for window_size in window_sizes:
            print(f"Testing window size: {window_size}")
            
            # Select data for the specified number of days back
            selected_data = select_data_for_window(data, target_date, window_size, num_days_back)
            
            # Generate sliding window matrix
            X, y = sliding_window_matrix(selected_data, window_size, num_days_back)
            
            # Evaluate model performance
            mse, r2 = evaluate_model(X, y)
            
            # Store the results for analysis
            results.append({
                'window_size': window_size,
                'num_days_back': num_days_back,
                'mse': mse,
                'r2': r2
            })
            
            print(f"Window size: {window_size}, Days back: {num_days_back}, MSE: {mse}, R²: {r2}")
    
    return pd.DataFrame(results)

# Example test run
csv_hour_file = 'data/ta_metrics/hour_14_metrics.csv'
target_date = '2018-01-27 14:00:00'
# Columns
window_sizes = [3, 5, 7]  # Test different window sizes
# Rows
days_back_options = [10, 15, 20, 30]  # Test different number of days back

# Run the tests
results_df = test_window_and_days_back(csv_hour_file, target_date, window_sizes, days_back_options)

# Display all results
print("\nResults:")
print(results_df)
