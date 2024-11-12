'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/matrix_eval_more_metrics.py
'''

import pandas as pd

# File path
results_file = 'data/metrics/results_more_metrics.csv'

# Load the CSV without headers
data = pd.read_csv(results_file, header=None)

# Assign column names
data.columns = ["Datetime", "Rows", "Columns", "MSE", "RMSE", "MAE", "R2", "Adjusted R²", "Explained Variance", "Max Error", "MAPE"]

# Organize by lowest MSE, RMSE, MAE, MAPE, Max Error, and highest R², Adjusted R², Explained Variance
# Ascending for error metrics (e.g., MSE, RMSE, MAE, MAPE, Max Error)
#   prioritizing lower values, as these indicate better model accuracy.
# Descending for variance explained metrics (e.g., R², Adjusted R², Explained Variance)
#   prioritizing higher values, indicating better fit quality.
sorted_data = data.sort_values(
    by=["MSE", "RMSE", "MAE", "MAPE", "Max Error", "R2", "Adjusted R²", "Explained Variance"],
    ascending=[True, True, True, True, True, False, False, False]
)

# Display the top configurations
print(sorted_data.head())

# Print the top 1000 results
print(sorted_data.head(1000))

# # Remove the ones with a negative R²
# positive_r2 = data[data["R2"] > 0]

# # Print the top 1000 results
# print(positive_r2.head(1000))

# Analyzing the results
# Which dimensions are the best?
# Which window size and number of days back are the best?

# Find the most common dimensions in the top 1000 results
top_dimensions = sorted_data.head(1000)[["Rows", "Columns"]].value_counts()

# Print the top dimensions
print(top_dimensions)
