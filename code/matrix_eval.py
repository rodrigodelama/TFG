'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/matrix_eval.py

This script reads the results from the regression model evaluation
Organizes the results by the best metrics, in this case just lowest MSE and highest R²
And prints the most common dimensions in the top 1000 results
'''

import pandas as pd

# File path
# results_file = 'data/metrics/results.csv'
# results_file = 'data/metrics/results_no_weekends.csv'
results_file = 'data/metrics/results_return_data.csv'

# Load the CSV without headers
data = pd.read_csv(results_file, header=None)

# Assign column names
data.columns = ["Datetime", "Rows", "Columns", "MSE", "R2"]

# Organize by lowest MSE and highest R²
sorted_data = data.sort_values(by=["MSE", "R2"], ascending=[True, False])

# # Print the top results
# print(sorted_data.head())

# Print the top 1000 results
print(sorted_data.head(1000))

# Remove the ones with a negative R²
positive_r2 = data[data["R2"] > 0]

# Print the top 1000 results
print(positive_r2.head(1000))

# Analyzing the results
# Which dimensions are the best?
# Which window size and number of days back are the best?

# Find the most common dimensions in the top 1000 results
top_dimensions = sorted_data.head(1000)[["Rows", "Columns"]].value_counts()

# Print the top dimensions
print(top_dimensions)
