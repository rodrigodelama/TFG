'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/ta_eval.py

This script reads the technical analysis metrics from the CSV file
Chooses the metrics we want to use for the regression model and evaluates the model
'''

import pandas as pd
import regression as reg
import numpy as np

hour_to_predict = 14
rw = 3

# Load data
data = pd.read_csv(f'data/ta_metrics/metrics_hour_{hour_to_predict}_rw_{rw}.csv', parse_dates=['Datetime'])

#! REVIEW THIS PROCESS
# Find columns with NaN or infinity values
print("NaN values per column:\n", data.isna().sum())
print("Infinity values per column:\n", np.isinf(data).sum())

# Option 1: Drop rows with NaN or infinity values
# data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Option 2: Replace NaN or infinity values with the mean or median
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# Replace NaN or infinity values with the mean or median
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# Review columns with NaN or infinity values
print("NaN values per column:\n", data.isna().sum())
print("Infinity values per column:\n", np.isinf(data).sum())
#! REVIEW THIS PROCESS

# Define the target variable
target = data['MarginalES']

# Select the desired features
features = data[['SMA_3', 'EMA_3', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ROC_3', 'RSI_3']]

# Evaluate the model
mse, r2 = reg.evaluate_model(features, target)

# Save the results to a CSV file
results = pd.DataFrame({
    'MSE': [mse],
    'R2': [r2]
})
results.to_csv('data/ta_metrics/results_hour_14_rw_3.csv', index=False)
