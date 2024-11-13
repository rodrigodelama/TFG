'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: code/ta_eval.py

This script reads the technical analysis metrics from the CSV file
Chooses the metrics we want to use for the regression model and evaluates the model
'''

#! REVIEW THIS CODE
#! this is not exactly the correct way to do it

import pandas as pd
import regression as reg
import numpy as np

hour_to_predict = 14
rw = 3

# Load data
data = pd.read_csv(f'data/ta_metrics/metrics_hour_{hour_to_predict}_rw_{rw}.csv', parse_dates=['Datetime'])

# print("NaN values per column:\n", data.isna().sum())
# print("Infinity values per column:\n", np.isinf(data).sum())

data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())

# Define the target variable
target = data['MarginalES']

# Select the desired features
features = data[['SMA_3', 'EMA_3', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ROC_3', 'RSI_3']]

# Train the model
model = reg.train_model(features, target)

# Calculate new metrics based on the last available rows in your dataset
# This is an example; you may need to adjust based on your specific metrics
last_values = data.tail(3)  # Example for a 3-day moving average

new_sma_3 = last_values['MarginalES'].mean()
new_ema_3 = last_values['MarginalES'].ewm(span=3, adjust=False).mean().iloc[-1]
new_bb_upper = last_values['MarginalES'].mean() + 2 * last_values['MarginalES'].std()
new_bb_middle = last_values['MarginalES'].mean()
new_bb_lower = last_values['MarginalES'].mean() - 2 * last_values['MarginalES'].std()
new_roc_3 = (last_values['MarginalES'].iloc[-1] - last_values['MarginalES'].iloc[0]) / last_values['MarginalES'].iloc[0]
new_rsi_3 = 100 - 100 / (1 + new_roc_3)
# Similarly calculate BB_Upper, BB_Lower, ROC, RSI, etc.

next_day_data = pd.DataFrame({
    'SMA_3': [new_sma_3],
    'EMA_3': [new_ema_3],
    'BB_Upper': [new_bb_upper],
    'BB_Middle': [new_bb_middle],
    'BB_Lower': [new_bb_lower],
    'ROC_3': [new_roc_3],
    'RSI_3': [new_rsi_3]
    # Add other metrics here like BB_Upper, BB_Middle, BB_Lower, ROC, RSI
})

# Predict the next day's MarginalES (or target variable)
next_day_prediction = model.predict(next_day_data)
print(f"Predicted value for the next day: {next_day_prediction[0]}")

