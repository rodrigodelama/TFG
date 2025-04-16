'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-02-27
File: code/utils/plot_metrics.py
'''

import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a CSV file
# data = pd.read_csv('data/ta_metrics/new_price_metrics_hour_14.csv', parse_dates=['Datetime'])
data = pd.read_csv('data/ta_metrics/new_return_metrics_hour_14.csv', parse_dates=['Datetime'])

# Set the Datetime column as the index
data.set_index('Datetime', inplace=True)

# List of metrics to plot
# metrics = ['MarginalES', 'SMA_3', 'SMA_5', 'SMA_7', 'SMA_10', 'SMA_30', 'SMA_50', 'SMA_60', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'ROC_12', 'ROC_50', 'RSI_5', 'RSI_7', 'RSI_14']
metrics = ['Return', 'SMA_3', 'SMA_5', 'SMA_7', 'SMA_10', 'SMA_30', 'SMA_50', 'SMA_60', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'ROC_12', 'ROC_50', 'RSI_5', 'RSI_7', 'RSI_14']

# Create a plot for each metric
for metric in metrics:
    plt.figure(figsize=(12, 6))
    
    # Plot the actual price and the moving averages
    if 'SMA' in metric or 'EMA' in metric:
        # plt.plot(data.index, data['MarginalES'], label='MarginalES', color='black', linewidth=1.5)
        plt.plot(data.index, data['Return'], label='Return', color='black', linewidth=1.5)
    
    plt.plot(data.index, data[metric], label=metric)
    
    # Add titles and labels
    plt.title(f'Time Series Plot of {metric}')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    
    # Save the plot as a PNG file
    # plt.savefig(f'data/plots/price/{metric}_plot.png')
    plt.savefig(f'data/plots/return/{metric}_plot.png')
    plt.close()

print("Plots have been created and saved as PNG files.")
