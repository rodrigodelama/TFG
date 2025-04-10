'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-02-20
File: code/utils/return_plot.py
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_returns(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, parse_dates=['Datetime'])
    
    # Drop NaN values if present
    data = data.dropna()
    
    # Plot the Return Values over time in logarithmic form
    plt.figure(figsize=(10, 6))
    plt.plot(data['Datetime'], np.log(data['Return']), marker='o', linestyle='-', color='b')
    
    # Set plot labels and title
    plt.xlabel('Datetime')
    plt.ylabel('Return Value')
    plt.title('Energy Price Returns Over Time')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Example usage
plot_returns('../../data/clean_return_data.csv')
