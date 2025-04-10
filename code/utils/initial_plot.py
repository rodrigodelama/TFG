'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/utils/initial_plot.py
'''

import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, parse_dates=['Datetime'])
    
    # Plot the 'MarginalES' column over time
    plt.figure(figsize=(10, 6))
    plt.plot(data['Datetime'], data['MarginalES'], marker='o', linestyle='-')
    
    # Set plot labels and title
    plt.xlabel('Datetime')
    plt.ylabel('Price €/MWh (MarginalES)')
    plt.title('Energy Prices Over Time')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Example usage
# plot_csv('../../data/processed_data.csv')

plot_csv('../../data/hour_14_metrics.csv')
