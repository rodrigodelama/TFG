'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/data_processing/db_cleanup_weekends.py
'''

import pandas as pd

# Load the processed data
data_path = '../../data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime']) # df means DataFrame

# Filter data for a specific hour (e.g., 14:00) to a new DataFrame (df_hour)
hour_to_predict = 14
df_hour = df[df['Datetime'].dt.hour == hour_to_predict].copy() # df_hour means DataFrame for the selected hour

# Filter data removing weekends
df_hour = df_hour[df_hour['Datetime'].dt.dayofweek < 5]

# Save the weekend-free data
df_hour.to_csv('../../data/processed_data_no_weekends.csv', index=False)
