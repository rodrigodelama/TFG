'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/data_processing/2_db_14_and_return_builder.py
'''

import pandas as pd

# Load the original dataset
data_path = '../../data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

# Define the hour to predict
hour_to_predict = 14

# Filter data for a specific hour (e.g., 14:00) to a new DataFrame (df_hour)
df_hour = df[df['Datetime'].dt.hour == hour_to_predict].copy()

df_hour.to_csv("../../data/hour_14_metrics.csv", index=False)
