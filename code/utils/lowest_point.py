'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-02-13
File: code/utils/lowest_point.py
'''

import pandas as pd

def find_zero_or_negative_prices(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['Datetime'])  # Read CSV and parse dates
    df_invalid = df[df['MarginalES'] <= 0]  # Filter rows where MarginalES is 0 or negative
    
    if df_invalid.empty:
        print("No zero or negative values found.")
    else:
        print(f"Found {len(df_invalid)} zero or negative values:")
        print(df_invalid)

        # Find the lowest MarginalES value
        min_value = df_invalid['MarginalES'].min()
        min_entry = df_invalid[df_invalid['MarginalES'] == min_value]

        print("\nLowest MarginalES value found:")
        print(min_entry)

# Example usage
csv_file = "../../data/hour_14_metrics.csv"
# csv_file = '../../data/clean_return_data.csv'
find_zero_or_negative_prices(csv_file)
