import pandas as pd

# Load the original dataset
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

# Define the hour to predict
hour_to_predict = 14

# Filter data for a specific hour (e.g., 14:00) to a new DataFrame (df_hour)
df_hour = df[df['Datetime'].dt.hour == hour_to_predict].copy()

df_hour.to_csv("data/hour_14_metrics.csv", index=False)

# Calculate the return values as the percentage change between consecutive rows
df_hour['Return'] = df_hour['MarginalES'].pct_change()

# # Drop the first row since its return value will be NaN
# df_hour = df_hour.dropna()

# Keep only the Datetime and Return columns
df_hour = df_hour[['Datetime', 'Return']]

# Change the Return column name to the original MarginalES column name
df_hour = df_hour.rename(columns={'Return': 'MarginalES'})

# Save the new dataset to a CSV file
df_hour.to_csv("data/return_data.csv", index=False)
