import pandas as pd
import numpy as np

# Load the processed data
data_path = 'data/return_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

df_cleaned = df.copy()  # Work on a copy
df_cleaned['Return'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned['Return'].interpolate(inplace=True)
# df_cleaned['Return'] = df_cleaned['Return'].clip(lower=-0.9)
df_cleaned.to_csv("data/clean_return_data.csv", index=False)
