'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

File: new_ta_metrics.py
'''

import pandas as pd
import ta  # Technical Analysis Library
import numpy as np

# Load the data
    # the regular price data
data_path = 'data/processed_data.csv'

# Second run
    # the return data
# data_path = 'data/clean_return_data.csv'

# Retrieve and filter data for a specific hour (e.g., 14:00) to a new DataFrame (df_hour)
df = pd.read_csv(data_path, parse_dates=['Datetime'])
hour_to_predict = 14
df_hour = df[df['Datetime'].dt.hour == hour_to_predict].copy()

'''
- Simple Moving Average (SMA) (shorter window)
    A shorter SMA could be useful to identify short-term trends in energy prices. For example, a 5-day or 10-day
    SMA could help capture recent price movements

- Simple Moving Average (longer window)
    A longer SMA could be useful to capture long-term trends in energy prices. For example, a 50-day or 200-day
    SMA could help identify the overall direction of the market

- Exponential Moving Average (EMA): Similar to SMA but gives more weight to recent prices. This could be valuable
    to capture more immediate trends in energy prices

- Volatility ?
- Momentum ?

- Price Rate of Change (ROC)
	The ROC measures the percentage change between the current price and a price from a previous time period
    (e.g., 1 day, 7 days ago). It helps identify momentum in the market and is useful to highlight trends or
    sudden shifts in energy prices


- Relative Strength Index (RSI)
	A momentum indicator that can identify overbought or oversold conditions in the energy market. You can use
    it to track whether the price is approaching extreme values, which could indicate a reversal

    Is RSI really relevant ??
    In energy there is no such thing as overbought or oversold conditions, so idk if its actually useful
'''

# Test with various rolling windows

# Simple Moving Average (SMA) (shorter window)
df_hour[f'SMA_{3}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=3).sma_indicator()
df_hour[f'SMA_{5}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=5).sma_indicator()
df_hour[f'SMA_{7}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=7).sma_indicator()

# Simple Moving Average (longer window)
df_hour[f'SMA_{10}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=10).sma_indicator() # short term in trading
df_hour[f'SMA_{30}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=30).sma_indicator()
df_hour[f'SMA_{50}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=50).sma_indicator() # medium term in trading
df_hour[f'SMA_{60}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=60).sma_indicator()
df_hour[f'SMA_{100}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=100).sma_indicator()
df_hour[f'SMA_{200}'] = ta.trend.SMAIndicator(close=df_hour['MarginalES'], window=200).sma_indicator() # long term in trading

# Exponential Moving Average (EMA) (Rolling Window set as typically done in trading) - da mas peso a los precios recientes
df_hour[f'EMA_{12}'] = ta.trend.EMAIndicator(close=df_hour['MarginalES'], window=12).ema_indicator()
df_hour[f'EMA_{26}'] = ta.trend.EMAIndicator(close=df_hour['MarginalES'], window=26).ema_indicator()

# Rate of Change (ROC, over N days)
df_hour[f'ROC_{12}'] = ta.momentum.ROCIndicator(close=df_hour['MarginalES'], window=12).roc() # as done in trading for momentum
df_hour[f'ROC_{50}'] = ta.momentum.ROCIndicator(close=df_hour['MarginalES'], window=50).roc()

#! Review if this is useful
# Relative Strength Index (RSI, N-day window)
df_hour[f'RSI_{5}'] = ta.momentum.RSIIndicator(close=df_hour['MarginalES'], window=5).rsi() # rapid momentum changes
df_hour[f'RSI_{7}'] = ta.momentum.RSIIndicator(close=df_hour['MarginalES'], window=7).rsi() # rapid momentum changes
df_hour[f'RSI_{14}'] = ta.momentum.RSIIndicator(close=df_hour['MarginalES'], window=14).rsi() # standard momentum

# # Return
# # Simple Moving Average (SMA) (shorter window)
# df_hour[f'SMA_{3}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=3).sma_indicator()
# df_hour[f'SMA_{5}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=5).sma_indicator()
# df_hour[f'SMA_{7}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=7).sma_indicator()

# # Simple Moving Average (longer window)
# df_hour[f'SMA_{10}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=10).sma_indicator() # short term in trading
# df_hour[f'SMA_{30}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=30).sma_indicator()
# df_hour[f'SMA_{50}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=50).sma_indicator() # medium term in trading
# df_hour[f'SMA_{60}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=60).sma_indicator()
# df_hour[f'SMA_{100}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=100).sma_indicator()
# df_hour[f'SMA_{200}'] = ta.trend.SMAIndicator(close=df_hour['Return'], window=200).sma_indicator() # long term in trading

# # Exponential Moving Average (EMA) (Rolling Window set as typically done in trading) - da mas peso a los precios recientes
# df_hour[f'EMA_{12}'] = ta.trend.EMAIndicator(close=df_hour['Return'], window=12).ema_indicator()
# df_hour[f'EMA_{26}'] = ta.trend.EMAIndicator(close=df_hour['Return'], window=26).ema_indicator()

# # Rate of Change (ROC, over N days)
# df_hour[f'ROC_{12}'] = ta.momentum.ROCIndicator(close=df_hour['Return'], window=12).roc() # as done in trading for momentum
# df_hour[f'ROC_{50}'] = ta.momentum.ROCIndicator(close=df_hour['Return'], window=50).roc()

# #! Review if this is useful
# # Relative Strength Index (RSI, N-day window)
# df_hour[f'RSI_{5}'] = ta.momentum.RSIIndicator(close=df_hour['Return'], window=5).rsi() # rapid momentum changes
# df_hour[f'RSI_{7}'] = ta.momentum.RSIIndicator(close=df_hour['Return'], window=7).rsi() # rapid momentum changes
# df_hour[f'RSI_{14}'] = ta.momentum.RSIIndicator(close=df_hour['Return'], window=14).rsi() # standard momentum


# Drop missing values generated by rolling calculations
df_hour.dropna(inplace=True)

# Check for NaN or infinity values
print("NaN values per column:\n", df_hour.isna().sum())
print("Infinity values per column:\n", df_hour.isin([np.inf, -np.inf]).sum())

# Drop NaN or infinity values
df_hour = df_hour.replace([np.inf, -np.inf], np.nan).dropna()
# Interpolate missing values
df_hour = df_hour.interpolate()

# Check for NaN or infinity values
print("NaN values per column:\n", df_hour.isna().sum())
print("Infinity values per column:\n", df_hour.isin([np.inf, -np.inf]).sum())


# Save the dataset with metrics for the selected hour
df_hour.to_csv(f'data/ta_metrics/new_price_metrics_hour_{hour_to_predict}.csv', index=False)
# df_hour.to_csv(f'data/ta_metrics/new_return_metrics_hour_{hour_to_predict}.csv', index=False)

# Output message
print(f"Metrics for hour {hour_to_predict} calculated and saved to 'data/ta_metrics/new_metrics_hour_{hour_to_predict}.csv'.")
