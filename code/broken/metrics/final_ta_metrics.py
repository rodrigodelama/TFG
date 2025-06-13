'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-06-06
File: code/metrics/final_ta_metrics.py
'''

import pandas as pd
import ta  # bukosabino's Technical Analysis Library
import numpy as np

# Load the data
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

# Reduce dataset to only to the 14th hour
hour_to_predict = 14
df = df[df['Datetime'].dt.hour == hour_to_predict].copy() # df_hour means DataFrame for the selected hour

'''
Trend-following:
- Simple Moving Average (SMA) (shorter window)
    A shorter SMA could be useful to identify short-term trends in energy prices. For example, a 5-day or 10-day
    SMA could help capture recent price movements

- Simple Moving Average (longer window)
    A longer SMA could be useful to capture long-term trends in energy prices. For example, a 50-day or 200-day
    SMA could help identify the overall direction of the market

- Exponential Moving Average (EMA): Similar to SMA but gives more weight to recent prices. This could be valuable
    to capture more immediate trends in energy prices

Momentum:
- Price Rate of Change (ROC)
	The ROC measures the percentage change between the current price and a price from a previous time period
    (e.g., 1 day, 7 days ago). It helps identify momentum in the market and is useful to highlight trends or
    sudden shifts in energy prices

- Relative Strength Index (RSI)
	A momentum indicator that can identify overbought or oversold conditions in the energy market. You can use
    it to track whether the price is approaching extreme values, which could indicate a reversal

    Is RSI really relevant ??
    In energy there is no such thing as overbought or oversold conditions, so idk if its actually useful

Volatility:
- Bollinger Bands Width (BB Width)
    A volatility indicator that measures the width of the Bollinger Bands. It can help identify periods of
    high or low volatility in energy prices
- Average True Range (ATR)
    A volatility indicator that measures the average range between the high and low prices over a specified period.
'''

# The "window" parameter is measured in days

# Trend-following - SMA and EMA
# Simple Moving Average
df[f'SMA_{3}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=3).sma_indicator()
df[f'SMA_{5}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=5).sma_indicator()
df[f'SMA_{7}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=7).sma_indicator() # week trend
df[f'SMA_{14}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=14).sma_indicator()
df[f'SMA_{30}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=30).sma_indicator() # month trend
df[f'SMA_{90}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=90).sma_indicator()
df[f'SMA_{180}'] = ta.trend.SMAIndicator(close=df['MarginalES'], window=180).sma_indicator() # half year trend

# Exponential Moving Average
df[f'EMA_{3}'] = ta.trend.EMAIndicator(close=df['MarginalES'], window=3).ema_indicator()
df[f'EMA_{5}'] = ta.trend.EMAIndicator(close=df['MarginalES'], window=5).ema_indicator()
df[f'EMA_{7}'] = ta.trend.EMAIndicator(close=df['MarginalES'], window=7).ema_indicator()
df[f'EMA_{14}'] = ta.trend.EMAIndicator(close=df['MarginalES'], window=14).ema_indicator()
df[f'EMA_{30}'] = ta.trend.EMAIndicator(close=df['MarginalES'], window=30).ema_indicator()

# Momentum - Rate of Change (ROC) and RSI
# Rate of Change
df[f'ROC_{3}'] = ta.momentum.ROCIndicator(close=df['MarginalES'], window=3).roc()
df[f'ROC_{5}'] = ta.momentum.ROCIndicator(close=df['MarginalES'], window=5).roc()
df[f'ROC_{7}'] = ta.momentum.ROCIndicator(close=df['MarginalES'], window=7).roc()
df[f'ROC_{14}'] = ta.momentum.ROCIndicator(close=df['MarginalES'], window=14).roc()
df[f'ROC_{30}'] = ta.momentum.ROCIndicator(close=df['MarginalES'], window=30).roc()

# Relative Strength Index
df[f'RSI_{5}'] = ta.momentum.RSIIndicator(close=df['MarginalES'], window=5).rsi() # rapid momentum changes
df[f'RSI_{7}'] = ta.momentum.RSIIndicator(close=df['MarginalES'], window=7).rsi()
df[f'RSI_{14}'] = ta.momentum.RSIIndicator(close=df['MarginalES'], window=14).rsi() # biweekly momentum
df[f'RSI_{30}'] = ta.momentum.RSIIndicator(close=df['MarginalES'], window=30).rsi()

# Volatility - Bollinger Bands Width and Standard Deviation
# BB Width (Bollinger Bands Width)
df['BB_Width_7'] = ta.volatility.BollingerBands(close=df['MarginalES'], window=7, window_dev=2).bollinger_wband()
df['BB_Width_14'] = ta.volatility.BollingerBands(close=df['MarginalES'], window=14, window_dev=2).bollinger_wband()

# Standard Deviation (STD)
df['STD_7'] = df['MarginalES'].rolling(window=7).std()
df['STD_14'] = df['MarginalES'].rolling(window=14).std()
df['STD_30'] = df['MarginalES'].rolling(window=30).std()
df['STD_90'] = df['MarginalES'].rolling(window=90).std()

# Additional contextual features
df['month'] = df['Datetime'].dt.month
df['day_of_week'] = df['Datetime'].dt.dayofweek  # Monday=0
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Drop missing values generated by rolling calculations
df.dropna(inplace=True)

# Check for NaN or infinity values (specially in ROC)
print("NaN values per column:\n", df.isna().sum())
print("Infinity values per column:\n", df.isin([np.inf, -np.inf]).sum())

# Drop NaN or infinity values and interpolate
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.interpolate()

# Save the dataset with metrics
df.to_csv(f'data/ta_metrics/final_price_ta_metrics.csv', index=False)

# Output message
print(f"Metrics calculated and saved")
