import pandas as pd

# Load the processed data
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

# Calculate TA metrics
# 1. Simple Moving Averages (SMA)
df['SMA_10'] = df['MarginalES'].rolling(window=10).mean()  # 10-hour SMA
df['SMA_20'] = df['MarginalES'].rolling(window=20).mean()  # 20-hour SMA

# 2. Exponential Moving Average (EMA)
df['EMA_10'] = df['MarginalES'].ewm(span=10, adjust=False).mean()  # 10-hour EMA
df['EMA_20'] = df['MarginalES'].ewm(span=20, adjust=False).mean()  # 20-hour EMA

# 3. Bollinger Bands (20-hour window, 2 standard deviations)
df['BB_Middle'] = df['MarginalES'].rolling(window=20).mean()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['MarginalES'].rolling(window=20).std()
df['BB_Lower'] = df['BB_Middle'] - 2 * df['MarginalES'].rolling(window=20).std()

# 4. Rate of Change (ROC, over 12 hours)
df['ROC'] = df['MarginalES'].pct_change(periods=12) * 100  # Rate of change over 12 hours

# 5. Relative Strength Index (RSI, 14-hour window)
window_length = 14
delta = df['MarginalES'].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Handle missing values generated during the rolling calculations
df.fillna(0, inplace=True)

# Save the dataset with TA metrics to a new CSV file
df.to_csv('data/processed_data_with_TA_metrics.csv', index=False)

# Final output message
print("Technical Analysis metrics have been calculated and saved to 'processed_data_with_TA_metrics.csv'.")