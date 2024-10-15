import pandas as pd

# Load the processed data
data_path = 'data/processed_data.csv'
df = pd.read_csv(data_path, parse_dates=['Datetime'])

# Filter data for a specific hour (e.g., 15:00)
# This will narrow down the data to only that hour for each day
hour_to_predict = 15
df_hour = df[df['Datetime'].dt.hour == hour_to_predict].copy()

# Sort data by date to ensure proper chronological order
df_hour = df_hour.sort_values(by='Datetime')

# Let's consider a 7-day (weekly) window as an example
rolling_window = 7
rw = rolling_window
rw_str = str(rw)

# Calculate metrics considering the past few days or weeks
'''
1. Simple Moving Average (SMA)
    A basic moving average for energy prices (daily or hourly). We could compute SMAs for different window
    lengths (e.g., 5-day, 10-day, or 20-day) to smooth out price fluctuations.
2. Exponential Moving Average (EMA): Similar to SMA but gives more weight to recent prices. This could be 
    valuable to capture more immediate trends in energy prices.

3. Bollinger Bands
	Bands based on a moving average (SMA or EMA) that measure volatility. They will allow you to observe how
    energy prices behave within a “normal” range and identify periods of high volatility, which could be linked
    to demand/supply fluctuations.

4. Price Rate of Change (ROC)
	The ROC measures the percentage change between the current price and a price from a previous time period
    (e.g., 1 day, 7 days ago). It helps identify momentum in the market and is useful to highlight trends or
    sudden shifts in energy prices.

5. Relative Strength Index (RSI)
	A momentum indicator that can identify overbought or oversold conditions in the energy market. You can use
    it to track whether the price is approaching extreme values, which could indicate a reversal.
'''
# 1. Simple Moving Average (SMA)
df_hour['SMA_'+rw_str] = df_hour['MarginalES'].rolling(window=rw).mean()  # 7-day SMA

# 2. Exponential Moving Average (EMA)
df_hour['EMA_'+rw_str] = df_hour['MarginalES'].ewm(span=rw, adjust=False).mean()  # 7-day EMA

# 3. Bollinger Bands (7-day window, 2 standard deviations)
df_hour['BB_Middle'] = df_hour['MarginalES'].rolling(window=rw).mean()
df_hour['BB_Upper'] = df_hour['BB_Middle'] + 2 * df_hour['MarginalES'].rolling(window=rw).std()
df_hour['BB_Lower'] = df_hour['BB_Middle'] - 2 * df_hour['MarginalES'].rolling(window=rw).std()

# 4. Rate of Change (ROC, over 7 days)
df_hour['ROC_'+rw_str] = df_hour['MarginalES'].pct_change(periods=rw) * 100  # ROC over rolling_window days

# 5. Relative Strength Index (RSI, 7-day window)
delta = df_hour['MarginalES'].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=rw).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rw).mean()

rs = gain / loss
df_hour['RSI_'+rw_str] = 100 - (100 / (1 + rs))

# Drop missing values generated by rolling calculations
df_hour.dropna(inplace=True)

# Save the dataset with metrics for the selected hour
df_hour.to_csv(f'data/hour_{hour_to_predict}_metrics.csv', index=False)

# Model preparation (Example using simple Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the target (price) and features (the metrics)
X = df_hour[['SMA_'+rw_str, 'EMA_'+rw_str, 'BB_Middle', 'BB_Upper', 'BB_Lower', 'ROC_'+rw_str, 'RSI_'+rw_str]]
y = df_hour['MarginalES']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for hour {hour_to_predict}: {mse}")

# After evaluating the performance, you can expand this to other hours