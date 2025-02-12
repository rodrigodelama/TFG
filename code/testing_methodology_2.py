import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/hour_14_metrics.csv', parse_dates=['Datetime'])

# Date range for the subset
start_date = '2018-01-01'
end_date = '2023-01-01'
subset_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]

# Sliding window size
window_size = 30

# Function to create sliding windows of data
def create_sliding_window(data, window_size):
    X, y = [], []  # Initialize lists for input features (X) and target values (y)
    for i in range(len(data) - window_size):
        # Extract a window of size `window_size` from the data
        X.append(data.iloc[i:i+window_size, 1:].values.flatten())  
        
        # The target is the value right after the current window
        y.append(data.iloc[i + window_size, 1])  
    
    # Convert the lists to DataFrame/Series for easier use in training
    return pd.DataFrame(X), pd.Series(y)

# Create sliding window matrix
X, y = create_sliding_window(subset_df, window_size)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
train_ratio = 0.8
train_index = int(train_ratio * len(X_scaled))

X_train, X_test = X_scaled[:train_index], X_scaled[train_index:]
y_train, y_test = y[:train_index], y[train_index:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on the training set
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Evaluate on the test set
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print run results
print(f'Window Size: {window_size}')
print(f'Date Range: {start_date} to {end_date}')
print(f'Number of Observations: {len(subset_df)}')

# Print the error metrics
print(f"Training MSE: {mse_train}, R² Score: {r2_train}")
print(f"Test MSE: {mse_test}, R² Score: {r2_test}")

# Baseline comparison (Persistence model: predicts the last known value)
baseline_pred = X.iloc[:, -1].iloc[train_index:]  # Last known value in the window
mse_baseline = mean_squared_error(y_test, baseline_pred)
print(f"Baseline MSE: {mse_baseline}")

# Visualize residuals for the test set
errors = y_test - y_pred_test
plt.figure(figsize=(10, 5))
plt.scatter(range(len(errors)), errors, alpha=0.7, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
plt.title('Residuals of Predictions')
plt.xlabel('Test Observation Index')
plt.ylabel('Prediction Error')
plt.legend()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred_test, label='Predicted', color='orange', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Test Observation Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()
