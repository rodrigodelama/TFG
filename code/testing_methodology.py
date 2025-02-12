import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('data/hour_14_metrics.csv', parse_dates=['Datetime'])

# Date range for the subset
start_date = '2022-01-01'
end_date = '2023-02-01'
subset_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]

# Sliding window size
window_size = 14

# Function to create sliding windows of data
def create_sliding_window(data, window_size):
    X, y = [], []  # Initialize lists for input features (X) and target values (y)
    for i in range(len(data) - window_size):
        # Extract a window of size `w
        # 
        # indow_size` from the data
        X.append(data.iloc[i:i+window_size, 1:].values.flatten())  
        
        # The target is the value right after the current window
        y.append(data.iloc[i + window_size, 1])  
    
    # Convert the lists to DataFrame/Series for easier use in training
    return pd.DataFrame(X), pd.Series(y)

# Create sliding window matrix
X, y = create_sliding_window(subset_df, window_size)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate error metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print run results
print(f'Window Size: {window_size}')
print(f'Date Range: {start_date} to {end_date}')
print(f'Number of Observations: {len(subset_df)}')

# Print the error metrics
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Print the predictions alongside actual values for comparison
predictions = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print(predictions)

# Calculate the error for each prediction
errors = y - y_pred
print(errors)
