import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from code.utils.matrix_builder import create_weight_matrix_with_features

# Load CSV
csv_hour_file = '/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/data/ta_metrics/new_price_metrics_hour_14.csv'
df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

# Decide window size for this run
window_size = 3

# Select features (excluding 'Datetime')
feature_columns = df.columns[1:]
df = df[['Datetime'] + list(feature_columns)]

# df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek  # Monday=0
df['month'] = df['Datetime'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# scaler = StandardScaler()
# columns_to_scale = df.columns[window_size-1:]
# print(f"Columns to scale: {columns_to_scale.tolist()}")
# df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Filter date range
train_start_date = '2018-12-25'
train_end_date = '2021-01-01'
train_subset_df = df[(df['Datetime'] >= train_start_date) & (df['Datetime'] <= train_end_date)]

# Create sliding window matrices
X, y = create_weight_matrix_with_features(train_subset_df, window_size)

# Split into training and testing sets
# Using shuffle=False since the data is a time series and needs to preserve chronology
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_test_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2: {r2:.4f}")

# Predict next value based on last test row
# X_last = X_test.iloc[-1].values.reshape(1, -1)
X_last = X_test.iloc[[-1]]  # double brackets preserve DataFrame structure with column names
y_pred_last = model.predict(X_last)[0]
y_actual_last = y_test.iloc[-1]

# Show prediction vs actual
print(f"\nLast test window input:\n{X_test.iloc[-1]}")
print(f"\nPredicted next price: {y_pred_last:.2f}")
print(f"Actual next price:    {y_actual_last:.2f}")
print(f"Prediction error:     {abs(y_pred_last - y_actual_last):.2f}")