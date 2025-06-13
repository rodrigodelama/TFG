import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from code.utils.matrix_builder import create_weight_matrix_with_features

# Load CSV
csv_hour_file = '/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/data/ta_metrics/new_price_metrics_hour_14.csv'
df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

# Select features (excluding 'Datetime')
feature_columns = df.columns[1:]
df = df[['Datetime'] + list(feature_columns)]

# Filter date range
train_start_date = '2018-12-25'
train_end_date = '2021-01-01'
train_subset_df = df[(df['Datetime'] >= train_start_date) & (df['Datetime'] <= train_end_date)]

# Create sliding window matrices
window_size = 3
X_0606, y_0606 = create_weight_matrix_with_features(train_subset_df, window_size)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_0606, y_0606, test_size=0.5, shuffle=False)

# Train Lasso model (adjust alpha as needed)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Predict on test set
y_test_pred = lasso_model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2: {r2:.4f}")

# Predict next value based on last test row
X_last = X_test.iloc[[-1]]
y_pred_last = lasso_model.predict(X_last)[0]
y_actual_last = y_test.iloc[-1]

# Show prediction vs actual
print(f"\nLast test window input:\n{X_test.iloc[-1]}")
print(f"\nPredicted next price: {y_pred_last:.2f}")
print(f"Actual next price:    {y_actual_last:.2f}")
print(f"Prediction error:     {abs(y_pred_last - y_actual_last):.2f}")