'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-08
File: code/utils/graphs_test.py
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def create_weight_matrix(data, window_size, target_col='MarginalES'):
    """
    Creates a feature matrix X with sliding windows of the target column and all other features,
    and a corresponding target vector y containing the next price value.
    
    Parameters:
    - data: DataFrame with 'Datetime' and multiple feature columns
    - window_size: Size of the sliding window
    - target_col: Name of the column to predict (default: 'MarginalES')
    
    Returns:
    - X: DataFrame with sliding windows features
    - y: Series with target values (next price after each window)
    """
    # Remove datetime column if it exists
    if 'Datetime' in data.columns:
        data = data.drop('Datetime', axis=1)
    
    # Get the number of features (columns)
    n_features = data.shape[1]
    
    # Initialize empty lists
    X_data = []
    y_data = []
    
    # For each possible window start position
    for i in range(len(data) - window_size):
        # Get the window of data
        window = data.iloc[i:i+window_size]
        
        # Flatten the window into a single row
        row = window.values.flatten()
        
        # Add to X
        X_data.append(row)
        
        # Get the target value (next value of target column after window)
        y_data.append(data[target_col].iloc[i + window_size])
    
    # Convert to DataFrame and Series
    X_df = pd.DataFrame(X_data)
    y_series = pd.Series(y_data)
    
    return X_df, y_series


# Read DB w features
# csv_db_file = '../data/hour_14_metrics.csv'
csv_db_file = '../../data/ta_metrics/0401_price_metrics_hour_14.csv'
df_sw = pd.read_csv(csv_db_file, parse_dates=['Datetime'])
df_sw = df_sw[['Datetime', 'MarginalES']]

# Date range
start_date = '2018-01-01'
end_date = '2025-03-18'

subset_df_features = df_sw[(df_sw['Datetime'] >= start_date) & (df_sw['Datetime'] <= end_date)]


###  Long Stationarity supposition (lots of rows)
stationarity_depth = 3

window_size = 3

# Create sliding window weight matrix
X_subset_to_trim, y_subset_to_trim = create_weight_matrix(subset_df_features, window_size)

print("____________ X_subset_to_trim")
print(X_subset_to_trim)
print("____________ END X_subset_to_trim")

# Initialize y_pred with same index as y_subset_to_trim
y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

for i in range(y_subset_to_trim.size - stationarity_depth): # to avoid going out of bounds
    # Make subsets for training of the specified depth
    X_train = X_subset_to_trim.iloc[i:i + stationarity_depth]
    y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the NEXT point after the training window
    X_predict = X_subset_to_trim.iloc[i + stationarity_depth].values.reshape(1, -1)
    y_predict = model.predict(X_predict)[0]

    # Save the predicted variable
    y_pred.iloc[i + stationarity_depth] = y_predict

# Compare y_subset_to_trim with y_pred in the available indexes
print("Actual vs Predicted:")
print(pd.DataFrame({'Actual': y_subset_to_trim, 'Predicted': y_pred}))

#calculate error.
valid_pred = y_pred.dropna()
valid_actual = y_subset_to_trim[valid_pred.index]
error = valid_actual - valid_pred
print("\nError:")
print(error)



# Plotting the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(valid_actual.index, valid_actual.values, label='Actual Values', marker='.')
plt.plot(valid_pred.index, valid_pred.values, label='Predicted Values', marker='.')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values w Features - Stationarity 3 days')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the error
error = valid_actual - valid_pred
plt.figure(figsize=(12, 6))
plt.plot(error.index, error.values, label='Error', marker='.')
plt.xlabel('Index')
plt.ylabel('Error')
plt.title('Error (Actual - Predicted)')
plt.grid(True)
plt.show()
