'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/regression_tests/test_model.py
'''

# Model preparation (Example using simple Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the target (price) and features (the metrics)
X = df_hour[['SMA_'+rolling_window, 'EMA_'+rolling_window, 'BB_Middle', 'BB_Upper', 'BB_Lower', 'ROC_'+rolling_window, 'RSI_'+rolling_window]]
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
