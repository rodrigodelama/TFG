import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.matrix_builder import create_ta_feature_matrix

# Load CSV
csv_hour_file = '../data/ta_metrics/final_price_ta_metrics.csv'

df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

# Define the sliding window size for this run
sliding_window = 10

# Decide price features window size for this run
lag_price_window = 3

# compute number of sliding window models to train in the dataset
num_sliding_windows = len(df) - sliding_window
print(f"Number of rows in the dataset: {len(df)}")
print(f"Number of sliding windows to train: {num_sliding_windows}")

# Display the first few rows of the DataFrame
print(f"Database:\n{df.head()}")

# initial_date = '2018-06-29 14:00:00'
# initial_date = pd.to_datetime(initial_date)

# Create Training and Test loop
# for i in range(3):
for i in range(num_sliding_windows):
    # Note: This is a sliding window approach, where each iteration uses the next row as the test set

    # Ensure we do not exceed the DataFrame length
    if i + sliding_window >= len(df):
        break  # Avoid index out of bounds

    # train_end_date = df.iloc[i + sliding_window - 1]['Datetime']
    # print(f"Training end date: {train_end_date}")

    # We will take a window of 'sliding_window' rows for training and predict on the next row

    # Create training set
    train = df.iloc[i:i + sliding_window]
    test = df.iloc[i + sliding_window:i + sliding_window + 1]
    print(f"Training on:\n{train}\nTesting on:\n{test}\n")

    # Create feature matrix and target variable
    X_train, y_train = create_ta_feature_matrix(train, lag_price_window)

    print(f"X_train:\n{X_train.head()}")
    print(f"y_train:\n{y_train.head()}")

    # Determine the training and testing sets

    # Fit the Lasso model
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")

    # Predict on the test set
    X_test, y_test = create_ta_feature_matrix(test, lag_price_window)
    print(f"X_test:\n{X_test.head()}")
    print(f"y_test:\n{y_test.head()}")

    print("Predicting on the test set...")
    y_pred = model.predict(X_test)
    print(f"y_pred:\n{y_pred[:5]}")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")
    print("Evaluation metrics calculated.\n")

    # Print the results
    print(f"Test set predictions:\n{y_pred}")
    print(f"Actual values:\n{y_test.values}\n")
