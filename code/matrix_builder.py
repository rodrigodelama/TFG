'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-24

File: code/utils/matrix_builder.py
'''

import pandas as pd
import numpy as np

def create_feature_matrix(data, lag_price_window):
    """
    Creates a feature matrix where each row is a sliding window of prices,
    and a corresponding target vector containing the next price value.
    
    Parameters:
    ----------
    - data: DataFrame with 'Datetime' and 'MarginalES' columns
    - lag_price_window: Size of the prices sliding window
    
    Returns:
    -------
    - X: DataFrame with sliding windows as rows
    - y: Series with target values (next price after each window)
    """
    # Extract the MarginalES column
    if 'MarginalES' in data.columns:
        prices = data['MarginalES'].values
    else:
        # Assume it's the second column (index 1)
        prices = data.iloc[:, 1].values
    
    # Create empty matrices
    X = np.zeros((len(prices) - lag_price_window, lag_price_window))
    y = np.zeros(len(prices) - lag_price_window)
    
    # Fill the matrices with the sliding windows and targets
    for i in range(len(prices) - lag_price_window):
        X[i, :] = prices[i:i+lag_price_window]  # Window of prices
        y[i] = prices[i+lag_price_window]       # Next price after window
    
    # Convert to DataFrame/Series for easier use in training
    return pd.DataFrame(X), pd.Series(y)

def create_expanded_feature_matrix(dataframe, lag_price_window, debug=False):
    """
    Creates a sliding window dataset for time series forecasting where each row contains:
    1. A window of historical prices (right-aligned)
    2. Feature values from the most recent point in the window
    3. Target value (next price after the window)
    
    Parameters:
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing at minimum 'Datetime' and 'MarginalES' columns, 
        plus any additional feature columns
    lag_price_window : int
        Number of data points to include in each window of prices
        
    Returns:
    -------
    X : pandas.DataFrame
        Features DataFrame with:
        - Historical prices labeled as 'price_t-n' through 'price_t-1'
        - All additional features from the original dataframe
    y : pandas.Series
        Target values (price at time t)
    """

    # Input validation
    if lag_price_window < 1:
        raise ValueError("Feature window size must be at least 1")
    if len(dataframe) <= lag_price_window:
        raise ValueError(f"DataFrame must have more rows ({len(dataframe)}) than lag_price_window ({lag_price_window})")
    if 'Datetime' not in dataframe.columns or 'MarginalES' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'Datetime' and 'MarginalES' columns")

    X, y = [], []

    # Extract price data and features
    df_prices = dataframe[['Datetime', 'MarginalES']]
    df_features = dataframe.iloc[:, 2:] # Exclude 'Datetime' and 'MarginalES'
    feature_names = df_features.columns.tolist()

    if debug:
        print(f"Feature columns identified: {feature_names}")

    # Create samples from the data
    for i in range(lag_price_window, len(df_prices)):
        # Extract the window for prices as features (right-aligned)
        window = df_prices.iloc[i-lag_price_window:i, 1].values.flatten()

        # Extract corresponding feature row (from the most recent point in the window)
        feature_row = df_features.iloc[i-1].values.flatten()

        # Concatenate window prices with feature row
        X.append(np.concatenate((window, feature_row)))
        y.append(df_prices.iloc[i, 1])  # Predict current price

    # Return DataFrame and Series with proper column names
    price_columns = [f'price_t-{lag_price_window-i}' for i in range(lag_price_window)]
    X_df = pd.DataFrame(X, columns=price_columns + feature_names)
    y_series = pd.Series(y, name='price_t')

    if debug:
        print(f"X DataFrame shape: {X_df.shape}")
        print(f"Sample size: {len(X_df)}")

    return X_df, y_series
