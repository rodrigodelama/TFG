'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-24
File: code/utils/matrix_builder.py
'''

# import os
# import logging

# if os.getenv('TFG_DEBUG', '1'):
# if os.getenv('TFG_DEBUG') == '1':
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger(__name__)
#     logger.debug("Debugging is enabled. [Sliding Window]")

# Original function
# def create_feature_matrix(data, window_size):
#     X, y = [], []  # Initialize lists for input features (X) and target values (y)
    
#     for i in range(len(data) - window_size):
#         # Extract a window of size 'window_size' from the data
#         X.append(data.iloc[i:i+window_size, 1:].values.flatten())  
        
#         # The label is the value right after the current window
#         y.append(data.iloc[i + window_size, 1])  
    
#     # Convert the lists to DataFrame/Series for easier use in training
#     return pd.DataFrame(X), pd.Series(y)

# Updated for mini models
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



# Function to create the feature matrix, with the sliding window, and the columns of extra data points
# Right Alligned
# x1 x2 x3 sma_3
# sma_3 = x1+x2+x3 / 3
# Original function
# def create_sliding_window_more_columns(dataframe, window_size):
#     X, y = [], []
    
#     df_prices = dataframe[['Datetime', 'MarginalES']]
#     df_features = dataframe.iloc[:, 2:]  # Exclude 'Datetime' and 'MarginalES'
    
#     for i in range(window_size, len(df_prices)):
#         # Extract sliding window for prices (right-aligned)
#         window = df_prices.iloc[i-window_size:i, 1:].values.flatten()
        
#         # Extract corresponding feature row
#         feature_row = df_features.iloc[i-1].values.flatten()
        
#         # Concatenate sliding window prices with feature row
#         X.append(np.concatenate((window, feature_row)))
#         y.append(df_prices.iloc[i, 1])
    
#     return pd.DataFrame(X), pd.Series(y)

# # Original function improved
# def create_sliding_window_with_features(dataframe, window_size):
#     """
#     Creates a sliding window dataset for time series forecasting where each row contains:
#     1. A window of historical prices (right-aligned)
#     2. Feature values from the most recent point in the window
#     3. Target value (next price after the window)
    
#     Parameters:
#     ----------
#     dataframe : pandas.DataFrame
#         DataFrame containing at minimum 'Datetime' and 'MarginalES' columns, 
#         plus any additional feature columns
#     window_size : int
#         Number of historical price points to include in each window
        
#     Returns:
#     -------
#     X : pandas.DataFrame
#         Features DataFrame with:
#         - Historical prices labeled as 'price_t-n' through 'price_t-1'
#         - All additional features from the original dataframe
#     y : pandas.Series
#         Target values (price at time t)
        
#     Example:
#     -------
#     X, y = create_sliding_window_with_features(price_data, window_size=24)
#     """
#     # Input validation
#     if window_size < 1:
#         raise ValueError("Window size must be at least 1")
#     if len(dataframe) <= window_size:
#         raise ValueError(f"DataFrame must have more rows ({len(dataframe)}) than window_size ({window_size})")
#     if 'Datetime' not in dataframe.columns or 'MarginalES' not in dataframe.columns:
#         raise ValueError("DataFrame must contain 'Datetime' and 'MarginalES' columns")
    
#     X, y = [], []
    
#     # Extract price data and features
#     df_prices = dataframe[['Datetime', 'MarginalES']]
#     df_features = dataframe.iloc[:, 2:]  # Exclude 'Datetime' and 'MarginalES'
#     feature_names = df_features.columns.tolist()
    
#     for i in range(window_size, len(df_prices)):
#         # Extract sliding window for prices (right-aligned)
#         window = df_prices.iloc[i-window_size:i, 1:].values.flatten()
        
#         # Extract corresponding feature row (from the most recent point in the window)
#         feature_row = df_features.iloc[i-1].values.flatten()
        
#         # Concatenate sliding window prices with feature row
#         X.append(np.concatenate((window, feature_row)))
#         y.append(df_prices.iloc[i, 1])  # Next price point as target
    
#     # Create column names for the price window
#     price_columns = [f'price_t-{window_size-i}' for i in range(window_size)]
    
#     # Return DataFrame with proper column names
#     X_df = pd.DataFrame(X, columns=price_columns + feature_names)
#     y_series = pd.Series(y, name='price_t')
    
#     return X_df, y_series


#! REVIEW

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


# FROM LATEX
"""
import pandas as pd
import numpy as np

def create_feature_matrix_with_features(dataframe, window_size, debug=False):
    ""
    Creates a sliding window dataset for time series forecasting where each row contains:
    1. A window of historical prices (right-aligned)
    2. Feature values from the most recent point in the window
    3. Target value (next price after the window)
    
    Parameters:
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing at minimum 'Datetime' and 'MarginalES' columns, 
        plus any additional feature columns
    window_size : int
        Number of historical price points to include in each window
    debug : bool, optional
        If True, print debugging information instead of using logger
        
    Returns:
    -------
    X : pandas.DataFrame
        Features DataFrame with:
        - Historical prices labeled as 'price_t-n' through 'price_t-1'
        - All additional features from the original dataframe
    y : pandas.Series
        Target values (price at time t)
    ""
    
    # Input validation
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    if len(dataframe) <= window_size:
        raise ValueError(f"DataFrame must have more rows ({len(dataframe)}) than window_size ({window_size})")
    if 'Datetime' not in dataframe.columns or 'MarginalES' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'Datetime' and 'MarginalES' columns")
    
    X, y = [], []
    
    # Extract price data and features
    df_prices = dataframe[['Datetime', 'MarginalES']]
    df_features = dataframe.iloc[:, 2:]  # Exclude 'Datetime' and 'MarginalES'
    feature_names = df_features.columns.tolist()
    
    if debug:
        print(f"Feature columns identified: {feature_names}")
    
    # Create samples from the data
    for i in range(window_size, len(df_prices)):
        # Extract sliding window for prices (right-aligned)
        window = df_prices.iloc[i-window_size:i, 1:].values.flatten()
        
        # Extract corresponding feature row (from the most recent point in the window)
        feature_row = df_features.iloc[i-1].values.flatten()
        
        # Concatenate sliding window prices with feature row
        X.append(np.concatenate((window, feature_row)))
        y.append(df_prices.iloc[i, 1])  # Next price point as target
    
    # Create column names for the price window
    price_columns = [f'price_t-{window_size-i}' for i in range(window_size)]
    
    # Return DataFrame with proper column names
    X_df = pd.DataFrame(X, columns=price_columns + feature_names)
    y_series = pd.Series(y, name='price_t')
    
    if debug:
        print(f"X DataFrame shape: {X_df.shape}")
        print(f"Feature columns count: {len(feature_names)}")
        print(f"Price window columns: {price_columns}")
        print(f"Total columns: {len(X_df.columns)}")
        print(f"Sample size: {len(X_df)}")
    
    return X_df, y_series
"""

# def create_ta_feature_matrix(dataframe, lag_price_window, debug=False):
#     """
#     Creates a sliding window dataset for time series forecasting where each row contains:
#     1. A window of historical prices (right-aligned)
#     2. Feature values from the most recent point in the window
#     3. Target value (next price after the window)
    
#     Parameters:
#     ----------
#     dataframe : pandas.DataFrame
#         DataFrame containing at minimum 'Datetime' and 'MarginalES' columns, 
#         plus any additional feature columns
#     lag_price_window : int
#         Number of historical price points to include in each window
#     debug : bool, optional
#         If True, print debugging information instead of using logger
        
#     Returns:
#     -------
#     X : pandas.DataFrame
#         Features DataFrame with:
#         - Historical prices labeled as 'price_t-n' through 'price_t-1'
#         - All additional features from the original dataframe
#     y : pandas.Series
#         Target values (price at time t)
        
#     Example:
#     -------
#     X, y = create_sliding_window_with_features(price_data, lag_price_window=24)
#     """
#     # Input validation
#     if lag_price_window < 1:
#         raise ValueError("Feature window size must be at least 1")
#     if len(dataframe) <= lag_price_window:
#         raise ValueError(f"DataFrame must have more rows ({len(dataframe)}) than lag_price_window ({lag_price_window})")
#     if 'Datetime' not in dataframe.columns or 'MarginalES' not in dataframe.columns:
#         raise ValueError("DataFrame must contain 'Datetime' and 'MarginalES' columns")
    
#     X, y = [], []
    
#     # Extract price data and features
#     df_prices = dataframe[['Datetime', 'MarginalES']]
#     df_features = dataframe.iloc[:, 2:]  # Exclude 'Datetime' and 'MarginalES'
#     feature_names = df_features.columns.tolist()
    
#     if debug:
#         print(f"Feature columns identified: {feature_names}")
    
#     # Create samples from the data
#     for i in range(lag_price_window, len(df_prices)):
#         # Extract the window for prices as features (right-aligned)
#         window = df_prices.iloc[i-lag_price_window:i, 1:].values.flatten()
        
#         # Extract corresponding feature row (from the most recent point in the window)
#         feature_row = df_features.iloc[i-1].values.flatten()

#         # Concatenate window prices with feature row
#         X.append(np.concatenate((window, feature_row)))
#         y.append(df_prices.iloc[i, 1])  # Next price point as target
    
#     # Create column names for the prices feature window
#     price_columns = [f'price_t-{lag_price_window-i}' for i in range(lag_price_window)]
    
#     # Return DataFrame with proper column names
#     X_df = pd.DataFrame(X, columns=price_columns + feature_names)
#     y_series = pd.Series(y, name='price_t')
    
#     if debug:
#         print(f"X DataFrame shape: {X_df.shape}")
#         print(f"Feature columns count: {len(feature_names)}")
#         print(f"Price window columns: {price_columns}")
#         print(f"Total columns: {len(X_df.columns)}")
#         print(f"Sample size: {len(X_df)}")
    
#     return X_df, y_series
#!



# # Updated for mini models
# def create_feature_matrix_with_features(data, window_size):
#     """
#     Creates a feature matrix where each row is a sliding window of prices,
#     along with technical indicators as features, and a target vector with the next price.
    
#     Parameters:
#     ----------
#     - data: DataFrame with 'Datetime', 'MarginalES', and technical indicators
#     - window_size: Size of the sliding window
    
#     Returns:
#     -------
#     - X: DataFrame with sliding windows as rows and features as columns
#     - y: Series with target values (next price after each window)
#     """
#     # Extract the MarginalES column (price data)
#     if 'MarginalES' in data.columns:
#         prices = data['MarginalES'].values
#     else:
#         # If not, assume it's the second column (index 1)
#         prices = data.iloc[:, 1].values
    
#     # Number of samples we'll have after creating windows
#     n_samples = len(prices) - window_size - 1
    
#     # Create matrices for windows and target
#     X_windows = np.zeros((n_samples, window_size))
#     y = np.zeros(n_samples)
    
#     # Fill the matrices with the sliding windows and targets
#     for i in range(n_samples):
#         X_windows[i, :] = prices[i:i+window_size]  # Window of prices
#         y[i] = prices[i+window_size]  # Next price after window
    
#     # Convert window matrix to DataFrame
#     X_df = pd.DataFrame(X_windows, columns=[f'price_t-{window_size-i}' for i in range(window_size)])
    
#     # Extract features from the data, excluding Datetime and MarginalES columns
#     feature_cols = [col for col in data.columns if col not in ['Datetime', 'MarginalES']]
    
#     # DEBUG
#     logger.debug("Feature columns identified:", feature_cols)
    
#     # For each sample i, the features should correspond to the window [i:i+window_size-1]
#     # So for the first window positions 0,1,2, we want features from position 2 (window_size-1)
#     # That means for all samples, we want features from position i+window_size-1
#     features_idx = range(window_size-1, window_size-1+n_samples)
#     features = data[feature_cols].iloc[features_idx].reset_index(drop=True)
    
#     # DEBUG
#     logger.debug("Features DataFrame shape:", features.shape)
#     logger.debug("X_df shape:", X_df.shape)
    
#     # Combine the window matrix with the features
#     X_combined = pd.concat([X_df, features], axis=1)
    
#     # DEBUG
#     logger.debug("X_combined shape after concatenation:", X_combined.shape)
#     logger.debug("X_combined columns:", X_combined.columns.tolist())
    
#     return X_combined, pd.Series(y)
