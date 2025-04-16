'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-14
File: code/utils/mini_models.py

This script houses the function where the mini models are created and trained
'''

import pandas as pd
import numpy as np

from utils.sliding_window import create_weight_matrix_with_features

def mini_model(model, subset_df_features, window_size, stationarity_depth):
    """
    Create a mini model using Lasso regression with a sliding window approach.

    Parameters:
    - subset_df_features: DataFrame containing the features for the model
    - window_size: Size of the sliding window for creating features
    - stationarity_depth: Depth of the training window for the model
    """

    X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)

    # logger.debug("X_combined shape: %s", X_combined.shape)

    print("____________ X_combined")
    print(X_combined)
    print("____________ END X_combined")

    # Initialize y_pred with same index as y_subset_to_trim
    y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

    for i in range(y_subset_to_trim.size - stationarity_depth): # to avoid going out of bounds
        # Make subsets for training of the specified depth
        X_train = X_combined.iloc[i:i + stationarity_depth]
        y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]

        model.fit(X_train, y_train)

        # Predict the NEXT point after the training window
        # X_predict = X_combined.iloc[i + stationarity_depth].values.reshape(1, -1)
        # Use Pandas instead of NumPy
        X_predict = X_combined.iloc[[i + stationarity_depth]]  # Keep as DataFrame with column names - useful for feature importance
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

    return valid_pred, valid_actual, error
