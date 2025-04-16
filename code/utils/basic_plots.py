'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-14
File: code/utils/basic_plots.py
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_predicted(model, valid_actual, valid_pred, stationarity_depth):
    """
    Plots the actual vs. predicted values for a given validation set.
    """
    model_name = model.__class__.__name__
    plt.figure(figsize=(12, 6))
    plt.plot(valid_actual.index, valid_actual.values, label='Actual Values', marker='.')
    plt.plot(valid_pred.index, valid_pred.values, label='Predicted Values', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # plt.title('Actual vs. Predicted Values - Stationarity 365 days')
    plt.title(f'{model_name} Actual vs. Predicted Values - Stationarity {stationarity_depth} days')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_error(valid_actual, valid_pred):
def plot_error(error):
    """
    Plots the error between actual and predicted values.
    """
    # error = valid_actual - valid_pred
    plt.figure(figsize=(12, 6))
    plt.plot(error.index, error.values, label='Error', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Error (Actual - Predicted)')
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, X_combined):
    """
    Plots the feature importance of the model.
    """
    feature_importance = model.feature_importances_
    # Sort and plot
    indices = np.argsort(feature_importance)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    # Feature names
    feature_names = X_combined.columns.tolist()
    # Sort feature names according to importance
    sorted_feature_names = [feature_names[i] for i in indices]
    plt.bar(range(X_combined.shape[1]), feature_importance[indices], align="center")
    plt.xticks(range(X_combined.shape[1]), sorted_feature_names, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()