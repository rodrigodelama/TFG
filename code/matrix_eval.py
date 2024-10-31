'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado 
'''

import pandas as pd

# File path
results_file = 'data/metrics/results.csv'

# Load the CSV without headers
data = pd.read_csv(results_file, header=None)

# Assign column names
data.columns = ["Datetime", "Rows", "Columns", "MSE", "R2"]

# Organize by lowest MSE and highest R²
sorted_data = data.sort_values(by=["MSE", "R2"], ascending=[True, False])

# # Print the top results
# print(sorted_data.head())

# Print the top 1000 results
print(sorted_data.head(1000))

# Remove the ones with a negative R²
positive_r2 = data[data["R2"] > 0]

# Print the top 1000 results
print(positive_r2.head(1000))

# Analyzing the results
# Which dimensions are the best?
# Which window size and number of days back are the best?

# Find the most common dimensions in the top 1000 results
top_dimensions = sorted_data.head(1000)[["Rows", "Columns"]].value_counts()

# Print the top dimensions
print(top_dimensions)



# # Group by matrix dimensions (Rows and Columns)
# metrics_by_dimension = data.groupby(["Rows", "Columns"]).agg(
#     avg_mse=("MSE", "mean"),
#     avg_r2=("R2", "mean")
# ).reset_index()

# # Sort for optimal values (low MSE, high R²)
# optimal_dimensions = metrics_by_dimension.sort_values(by=["avg_mse", "avg_r2"], ascending=[True, False])

# # View top results
# print(optimal_dimensions.head())

# # Group by the window size and number of days back
# grouped_results = results.groupby(['window_size', 'num_days_back']).mean()

# # Find the best configuration
# best_mse = grouped_results['mse'].min()
# best_r2 = grouped_results['r2'].max()

# best_config = grouped_results[(grouped_results['mse'] == best_mse) & (grouped_results['r2'] == best_r2)]

# print(f"Best configuration: {best_config}")