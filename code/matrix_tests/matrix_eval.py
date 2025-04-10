'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2024-11-27
File: code/matrix_tests/matrix_eval.py

This script reads the results from the regression model evaluation
Organizes the results by the best metrics, in this case just lowest MSE and highest R²
And prints the most common dimensions in the top 1000 results
'''

'''
NO USAR LA MODA PARA LAS DIMENSIONES

MEJOR USAR MEDIA O MEDIANA PARA LOS RESULTADOS

SI HAY MCUHOS DIAS DODNE SE PIRA, MEJOR AL MEDIANA

SI NO HAY GRANDES DIFERENCIAS ENTRE LOS RESULTADOS, MEJOR LA MEDIA

CONSTRUIR LA MATRIZ 3D Y APLANARLA PARA HACER LA MEDIA O LA MEDIANA
ELEGIMOS LA MENOS MALA
'''

import pandas as pd
import numpy as np

# File path
results_file = 'data/metrics/results.csv'
# results_file = 'data/metrics/results_no_weekends.csv'
# results_file = 'data/metrics/results_return_data.csv'

# Load the CSV without headers
data = pd.read_csv(results_file, header=None)

'''
RAW DATA
2018-05-23 14:00:00,3,10,26.980081724030207,-64.86934014655874
2018-05-23 14:00:00,5,10,204.0733487376421,-0.5100814336003117
2018-05-23 14:00:00,7,10,249.89408706716756,-7.569756072262265
2018-05-23 14:00:00,3,15,168.71173140597804,-1.7268922282668404
2018-05-23 14:00:00,5,15,90.28369192238102,-5.792798744868569
2018-05-23 14:00:00,7,15,2773.820104680621,-1224.7991801022108
2018-05-23 14:00:00,3,20,682.3905221439018,-1.1878059771436238
2018-05-23 14:00:00,5,20,261.53597058716434,-7.122771163847439
2018-05-23 14:00:00,7,20,117.33990308074367,-1.232313049741487
2018-05-23 14:00:00,3,30,39.60290072101041,-0.1507423632950846
2018-05-23 14:00:00,5,30,405.51076382725455,-0.6329597906776467
2018-05-23 14:00:00,7,30,46.73926379382787,-0.6940251274463329
'''

# Assign column names
data.columns = ["Datetime", "Rows", "Columns", "MSE", "R2"]

# Print head
print(data.head())

# MSE only data set
mse_matrix = data[["Datetime", "Rows", "Columns", "MSE"]]

# Print head
print(mse_matrix.head())

'''
PROCESSED DATA (MSE)
2018-05-23 14:00:00,3,10,26.980081724030207
2018-05-23 14:00:00,5,10,204.0733487376421
2018-05-23 14:00:00,7,10,249.89408706716756
2018-05-23 14:00:00,3,15,168.71173140597804
2018-05-23 14:00:00,5,15,90.28369192238102
2018-05-23 14:00:00,7,15,2773.820104680621
2018-05-23 14:00:00,3,20,682.3905221439018
2018-05-23 14:00:00,5,20,261.53597058716434
2018-05-23 14:00:00,7,20,117.33990308074367
2018-05-23 14:00:00,3,30,39.60290072101041
2018-05-23 14:00:00,5,30,405.51076382725455
2018-05-23 14:00:00,7,30,46.73926379382787
'''

# Create 3D matrix of all the results

# First, we need to know the dimensions of the matrix
unique_dates = mse_matrix["Datetime"].unique()
unique_rows = mse_matrix["Rows"].unique()
unique_columns = mse_matrix["Columns"].unique()

# Create the 3D matrix
matrix = np.zeros((len(unique_dates), len(unique_rows), len(unique_columns)))

# Fill the matrix with the MSE values
for i, date in enumerate(unique_dates):
    for j, row in enumerate(unique_rows):
        for k, column in enumerate(unique_columns):
            mse = mse_matrix[(mse_matrix["Datetime"] == date) & (mse_matrix["Rows"] == row) & (mse_matrix["Columns"] == column)]["MSE"]
            if mse.empty:
                matrix[i, j, k] = np.nan
            else:
                matrix[i, j, k] = mse.values[0]

# Print the matrix
print(matrix)

# Save the matrix
np.save('data/metrics/mse_matrix.npy', matrix)

# AVERAGE MATRIX
# Calculate the average of each matrix position
# Average out all the individual results to get a single value that takes into account all layers


# MEDIAN MATRIX
# Calculate the median of each matrix position
# Median out all the individual results to get a single value that takes into account all layers









# Repeat the process for the R² values

# R² only data set
# r2_matrix = data[["Rows", "Columns", "R2"]]

#####! ChatGPT code
'''
# Convert the 'Datetime' column to datetime object and extract date (ignoring time)
mse_matrix['Datetime'] = pd.to_datetime(mse_matrix['Datetime'])
mse_matrix['Date'] = mse_matrix['Datetime'].dt.date

# Define function to check matrix completeness
def is_complete(subset, rows, columns):
    unique_row_col = subset[['Rows', 'Columns']].drop_duplicates()
    return len(unique_row_col) == rows * columns

# Get unique row and column identifiers
unique_rows = sorted(mse_matrix['Rows'].unique())
unique_columns = sorted(mse_matrix['Columns'].unique())
rows, columns = len(unique_rows), len(unique_columns)

# Dictionary to map row/column values to indices for easy access in matrix
row_mapping = {val: i for i, val in enumerate(unique_rows)}
column_mapping = {val: j for j, val in enumerate(unique_columns)}

# Prepare list for matrices
matrices = []

# Group data by each day
for date, group in mse_matrix.groupby('Date'):
    # Check if group has complete data
    if is_complete(group, rows, columns):
        # Initialize an empty matrix
        matrix = np.zeros((rows, columns))
        
        # Fill the matrix with values
        for _, row in group.iterrows():
            r_idx = row_mapping[row['Rows']]
            c_idx = column_mapping[row['Columns']]
            matrix[r_idx, c_idx] = row['MSE']
        
        # Add the matrix to our list of matrices
        matrices.append(matrix)
    else:
        print(f"Data incomplete for date {date}, skipping.")

# Stack matrices into a 3D array if any matrices exist
if matrices:
    big_matrix = np.stack(matrices, axis=0)
    print("3D Matrix created with shape:", big_matrix.shape)
else:
    print("No complete matrices were found.")

# Display the final 3D matrix if needed
print(big_matrix)

# save the 3D matrix
np.save('data/metrics/mse_matrix.npy', big_matrix)

# Get the mean of the 3D matrix
mean_matrix = np.mean(big_matrix, axis=0)
print("Mean matrix created with shape:", mean_matrix.shape)
print(mean_matrix)
# Print the lowest value and its index
min_value = np.min(mean_matrix)
min_index = np.unravel_index(np.argmin(mean_matrix, axis=None), mean_matrix.shape)
print(f"Lowest value: {min_value} at index {min_index}")

# Get the median of the 3D matrix
median_matrix = np.median(big_matrix, axis=0)
print("Median matrix created with shape:", median_matrix.shape)
print(median_matrix)
# Print the lowest value and its index
min_value = np.min(median_matrix)
min_index = np.unravel_index(np.argmin(median_matrix, axis=None), median_matrix.shape)
print(f"Lowest value: {min_value} at index {min_index}")
'''
#####! ChatGPT code
