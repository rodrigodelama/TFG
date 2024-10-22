#Split your data into training and testing (70/30) is indeed a good starting point!)
#For simplicity, use the names X_train, X_test, y_train, and y_test for the corresponding numpy arrays.
#Note: Set random_state to a fixed value, for example, 42

#YOUR CODE HERE
import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state=42)
# 506 rows, 13 columns
print("X_train sample:", X_train.shape) #70/100
print("X_test:", X_test.shape) #30/100
# 506 rows, 1 column
print("y_train sample:", y_train.shape) #70/100
print("y_test:", y_test.shape) #30/100

#Now, split your training data into two subsets: train_val and val (70/30). 
#For simplicity, use the names X_train_val, X_val, y_train_val, and y_val for the corresponding numpy arrays.
#Note: Set random_state to a fixed value, for example, 42

#YOUR CODE HERE

#train val is 70% of 354 and val is 30% of 354. The test sample is the remaining 152 (original 30%)
X_train_val, X_val, y_train_val, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

# train to develop our models
# validation for regressors with hyperparameter
# example for ridge ¡, the param thats gives us lower error shall be alpha

# test for final results

#YOUR CODE HERE

# we only sepparate into a further 70/30 the X_train and y_train values
print("X_train_val sample:", X_train_val.shape) # 70% of 70%
print("X_val sample:", X_val.shape) # 30% of 70%
print("X_test sample:", X_test.shape) # 30%
print()
print("y_train_val sample:", y_train_val.shape) # 70% of 70%
print("y_val sample:", y_val.shape) # 30% of 70%
print("y_test sample:", y_test.shape) # 30%
