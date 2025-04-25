'''
uc3m
Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices
Author: Rodrigo De Lama Fernández
Professor: Emilio Parrado

Date: 2025-04-14
File: code/utils/mini_models.py

This script houses the function where the mini models are created and trained
'''

# # # # import pandas as pd
# # # # import numpy as np

# # # # from utils.sliding_window import create_weight_matrix_with_features

# # # # def mini_model(model, subset_df_features, window_size, stationarity_depth):
# # # #     """
# # # #     Create a mini model using Lasso regression with a sliding window approach.

# # # #     Parameters:
# # # #     - subset_df_features: DataFrame containing the features for the model
# # # #     - window_size: Size of the sliding window for creating features
# # # #     - stationarity_depth: Depth of the training window for the model
# # # #     """

# # # #     X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)

# # # #     # logger.debug("X_combined shape: %s", X_combined.shape)

# # # #     # print("____________ X_combined")
# # # #     # print(X_combined)
# # # #     # print("____________ END X_combined")

# # # #     # Initialize y_pred with same index as y_subset_to_trim
# # # #     y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

# # # #     for i in range(y_subset_to_trim.size - stationarity_depth): # to avoid going out of bounds
# # # #         # Make subsets for training of the specified depth
# # # #         X_train = X_combined.iloc[i:i + stationarity_depth]
# # # #         y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]

# # # #         model.fit(X_train, y_train)

# # # #         # Predict the NEXT point after the training window
# # # #         # X_predict = X_combined.iloc[i + stationarity_depth].values.reshape(1, -1)
# # # #         # Use Pandas instead of NumPy
# # # #         X_predict = X_combined.iloc[[i + stationarity_depth]]  # Keep as DataFrame with column names - useful for feature importance
# # # #         y_predict = model.predict(X_predict)[0]

# # # #         # Save the predicted variable
# # # #         y_pred.iloc[i + stationarity_depth] = y_predict

# # # #     # Compare y_subset_to_trim with y_pred in the available indexes
# # # #     print("Actual vs Predicted:")
# # # #     print(pd.DataFrame({'Actual': y_subset_to_trim, 'Predicted': y_pred}))

# # # #     #calculate error.
# # # #     valid_pred = y_pred.dropna()
# # # #     valid_actual = y_subset_to_trim[valid_pred.index]
# # # #     error = valid_actual - valid_pred
# # # #     print("\nError:")
# # # #     print(error)

# # # #     return valid_pred, valid_actual, error, X_combined

# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # import joblib

# # # from utils.sliding_window import create_weight_matrix_with_features

# # # def mini_model(model, subset_df_features, window_size, stationarity_depth, model_name=None, save_dir="models", debug=False):
# # #     """
# # #     Create or load a mini model using Lasso regression with a sliding window approach.

# # #     Parameters:
# # #     - model: An untrained sklearn model object (Lasso, Ridge, etc.)
# # #     - subset_df_features: DataFrame containing the features for the model
# # #     - window_size: Size of the sliding window for creating features
# # #     - stationarity_depth: Depth of the training window for the model
# # #     - model_name: Optional name for saving/loading the model
# # #     - save_dir: Directory to store models
# # #     - debug: If True, print intermediate steps for debugging

# # #     Returns:
# # #     - valid_pred: Series of predicted values
# # #     - valid_actual: Series of actual values
# # #     - error: Series of errors
# # #     - X_combined: Matrix of features used
# # #     """

# # #     if debug:
# # #         print("🔍 Creating weight matrix with features...")

# # #     X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)
# # #     y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

# # #     os.makedirs(save_dir, exist_ok=True)

# # #     for i in range(y_subset_to_trim.size - stationarity_depth):
# # #         X_train = X_combined.iloc[i:i + stationarity_depth]
# # #         y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]
# # #         X_predict = X_combined.iloc[[i + stationarity_depth]]
# # #         model_file = None

# # #         if model_name:
# # #             model_file = os.path.join(save_dir, f"{model_name}_step_{i}.joblib")

# # #         if model_file and os.path.exists(model_file):
# # #             if debug:
# # #                 print(f"📦 Loading model from {model_file}")
# # #             model = joblib.load(model_file)
# # #         else:
# # #             if debug:
# # #                 print(f"⚙️ Training model at step {i}")
# # #             model.fit(X_train, y_train)
# # #             if model_file:
# # #                 joblib.dump(model, model_file)
# # #                 if debug:
# # #                     print(f"💾 Model saved to {model_file}")

# # #         y_predict = model.predict(X_predict)[0]
# # #         y_pred.iloc[i + stationarity_depth] = y_predict

# # #     if debug:
# # #         print("\n📊 Actual vs Predicted:")
# # #         print(pd.DataFrame({'Actual': y_subset_to_trim, 'Predicted': y_pred}))

# # #     valid_pred = y_pred.dropna()
# # #     valid_actual = y_subset_to_trim[valid_pred.index]
# # #     error = valid_actual - valid_pred

# # #     if debug:
# # #         print("\n📉 Prediction Error:")
# # #         print(error.describe())
# # #         print(error)

# # #     return valid_pred, valid_actual, error, X_combined

# # import os
# # import joblib
# # import pandas as pd
# # import numpy as np

# # from utils.sliding_window import create_weight_matrix_with_features

# # def mini_model(model, subset_df_features, window_size, stationarity_depth,
# #                model_name=None, save_dir="models", debug=False, save_every_step=False):
# #     """
# #     Create a mini model using regression with a sliding window approach.

# #     Parameters:
# #     - model: A scikit-learn style regression model (must implement fit/predict)
# #     - subset_df_features: DataFrame with input features
# #     - window_size: Size of sliding window
# #     - stationarity_depth: Training window depth per step
# #     - model_name: Optional name for saving the model
# #     - save_dir: Directory to store the trained model(s)
# #     - debug: If True, print detailed internal steps
# #     - save_every_step: If True, save model at each step; else only final
# #     """

# #     if debug:
# #         print("🔍 Creating weight matrix with features...")

# #     X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)

# #     if debug:
# #         print(f"📐 X_combined shape: {X_combined.shape}")
# #         print(f"📏 y_subset_to_trim length: {len(y_subset_to_trim)}")

# #     # Prepare prediction holder
# #     y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

# #     # Ensure save directory exists
# #     if model_name:
# #         os.makedirs(save_dir, exist_ok=True)

# #     for i in range(y_subset_to_trim.size - stationarity_depth):
# #         if debug and i % 100 == 0:
# #             print(f"⚙️ Training model at step {i}")

# #         # Define window slices
# #         X_train = X_combined.iloc[i:i + stationarity_depth]
# #         y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]
# #         X_predict = X_combined.iloc[[i + stationarity_depth]]

# #         # Train model
# #         model.fit(X_train, y_train)
# #         y_predict = model.predict(X_predict)[0]
# #         y_pred.iloc[i + stationarity_depth] = y_predict

# #         # Save per-step model if enabled
# #         if save_every_step and model_name:
# #             model_file = os.path.join(save_dir, f"{model_name}_step_{i}.joblib")
# #             joblib.dump(model, model_file)
# #             if debug:
# #                 print(f"💾 Model saved to {model_file}")

# #     # Save final model if needed
# #     if not save_every_step and model_name:
# #         final_model_file = os.path.join(save_dir, f"{model_name}_final.joblib")
# #         joblib.dump(model, final_model_file)
# #         if debug:
# #             print(f"✅ Final model saved to {final_model_file}")

# #     # Evaluate predictions
# #     valid_pred = y_pred.dropna()
# #     valid_actual = y_subset_to_trim[valid_pred.index]
# #     error = valid_actual - valid_pred

# #     if debug:
# #         print("\n📊 Actual vs Predicted:")
# #         print(pd.DataFrame({'Actual': valid_actual, 'Predicted': valid_pred}))
# #         print("\n📉 Error:")
# #         print(error.describe())

# #     return valid_pred, valid_actual, error, X_combined

# import os
# import joblib
# import logging
# import pandas as pd
# import numpy as np

# from utils.sliding_window import create_weight_matrix_with_features

# # Configure module-level logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # Default level
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# if not logger.handlers:
#     logger.addHandler(handler)

# def mini_model(model, subset_df_features, window_size, stationarity_depth,
#                model_name=None, save_dir="models", debug=False, save_every_step=False):
#     """
#     Trains a model using a sliding window approach and optionally saves models.

#     Parameters:
#     - model: A scikit-learn model implementing fit() and predict()
#     - subset_df_features: DataFrame with features
#     - window_size: Feature creation window size
#     - stationarity_depth: Training window size for each prediction step
#     - model_name: Base name for saving models
#     - save_dir: Directory to save trained models
#     - debug: Enables debug-level logs if True
#     - save_every_step: If True, saves model after every training step
#     """

#     if debug:
#         logger.setLevel(logging.DEBUG)

#     logger.info("🔍 Creating weight matrix with features...")
#     X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)

#     logger.debug(f"📐 X_combined shape: {X_combined.shape}")
#     logger.debug(f"📏 y_subset_to_trim length: {len(y_subset_to_trim)}")

#     y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')

#     if model_name:
#         os.makedirs(save_dir, exist_ok=True)

#     for i in range(y_subset_to_trim.size - stationarity_depth):
#         if debug and i % 100 == 0:
#             logger.debug(f"⚙️ Training model at step {i}")

#         X_train = X_combined.iloc[i:i + stationarity_depth]
#         y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]
#         X_predict = X_combined.iloc[[i + stationarity_depth]]

#         model.fit(X_train, y_train)
#         y_predict = model.predict(X_predict)[0]
#         y_pred.iloc[i + stationarity_depth] = y_predict

#         if save_every_step and model_name:
#             model_file = os.path.join(save_dir, f"{model_name}_step_{i}.joblib")
#             joblib.dump(model, model_file)
#             logger.debug(f"💾 Model saved to {model_file}")

#     if not save_every_step and model_name:
#         final_model_file = os.path.join(save_dir, f"{model_name}_final.joblib")
#         joblib.dump(model, final_model_file)
#         logger.info(f"✅ Final model saved to {final_model_file}")

#     valid_pred = y_pred.dropna()
#     valid_actual = y_subset_to_trim[valid_pred.index]
#     error = valid_actual - valid_pred

#     if debug:
#         logger.debug("📊 Actual vs Predicted:")
#         logger.debug(pd.DataFrame({'Actual': valid_actual, 'Predicted': valid_pred}).to_string())
#         logger.debug("📉 Error summary:")
#         logger.debug(error.describe().to_string())

#     return valid_pred, valid_actual, error, X_combined

import os
import joblib
import logging
import pandas as pd
import numpy as np

from utils.sliding_window import create_weight_matrix_with_features
from utils.logger import get_logger
logger = get_logger(__name__)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def mini_model(model, subset_df_features, window_size, stationarity_depth,
               model_name=None, save_dir=None, debug=False, save_every_step=False):
    """
    Trains a model using a sliding window approach. Skips training if all step models exist.

    Parameters:
    - model: A scikit-learn model implementing fit() and predict()
    - subset_df_features: DataFrame with features
    - window_size: Feature creation window size
    - stationarity_depth: Training window size for each prediction step
    - model_name: Base name for saving models
    - save_dir: Directory to save trained models
    - debug: Enables debug-level logs
    - save_every_step: If True, saves model after every training step
    """

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info("🔍 Creating weight matrix with features...")
    X_combined, y_subset_to_trim = create_weight_matrix_with_features(subset_df_features, window_size)
    y_pred = pd.Series(index=y_subset_to_trim.index, dtype='float64')
    total_steps = y_subset_to_trim.size - stationarity_depth

    logger.debug(f"📐 X_combined shape: {X_combined.shape}")
    logger.debug(f"🧮 Total training steps: {total_steps}")

    # ⚙️ Auto-generate save_dir if not specified
    if model_name and save_dir is None:
        save_dir = os.path.join("models", model_name)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_models_exist = False
    step_model_paths = []

    if model_name:
        os.makedirs(save_dir, exist_ok=True)
        step_model_paths = [
            os.path.join(save_dir, f"{model_name}_step_{i}.joblib") for i in range(total_steps)
        ]
        all_models_exist = all(os.path.exists(path) for path in step_model_paths)
        existing = sum(os.path.exists(p) for p in step_model_paths)
        logger.info(f"📦 Found {existing}/{total_steps} step models.")

    if all_models_exist:
        logger.info("♻️ All step models found — skipping training and loading predictions.")
        for i, model_path in enumerate(step_model_paths):
            trained_model = joblib.load(model_path)
            X_predict = X_combined.iloc[[i + stationarity_depth]]
            y_pred.iloc[i + stationarity_depth] = trained_model.predict(X_predict)[0]
    else:
        logger.info("⚙️ Starting training loop...")
        for i in range(total_steps):
            if debug and i % 100 == 0:
                logger.debug(f"⚙️ Training model at step {i}")

            X_train = X_combined.iloc[i:i + stationarity_depth]
            y_train = y_subset_to_trim.iloc[i:i + stationarity_depth]
            X_predict = X_combined.iloc[[i + stationarity_depth]]

            model.fit(X_train, y_train)
            y_predict = model.predict(X_predict)[0]
            y_pred.iloc[i + stationarity_depth] = y_predict

            if model_name and save_every_step:
                joblib.dump(model, step_model_paths[i])
                # logger.debug(f"💾 Model saved to {step_model_paths[i]}")
                logger.debug(f"💾 Saved step {i} → {os.path.basename(step_model_paths[i])}")

        if model_name and not save_every_step:
            final_model_path = os.path.join(save_dir, f"{model_name}_final.joblib")
            joblib.dump(model, final_model_path)
            logger.info(f"✅ Final model saved to {final_model_path}")

    valid_pred = y_pred.dropna()
    valid_actual = y_subset_to_trim[valid_pred.index]
    error = valid_actual - valid_pred

    # if debug:
        # logger.debug("📊 Actual vs Predicted:")
        # logger.debug(pd.DataFrame({'Actual': valid_actual, 'Predicted': valid_pred}).to_string())
        # logger.debug("📉 Error summary:")
        # logger.debug(error.describe().to_string())

    return valid_pred, valid_actual, error, X_combined
