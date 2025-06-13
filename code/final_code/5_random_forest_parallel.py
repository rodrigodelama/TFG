import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.matrix_builder import create_feature_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.ensemble import RandomForestRegressor
import time
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Load the dataset
csv_hour_file = '/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/data/ta_metrics/final_price_ta_metrics.csv'
df = pd.read_csv(csv_hour_file, parse_dates=['Datetime'])

def random_forest_training_gt(df, model, sliding_window, lag_price_window, DEBUG):
    """
    Train a Random Forest model using a sliding window approach.

    Parameters:
    - df: DataFrame containing the dataset with features and target variable.
    - model: The machine learning model to be trained (e.g., RandomForest).
    - sliding_window: Number of rows to use for training in each sliding window.
    - lag_price_window: Number of previous days to use as features.
    
    Returns:
    - prediction_df: DataFrame containing predictions and actual values.
    """
    if DEBUG:
        print("Debug mode is ON. Detailed output will be printed.")

    # Validate input parameters
    if sliding_window <= lag_price_window:
        raise ValueError("Sliding window must be greater than to the price feature window.")

    training_sliding_window = sliding_window + 1  # +1 to include the next row as the test set

    # Calculate number of sliding window models to train in the dataset
    num_sliding_windows = len(df) - training_sliding_window
    if DEBUG:
        # Training for sliding windows and price feature window
        print(f"Training sliding window size: {training_sliding_window}, Price feature window size: {lag_price_window}")
        print(f"Number of rows in the dataset: {len(df)}")

    print(f"Number of models to train: {num_sliding_windows}")

    # Initialize lists to store predictions, actuals, and timestamps
    predictions_list = []
    actuals_list = []
    timestamps_list = []

    model = model  # This is done in order to use different model configurations of RandomForestRegressor

    for i in range(num_sliding_windows):
        if DEBUG:
            print(f"Processing sliding window {i + 1}/{num_sliding_windows}...")

        # Ensure we do not exceed the DataFrame length
        if i + training_sliding_window >= len(df):
            break  # Avoid index out of bounds
        
        sliding_window_set = df.iloc[i : i + training_sliding_window]

        if DEBUG:
            print(f"Sliding window should have {training_sliding_window} rows, got {len(sliding_window_set)} rows.")
            print(f"Sliding window set:\n{sliding_window_set}")
        
        # Create feature matrix and target variable for training
        X_train, y_train = create_feature_matrix(sliding_window_set, lag_price_window)
        if DEBUG:
            print(f"Feature matrix shape: {X_train.shape}, Target variable shape: {y_train.shape}")
            print(f"Feature matrix:\n{X_train.head()}")
            print(f"Target variable:\n{y_train.head()}")

        # Split for training and prediction
        X_train_fit = X_train.iloc[:-1]
        y_train_fit = y_train.iloc[:-1]

        X_to_predict = X_train.iloc[-1:]
        y_to_predict = y_train.iloc[-1]

        if DEBUG:
            print(f"Training features shape: {X_train_fit.shape}, Training target shape: {y_train_fit.shape}")
            print(f"Features to predict shape: {X_to_predict.shape}, Target to predict: {y_to_predict}")

        model.fit(X_train_fit, y_train_fit)
        y_predicted = model.predict(X_to_predict)
        
        # Add a lower bounds to set extreme negative predictions to 0, assuming prices cannot be negative
        if y_predicted[0] < 0:
            y_predicted[0] = 0

        # Store results
        predictions_list.append(y_predicted[0])
        actuals_list.append(y_to_predict)
        if 'Datetime' in sliding_window_set.columns:
            timestamps_list.append(sliding_window_set.iloc[-1]['Datetime'])
        else:
            timestamps_list.append(i + training_sliding_window - 1)

    # Create final prediction DataFrame
    prediction_df = pd.DataFrame({
        'Timestamp': timestamps_list,
        'Predicted': predictions_list,
        'Actual': actuals_list
    })

    return prediction_df

def process_single_combination(args):
    """Process a single (sliding_window, lag_price_window, n_estimators, max_depth) combination"""
    df, model_params, sliding_window, lag_price_window, percentiles, DEBUG = args
    
    # Create model instance for this process
    model = RandomForestRegressor(**model_params)
    
    print(f"Processing sliding_window={sliding_window}, lag_price_window={lag_price_window}, "
          f"n_estimators={model_params['n_estimators']}, max_depth={model_params['max_depth']}")
    
    try:
        # Run the training
        prediction_df = random_forest_training_gt(df, model, sliding_window, lag_price_window, DEBUG)

        # Calculate overall metrics
        actuals_list = prediction_df['Actual'].values
        predictions_list = prediction_df['Predicted'].values
        timestamps_list = prediction_df['Timestamp'].values

        mse = mean_squared_error(actuals_list, predictions_list)
        mae = mean_absolute_error(actuals_list, predictions_list)
        r2 = r2_score(actuals_list, predictions_list)

        # Calculate prediction errors for filtering
        prediction_errors = np.abs(np.array(predictions_list) - np.array(actuals_list))
        # Store results for this combination
        combination_results = []

        # Calculate ES-like metric using prediction_errors
        # FIXED: Calculate worst 5% of errors across all predictions
        worst_overall_error_threshold = np.percentile(prediction_errors, 95)  # worst 5% of errors
        worst_overall_indices = prediction_errors >= worst_overall_error_threshold
        avg_worst_error = np.mean(prediction_errors[worst_overall_indices])

        # Add overall results (100% percentile)
        combination_results.append({
            'sliding_window': sliding_window,
            'lag_price_window': lag_price_window,
            'n_estimators': model_params['n_estimators'],
            'max_depth': model_params['max_depth'],
            'percentile': 100,
            'data_points': len(predictions_list),
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'worst_avg_error': avg_worst_error
        })

        for p in percentiles:
            # Filter by best predictions (lowest errors)
            best_error_threshold = np.percentile(prediction_errors, p)
            
            # Get indices of best predictions 
            best_indices = prediction_errors <= best_error_threshold

            filtered_actuals = np.array(actuals_list)[best_indices]
            filtered_predictions = np.array(predictions_list)[best_indices]
            filtered_errors = prediction_errors[best_indices]

            if len(filtered_actuals) > 0:
                mse_p = mean_squared_error(filtered_actuals, filtered_predictions)
                mae_p = mean_absolute_error(filtered_actuals, filtered_predictions)
                r2_p = r2_score(filtered_actuals, filtered_predictions)
                
                # FIXED: Calculate worst 5% WITHIN the filtered subset
                worst_5_percent_threshold = np.percentile(filtered_errors, 95)
                worst_5_percent_indices = filtered_errors >= worst_5_percent_threshold
                worst_5_percent_errors = filtered_errors[worst_5_percent_indices]
                avg_worst_5_percent = np.mean(worst_5_percent_errors)

                combination_results.append({
                    'sliding_window': sliding_window,
                    'lag_price_window': lag_price_window,
                    'n_estimators': model_params['n_estimators'],
                    'max_depth': model_params['max_depth'],
                    'percentile': p,
                    'data_points': len(filtered_predictions),
                    'mse': mse_p,
                    'mae': mae_p,
                    'r2': r2_p,
                    'worst_avg_error': avg_worst_5_percent  # FIXED: Now actually the average of worst 5%
                })
            else:
                # Add entry for no data available
                combination_results.append({
                    'sliding_window': sliding_window,
                    'lag_price_window': lag_price_window,
                    'n_estimators': model_params['n_estimators'],
                    'max_depth': model_params['max_depth'],
                    'percentile': p,
                    'data_points': 0,
                    'mse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'worst_avg_error': np.nan
                })
        
        # Memory cleanup
        del prediction_df, actuals_list, predictions_list, prediction_errors
        gc.collect()
        
        return combination_results
        
    except Exception as e:
        print(f"Error processing combination: {e}")
        return []

def initialize_results_file(results_file):
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(results_file):
        columns = ['sliding_window', 'lag_price_window', 'n_estimators', 'max_depth', 
                  'percentile', 'data_points', 'mse', 'mae', 'r2', 'worst_avg_error']
        pd.DataFrame(columns=columns).to_csv(results_file, index=False)

def save_batch_results(results_file, batch_results):
    """Save batch results to CSV file using append mode"""
    for result in batch_results:
        if result:  # Check if result is not empty
            batch_df = pd.DataFrame(result)
            # Use append mode, no header for subsequent writes
            batch_df.to_csv(results_file, mode='a', header=False, index=False)

def load_completed_combinations(results_file):
    """Load completed combinations from existing results file"""
    try:
        existing_results_df = pd.read_csv(results_file)
        print(f"Loaded {len(existing_results_df)} existing results")
        
        # Create a set of completed combinations for quick lookup
        # Only consider unique (sliding_window, lag_price_window) pairs
        completed_combinations = set()
        for _, row in existing_results_df.iterrows():
            completed_combinations.add((int(row['sliding_window']), int(row['lag_price_window'])))
        
        return completed_combinations
        
    except FileNotFoundError:
        print("No existing results found. Starting fresh...")
        return set()

def main():
    # Debug parameter
    DEBUG = False

    # Define the Sliding Windows for the runs
    sliding_windows = [7, 10, 15, 30, 90]
    lag_price_windows = [1, 2, 4, 6]
    percentiles = [99, 95, 85, 75]

    # Random Forest hyperparameters (including max_depth)
    n_estimators = [100, 200, 300, 400]
    max_depth = [None, 10, 20, 30]

    # Use 6 workers for M1 Pro (6P cores)
    max_workers = 8 # Max workers for parallel processing at night
    print(f"Using {max_workers} parallel workers")

    # Process all combinations with parallel processing
    for n_est in n_estimators:
        for max_d in max_depth:
            # Initialize results file path
            results_file = f"/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/code/final_metrics/3_randomforest/random_forest_baseline_results_n_estimators_{n_est}_max_depth_{max_d}.csv"

            # Initialize CSV file if it doesn't exist
            initialize_results_file(results_file)
            
            # Load completed combinations
            completed_combinations = load_completed_combinations(results_file)
            
            # Prepare model parameters
            model_params = {
                'n_estimators': n_est,
                'max_depth': max_d,
                'n_jobs': 1,  # Each process uses 1 core
                'random_state': 42,

                # Uncomment below for more advanced configurations
                # 'warm_start': True,
                # 'max_samples': 0.8,
                # 'max_features': 'sqrt',
                # 'min_samples_split': 5,
                # 'min_samples_leaf': 2
            }
            
            # Prepare all combinations that haven't been completed
            combinations = []
            for sliding_window in sliding_windows:
                for lag_price_window in lag_price_windows:
                    if (sliding_window, lag_price_window) not in completed_combinations:
                        combinations.append((df, model_params, sliding_window, lag_price_window, percentiles, DEBUG))
            
            if not combinations:
                print(f"All combinations already completed for n_estimators={n_est}, max_depth={max_d}")
                continue
            
            print(f"Processing {len(combinations)} combinations for n_estimators={n_est}, max_depth={max_d}")
            
            # Progress monitoring
            start_time = time.time()
            
            # Process in parallel batches to manage memory
            batch_size = max_workers
            total_batches = len(combinations) // batch_size + (1 if len(combinations) % batch_size else 0)
            
            with tqdm(total=len(combinations), desc=f"RF n_est={n_est}, max_depth={max_d}") as pbar:
                for batch_idx in range(total_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, len(combinations))
                    batch_combinations = combinations[batch_start:batch_end]
                    
                    # Process batch in parallel
                    with ProcessPoolExecutor(max_workers=min(len(batch_combinations), max_workers)) as executor:
                        batch_results = list(executor.map(process_single_combination, batch_combinations))
                    
                    # Save batch results immediately (progressive saving)
                    save_batch_results(results_file, batch_results)
                    
                    # Update completed combinations tracking (for this session)
                    for i, result in enumerate(batch_results):
                        if result:  # Check if result is not empty
                            combo = batch_combinations[i]
                            completed_combinations.add((combo[2], combo[3]))  # sliding_window, lag_price_window
                    
                    # Update progress
                    pbar.update(len(batch_combinations))
                    
                    # Calculate and display timing information
                    elapsed = time.time() - start_time
                    completed_combos = batch_end
                    if completed_combos > 0:
                        avg_time_per_combo = elapsed / completed_combos
                        remaining_combos = len(combinations) - completed_combos
                        remaining_time = avg_time_per_combo * remaining_combos
                        
                        pbar.set_postfix({
                            'ETA': f"{remaining_time/3600:.1f}h",
                            'Avg': f"{avg_time_per_combo:.1f}s/combo",
                            'Batch': f"{batch_idx+1}/{total_batches}"
                        })
                    
                    # Memory cleanup after each batch
                    gc.collect()
            
            print(f"Completed all combinations for n_estimators={n_est}, max_depth={max_d}")
            print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")

    # Final comprehensive results aggregation
    print("\nAggregating all results...")
    comprehensive_gt_df = pd.DataFrame()

    for n_est in n_estimators:
        for max_d in max_depth:
            try:
                results_file = f"/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/code/final_metrics/3_randomforest/random_forest_baseline_results_n_estimators_{n_est}_max_depth_{max_d}.csv"
                results_df = pd.read_csv(results_file)
                comprehensive_gt_df = pd.concat([comprehensive_gt_df, results_df], ignore_index=True)
                print(f"Added results for n_estimators={n_est}, max_depth={max_d}: {len(results_df)} rows")
            except FileNotFoundError:
                print(f"Results file not found for n_estimators={n_est}, max_depth={max_d}")

    # Save comprehensive results
    comprehensive_gt_df.to_csv('/Users/rodrigodelama/Library/Mobile Documents/com~apple~CloudDocs/uc3m/TFG/code/final_metrics/3_randomforest/random_forest_full_baseline_results_optimized.csv', index=False)
    print(f"\nFinal results saved to 'random_forest_full_baseline_results_optimized.csv'")
    print(f"Total combinations processed: {len(comprehensive_gt_df)}")
    
    # Calculate expected vs actual combinations
    expected_total = len(sliding_windows) * len(lag_price_windows) * len(n_estimators) * len(max_depth) * (len(percentiles) + 1)  # +1 for 100% percentile
    print(f"Expected total result rows: {expected_total}")
    print(f"Actual result rows: {len(comprehensive_gt_df)}")

    # Memory cleanup
    gc.collect()

if __name__ == '__main__':
    mp.freeze_support()  # For Windows compatibility
    main()