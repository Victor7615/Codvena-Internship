# oil_forecast_script.py

import os
import logging
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Data Processing Functions ---

def load_and_prepare_data(file_path: str, well_code: str, target_variable: str) -> pd.DataFrame:
    """
    Loads data from an Excel file, filters for a specific well, and performs cleaning.

    Args:
        file_path (str): Path to the Volve production data Excel file.
        well_code (str): The code for the well to model (e.g., 'NO 15/9-F-1 C').
        target_variable (str): The name of the target column for prediction.

    Returns:
        pd.DataFrame: A cleaned and prepared DataFrame for the specified well.
    """
    logging.info(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_excel(file_path, sheet_name='Daily Production Data')
        logging.info(f"Successfully loaded data with shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        raise

    if well_code not in df['WELL_BORE_CODE'].unique():
        logging.error(f"Well code '{well_code}' not found in the dataset.")
        raise ValueError(f"Well '{well_code}' not found.")

    df_well = df[df['WELL_BORE_CODE'] == well_code].copy()
    logging.info(f"Filtered for well '{well_code}'. Shape: {df_well.shape}")

    # Data Cleaning
    df_well['DATEPRD'] = pd.to_datetime(df_well['DATEPRD'])
    df_well.set_index('DATEPRD', inplace=True)
    df_well.sort_index(inplace=True)

    relevant_cols = [
        'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
        'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
        'AVG_WHT_P', 'BORE_GAS_VOL', 'BORE_WAT_VOL', target_variable
    ]
    df_well = df_well[relevant_cols]

    for col in df_well.columns:
        df_well[col] = pd.to_numeric(df_well[col], errors='coerce')

    # Impute missing values using modern pandas methods
    df_well.ffill(inplace=True)
    df_well.bfill(inplace=True)

    logging.info(f"Data cleaning complete. Final shape: {df_well.shape}")
    return df_well

def create_features(df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    """
    Engineers time-series features (time-based, lags, rolling windows).

    Args:
        df (pd.DataFrame): The cleaned input DataFrame.
        target_variable (str): The name of the target column.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logging.info("Starting feature engineering...")
    df_feat = df.copy()

    # Time-based features
    df_feat['year'] = df_feat.index.year
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['dayofweek'] = df_feat.index.dayofweek

    # Lag features
    for lag in [1, 7, 14]:
        df_feat[f'{target_variable}_lag_{lag}'] = df_feat[target_variable].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        df_feat[f'{target_variable}_roll_mean_{window}'] = df_feat[target_variable].rolling(window=window).mean()
        df_feat[f'{target_variable}_roll_std_{window}'] = df_feat[target_variable].rolling(window=window).std()

    df_feat.dropna(inplace=True)
    logging.info(f"Feature engineering complete. Shape after feature engineering: {df_feat.shape}")
    return df_feat

# --- 3. MLflow and Visualization Helper Functions ---

def log_plots_to_mlflow(model, X_train, y_train, X_test, y_test, split_date, target_variable, well_code):
    """Generates and logs feature importance and time-series plots to MLflow."""
    # 1. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    lgb.plot_importance(model, ax=ax, max_num_features=20)
    plt.title('Feature Importance')
    plt.tight_layout()
    plot_path = "feature_importance.png"
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path, "plots")
    plt.close(fig)
    logging.info("Logged feature importance plot.")

    # 2. Actual vs. Predicted Time Series Plot
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    fig_ts, ax_ts = plt.subplots(figsize=(15, 7))
    ax_ts.plot(y_train.index, y_train, label='Actual Train', color='blue')
    ax_ts.plot(y_test.index, y_test, label='Actual Test', color='cyan')
    ax_ts.plot(y_train.index, train_predictions, label='Predicted Train', color='red', linestyle='--')
    ax_ts.plot(y_test.index, test_predictions, label='Predicted Test', color='orange', linestyle='--')
    ax_ts.axvline(x=split_date, color='green', linestyle=':', label='Train-Test Split')
    ax_ts.set_title(f'{target_variable} Actual vs. Predicted for {well_code}')
    ax_ts.set_xlabel('Date')
    ax_ts.set_ylabel(target_variable)
    ax_ts.legend()
    ax_ts.grid(True)
    plt.tight_layout()
    ts_plot_path = "actual_vs_predicted_time_series.png"
    fig_ts.savefig(ts_plot_path)
    mlflow.log_artifact(ts_plot_path, "plots")
    plt.close(fig_ts)
    logging.info("Logged actual vs. predicted time series plot.")

# --- 4. Main Model Training Function ---

def train_and_log_model(df_feat: pd.DataFrame, well_code: str, target_variable: str, experiment_name: str):
    """
    Trains, tunes, and logs the LightGBM model using MLflow.

    Args:
        df_feat (pd.DataFrame): DataFrame with all features.
        well_code (str): The well code for logging purposes.
        target_variable (str): The name of the target column.
        experiment_name (str): Name for the MLflow experiment.
    """
    logging.info("Starting model training and logging process...")
    
    # Split data
    X = df_feat.drop(target_variable, axis=1)
    y = df_feat[target_variable]
    split_date = X.index.max() - pd.DateOffset(months=6)
    X_train, X_test = X.loc[X.index <= split_date], X.loc[X.index > split_date]
    y_train, y_test = y.loc[y.index <= split_date], y.loc[y.index > split_date]
    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        logging.info(f"MLflow Run Started. Run ID: {run.info.run_id}")
        mlflow.log_params({"well_code": well_code, "target_variable": target_variable})

        # Hyperparameter Tuning with GridSearchCV
        tscv = TimeSeriesSplit(n_splits=3)
        lgbm = lgb.LGBMRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31], 'max_depth': [-1, 10]
        }

        logging.info("Starting GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(lgbm, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        logging.info(f"Best parameters found: {best_params}")
        mlflow.log_params(best_params)

        # Train final model with best parameters
        final_model = lgb.LGBMRegressor(**best_params, random_state=42)
        final_model.fit(X_train, y_train)

        # Evaluation
        predictions = final_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        logging.info(f"Test Set MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse})

        # Log artifacts
        log_plots_to_mlflow(final_model, X_train, y_train, X_test, y_test, split_date, target_variable, well_code)
        
        # Log model
        signature = infer_signature(X_train, final_model.predict(X_train))
        model_name = f"oil-prod-forecast-{well_code.replace('/', '_')}"
        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )
        logging.info(f"Logged and registered model as '{model_name}'.")

# --- 5. Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oil Production Forecasting Script")
    parser.add_argument(
        "--file_path",
        type=str,
        default="Volve production data.xlsx",
        help="Path to the input data file."
    )
    parser.add_argument(
        "--well_code",
        type=str,
        default="NO 15/9-F-1 C",
        help="Well bore code to model."
    )
    parser.add_argument(
        "--target_variable",
        type=str,
        default="BORE_OIL_VOL",
        help="Name of the target variable column."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Oil_Production_Forecasting_Volve_Script",
        help="Name of the MLflow experiment."
    )
    
    args = parser.parse_args()

    # Execute the pipeline
    try:
        df_well_clean = load_and_prepare_data(args.file_path, args.well_code, args.target_variable)
        df_featured = create_features(df_well_clean, args.target_variable)
        train_and_log_model(df_featured, args.well_code, args.target_variable, args.experiment_name)
        logging.info("--- Script finished successfully ---")
    except Exception as e:
        logging.error(f"An error occurred during the script execution: {e}")