"""Part 2: Streamlit Forecasting App
This script sets up a Streamlit application to forecast oil production
using a pre-trained MLflow model. It allows users to input expected
operational parameters for the next day and generates a forecast based
on the latest historical data.
It uses the MLflow model registry to load the latest version of the model
and provides a user-friendly interface for inputting parameters and viewing
the forecast result.
It also includes caching for performance optimization and error handling
to ensure the application runs smoothly.
This script is designed to be run in a Streamlit environment.
Make sure to have the necessary libraries installed and the MLflow server
running with the model registered.
"""

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import time
from datetime import timedelta
import os

# --- Configuration ---
# This should match the experiment name in your training script
MLFLOW_EXPERIMENT_NAME = "Oil_Production_Forecasting_Volve"
# This should match the well code you trained the model on
WELL_TO_MODEL = 'NO 15/9-F-1 C'
# Path to the original data, needed to create features for new predictions
# NOTE: This path should be relative to the repository, not a local drive.
# Ensure this file is committed to your GitHub repository.
DATA_FILE_PATH = "Volve production data.xlsx"
# Target variable
TARGET_VARIABLE = 'BORE_OIL_VOL'

# We are bypassing the MLflow registry's stored file path by directly
# loading from the artifact folder, so we do not need to set these.
# os.environ["MLFLOW_TRACKING_URI"] = "mlruns/"
# os.environ["MLFLOW_TRACKING_SERVER_ARTIFACT_URI"] = "mlruns/"

@st.cache_resource
def load_mlflow_model():
    """
    Loads the latest version of the registered MLflow model by
    directly referencing its artifact path in the repository.
    The resource is cached to avoid reloading on every interaction.
    """
    # The direct relative path to the model's artifact directory.
    # This bypasses the MLflow registry's hardcoded paths.
    # It assumes the mlruns folder is committed to the repo.
    # The path is constructed from your specific experiment and run ID.
    model_path = "mlruns/581584146196918816/bc79f9adb1c3474cb17952947f37a4ba/artifacts/model"
    try:
        st.info(f"Attempting to load model from: {model_path}")
        # Load the model directly using the file path URI.
        model = mlflow.pyfunc.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load the model from MLflow artifacts: {e}")
        st.info("Please verify the model artifact path and ensure the `mlruns` folder is committed to your repository.")
        return None

@st.cache_data
def load_historical_data(file_path, well_code):
    """
    Loads the full historical data for a well to be used for display
    and for feature calculation.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='Daily Production Data')
        df_well = df[df['WELL_BORE_CODE'] == well_code].copy()
        df_well['DATEPRD'] = pd.to_datetime(df_well['DATEPRD'])
        df_well.set_index('DATEPRD', inplace=True)
        df_well.sort_index(inplace=True)

        # Select and clean relevant columns as in the training script
        relevant_cols = [
            'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P',
            'AVG_WHT_P', 'BORE_GAS_VOL', 'BORE_WAT_VOL', TARGET_VARIABLE
        ]
        df_well = df_well[relevant_cols]
        for col in df_well.columns:
            df_well[col] = pd.to_numeric(df_well[col], errors='coerce')

        df_well.ffill(inplace=True)
        df_well.bfill(inplace=True)

        return df_well
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return None

def create_features_for_prediction(user_inputs, historical_data_for_features):
    """
    Takes user inputs for the next day and combines them with historical
    data to generate the full feature set required by the model.
    """
    # Get the last known date from historical data
    last_date = historical_data_for_features.index.max()
    next_date = last_date + pd.DateOffset(days=1)

    # Create a new row for the prediction day
    # Start by copying the last known data point
    new_row = historical_data_for_features.iloc[[-1]].copy()
    new_row.index = [next_date]

    # Update the row with the user's inputs
    for key, value in user_inputs.items():
        if key in new_row.columns:
            new_row[key] = value

    # Combine with historical data to calculate lags/rolls
    combined_df = pd.concat([historical_data_for_features, new_row])

    # --- Recreate features exactly as in training ---
    # Time-based features
    combined_df['year'] = combined_df.index.year
    combined_df['month'] = combined_df.index.month
    combined_df['dayofyear'] = combined_df.index.dayofyear
    combined_df['dayofweek'] = combined_df.index.dayofweek

    # Lag features
    for lag in [1, 7, 14]:
        combined_df[f'{TARGET_VARIABLE}_lag_{lag}'] = combined_df[TARGET_VARIABLE].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        combined_df[f'{TARGET_VARIABLE}_roll_mean_{window}'] = combined_df[TARGET_VARIABLE].rolling(window=window).mean()
        combined_df[f'{TARGET_VARIABLE}_roll_std_{window}'] = combined_df[TARGET_VARIABLE].rolling(window=window).std()

    # Return only the last row, which has all the features for the prediction
    return combined_df.iloc[[-1]].drop(columns=[TARGET_VARIABLE])


# --- Streamlit App UI ---
st.set_page_config(page_title="Oil Production Forecast", layout="wide")
st.title("Future Oil Production Forecast 🛢️")

# Load model and data
model = load_mlflow_model()
historical_data = load_historical_data(DATA_FILE_PATH, WELL_TO_MODEL)

if model is None or historical_data is None or historical_data.empty:
    st.warning("Application cannot start because the model or data is unavailable. Please check the logs above for more details.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Forecast Inputs")
        st.write(f"Enter the expected operational parameters for the next day for well **{WELL_TO_MODEL}**.")

        # Get the most recent values as defaults for the input fields
        last_known_values = historical_data.iloc[-1]

        user_inputs = {
            # Primary inputs that are frequently changed
            "ON_STREAM_HRS": st.slider("On Stream Hours", 0.0, 24.0, float(last_known_values['ON_STREAM_HRS']), 0.5),
            "AVG_CHOKE_SIZE_P": st.number_input("Average Choke Size (%)", value=float(last_known_values['AVG_CHOKE_SIZE_P'])),
            "AVG_WHP_P": st.number_input("Average Wellhead Pressure", value=float(last_known_values['AVG_WHP_P'])),
            "AVG_WHT_P": st.number_input("Average Wellhead Temperature", value=float(last_known_values['AVG_WHT_P'])),
        }

        # Use an expander for other parameters that have default values
        with st.expander("Show/Hide Other Operational Parameters"):
            user_inputs["AVG_DOWNHOLE_PRESSURE"] = st.number_input(
                "Average Downhole Pressure",
                value=float(last_known_values['AVG_DOWNHOLE_PRESSURE'])
            )
            user_inputs["AVG_DOWNHOLE_TEMPERATURE"] = st.number_input(
                "Average Downhole Temperature",
                value=float(last_known_values['AVG_DOWNHOLE_TEMPERATURE'])
            )
            user_inputs["AVG_DP_TUBING"] = st.number_input(
                "Average DP Tubing",
                value=float(last_known_values['AVG_DP_TUBING'])
            )
            user_inputs["AVG_ANNULUS_PRESS"] = st.number_input(
                "Average Annulus Pressure",
                value=float(last_known_values['AVG_ANNULUS_PRESS'])
            )
            user_inputs["BORE_GAS_VOL"] = st.number_input(
                "Bore Gas Volume (Sm³)",
                value=float(last_known_values['BORE_GAS_VOL'])
            )
            user_inputs["BORE_WAT_VOL"] = st.number_input(
                "Bore Water Volume (Sm³)",
                value=float(last_known_values['BORE_WAT_VOL'])
            )

    with col2:
        st.header("Forecast Result")
        if st.button("Forecast Next Day's Production", type="primary"):
            with st.spinner("Generating features and running prediction..."):
                # Always use the last 30 days of the full dataset for feature engineering
                latest_data_for_features = historical_data.tail(30)

                # Create the feature vector for the prediction
                features_df = create_features_for_prediction(user_inputs, latest_data_for_features)

                # Make prediction
                prediction = model.predict(features_df)[0]

            st.success("Forecast Generated!")
            st.metric(
                label=f"Predicted Oil Production for { (historical_data.index.max() + pd.DateOffset(days=1)).strftime('%Y-%m-%d') }",
                value=f"{prediction:,.2f} Sm³",
                help="This is a model-based estimation and may differ from actual production."
            )

    st.header("Historical Data Viewer")
    st.write(f"Select a date range to view historical production data for well **{WELL_TO_MODEL}**.")

    # --- DATE RANGE SELECTOR ---
    min_date = historical_data.index.min().date()
    max_date = historical_data.index.max().date()

    # Set default range to the last 30 days
    default_start_date = max_date - timedelta(days=29)

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", value=default_start_date, min_value=min_date, max_value=max_date)
    with date_col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
    else:
        # Filter data based on user selection
        filtered_data = historical_data.loc[start_date:end_date]

        st.line_chart(filtered_data[TARGET_VARIABLE])
        st.dataframe(filtered_data)
