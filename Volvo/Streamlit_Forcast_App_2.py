
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

# --- Configuration ---
# This should match the experiment name in your training script
MLFLOW_EXPERIMENT_NAME = "Oil_Production_Forecasting_Volve"
# This should match the well code you trained the model on
WELL_TO_MODEL = 'NO 15/9-F-1 C'
# Path to the original data, needed to create features for new predictions
DATA_FILE_PATH = "Volve production data.xlsx"
# Target variable
TARGET_VARIABLE = 'BORE_OIL_VOL'

@st.cache_resource
def load_mlflow_model():
    """
    Loads the latest version of the registered MLflow model.
    The resource is cached to avoid reloading on every interaction.
    """
    model_name = f"oil-prod-forecast-{WELL_TO_MODEL.replace('/', '_')}"
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Could not load the model from MLflow registry: {e}")
        st.info("Please ensure the training script has been run successfully and the model is registered.")
        return None

@st.cache_data
def get_latest_data(file_path, well_code):
    """
    Loads the original data to get the most recent records, which are
    needed to calculate lag and rolling features for the next prediction.
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

        df_well.fillna(method='ffill', inplace=True)
        df_well.fillna(method='bfill', inplace=True)

        return df_well.tail(30) # Return last 30 days for feature calculation
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return None

def create_features_for_prediction(user_inputs, historical_data):
    """
    Takes user inputs for the next day and combines them with historical
    data to generate the full feature set required by the model.
    """
    # Get the last known date from historical data
    last_date = historical_data.index.max()
    next_date = last_date + pd.DateOffset(days=1)

    # Create a new row for the prediction day
    # Start by copying the last known data point
    new_row = historical_data.iloc[[-1]].copy()
    new_row.index = [next_date]

    # Update the row with the user's inputs
    for key, value in user_inputs.items():
        if key in new_row.columns:
            new_row[key] = value

    # Combine with historical data to calculate lags/rolls
    combined_df = pd.concat([historical_data, new_row])

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
latest_data = get_latest_data(DATA_FILE_PATH, WELL_TO_MODEL)

if model is None or latest_data is None:
    st.warning("Application cannot start because the model or data is unavailable.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Forecast Inputs")
        st.write(f"Enter the expected operational parameters for the next day for well **{WELL_TO_MODEL}**.")

        # Get the most recent values as defaults for the input fields
        last_known_values = latest_data.iloc[-1]

        # --- MODIFIED SECTION STARTS HERE ---
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
        # --- MODIFIED SECTION ENDS HERE ---


    with col2:
        st.header("Forecast Result")
        if st.button("Forecast Next Day's Production", type="primary"):
            with st.spinner("Generating features and running prediction..."):
                # Create the feature vector for the prediction
                features_df = create_features_for_prediction(user_inputs, latest_data)

                # Make prediction
                prediction = model.predict(features_df)[0]

            st.success("Forecast Generated!")
            st.metric(
                label=f"Predicted Oil Production for { (latest_data.index.max() + pd.DateOffset(days=1)).strftime('%Y-%m-%d') }",
                value=f"{prediction:,.2f} Sm³",
                help="This is a model-based estimation and may differ from actual production."
            )

    st.header("Historical Data")
    st.write(f"Showing the last 30 days of production data for well **{WELL_TO_MODEL}** used to generate features.")
    st.line_chart(latest_data[TARGET_VARIABLE])
    st.dataframe(latest_data)