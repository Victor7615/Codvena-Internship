# Data Science Internship Projects

Welcome to the repository for my data science internship projects. This document outlines two major projects completed during the internship:

1.  **Predictive Maintenance**: A classification project to predict machine failure based on sensor data.
2.  **Oil Production Forecasting**: A time-series analysis project to forecast oil production for a specific well, complete with a deployment-ready script and a Streamlit web application.

These projects were developed to meet the requirements outlined in the **Data Science Task List**, demonstrating a comprehensive application of data science principles from basic data handling to advanced modeling and deployment.

---
## Alignment with Data Science Task List

The task list requires the completion of any three tasks. The two projects presented here collectively fulfill requirements from all three levels, including **Data Cleaning and Preprocessing (Level 1)**, **Exploratory Data Analysis (Level 1)**, **Predictive Modeling (Classification - Level 2)**, and **Time Series Analysis (Advanced - Level 3)**.

### Project 1: Predictive Maintenance (Classification)

This project focuses on predicting machine failure using a maintenance dataset. It aligns with the following tasks from the list:

#### ✅ **Level 1, Task 2: Data Cleaning and Preprocessing**
This task involves cleaning and preparing a raw dataset for analysis.

* **Objective Met**: The `Maintainence.ipynb` notebook accomplishes this through several steps:
    * **Handling Missing Data**: The initial data inspection confirms there are no missing values (`df.isnull().sum()`).
    * **Feature Scaling**: A `StandardScaler` is applied to all numerical features to normalize their scale, which is crucial for models like Logistic Regression.
    * **Categorical Variable Conversion**: The `Type` column, which is a categorical feature, is converted into a numerical format using `OneHotEncoder`.
    * **Implementation**: These steps are encapsulated within a `ColumnTransformer` and a `Pipeline` for a robust and reproducible preprocessing workflow.

#### ✅ **Level 1, Task 3: Exploratory Data Analysis (EDA)**
This task requires a deep dive into the data to understand its structure, identify trends, and visualize relationships.

* **Objective Met**: The notebook features a comprehensive EDA section that includes:
    * **Summary Statistics**: `df.describe()` is used to compute the mean, median, standard deviation, and other key statistics for all numerical features.
    * **Data Visualization**:
        * **Histograms**: The distribution of each numerical feature is visualized to understand its underlying pattern.
        * **Box Plots**: The relationship between each numerical feature and the target variable (Machine Failure) is examined using box plots, highlighting how features like `Torque` and `Rotational speed` differ between failure and non-failure cases.
        * **Count Plots**: The distribution of the target variable and categorical features like `Type` are visualized.
    * **Correlation Analysis**: A `heatmap` of the correlation matrix is generated to identify relationships between numerical features, revealing strong correlations between temperature metrics and between torque and speed.
    * **Insight Generation**: Each step of the EDA is accompanied by an "Insight" section that summarizes the key findings.

#### ✅ **Level 2, Task 2: Classification**
This task focuses on building and evaluating a model to predict a categorical outcome.

* **Objective Met**: The project successfully builds and evaluates multiple classification models:
    * **Model Training**: The notebook trains several classifiers, including **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**.
    * **Evaluation Metrics**: The performance of each model is rigorously evaluated using standard classification metrics:
        * **Accuracy**: Overall correctness of the model.
        * **Precision**: The ratio of true positives to all positive predictions.
        * **Recall**: The ability of the model to find all the positive samples.
        * **F1-Score**: The harmonic mean of precision and recall.
        * **ROC AUC Score**: A measure of the model's ability to distinguish between classes.
        * A **Confusion Matrix** is also generated and logged for each model to visualize its performance in detail.
    * **Model Comparison and Tuning**: `GridSearchCV` is used to tune the hyperparameters for each model, ensuring optimal performance. The results and best parameters are logged using **MLflow**, allowing for easy comparison.

---
### Project 2: Oil Production Forecasting (Time Series Analysis)

This project involves analyzing time-series data from the Volve oil field to forecast future oil production. It is a complete workflow, from data processing to model deployment with a Streamlit app.

#### ✅ **Level 1, Task 2: Data Cleaning and Preprocessing**
This task is a foundational step in preparing the time-series data for modeling.

* **Objective Met**: The `oil_forecast_script.py` script performs these steps within the `load_and_prepare_data` function:
    * **Handling Missing Data**: After converting columns to numeric types, missing values are imputed using a forward-fill and back-fill strategy (`ffill` and `bfill`), which is a standard technique for time-series data.
    * **Data Structuring**: The script loads data from an Excel file, filters it for a specific well, converts the date column to a datetime index, and sorts the data chronologically.

#### ✅ **Level 3, Task 1: Time Series Analysis**
This advanced task involves analyzing and modeling time-series data to make future forecasts.

* **Objective Met**: The project directly addresses all objectives of this task:
    * **Feature Engineering for Time Series**: The `create_features` function engineers features specifically for time-series forecasting:
        * **Trend and Seasonality Components**: Time-based features like `year`, `month`, and `dayofyear` are created to help the model capture trends and seasonality.
        * **Lag Features**: Production values from previous time steps (e.g., 1, 7, and 14 days ago) are included as features.
        * **Moving Averages**: Rolling window features (mean and standard deviation over 7, 14, and 30 days) are calculated to capture recent trends.
    * **Forecasting Model**: A **LightGBM Regressor**, a powerful gradient-boosting model suitable for time-series forecasting, is built and trained.
    * **Model Evaluation**: The model's performance is evaluated using metrics appropriate for time-series forecasting:
        * **Mean Absolute Error (MAE)**.
        * **Root Mean Squared Error (RMSE)**.
    * **Visualization**: The script generates and logs plots of **actual vs. predicted** values over time, providing a clear visual assessment of the forecast accuracy.

#### ✅ **Bonus: Model Deployment and Application**
Though not an explicit task, the included `Streamlit_Forcast_App_3.py` script demonstrates an advanced, practical application of the trained model.

* **Objective Met**: It creates an interactive web application that:
    * Loads the latest version of the trained model from the **MLflow Model Registry**.
    * Allows a user to input operational parameters for the next day.
    * Generates and displays a forecast for future oil production based on these inputs.
    * Includes a historical data viewer for context.