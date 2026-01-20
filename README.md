# Random Forest Regression Analysis

This directory contains an implementation of **Random Forest Regression**, an ensemble learning method that uses multiple decision trees to provide highly accurate predictions for non-linear data.

## Dataset Overview

The model uses the `Position_Salaries.csv` dataset, which includes:

- **Features**:
  - `Level`: Numerical level of the position (1 to 10).
- **Target**:
  - `Salary`: The annual salary associated with the position level.

## Implementation Steps

The implementation follows these key steps:

1.  **Data Preprocessing**:
    - Importing libraries (`numpy`, `matplotlib`, `pandas`).
    - Note: Random Forest Regression is not sensitive to outliers and does not require feature scaling for this dataset.
2.  **Model Training**:
    - Used `sklearn.ensemble.RandomForestRegressor` with `n_estimators=100`.
    - Trained on the entire dataset to capture the full range of salary levels.
3.  **Prediction**:
    - Predicted the salary for a specific level (6.5).
    - **Result**: $158,300.
4.  **Visualization**:
    - Plotted results at high resolution (0.01 step size) to show the ensemble's sophisticated step-function approach.

## Results

Random Forest Regression provides more precise results for non-linear trends than a single decision tree.

- By averaging 100 different trees, the model achieves better generalization.
- The prediction for level 6.5 ($158.3k) is a more refined estimate than the $150k baseline from a single tree, closer to the expected non-linear curve.
