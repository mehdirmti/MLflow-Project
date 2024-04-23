# -*- coding: utf-8 -*-
"""
Created on Monday April 22 23:00:00 2024

@author: m.rahmati
"""
# -----------------------------------------------------------------------------
# Import needed Librarries ---------------------------------------
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # to devide data into tarin and test subsets
from sklearn.ensemble import RandomForestRegressor # To do Random Forest Regression modleing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import sys

# -----------------------------------------------------------------------------
# Read data -----------------------------------------------------
# -----------------------------------------------------------------------------
dataset = pd.read_csv('Salary_Data.csv')

# -----------------------------------------------------------------------------
# Specify dependent and independent variables --------------------
# -----------------------------------------------------------------------------
X = dataset.iloc[:,:-1].values # take all columns except last one as independent variables
y = dataset.iloc[:,-1].values # take last column as dependent variable

# -----------------------------------------------------------------------------
# Train and Test Data Devision ---------------------------------
# -----------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# -----------------------------------------------------------------------------
# Simple linear regression modeling ------------------------------
# -----------------------------------------------------------------------------
# Define parameters
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 10
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Build the model
regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth, random_state=0)

# Fit the model
regressor.fit(X_train, y_train)

# Do prediction for test data
y_pred = regressor.predict(X_test)

# -----------------------------------------------------------------------------
# Evalaute Results ---------------------------------------------------
# -----------------------------------------------------------------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

(rmse, mae, r2) = eval_metrics(y_test, y_pred)

print("RF Regressor (n_estimators={:f}, max_depth={:f}):".format(n_estimators, max_depth))
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)


# for remote serever only
remote_server_uri = "https://dagshub.com/mehdirmti/MLflow-Tracking.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("My_Model")
#run_name = "run_{}".format(int(sys.argv[3]) if len(sys.argv) > 3 else 0)

#with mlflow.start_run(run_name=run_name) as run:
with mlflow.start_run() as run:
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # Model registry on remote server
    mlflow.sklearn.log_model(regressor, "RF_Regressor", registered_model_name="RandomForestRegressor")