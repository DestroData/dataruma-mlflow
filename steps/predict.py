import pandas as pd
import yaml
import os
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error


def predict(X, y, train_model):

    y_pred = train_model.predict(X)

    metrics_result = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': root_mean_squared_error(y, y_pred),
            'msle': root_mean_squared_log_error(y, y_pred),
            'r2': r2_score(y, y_pred)
            }

    return metrics_result

