import pandas as pd
import mlflow
from sklearn.metrics import root_mean_squared_error, median_absolute_error, root_mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split


def ingest(data_version):

    data = pd.read_csv(f'data/data-{data_version}.csv', index_col=0)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data, test_data


def separate_target(data):
    """
    separate target and features
    """

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return X, y


def get_or_create_experiment(experiment_name, tags):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return mlflow.create_experiment(experiment_name, tags=tags)
    else:
        return experiment.experiment_id


def cal_all_metrics(test_y, pred_y):

    metrics = {
        'rmse' : root_mean_squared_error(test_y, pred_y),
        'median_ae' : median_absolute_error(test_y, pred_y),
        'rmsle' : root_mean_squared_log_error(test_y, pred_y),
        'r2' : r2_score(test_y, pred_y)
    }

    return metrics






