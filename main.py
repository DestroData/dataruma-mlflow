import pandas as pd
import optuna
import mlflow
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

from figures import (plot_correlations,
                    plot_feature_importance,
                    plot_residuals,
                    plot_residuals_hist)

from utils import (separate_target,
                            get_or_create_experiment,
                            cal_all_metrics, ingest)


# to control logs
optuna.logging.set_verbosity(optuna.logging.ERROR)


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values. Not for every run.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    model = GradientBoostingRegressor(**params)
    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)
    metrics = cal_all_metrics(test_y, pred_y)
    rmse = root_mean_squared_error(test_y, pred_y)

    with mlflow.start_run(experiment_id=experiment_id, run_name=child_run_name, nested=True):

        mlflow.log_params(params)
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)


        return rmse

def start_optimization(main_run_name, n_trials):

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    study = optuna.create_study(direction="minimize")

    with mlflow.start_run(experiment_id=experiment_id, run_name=main_run_name):

        study.optimize(objective , n_trials=n_trials, callbacks=[champion_callback])

        # create usefull variables of the best model
        best_params = study.best_params
        best_model = GradientBoostingRegressor(**best_params, random_state=42)
        best_model.fit(train_X, train_y)
        pred_y = best_model.predict(test_X)
        metrics = cal_all_metrics(test_y, pred_y)
        signature = infer_signature(test_X, pred_y)
        dataset = mlflow.data.from_pandas(trainset)

        # creat graph for analysis
        corelation_plot = plot_correlations(data, "nombre_pizza_soir")
        importances = plot_feature_importance(best_model, train_X)
        residuals = plot_residuals(pred_y, test_y)
        residuals_hist = plot_residuals_hist(pred_y, test_y)

        #logs graph in mlflow
        mlflow.log_figure(figure=corelation_plot, artifact_file="plots/correlation_plot.png")
        mlflow.log_figure(figure=importances, artifact_file="plots/feature_importances.png")
        mlflow.log_figure(figure=residuals, artifact_file="plots/residuals.png")
        mlflow.log_figure(figure=residuals_hist, artifact_file="plots/residuals_histogramme.png")

        # logs tags, params and metrics in mlflow
        mlflow.log_params(study.best_params)
        mlflow.set_tags(tags=tags)
        mlflow.log_metrics(metrics)
        mlflow.log_input(dataset, context="training")

        #log models in mlflow
        mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="",
        signature=signature,
        input_example=train_X,
        registered_model_name="XGB_from_sk",
        )


if __name__ == "__main__":

#################################### parameters #########################################

    experiment_description=(
        "the goal of the experience is to optimze "
        "the hyperparameters of the Gradient Boosting Regressors"""
    )

    tags={
    "project": "dataruma",
    "optimizer_engine": "optuna",
    "model_family": "xgboost",
    "mlflow.note.content": experiment_description,
    }

    data_version = 'v1'
    exp_name = 'dataruma'
    main_run_name = "main_run_v1"
    child_run_name = "child_run"
    n_trials = 250

##########################################################################################

    experiment_id = get_or_create_experiment(exp_name, tags)
    trainset, testset = ingest(data_version)
    data = pd.concat([trainset, testset], ignore_index=True)
    train_X, train_y, = separate_target(trainset)
    test_X, test_y = separate_target(testset)

    start_optimization(main_run_name, n_trials)

