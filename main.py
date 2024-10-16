import logging
from steps.ingest import ingest
from steps.training import training
from steps.predict import predict
from steps.others_functions import separate_target
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def train_with_mlflow(data_version, model, params, tags, exp_name):

    # Load data
    trainset, testset = ingest(data_version)
    logging.info("Data ingestion completed successfully")

    # Prepare and train model
    x_train, y_train = separate_target(trainset)
    train_model = training(x_train, y_train, model, params)
    logging.info("Model training completed successfully")

    # Evaluate model
    x_test, y_test = separate_target(testset)
    metrics = predict(x_test, y_test, train_model)
    logging.info("Model evaluation completed successfully")



    client = mlflow.MlflowClient()
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    experiment = client.get_experiment_by_name(exp_name)

    if experiment is None:
        experiment_id = client.create_experiment(name=exp_name, tags=tags)

    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train, train_model.predict(x_train))

        mlflow.sklearn.log_model(
        sk_model=train_model,
        artifact_path="",
        signature=signature,
        input_example=x_train,
        registered_model_name="registered_model_name",
        )

        logging.info("MLflow tracking completed successfully")

    print(f"Tags: {experiment.tags}")

if __name__ == "__main__":

    experiment_description= (
        "expérimentation de test"
    )

    tags= {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
    }

    params = {'n_estimators':450,
    'learning_rate':0.1,
    'max_depth':3,
    'subsample':1.0,
    'min_samples_split':2,
    'min_samples_leaf':1,
    'max_features':None,
    'random_state':42
    }

    exp_name = f'Dataruma'

    train_with_mlflow('v1', GradientBoostingRegressor, params, tags, exp_name)

