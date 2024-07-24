import mlflow
import os
import warnings

warnings.filterwarnings('ignore')
def set_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment.
    
    ---------------
    Parameters
    ----------
    - `experiment_name`: Name of the experiment.   
    
    Returns
    -------
    - Experiment ID. 
    ----------------
    
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def register_model_with_client(model_name: str, run_id: str, artifact_path: str) -> None: 
    """
    Register a model with the MLflow tracking server.
    
    --------------------
    Parameters
    ----------
    - `model_name` (str): The name of the model to be registered.
    - `run_id` (str): The unique identifier for the run that generated the model artifact.
    - `artifact_path` (str): The relative path to the model artifact within the run's artifact directory.
    
    Returns
    -------
    - `None`
    
    --------------------
    """
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model(model_name)
    client.create_model_version(name=model_name, source=f"runs:/{run_id}/{artifact_path}")
    
    
def log_model(model, params: dict) -> None:
    mlflow.log_params(params)
    
    # Log model summary
    with open("model_summary.txt", "w") as f:
        f.write(str(model))
    mlflow.log_artifact("model_summary.txt")
    os.remove("model_summary.txt")
    