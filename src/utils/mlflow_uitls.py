import mlflow

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


def register_model_with_client(model_name: str, run_id: str, artifact_path: str):
    """
    Register a model.

    :param model_name: Name of the model.
    :param run_id: Run ID.  
    :param artifact_path: Artifact path.

    :return: None.
    """
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model(model_name)
    client.create_model_version(name=model_name, source=f"runs:/{run_id}/{artifact_path}")