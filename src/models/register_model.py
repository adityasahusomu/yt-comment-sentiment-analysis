import json
import mlflow
import logging

# point to the same tracking server
mlflow.set_tracking_uri("http://ec2-3-135-238-101.us-east-2.compute.amazonaws.com:5000/")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from experiment_info.json"""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):

    try:
        run_id = model_info["run_id"]
        artifact_subpath = model_info["artifact_subpath"]

        model_uri = f"runs:/{run_id}/{artifact_subpath}"
        logger.debug(f"Registering model from {model_uri}")

        model_version = mlflow.register_model(model_uri, model_name)

        # move it to Staging automatically
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(
            f"Model {model_name} version {model_version.version} "
            f"registered and transitioned to Staging"
        )

        return model_version

    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        raise


def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)

    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()