import mlflow
from mlflow.tracking import MlflowClient

# mlflow tracking uri
mlflow.set_tracking_uri('http://ec2-3-135-238-101.us-east-2.compute.amazonaws.com:5000/')

def load_model_from_registry(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model_from_registry("yt_chrome_plugin_model", "3")
print("model loaded successfully")