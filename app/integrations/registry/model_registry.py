"""
Model registry interface for Cyrex runtime
Uses deepiri-modelkit registry client
"""
import os
from typing import Optional
from deepiri_modelkit import ModelRegistryClient as BaseRegistryClient


class CyrexModelRegistry:
    """Model registry for Cyrex runtime"""
    
    def __init__(self):
        """Initialize model registry"""
        self.client = BaseRegistryClient(
            registry_type=os.getenv("MODEL_REGISTRY_TYPE", "mlflow"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            s3_endpoint=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            s3_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            s3_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            s3_bucket=os.getenv("S3_BUCKET", "mlflow-artifacts")
        )
    
    def load_model(self, model_name: str, version: Optional[str] = None):
        """Load model from registry"""
        return self.client.get_model(model_name, version)
    
    def download_model(self, model_name: str, version: str, destination: str) -> str:
        """Download model to destination"""
        return self.client.download_model(model_name, version, destination)
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """List available models"""
        return self.client.list_models(model_name)

