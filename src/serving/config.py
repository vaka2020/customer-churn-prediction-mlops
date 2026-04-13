from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
METADATA_PATH = MODEL_DIR / "preprocessing_metadata.json"

MODEL_NAME = "Customer Churn Prediction"
MODEL_VERSION = "1.0.0"
MODEL_TYPE = "Logistic Regression"

API_TITLE = "Customer Churn Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Production API for predicting customer churn in telecommunications.

## Features
- Single customer prediction
- Batch predictions
- Model health monitoring
- Automatic input validation

## Model Details
- Algorithm: Logistic Regression with L2 regularization
- Training samples: 5,634
- Test accuracy: 74.4%
- Test recall: 79.9%
- Features: 48 (18 numerical + 30 one-hot encoded)
"""
