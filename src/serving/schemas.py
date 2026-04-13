from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class CustomerInput(BaseModel):
    """
    Input schema for a single customer prediction request.
    All fields match the features expected by the model.
    """
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen (0 or 1)")
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    
    gender: str = Field(..., description="Customer gender (Male/Female)")
    Partner: str = Field(..., description="Whether customer has a partner (Yes/No)")
    Dependents: str = Field(..., description="Whether customer has dependents (Yes/No)")
    PhoneService: str = Field(..., description="Whether customer has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Whether customer has multiple lines (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Type of internet service (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Whether customer has online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Whether customer has online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Whether customer has device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Whether customer has tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Whether customer has streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies (Yes/No/No internet service)")
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be Male or Female')
        return v
    
    @validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError(f'must be Yes or No')
        return v
    
    @validator('Contract')
    def validate_contract(cls, v):
        valid = ['Month-to-month', 'One year', 'Two year']
        if v not in valid:
            raise ValueError(f'Contract must be one of {valid}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }

class PredictionResponse(BaseModel):
    """
    Response schema for prediction results.
    """
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    churn_prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Risk category (Low/Medium/High)")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST-12345",
                "churn_prediction": 1,
                "churn_probability": 0.78,
                "risk_level": "High",
                "timestamp": "2026-04-13T10:30:00"
            }
        }

class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch predictions.
    """
    customers: List[CustomerInput] = Field(..., description="List of customers to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "SeniorCitizen": 0,
                        "tenure": 12,
                        "MonthlyCharges": 65.50,
                        "TotalCharges": 786.00,
                        "gender": "Female",
                        "Partner": "Yes",
                        "Dependents": "No",
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check"
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelInfoResponse(BaseModel):
    """
    Response schema for model information endpoint.
    """
    model_name: str
    model_version: str
    model_type: str
    features_count: int
    training_samples: int
    test_accuracy: float
    test_recall: float
    test_f1: float
    test_roc_auc: float