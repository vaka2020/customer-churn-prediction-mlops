from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import json

from .schemas import (
    CustomerInput, PredictionResponse, BatchPredictionRequest,
    HealthResponse, ModelInfoResponse
)
from .model_loader import model_manager
from .config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    MODEL_NAME, MODEL_VERSION, MODEL_TYPE, METADATA_PATH
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Customer Churn Prediction API",
        "version": API_VERSION,
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and model health status.
    """
    model_loaded = model_manager.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get detailed information about the loaded model.
    """
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        features_count = metadata.get('feature_count', 48)
        training_samples = metadata.get('train_samples', 5634)
        
    except Exception as e:
        features_count = 48
        training_samples = 5634
    
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        features_count=features_count,
        training_samples=training_samples,
        test_accuracy=0.7438,
        test_recall=0.7995,
        test_f1=0.6236,
        test_roc_auc=0.8436
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput, customer_id: str = None):
    """
    Predict churn for a single customer.
    """
    try:
        customer_dict = customer.dict()
        
        prediction, probability = model_manager.predict(customer_dict)
        
        risk_level = model_manager.get_risk_level(probability)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers in a single request.
    """
    try:
        customers_dict = [customer.dict() for customer in request.customers]
        
        results = model_manager.predict_batch(customers_dict)
        
        predictions = []
        for i, (pred, prob) in enumerate(results):
            predictions.append({
                "customer_index": i,
                "churn_prediction": int(pred),
                "churn_probability": round(float(prob), 4),
                "risk_level": model_manager.get_risk_level(prob)
            })
        
        return {
            "total_customers": len(predictions),
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)