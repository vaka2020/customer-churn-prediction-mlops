import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("\nTesting /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    print("\nTesting /model/info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_high_risk():
    print("\nTesting /predict endpoint (High Risk Customer)...")
    
    customer = {
        "SeniorCitizen": 0,
        "tenure": 1,
        "MonthlyCharges": 85.50,
        "TotalCharges": 85.50,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_low_risk():
    print("\nTesting /predict endpoint (Low Risk Customer)...")
    
    customer = {
        "SeniorCitizen": 0,
        "tenure": 60,
        "MonthlyCharges": 45.00,
        "TotalCharges": 2700.00,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    print("\nTesting /predict/batch endpoint...")
    
    batch_request = {
        "customers": [
            {
                "SeniorCitizen": 0,
                "tenure": 1,
                "MonthlyCharges": 85.50,
                "TotalCharges": 85.50,
                "gender": "Female",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            },
            {
                "SeniorCitizen": 0,
                "tenure": 60,
                "MonthlyCharges": 45.00,
                "TotalCharges": 2700.00,
                "gender": "Male",
                "Partner": "Yes",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_request)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("="*70)
    print("API Testing Suite")
    print("="*70)
    
    try:
        test_health()
        test_model_info()
        test_predict_high_risk()
        test_predict_low_risk()
        test_batch_prediction()
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API at", BASE_URL)
        print("Make sure the API is running:")
        print("  python -m uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\nERROR: {e}")