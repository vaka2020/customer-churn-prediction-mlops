import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from .config import (
    PREPROCESSOR_PATH, MODEL_PATH, FEATURE_NAMES_PATH, METADATA_PATH
)

class ModelManager:
    """
    Manages loading and using the trained model and preprocessor.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.metadata = None
        self.original_feature_order = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """
        Load model, preprocessor, feature names, and metadata.
        """
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(PREPROCESSOR_PATH, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self.original_feature_order = [
                'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod',
                'is_new_customer', 'is_loyal_customer', 'tenure_group',
                'avg_monthly_charge', 'high_monthly_charge', 'total_services',
                'has_protection', 'has_support', 'is_monthly_contract',
                'automatic_payment', 'risky_payment', 'family_size',
                'senior_with_family', 'tenure_contract_interaction', 'charges_per_service'
            ]
            
            print("All model artifacts loaded successfully")
            
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering as training.
        """
        df = df.copy()
        
        df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
        df['is_loyal_customer'] = (df['tenure'] >= 48).astype(int)
        
        bins = [0, 12, 24, 48, float('inf')]
        labels = ['0-1yr', '1-2yr', '2-4yr', '4+yr']
        df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
        
        df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['high_monthly_charge'] = (df['MonthlyCharges'] > 70).astype(int)
        
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['total_services'] = sum((df[col] == 'Yes').astype(int) for col in service_cols if col in df.columns)
        
        df['has_protection'] = ((df['OnlineSecurity'] == 'Yes') | 
                               (df['DeviceProtection'] == 'Yes')).astype(int)
        df['has_support'] = (df['TechSupport'] == 'Yes').astype(int)
        
        df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['automatic_payment'] = ((df['PaymentMethod'] == 'Bank transfer (automatic)') | 
                                   (df['PaymentMethod'] == 'Credit card (automatic)')).astype(int)
        df['risky_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        
        df['family_size'] = ((df['Partner'] == 'Yes').astype(int) + 
                            (df['Dependents'] == 'Yes').astype(int))
        df['senior_with_family'] = (df['SeniorCitizen'] * df['family_size'])
        
        df['tenure_contract_interaction'] = df['tenure'] * df['is_monthly_contract']
        df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
        
        return df
    
    def preprocess_input(self, customer_data: Dict) -> np.ndarray:
        """
        Preprocess a single customer input.
        """
        df = pd.DataFrame([customer_data])
        df = self.engineer_features(df)
        
        df = df[self.original_feature_order]
        
        X_processed = self.preprocessor.transform(df)
        
        return X_processed
    
    def predict(self, customer_data: Dict) -> Tuple[int, float]:
        """
        Make prediction for a single customer.
        Returns: (prediction, probability)
        """
        X = self.preprocess_input(customer_data)
        
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0, 1])
        
        return prediction, probability
    
    def predict_batch(self, customers: List[Dict]) -> List[Tuple[int, float]]:
        """
        Make predictions for multiple customers.
        Returns: List of (prediction, probability) tuples
        """
        df = pd.DataFrame(customers)
        df = self.engineer_features(df)
        
        df = df[self.original_feature_order]
        
        X_processed = self.preprocessor.transform(df)
        
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return list(zip(predictions.astype(int), probabilities.astype(float)))
    
    @staticmethod
    def get_risk_level(probability: float) -> str:
        """
        Categorize churn probability into risk levels.
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

model_manager = ModelManager()