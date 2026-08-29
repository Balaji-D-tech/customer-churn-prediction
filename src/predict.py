"""
Prediction Inference Module for Customer Churn Application.
Loads exported final pipeline artifact and provides user-friendly inference API.
"""

import os
import json
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = os.path.join("models", "final_model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")


class ChurnPredictor:
    """
    Singleton Wrapper for Customer Churn Model Inference.
    """

    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python src/train.py' first."
            )

        self.pipeline = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.threshold = self.metadata.get("optimal_threshold", 0.5)
            self.model_name = self.metadata.get("best_model_name", "Tuned Classifier")
        else:
            self.metadata = {}
            self.threshold = 0.5
            self.model_name = "Tuned Classifier"

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts churn status for a single customer record dictionary.

        Args:
            customer_data: Dictionary containing key-value pairs matching dataset columns.

        Returns:
            Dictionary with prediction label, numeric probability, risk tier, and confidence score.
        """
        df_input = pd.DataFrame([customer_data])
        return self.predict_dataframe(df_input)[0]

    def predict_dataframe(self, df_input: pd.DataFrame) -> list:
        """
        Predicts churn status for a DataFrame of customer records.
        """
        # Calculate positive class probability (Churn=1)
        probabilities = self.pipeline.predict_proba(df_input)[:, 1]

        results = []
        for prob in probabilities:
            is_churn = prob >= self.threshold
            churn_label = "Churn (Yes)" if is_churn else "No Churn (No)"

            # Assign risk level based on probability
            if prob < 0.30:
                risk_level = "Low Risk"
                risk_color = "green"
            elif prob < 0.60:
                risk_level = "Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "High Risk"
                risk_color = "red"

            results.append(
                {
                    "prediction": churn_label,
                    "churn_binary": int(is_churn),
                    "churn_probability": float(prob),
                    "churn_percentage": f"{prob * 100:.1f}%",
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "model_used": self.model_name,
                    "threshold_applied": self.threshold,
                }
            )

        return results


def get_sample_customer() -> Dict[str, Any]:
    """
    Returns a sample high-risk customer profile for quick UI testing.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.70,
        "TotalCharges": 171.40,
    }


if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample = get_sample_customer()
    res = predictor.predict_single(sample)
    print("Sample Churn Prediction Result:")
    print(json.dumps(res, indent=4))
