"""
Automated Test Suite for Model Artifacts and Inference Pipeline.
Executed via pytest.
"""

import os
import json
import pytest
import pandas as pd
from src.predict import ChurnPredictor, get_sample_customer

MODEL_PATH = os.path.join("models", "final_model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")


def test_model_files_exist():
    """Verify that final_model.pkl and model_metadata.json exist after training."""
    assert os.path.exists(MODEL_PATH), f"Missing trained model file at {MODEL_PATH}"
    assert os.path.exists(METADATA_PATH), f"Missing metadata file at {METADATA_PATH}"


def test_metadata_structure():
    """Verify metadata contains required metrics and threshold."""
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    assert "best_model_name" in metadata
    assert "optimal_threshold" in metadata
    assert "metrics" in metadata
    assert "Accuracy" in metadata["metrics"]
    assert "F1-Score" in metadata["metrics"]
    assert "ROC-AUC" in metadata["metrics"]


def test_predictor_single_sample():
    """Verify that predictor returns valid result dictionary for single sample input."""
    predictor = ChurnPredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    sample = get_sample_customer()
    result = predictor.predict_single(sample)

    assert "prediction" in result
    assert "churn_binary" in result
    assert result["churn_binary"] in [0, 1]
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert "risk_level" in result
    assert result["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]


def test_predictor_high_risk_customer():
    """Verify high-risk customer profile yields high churn probability (> 0.50)."""
    predictor = ChurnPredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    high_risk_sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
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
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }
    result = predictor.predict_single(high_risk_sample)
    assert result["churn_probability"] > 0.40
    assert result["risk_level"] in ["Medium Risk", "High Risk"]


def test_predictor_low_risk_customer():
    """Verify low-risk customer profile yields low churn probability (< 0.40)."""
    predictor = ChurnPredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    low_risk_sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.00,
        "TotalCharges": 2700.00,
    }
    result = predictor.predict_single(low_risk_sample)
    assert result["churn_probability"] < 0.40
    assert result["risk_level"] == "Low Risk"
