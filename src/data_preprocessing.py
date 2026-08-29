"""
Data Preprocessing Module for Customer Churn Prediction.
Handles data cleaning, type conversion, missing value imputation,
feature scaling, and encoding using Scikit-Learn ColumnTransformer.
"""

import os
import sys
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Paths
RAW_DATA_PATH = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRANSFORMER_PATH = os.path.join("models", "preprocessor.pkl")


def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV data into a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        from src.data_downloader import download_dataset
        download_dataset(file_path)
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded raw dataset with shape: {df.shape}")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning:
    1. Converts TotalCharges to numeric (handles blank strings for tenure=0).
    2. Drops non-predictive customerID column.
    3. Converts Churn target column ('Yes'/'No') to binary integer (1/0).
    4. Checks and removes duplicates.
    """
    df = df.copy()

    # 1. Handle TotalCharges blank strings
    blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
    num_blanks = blank_mask.sum()
    if num_blanks > 0:
        print(f"[INFO] Found {num_blanks} blank strings in TotalCharges. Converting to 0.0 (tenure=0).")
        df.loc[blank_mask, "TotalCharges"] = "0.0"
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("[INFO] Dropped non-predictive column 'customerID'.")

    # 3. Handle duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"[INFO] Removing {dup_count} duplicate rows.")
        df = df.drop_duplicates()

    # 4. Target encoding - Ensure strictly integer 1/0
    if "Churn" in df.columns:
        churn_map = {"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        df["Churn"] = df["Churn"].map(churn_map).astype(int)
        print(f"[INFO] Target 'Churn' distribution: {df['Churn'].value_counts().to_dict()}")

    return df


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Returns numerical and categorical feature column names.
    """
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    return num_cols, cat_cols


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a Scikit-Learn ColumnTransformer pipeline for scaling numerical features
    and one-hot encoding categorical features.
    """
    num_cols, cat_cols = get_feature_lists()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor


def prepare_train_test_data(
    file_path: str = RAW_DATA_PATH, test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete pipeline to load, clean, split, scale, and encode churn dataset.
    Saves preprocessor artifact and processed CSV datasets.
    """
    df_raw = load_raw_data(file_path)
    df_clean = clean_raw_data(df_raw)

    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].astype(int)

    num_cols, cat_cols = get_feature_lists()

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"[INFO] Train class balance: {y_train.value_counts(normalize=True).to_dict()}")

    # Fit preprocessor strictly on X_train to prevent data leakage
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + encoded_cat_cols

    # Save preprocessor artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, TRANSFORMER_PATH)
    print(f"[SUCCESS] Saved preprocessor artifact to '{TRANSFORMER_PATH}'.")

    # Save processed data CSVs for reference
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df_train_proc = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_train_proc["Churn"] = y_train.values
    df_train_proc.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_processed.csv"), index=False)

    df_test_proc = pd.DataFrame(X_test_transformed, columns=feature_names)
    df_test_proc["Churn"] = y_test.values
    df_test_proc.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_processed.csv"), index=False)

    print(f"[SUCCESS] Saved processed train/test datasets to '{PROCESSED_DATA_DIR}'.")

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    prepare_train_test_data()
