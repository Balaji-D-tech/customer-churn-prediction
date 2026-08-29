"""
Model Training, Comparison, Hyperparameter Tuning, and Pipeline Export Module.
Implements baseline model training, class imbalance handling, grid search tuning,
evaluation benchmarking, and exports final scikit-learn Pipeline artifact.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, f1_score

from src.data_preprocessing import prepare_train_test_data, get_feature_lists
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
)

MODELS_DIR = "models"
FIGURES_DIR = os.path.join("reports", "figures")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def train_and_evaluate_all() -> Dict[str, Any]:
    """
    Executes complete training, benchmarking, tuning, and export workflow.
    """
    print("=" * 70, flush=True)
    print("      CUSTOMER CHURN PREDICTION — MODEL TRAINING & EVALUATION     ", flush=True)
    print("=" * 70, flush=True)

    # 1. Prepare Data & Preprocessor
    data = prepare_train_test_data()
    X_train_raw = data["X_train_raw"]
    X_test_raw = data["X_test_raw"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    preprocessor = data["preprocessor"]

    # 2. Define Baseline Classifiers
    baseline_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100),
    }

    print("\n--- STAGE 1: BASELINE MODEL EVALUATION ---", flush=True)
    results = []
    roc_curves_data = {}

    for name, model in baseline_models.items():
        print(f"[TRAINING] {name}...", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves_data[name] = (fpr, tpr, metrics["ROC-AUC"])

        print(
            f"  -> Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | "
            f"Rec: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['ROC-AUC']:.4f}",
            flush=True,
        )

    results_df = pd.DataFrame(results)

    # 3. Model Improvement & Hyperparameter Tuning
    print("\n--- STAGE 2: MODEL IMPROVEMENT & HYPERPARAMETER TUNING ---", flush=True)
    print("[INFO] Handling class imbalance via class_weight='balanced' & scale_pos_weight...", flush=True)

    scale_pos_weight_val = float((y_train == 0).sum() / (y_train == 1).sum())

    # Tuned Candidate 1: Random Forest with Class Weighting & Grid Search
    rf_grid = {
        "n_estimators": [100, 150],
        "max_depth": [6, 10],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    print("[TUNING] Tuning Random Forest (GridSearchCV)...", flush=True)
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid,
        cv=3,
        scoring="f1",
        n_jobs=1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"  -> Best RF Params: {rf_search.best_params_}", flush=True)

    # Tuned Candidate 2: XGBoost with Class Imbalance Weighting & Grid Search
    xgb_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "scale_pos_weight": [scale_pos_weight_val],
    }
    print("[TUNING] Tuning XGBoost (GridSearchCV)...", flush=True)
    xgb_search = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        xgb_grid,
        cv=3,
        scoring="f1",
        n_jobs=1,
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  -> Best XGB Params: {xgb_search.best_params_}", flush=True)

    # Evaluate Tuned Models
    improved_models = {
        "Tuned Random Forest (Balanced)": best_rf,
        "Tuned XGBoost (Balanced)": best_xgb,
    }

    for name, model in improved_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, metrics["ROC-AUC"])

        print(
            f"[IMPROVED] {name} -> Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | "
            f"Rec: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['ROC-AUC']:.4f}",
            flush=True,
        )

    all_results_df = pd.DataFrame(results)

    # 4. Select Best Model Based on F1-Score & ROC-AUC
    sorted_df = all_results_df.sort_values(by=["F1-Score", "ROC-AUC"], ascending=False)
    best_model_row = sorted_df.iloc[0]
    best_model_name = best_model_row["Model"]
    print(f"\n[WINNER] Selected Best Model: '{best_model_name}'", flush=True)

    if best_model_name == "Tuned XGBoost (Balanced)":
        best_classifier = best_xgb
    elif best_model_name == "Tuned Random Forest (Balanced)":
        best_classifier = best_rf
    else:
        best_classifier = baseline_models[best_model_name]

    # 5. Threshold Optimization
    y_prob_best = best_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
    f1_scores = [f1_score(y_test, (y_prob_best >= t).astype(int), zero_division=0) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"[INFO] Optimal Decision Threshold: {optimal_threshold:.3f} (Max F1-Score: {f1_scores[optimal_idx]:.4f})", flush=True)

    # Final predictions with optimal threshold
    y_pred_opt = (y_prob_best >= optimal_threshold).astype(int)
    final_metrics = calculate_metrics(y_test, y_pred_opt, y_prob_best)

    # 6. Save Evaluation Plots
    plot_roc_curves(roc_curves_data)
    plot_model_comparison(all_results_df)
    plot_confusion_matrix(y_test, y_pred_opt, model_name=best_model_name)

    # 7. Construct Full End-to-End Prediction Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_classifier),
    ])

    full_pipeline.fit(X_train_raw, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(full_pipeline, FINAL_MODEL_PATH)
    print(f"[SUCCESS] Exported final prediction pipeline to '{FINAL_MODEL_PATH}'.", flush=True)

    # Extract Feature Importances if available
    num_cols, cat_cols = get_feature_lists()
    if hasattr(best_classifier, "feature_importances_"):
        importances = best_classifier.feature_importances_.tolist()
        feat_imp_dict = dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15])
    else:
        feat_imp_dict = {}

    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": optimal_threshold,
        "metrics": final_metrics,
        "feature_names": feature_names,
        "top_features": feat_imp_dict,
        "all_model_results": all_results_df.to_dict(orient="records"),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Exported metadata to '{METADATA_PATH}'.", flush=True)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_all()
