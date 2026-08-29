"""
Model Evaluation Module for Customer Churn Prediction.
Calculates key metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
and generates evaluation figures (Confusion Matrix, ROC Curves).
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

FIGURES_DIR = os.path.join("reports", "figures")


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Computes classification metrics for binary churn prediction.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Best Model", save_path: str = None
) -> None:
    """
    Generates and saves a styled Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], save_path: str = None
) -> None:
    """
    Plots ROC curves for multiple models on a single graph.
    
    Args:
        roc_data: Dict mapping model_name -> (fpr, tpr, auc_score)
    """
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curves Comparison — Candidate Models", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Generates a grouped bar chart comparing all candidate models across metrics.
    """
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    df_melted = results_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Benchmark Comparison Across Evaluation Metrics", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Machine Learning Algorithm", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
