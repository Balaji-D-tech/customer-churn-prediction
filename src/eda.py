"""
Exploratory Data Analysis (EDA) Module for Customer Churn Prediction.
Generates publication-quality visualizations and statistical summaries saved in reports/figures/.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_all_eda_plots(df: pd.DataFrame) -> None:
    """
    Generates 10 comprehensive EDA charts for Customer Churn analysis.
    """
    print("[INFO] Generating EDA visualizations...")

    colors = ["#2b5c8f", "#d9534f"]

    # Ensure Churn is int 0/1 for plotting logic
    plot_df = df.copy()
    if plot_df["Churn"].dtype == object:
        plot_df["Churn_Numeric"] = plot_df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        plot_df["Churn_Label"] = plot_df["Churn"]
    else:
        plot_df["Churn_Numeric"] = plot_df["Churn"].astype(int)
        plot_df["Churn_Label"] = plot_df["Churn"].map({1: "Churn", 0: "No Churn"})

    # 1. Target Variable Churn Distribution
    plt.figure(figsize=(7, 5))
    churn_counts = plot_df["Churn_Numeric"].value_counts()
    ax = sns.barplot(x=churn_counts.index, y=churn_counts.values, hue=churn_counts.index, palette=colors, legend=False)
    plt.title("Target Distribution: Customer Churn Count", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Churn Status", fontsize=12)
    plt.ylabel("Number of Customers", fontsize=12)
    plt.xticks(ticks=[0, 1], labels=["No Churn", "Churn"])
    total = len(plot_df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}\n({height/total:.1%})",
            (p.get_x() + p.get_width() / 2.0, height / 2),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
    plt.savefig(os.path.join(FIGURES_DIR, "churn_distribution.png"), dpi=300)
    plt.close()

    # 2. Churn by Gender
    plt.figure(figsize=(7, 5))
    sns.countplot(data=plot_df, x="gender", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Gender", fontsize=14, fontweight="bold")
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_gender.png"), dpi=300)
    plt.close()

    # 3. Churn by Contract Type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x="Contract", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Contract Type", fontsize=14, fontweight="bold")
    plt.xlabel("Contract Duration", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_contract.png"), dpi=300)
    plt.close()

    # 4. Churn by Tenure (KDE plot)
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=plot_df, x="tenure", hue="Churn_Label", common_norm=False, palette=colors, fill=True, alpha=0.4)
    plt.title("Customer Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
    plt.xlabel("Tenure (Months)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_tenure.png"), dpi=300)
    plt.close()

    # 5. Churn by Monthly Charges
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=plot_df, x="MonthlyCharges", hue="Churn_Label", common_norm=False, palette=colors, fill=True, alpha=0.4)
    plt.title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold")
    plt.xlabel("Monthly Charges ($)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_monthly_charges.png"), dpi=300)
    plt.close()

    # 6. Churn by Payment Method
    plt.figure(figsize=(10, 5))
    sns.countplot(data=plot_df, x="PaymentMethod", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Payment Method", fontsize=14, fontweight="bold")
    plt.xlabel("Payment Method", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_payment_method.png"), dpi=300)
    plt.close()

    # 7. Churn by Internet Service
    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x="InternetService", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Internet Service Type", fontsize=14, fontweight="bold")
    plt.xlabel("Internet Service Provider", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_internet_service.png"), dpi=300)
    plt.close()

    # 8. Churn by Senior Citizen Status
    plt.figure(figsize=(7, 5))
    plot_df["Senior_Citizen_Label"] = plot_df["SeniorCitizen"].map({1: "Senior Citizen", 0: "Non-Senior"})
    sns.countplot(data=plot_df, x="Senior_Citizen_Label", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Senior Citizen Status", fontsize=14, fontweight="bold")
    plt.xlabel("Senior Citizen Status", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_senior_citizen.png"), dpi=300)
    plt.close()

    # 9. Churn by Tech Support Service
    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x="TechSupport", hue="Churn_Label", palette=colors)
    plt.title("Customer Churn by Tech Support Availability", fontsize=14, fontweight="bold")
    plt.xlabel("Tech Support Subscription", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)
    plt.legend(title="Status")
    plt.savefig(os.path.join(FIGURES_DIR, "churn_by_tech_support.png"), dpi=300)
    plt.close()

    # 10. Numerical Feature Correlation Heatmap
    plt.figure(figsize=(7, 6))
    num_df = plot_df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_Numeric"]].copy()
    num_df = num_df.rename(columns={"Churn_Numeric": "Churn"})
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Numerical Features & Churn", fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved 10 EDA charts to '{FIGURES_DIR}'.")


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, clean_raw_data

    df = clean_raw_data(load_raw_data())
    generate_all_eda_plots(df)
