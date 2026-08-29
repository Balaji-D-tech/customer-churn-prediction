# Dataset Documentation: IBM Telco Customer Churn

## Overview
This repository uses the official **IBM Telco Customer Churn** dataset.
The dataset contains information about 7,043 customers of a telecommunications service provider in California in Q3.

## Dataset Download
The raw dataset is stored under `data/raw/Telco-Customer-Churn.csv`.

If downloading manually, obtain the file from:
- **Public URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp-for-data/master/data/Telco-Customer-Churn.csv`
- **Kaggle**: `https://www.kaggle.com/datasets/blastchar/telco-customer-churn`

Place the downloaded CSV as `data/raw/Telco-Customer-Churn.csv`.

## Data Schema & Attributes

| Feature | Data Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `customerID` | String | Unique identifier for each customer | e.g., `7590-VHVEG` |
| `gender` | Categorical | Customer gender | `Female`, `Male` |
| `SeniorCitizen` | Numeric/Binary | Whether customer is a senior citizen | `0` (No), `1` (Yes) |
| `Partner` | Categorical | Whether customer has a partner | `Yes`, `No` |
| `Dependents` | Categorical | Whether customer has dependents | `Yes`, `No` |
| `tenure` | Numeric | Number of months customer has stayed with company | `0` - `72` months |
| `PhoneService` | Categorical | Whether customer has phone service | `Yes`, `No` |
| `MultipleLines` | Categorical | Whether customer has multiple lines | `Yes`, `No`, `No phone service` |
| `InternetService` | Categorical | Customer's internet service provider | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Categorical | Whether customer has online security service | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | Categorical | Whether customer has online backup service | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | Categorical | Whether customer has device protection plan | `Yes`, `No`, `No internet service` |
| `TechSupport` | Categorical | Whether customer has tech support service | `Yes`, `No`, `No internet service` |
| `StreamingTV` | Categorical | Whether customer has streaming TV service | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | Categorical | Whether customer has streaming movies service | `Yes`, `No`, `No internet service` |
| `Contract` | Categorical | Contract term of the customer | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | Categorical | Whether customer has paperless billing | `Yes`, `No` |
| `PaymentMethod` | Categorical | Customer's payment method | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | Numeric | Amount charged to customer monthly | `$18.25` - `$118.75` |
| `TotalCharges` | Numeric | Total amount charged to customer | `$18.80` - `$8684.80` (contains whitespace strings for 0-tenure) |
| **`Churn`** | **Target Binary** | **Whether customer churned/left within the last month** | **`Yes`, `No`** |

## Target Variable Analysis
- **Positive Class (`Yes`)**: Customer churned (~1,869 instances, 26.5%)
- **Negative Class (`No`)**: Customer stayed (~5,174 instances, 73.5%)
- **Imbalance Ratio**: Approximately ~1:2.77 ratio. Handled via `class_weight='balanced'` and threshold tuning.
