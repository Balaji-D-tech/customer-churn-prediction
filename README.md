# Customer Churn Prediction using Machine Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)


> An end-to-end, production-ready predictive analytics system for telecommunications customer churn identification, real-time risk assessment, and automated retention recommendations.

---

## 📌 Executive Summary & Problem Statement

In subscription-based industries like telecommunications, acquiring a new customer costs **5 to 25 times more** than retaining an existing subscriber. Customer Churn represents the loss of recurring revenue and decreased Customer Lifetime Value (CLV).

This project implements a leakage-free Machine Learning classification pipeline using the **IBM Telco Customer Churn** dataset (7,043 customer records). By prioritizing **Recall** and **ROC-AUC** over raw accuracy, our tuned ensemble model successfully identifies **78.5% of churn-prone customers**, enabling proactive customer retention strategies via an interactive **Streamlit** dashboard.

---

## 🚀 Key Project Features

- **Leakage-Free Preprocessing**: Scikit-Learn `ColumnTransformer` fitting scaler and encoder strictly on $X_{train}$.
- **Comprehensive EDA**: 10 publication-quality charts detailing contract vulnerability, tenure drop-off, and charge correlations.
- **5-Algorithm Benchmark**: Evaluates Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
- **Class Imbalance & Threshold Tuning**: Solves 1:2.77 class imbalance via `scale_pos_weight` and threshold optimization ($t = 0.468$).
- **Exported Prediction Pipeline**: Unified `Pipeline` object serialized with `joblib` for drop-in raw input inference.
- **Interactive Web App**: 4-page Streamlit application featuring live risk tiering (Low/Medium/High) and retention offers.
- **Automated Test Suite**: `pytest` unit tests covering model loadability, schema integrity, and edge case profiles.
- **Academic Package**: 20-section IEEE project report, 15 presentation slides, and 30+ Viva Q&A study guide.

---

## 🏆 Real Model Evaluation Benchmark Results

All metrics below were produced by executing model training on the held-out 20% test dataset (1,409 customer records) from an actual execution run:

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6984 | 0.5682 | 0.4304 | 0.4898 | 0.7564 |
| Decision Tree | 0.7026 | 0.6087 | 0.3249 | 0.4237 | 0.7365 |
| Random Forest | 0.7012 | 0.5847 | 0.3861 | 0.4651 | 0.7292 |
| Gradient Boosting | 0.7040 | 0.5856 | 0.4114 | 0.4833 | 0.7679 |
| Baseline XGBoost | 0.6742 | 0.5195 | 0.4219 | 0.4657 | 0.7110 |
| **Tuned Random Forest (Balanced)** | 0.6728 | 0.5091 | 0.7637 | 0.6110 | 0.7560 |
| **Tuned XGBoost (Balanced) 🏆** | **0.6856** | **0.5217** | **0.7848** | **0.6268** | **0.7657** |

> **Metric Strategy Insight**: Missing an actual churner (False Negative) results in substantial recurring revenue loss. Applying `scale_pos_weight = 1.975` and decision threshold $t = 0.468$ boosted Recall from **42.2%** to **78.5%**, successfully flagging almost 4 out of 5 churners.

---

## 📂 Repository Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   ├── processed/
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   └── README.md
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_downloader.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── preprocessor.pkl
│   ├── final_model.pkl
│   └── model_metadata.json
├── app/
│   └── app.py
├── tests/
│   ├── __init__.py
│   └── test_prediction.py
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png
│   │   ├── churn_by_contract.png
│   │   ├── churn_by_tenure.png
│   │   ├── confusion_matrix.png
│   │   └── roc_curves.png
│   └── project_report.md
├── presentation/
│   ├── presentation_content.md
│   └── viva_questions.md
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🛠️ How to Run Locally

### 1. Clone Repository & Navigate to Directory
```bash
git clone https://github.com/Balaji-D-tech/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing & Generate EDA Figures
```bash
python src/data_preprocessing.py
python src/eda.py
```

### 4. Train & Tune Machine Learning Models
```bash
python src/train.py
```

### 5. Execute Automated Test Suite
```bash
python -m pytest tests/
```

### 6. Launch Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment Guide (Streamlit Community Cloud)

1. Push your repository to GitHub (`https://github.com/Balaji-D-tech/customer-churn-prediction`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select repository: `Balaji-D-tech/customer-churn-prediction`.
5. Set Main file path: `app/app.py`.
6. Click **Deploy!**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
