# Project Report: Customer Churn Prediction Using Machine Learning

**Academic Evaluation & Technical Documentation Report**  
**Author**: Student  
**Degree / Course**: Bachelor of Technology / Computer Science / Data Science  
**Institution**: Department of Computer Science & Engineering  
**Date**: August 2026  

---

## Abstract
Customer churn—the rate at which customers discontinue subscriptions or service relationships—poses a major financial threat to telecommunications providers. Because acquiring a new customer costs 5 to 25 times more than retaining an existing subscriber, proactive churn prevention is a vital business capability. This project presents an end-to-end Machine Learning pipeline designed to predict customer churn using the benchmark **IBM Telco Customer Churn** dataset (7,043 customer records, 21 attributes). 

We construct a leakage-free preprocessing pipeline incorporating median imputation, standard scaling, and one-hot encoding. Five supervised classification algorithms were benchmarked: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost. To counter severe class imbalance (~26.5% positive churn rate), we applied class weighting (`class_weight='balanced'`), hyperparameter tuning via `GridSearchCV`, and optimal decision threshold selection. 

Our final tuned model achieved an **Accuracy of 76.8%**, **Recall of 80.2%**, **F1-Score of 0.648**, and **ROC-AUC of 0.846**. The production pipeline was serialized using `joblib` and integrated into a four-page interactive **Streamlit** web application featuring real-time risk classification, visual confidence gauges, and automated retention recommendations.

---

## 1. Introduction
In competitive industries such as telecommunications, subscription-based services rely heavily on recurring revenues. Customer churn directly erodes Monthly Recurring Revenue (MRR) and decreases Customer Lifetime Value (CLV). With the proliferation of mobile virtual network operators (MVNOs) and low switching costs, telecom providers must transition from reactive customer recovery to predictive, data-driven retention.

Machine Learning (ML) offers powerful tools to identify subtle behavioral patterns, billing anomalies, and contract vulnerabilities long before a customer formally cancels their subscription.

---

## 2. Problem Statement
Telecommunications companies lack accurate automated systems to detect subscribers at risk of churning prior to contract cancellation. Manual customer reviews are unscalable, subjective, and fail to capture multi-variable interactions.

### Objectives
1. **Develop an Automated Predictive Pipeline**: Build an end-to-end classification system capable of ingesting raw customer data and outputting a churn prediction (`Churn = Yes/No`).
2. **Handle Data Imbalance & Leakage**: Implement proper stratification, feature scaling, encoding, and class weighting to prevent data leakage and maximize positive class Recall.
3. **Compare Machine Learning Algorithms**: Systematically evaluate linear models, tree-based models, and ensemble boosting techniques.
4. **Deploy User-Friendly Application**: Construct a Streamlit web application enabling business stakeholders to perform real-time risk assessment and view actionable retention strategies.

---

## 3. Dataset Description
We utilize the public **IBM Telco Customer Churn** dataset containing 7,043 records and 21 features.

### Key Attributes Breakdown
- **Demographics**: `gender`, `SeniorCitizen` (0/1), `Partner` (Yes/No), `Dependents` (Yes/No).
- **Services**: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Account & Billing**: `tenure` (months), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Target Variable**: `Churn` (`Yes` = Customer left, `No` = Customer stayed).

---

## 4. Data Preprocessing & Cleaning
1. **Handling Missing Values**: `TotalCharges` contained 11 blank space strings (`' '`) corresponding to new customers with `tenure = 0`. These were converted to `0.0` to preserve data integrity without dropping records.
2. **Feature Dropping**: Non-predictive unique identifiers (`customerID`) were removed.
3. **Encoding & Scaling**:
   - Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` ($\mu=0, \sigma=1$).
   - Categorical columns were transformed using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
4. **Data Leakage Prevention**: The `ColumnTransformer` preprocessor was fitted *exclusively* on the 80% training split ($X_{train}$) and subsequently applied to the 20% test split ($X_{test}$).

---

## 5. Exploratory Data Analysis (EDA) Highlights
- **Class Distribution**: 5,174 Non-Churners (73.5%) vs. 1,869 Churners (26.5%).
- **Contract Type Impact**: Month-to-month contract holders exhibited a churn rate exceeding 42%, compared to under 3% for two-year contract holders.
- **Tenure Vulnerability**: Customers with tenure under 12 months accounted for over 50% of total churn instances.
- **Internet Service Effect**: Fiber optic subscribers showed significantly higher churn (~41%) than DSL subscribers (~19%), primarily driven by higher monthly charges ($80+) and service complaints.

---

## 6. Machine Learning Algorithms
We implemented and benchmarked five diverse classification algorithms:
1. **Logistic Regression**: Baseline linear model applying sigmoid activation function.
2. **Decision Tree Classifier**: Non-linear tree structure partitioning features by Gini impurity.
3. **Random Forest Classifier**: Ensemble bagging algorithm averaging decision trees to reduce variance.
4. **Gradient Boosting Classifier**: Sequential boosting algorithm minimizing log-loss residual errors.
5. **XGBoost Classifier**: Optimized extreme gradient boosting with regularized objective function.

---

## 7. Model Evaluation & Comparison

All models were evaluated on the held-out 20% test set (1,409 customer records).

### Real Evaluation Results Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8041 | 0.6522 | 0.5615 | 0.6034 | 0.8431 |
| Decision Tree | 0.7878 | 0.6094 | 0.5749 | 0.5916 | 0.7725 |
| Random Forest | 0.7928 | 0.6387 | 0.5187 | 0.5725 | 0.8329 |
| Gradient Boosting | 0.8055 | 0.6667 | 0.5348 | 0.5934 | 0.8447 |
| **Tuned XGBoost (Balanced)** | **0.7679** | **0.5435** | **0.8021** | **0.6480** | **0.8462** |
| **Tuned Random Forest (Balanced)** | **0.7686** | **0.5458** | **0.7941** | **0.6471** | **0.8451** |

> **Key Finding**: While baseline Logistic Regression and Gradient Boosting achieved higher raw Accuracy (~80.5%), their Recall was severely lacking (~53-56%), meaning they missed almost half of actual churners. By applying class weighting (`scale_pos_weight`) and decision threshold optimization ($t = 0.442$), the **Tuned XGBoost** model increased Recall to **80.21%**, capturing 300 out of 374 actual churners in the test set.

---

## 8. Model Improvement Strategy
1. **Class Weight Adjustment**: Implemented `scale_pos_weight = 2.77` in XGBoost and `class_weight='balanced'` in Random Forest to penalize false negative errors.
2. **Hyperparameter Tuning**: Performed 5-fold cross-validated `GridSearchCV` across parameters `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.
3. **Threshold Optimization**: Shifted classification threshold from default `0.50` to `0.442`, maximizing the F1-Score and catching high-risk customers earlier.

---

## 9. Final Model Serialization & Inference
The full preprocessor and tuned classifier were encapsulated into a single Scikit-Learn `Pipeline` object and exported via `joblib` to `models/final_model.pkl`. The pipeline receives raw Python dictionaries or DataFrames and automatically executes scaling, encoding, and probability estimation.

---

## 10. Web Application & User Interface
Built using **Streamlit**, the application includes:
- **Home Dashboard**: Executive metrics, KPI summaries, and domain introduction.
- **Interactive Predictor**: Form with sliders and dropdowns; computes churn probability, risk tier (Low/Medium/High), and actionable retention strategies.
- **Model Insights**: Interactive ROC curves, confusion matrix heatmaps, feature importance charts, and metric benchmarks.
- **Project Documentation**: Academic summary and embedded Viva examiner Q&A guide.

---

## 11. Verification & Automated Unit Testing
We created automated unit tests in `tests/test_prediction.py` covering:
1. File existence check for `final_model.pkl` and `model_metadata.json`.
2. Schema integrity of prediction result dictionaries.
3. High-risk profile validation (confirming high-probability prediction for month-to-month fiber optic profiles).
4. Low-risk profile validation (confirming low-probability prediction for long-tenure two-year contract profiles).

---

## 12. Limitations & Future Scope
- **Dataset Scope**: Dataset is static and represents a snapshot. Real-world implementation requires streaming customer activity logs (network usage, customer service call sentiment).
- **Future Enhancements**:
  1. Integrate survival analysis (Cox Proportional Hazards) to predict *when* a customer will churn.
  2. Implement automated model monitoring and drift detection (Evidently AI / MLflow).
  3. Deploy web app to Streamlit Community Cloud / AWS EC2 with CI/CD GitHub Actions pipeline.

---

## 13. Conclusion
This project successfully demonstrates a complete end-to-end Machine Learning solution for Customer Churn Prediction. By prioritizing Recall and ROC-AUC over superficial Accuracy, our tuned pipeline effectively detects 80.2% of churners while providing telecom operators with actionable retention recommendations via a polished web UI.

---

## 14. References
1. IBM Watson Analytics, *Telco Customer Churn Dataset*, 2019.
2. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
3. Chen, T. & Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, KDD '16.
4. Streamlit Inc., *Streamlit Documentation*, 2026. `https://docs.streamlit.io`
