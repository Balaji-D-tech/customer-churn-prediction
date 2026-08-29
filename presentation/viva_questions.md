# Viva Examination Questions & Answers: Customer Churn Prediction

Comprehensive study guide containing 30 likely Viva examination questions and technical answers.

---

## 1. General Machine Learning Questions

### Q1: What is Machine Learning?
**Answer**: Machine Learning is a branch of Artificial Intelligence where algorithms learn patterns from historical data to make predictions or decisions on unseen data without being explicitly programmed.

### Q2: What type of Machine Learning problem is Customer Churn Prediction?
**Answer**: It is a **Supervised Binary Classification** problem because the target variable (`Churn`) has known historical labels (`Yes` = 1, `No` = 0).

### Q3: What is the difference between Classification and Regression?
**Answer**: Classification predicts discrete categorical labels (e.g., Churn vs No Churn), whereas Regression predicts continuous numerical values (e.g., predicting exact customer tenure or monthly spend in dollars).

### Q4: Why did you choose Customer Churn Prediction for your project?
**Answer**: Because churn directly impacts business profitability (acquiring new customers costs 5–25x more than retaining existing ones), and telecom datasets provide rich multi-variable features suitable for ML modeling.

---

## 2. Dataset & Features

### Q5: What dataset did you use, and what is its size?
**Answer**: We used the official **IBM Telco Customer Churn** dataset containing **7,043 customer records** and **21 columns**.

### Q6: What is the target variable in your dataset?
**Answer**: The target variable is `Churn`, indicating whether a customer canceled their service in the last month (`Yes` = 1, `No` = 0).

### Q7: What are the main feature categories in the dataset?
**Answer**: 
1. Demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`)
2. Subscribed Services (`PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.)
3. Account Info (`tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`)

### Q8: Were there missing values in the dataset? How did you handle them?
**Answer**: `TotalCharges` contained 11 blank space strings (`' '`) for new customers with `tenure = 0`. We converted these strings to numeric `0.0` to preserve the rows without dropping data.

### Q9: Why did you drop the `customerID` column?
**Answer**: `customerID` is a unique arbitrary identifier with high cardinality. Keeping it would cause overfitting and add zero predictive power.

---

## 3. Data Preprocessing & Leakage

### Q10: What is Data Leakage, and how did you prevent it?
**Answer**: Data Leakage occurs when information from the test dataset is accidentally used to fit training transformers (like Scalers or Encoders). We prevented it by fitting `ColumnTransformer` strictly on $X_{train}$ and only applying `transform()` on $X_{test}$.

### Q11: What feature scaling method did you use and why?
**Answer**: We used `StandardScaler` ($\mu=0, \sigma=1$) on numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to prevent features with larger scales from dominating distance-based algorithms like Logistic Regression.

### Q12: How did you encode categorical variables?
**Answer**: We used `OneHotEncoder(drop='first', sparse_output=False)` to convert categorical strings into binary indicator variables while dropping the first category to avoid multicollinearity (the dummy variable trap).

### Q13: Why is a Train/Test split necessary?
**Answer**: It simulates real-world deployment by evaluating how well the model generalizes to unseen data, preventing fake high performance caused by memorizing training data (overfitting).

---

## 4. Machine Learning Algorithms

### Q14: How does Logistic Regression work?
**Answer**: Logistic Regression multiplies features by learned weights, sums them, and passes the result through a sigmoid function ($\frac{1}{1 + e^{-z}}$) to output a probability score between 0 and 1.

### Q15: What is the difference between Decision Trees and Random Forests?
**Answer**: A Decision Tree is a single tree that partitions data based on Gini impurity. A Random Forest is an ensemble bagging model that trains multiple decision trees on random subsets of data and averages their predictions to reduce variance.

### Q16: How does Gradient Boosting work compared to Random Forest?
**Answer**: Random Forest builds trees independently in parallel (bagging), whereas Gradient Boosting builds trees sequentially in series, where each new tree aims to correct the residual errors made by previous trees.

### Q17: Why did you test multiple algorithms?
**Answer**: According to the "No Free Lunch Theorem," no single algorithm outperforms all others on every dataset. Benchmarking multiple models allowed us to select the best performer based on empirical metrics.

---

## 5. Model Evaluation Metrics

### Q18: What is Accuracy, and why is it misleading for churn prediction?
**Answer**: Accuracy is $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$. It is misleading here because the dataset is imbalanced (~26.5% churn). A dummy model predicting "No Churn" for everyone gets 73.5% accuracy but misses 100% of actual churners.

### Q19: What is Precision and Recall?
**Answer**: 
- **Precision**: Of all predicted churners, how many were actual churners? ($\frac{TP}{TP + FP}$)
- **Recall**: Of all actual churners, how many did the model identify? ($\frac{TP}{TP + FN}$)

### Q20: Why did you prioritize Recall over Precision in this project?
**Answer**: In churn prediction, a False Negative (missing a churner who then leaves) costs hundreds of dollars in lost customer revenue. A False Positive (sending a discount to a loyal customer) costs very little. Thus, high Recall is prioritized.

### Q21: What is the F1-Score?
**Answer**: The F1-Score is the harmonic mean of Precision and Recall ($\frac{2 \times P \times R}{P + R}$). It balances both metrics into a single score.

### Q22: What is the ROC-AUC Curve?
**Answer**: The Receiver Operating Characteristic (ROC) curve plots True Positive Rate against False Positive Rate across all classification thresholds. The Area Under the Curve (AUC) measures the model's overall ability to distinguish between classes (1.0 is perfect, 0.5 is random).

### Q23: What is a Confusion Matrix?
**Answer**: A 2x2 matrix displaying True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN), summarizing classification performance.

---

## 6. Model Improvement & Imbalance

### Q24: What is Class Imbalance, and how did you address it?
**Answer**: Class Imbalance occurs when one class significantly outnumbers the other (~73.5% No vs ~26.5% Yes). We addressed it using class weighting (`class_weight='balanced'` in RF / `scale_pos_weight` in XGBoost) and decision threshold tuning.

### Q25: What is Hyperparameter Tuning?
**Answer**: Hyperparameters are configuration settings set *before* training (e.g., `max_depth`, `n_estimators`). Tuning optimizes these settings to achieve peak model performance.

### Q26: What is Cross-Validation?
**Answer**: K-Fold Cross-Validation splits training data into $K$ equal folds, iteratively training on $K-1$ folds and validating on the remaining fold to ensure stable, un-biased hyperparameter evaluation.

### Q27: What is Decision Threshold Optimization?
**Answer**: Instead of default $0.50$ probability cutoff, we evaluated thresholds from $0.1$ to $0.9$ to find the threshold ($0.442$) that yields maximum F1-Score and Recall.

---

## 7. Application & Deployment

### Q28: How does the Streamlit web application interact with the ML model?
**Answer**: The app collects user inputs via UI widgets, converts them into a Pandas DataFrame, loads the pre-trained `final_model.pkl` pipeline using `joblib`, calls `predict_proba()`, and formats the prediction, probability, and risk tier.

### Q29: Why did you save the complete Pipeline instead of just the classifier model?
**Answer**: Saving the complete pipeline ensures that raw input data undergoes identical preprocessing (scaling and one-hot encoding) seamlessly in production without needing separate data preprocessing code.

### Q30: How would a business use your application?
**Answer**: Customer service reps can enter customer profile details to view their real-time churn risk score. High-risk customers automatically trigger targeted retention campaigns (e.g., contract upgrade discounts or free tech support).
