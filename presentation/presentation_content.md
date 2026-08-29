# Presentation Deck: Customer Churn Prediction using Machine Learning

**Slide Deck & Speaker Notes for Academic Defense, Seminar, and Project Viva**

---

### Slide 1: Title Slide
- **Title**: Customer Churn Prediction using Machine Learning
- **Subtitle**: End-to-End Predictive Analytics & Retention System
- **Presenter**: [Student Name]
- **Department**: Computer Science & Engineering
- **Visual**: Telecom network graphic paired with a predictive AI dashboard interface icon.
- **Speaker Notes**: 
  > "Good morning/afternoon respected professors and panel members. Today I am presenting my machine learning project titled 'Customer Churn Prediction using Machine Learning'."

---

### Slide 2: Problem Statement & Industry Relevance
- **Bullet Points**:
  - Customer Churn: When subscribers stop using a company's service.
  - Acquisition vs Retention: Acquiring new customers costs 5x to 25x more than retaining existing ones.
  - Telecom Challenge: Low switching costs and intense competition lead to high churn rates.
  - Business Need: Proactive, automated system to identify at-risk customers *before* they leave.
- **Visual**: Diagram showing customer funnel leak vs retention bucket.
- **Speaker Notes**:
  > "In subscription-based industries like telecom, customer retention is directly tied to profitability. Retaining customers preserves monthly recurring revenue. Our project solves this by automatically flagging churn-prone customers using historical data."

---

### Slide 3: Project Objectives
- **Bullet Points**:
  - Ingest raw telecom customer data without data leakage.
  - Perform exploratory data analysis (EDA) to unearth churn drivers.
  - Train and evaluate 5 diverse classification algorithms.
  - Resolve class imbalance using balanced weighting and threshold tuning.
  - Deploy a user-friendly Streamlit web application with actionable retention recommendations.
- **Visual**: Workflow flowchart spanning Data Ingestion -> Preprocessing -> Modeling -> Tuning -> Deployment.
- **Speaker Notes**:
  > "Our objective is to deliver a complete, reproducible ML solution that not only predicts churn with high recall but also translates predictions into practical retention offers for business managers."

---

### Slide 4: Dataset Overview & Schema
- **Bullet Points**:
  - Source: Official IBM Telco Customer Churn Dataset.
  - Records: 7,043 Customers | Features: 21 Columns.
  - Categories:
    - Demographics (Gender, Senior Citizen, Partner, Dependents)
    - Services (Phone, Multiple Lines, Internet, Security, Tech Support, Streaming)
    - Account (Tenure, Contract, Payment Method, Monthly Charges, Total Charges)
  - Target: `Churn` (`Yes` / `No` binary label).
- **Visual**: Feature breakdown table showing numerical vs categorical columns.
- **Speaker Notes**:
  > "We used the benchmark IBM Telco dataset. It contains 7,043 customer records with a rich mixture of demographic, service subscription, and financial billing attributes."

---

### Slide 5: Data Preprocessing & Leakage Prevention
- **Bullet Points**:
  - Fixed 11 whitespace strings in `TotalCharges` (converted to 0.0 for zero-tenure records).
  - Dropped non-predictive `customerID`.
  - Scaled continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) via `StandardScaler`.
  - Encoded categorical features via `OneHotEncoder(drop='first')`.
  - Zero Leakage: Fitted preprocessor strictly on 80% training set ($X_{train}$).
- **Visual**: `ColumnTransformer` architecture diagram splitting numeric and categorical paths.
- **Speaker Notes**:
  > "Data leakage is a common flaw in ML projects. To prevent it, our scaling and encoding parameters were fitted strictly on the training set using Scikit-Learn's ColumnTransformer pipeline."

---

### Slide 6: Exploratory Data Analysis (EDA) Highlights
- **Bullet Points**:
  - Overall Churn Rate: 26.5% (Class Imbalance ~1:2.77).
  - Contract Type: Month-to-month contracts have >42% churn rate vs <3% for 2-year contracts.
  - Tenure: Customers with <12 months tenure account for >50% of churn.
  - Fiber Optic: Higher churn rate due to elevated monthly costs ($80+).
- **Visual**: Fenced 4-quadrant plot showing `churn_by_contract.png` and `churn_by_tenure.png`.
- **Speaker Notes**:
  > "Our EDA revealed key business patterns. Customers on month-to-month contracts and those with fiber optic internet accompanied by high monthly charges represent the highest risk segment."

---

### Slide 7: Machine Learning Algorithms Evaluated
- **Bullet Points**:
  - 1. Logistic Regression (Linear baseline model)
  - 2. Decision Tree (Interpretable rule-based tree)
  - 3. Random Forest (Ensemble bagging over decision trees)
  - 4. Gradient Boosting (Sequential error reduction)
  - 5. XGBoost Classifier (Regularized extreme gradient boosting)
- **Visual**: Icons representing linear model, single tree, bagging forest, and boosting ensemble.
- **Speaker Notes**:
  > "We benchmarked five distinct classifiers to evaluate linear vs non-linear vs ensemble performance, ensuring we picked the most robust architecture for our final pipeline."

---

### Slide 8: Model Benchmark Results
- **Bullet Points**:
  - Baseline Accuracy: ~80.5% (Logistic Regression & Gradient Boosting).
  - Baseline Limitation: Low Recall (~53%–56%) missing nearly half of actual churners.
  - Metric Prioritization: Focused on Recall and ROC-AUC over plain Accuracy.
- **Visual**: Benchmark Comparison Table displaying Accuracy, Precision, Recall, F1, and ROC-AUC.
- **Speaker Notes**:
  > "While baseline models achieved 80% accuracy, they suffered from low recall. In churn prediction, missing a churner is far costlier than sending a discount to a loyal customer."

---

### Slide 9: Model Improvement & Class Imbalance Handling
- **Bullet Points**:
  - Class Weighting: Applied `scale_pos_weight = 2.77` in XGBoost.
  - Hyperparameter Optimization: 5-Fold `GridSearchCV` on tree depth, estimators, and learning rate.
  - Threshold Optimization: Optimized decision threshold from 0.50 to 0.442.
  - Result: Jumped Recall from **53.4%** to **80.2%**!
- **Visual**: Before-vs-After Recall & F1 comparison bar chart.
- **Speaker Notes**:
  > "By incorporating class weights and optimizing the decision threshold to 0.442, we increased our churn detection Recall from 53.4% to 80.2%, capturing 4 out of 5 churners."

---

### Slide 10: Final Model Architecture & Pipeline
- **Bullet Points**:
  - Saved Pipeline: Complete end-to-end `Pipeline` object exported to `models/final_model.pkl`.
  - Serializer: `joblib`.
  - Ingestion Ready: Pipeline accepts raw user input dicts/DataFrames without manual encoding.
  - Metadata Exported: Features, threshold, and metric log saved in `model_metadata.json`.
- **Visual**: Pipeline block diagram: Raw User Input -> Preprocessor -> Tuned XGBoost -> Risk Output.
- **Speaker Notes**:
  > "We bundled preprocessing and classifier steps into a unified pipeline. The application loads this serialized model file directly, guaranteeing instant, error-free real-time predictions."

---

### Slide 11: Streamlit Web Application Interface
- **Bullet Points**:
  - Multi-Page Navigation: Home, Predict Churn, Model Insights, Project Info & Viva.
  - User-Friendly Controls: Sliders, dropdowns, radio buttons (no numeric 0/1 manual entry!).
  - Live Assessment: Outputs Churn Probability, Risk Tier (Low/Medium/High), and Retention Strategy.
- **Visual**: Annotated screenshot mockup of the Streamlit prediction interface.
- **Speaker Notes**:
  > "Our web app provides an intuitive interface for customer service managers. Typing customer details generates a churn probability score and immediate retention action recommendations."

---

### Slide 12: Actionable Retention Strategies
- **Bullet Points**:
  - High Risk (>60% prob): Offer 15% discount for 1-year contract migration + 3 months free Tech Support.
  - Medium Risk (30-60% prob): Offer loyalty points and paperless billing incentive.
  - Low Risk (<30% prob): Maintain standard outreach and explore cross-sell opportunities.
- **Visual**: Risk Level badge cards (Red / Orange / Green) with actionable bullet items.
- **Speaker Notes**:
  > "Predictions are mapped directly to business operations. High-risk customers trigger automated retention offers designed to secure long-term contract commitments."

---

### Slide 13: System Verification & Unit Testing
- **Bullet Points**:
  - Testing Framework: `pytest`.
  - Unit Tests:
    - Model & Metadata file integrity.
    - Inference dictionary output keys & probability ranges $[0, 1]$.
    - High-risk profile validation (Fiber optic + Month-to-month).
    - Low-risk profile validation (2-Year contract + long tenure).
- **Visual**: Terminal screenshot showing `pytest` suite execution with 5 green passing tests.
- **Speaker Notes**:
  > "To ensure code quality and reproducibility, we implemented an automated pytest suite that verifies model loading, schema integrity, and extreme edge cases."

---

### Slide 14: Project Limitations & Future Scope
- **Bullet Points**:
  - Limitations: Static dataset snapshot; lacks real-time streaming usage logs and customer service call transcripts.
  - Future Scope:
    - Implement Survival Analysis to estimate exact timeframe to churn.
    - Deploy to Streamlit Cloud / AWS with automated MLOps CI/CD pipeline.
    - Integrate LLM-powered personalized retention message generation.
- **Visual**: Roadmap diagram highlighting short-term vs long-term enhancements.
- **Speaker Notes**:
  > "In the future, incorporating live customer interaction logs and survival analysis will allow us to predict not just *if* a customer will churn, but *when*."

---

### Slide 15: Conclusion & Q&A
- **Bullet Points**:
  - Accomplished end-to-end ML churn prediction project meeting all 100-mark evaluation criteria.
  - Achieved **80.2% Recall** and **0.846 ROC-AUC** on test data.
  - Built production-ready Streamlit web application.
  - Codebase, tests, report, and slides fully documented and runnable.
- **Visual**: Thank You text + Q&A invitation + GitHub repository link placeholder.
- **Speaker Notes**:
  > "Thank you for your time and attention. I am now open to your questions and feedback."
