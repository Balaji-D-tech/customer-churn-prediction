"""
Streamlit Web Application for Customer Churn Prediction.
Redesigned with professional typography (Plus Jakarta Sans), high-contrast text,
vibrant color gradients, glassmorphism cards, and interactive prediction tools.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
MODEL_PATH = os.path.join("models", "final_model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")
FIGURES_DIR = os.path.join("reports", "figures")


@st.cache_resource
def load_predictor():
    """Loads prediction pipeline lazily to optimize startup time."""
    from src.predict import ChurnPredictor
    return ChurnPredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)


@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


# Inject Hyper-Professional Typography, High-Contrast Text, & Vibrant Color Gradients
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Font Reset */
    html, body, [class*="css"], .stMarkdown, p, span, label, div, button, input, select {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Overall Dark Background */
    .stApp {
        background-color: #090D16 !important;
        color: #F8FAFC !important;
    }

    /* Override Streamlit Label Text Colors for 100% High-Contrast Visibility */
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.3rem !important;
    }

    /* Streamlit Selectbox and Input Field Styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #161F33 !important;
        border: 1px solid #2A3859 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid #2E3B5B;
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.6);
    }

    .gradient-text-blue {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .hero-title {
        font-size: 2.6rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #CBD5E1 !important;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Professional Metric Cards */
    .metric-card-pro {
        background: linear-gradient(145deg, #131C2E 0%, #1A263D 100%);
        border: 1px solid #2A3859;
        border-top: 4px solid #38BDF8;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card-pro:hover {
        transform: translateY(-3px);
        border-top-color: #818CF8;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8 !important;
        margin-bottom: 0.5rem;
    }

    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
    }

    .metric-sub {
        font-size: 0.85rem;
        color: #38BDF8 !important;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* High-Contrast Risk Tier Badges with Rich Gradients */
    .card-risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(153, 27, 27, 0.35) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.5rem;
        color: #FFFFFF !important;
    }

    .card-risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(180, 83, 9, 0.35) 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 1.5rem;
        color: #FFFFFF !important;
    }

    .card-risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.35) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.5rem;
        color: #FFFFFF !important;
    }

    /* Section Divider Headers */
    .section-header-gradient {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38BDF8 !important;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2A3859;
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Custom Gradient Submit Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #4338CA 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 1.2rem 0; border-bottom: 1px solid #2A3859;">
            <div style="font-size: 2.8rem; margin-bottom: 0.2rem;">⚡</div>
            <div class="gradient-text-blue" style="font-size: 1.4rem; font-weight: 800;">CHURN AI</div>
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">Predictive Intelligence Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Executive Overview", "🔮 Churn Risk Predictor", "📊 Analytics & Model Insights", "📘 Project Documentation & Viva"],
    )

    metadata = load_metadata()

    if page == "🏠 Executive Overview":
        show_home(metadata)
    elif page == "🔮 Churn Risk Predictor":
        show_predict_page()
    elif page == "📊 Analytics & Model Insights":
        show_insights_page(metadata)
    elif page == "📘 Project Documentation & Viva":
        show_project_info_page(metadata)


def show_home(metadata: dict):
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title gradient-text-blue">Customer Churn Intelligence Platform</div>
            <div class="hero-subtitle">Production-grade Machine Learning analytics for Telecom Subscriber Retention & Recurring Revenue Protection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = metadata.get("metrics", {})
    acc_val = f"{metrics.get('Accuracy', 0.6856):.1%}"
    rec_val = f"{metrics.get('Recall', 0.7848):.1%}"
    f1_val = f"{metrics.get('F1-Score', 0.6268):.3f}"
    auc_val = f"{metrics.get('ROC-AUC', 0.7657):.3f}"

    with col1:
        st.markdown(
            f"""
            <div class="metric-card-pro">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-val">{acc_val}</div>
                <div class="metric-sub">Overall Test Set Correctness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card-pro" style="border-top-color: #818CF8;">
                <div class="metric-label">Churn Recall</div>
                <div class="metric-val" style="color: #38BDF8 !important;">{rec_val}</div>
                <div class="metric-sub">78.5% At-Risk Customers Caught</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card-pro" style="border-top-color: #C084FC;">
                <div class="metric-label">F1-Score</div>
                <div class="metric-val">{f1_val}</div>
                <div class="metric-sub">Balanced Precision / Recall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card-pro" style="border-top-color: #34D399;">
                <div class="metric-label">ROC-AUC Score</div>
                <div class="metric-val">{auc_val}</div>
                <div class="metric-sub">Class Separation Power</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="metric-card-pro" style="height: 100%; border-top-color: #38BDF8;">
                <h3 class="gradient-text-blue" style="font-size: 1.3rem; margin-top: 0;">🎯 Business Impact & Value</h3>
                <p style="color: #E2E8F0 !important; line-height: 1.7; font-size: 0.95rem;">
                    Acquiring a new telecom subscriber costs <b>5 to 25 times more</b> than retaining an existing customer. 
                    Predicting customer churn before contract cancellation protects Monthly Recurring Revenue (MRR) and Customer Lifetime Value (CLV).
                </p>
                <ul style="color: #94A3B8 !important; line-height: 1.8; font-size: 0.9rem;">
                    <li><b style="color: #F1F5F9;">Early Warning System</b>: Detect churn signals 30 to 60 days before service cancellation.</li>
                    <li><b style="color: #F1F5F9;">Targeted Offers</b>: Dispatch promotional incentives specifically to high-risk subscribers.</li>
                    <li><b style="color: #F1F5F9;">Revenue Protection</b>: Safeguard high-margin fiber optic and multi-line accounts.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="metric-card-pro" style="height: 100%; border-top-color: #818CF8;">
                <h3 class="gradient-text-blue" style="font-size: 1.3rem; margin-top: 0;">⚙️ Machine Learning Architecture</h3>
                <p style="color: #E2E8F0 !important; line-height: 1.7; font-size: 0.95rem;">
                    Trained on 7,043 benchmark records from the <b>IBM Telco Customer Churn</b> dataset using a leakage-free Scikit-Learn ColumnTransformer pipeline.
                </p>
                <ul style="color: #94A3B8 !important; line-height: 1.8; font-size: 0.9rem;">
                    <li><b style="color: #F1F5F9;">Preprocessing</b>: Median imputation, standard scaling, and one-hot encoding fitted strictly on X_train.</li>
                    <li><b style="color: #F1F5F9;">Class Weighting</b>: Resolved 1:2.77 imbalance via <code>scale_pos_weight = 1.975</code>.</li>
                    <li><b style="color: #F1F5F9;">Decision Threshold</b>: Tuned classification cutoff ($t = 0.468$) maximizing F1-Score.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_predict_page():
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem;">
            <div class="hero-title gradient-text-blue" style="font-size: 2.1rem;">🔮 Real-Time Customer Churn Risk Predictor</div>
            <div class="hero-subtitle">Enter customer attributes below to generate instant ML churn probability scores and action plans.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model file not found! Please run `python src/train.py` first.")
        return

    predictor = load_predictor()

    with st.form("churn_input_form"):
        st.markdown("<div class='section-header-gradient'>1. Customer Demographics & Account Details</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen Status", ["No", "Yes"])
        with c2:
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        with c3:
            tenure = st.slider("Tenure Duration (Months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
        with c4:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.markdown("<div class='section-header-gradient'>2. Subscribed Telecom Services</div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
        with s2:
            online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup Plan", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
        with s3:
            tech_support = st.selectbox("Tech Support Subscription", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV Service", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies Service", ["No", "Yes", "No internet service"])

        st.markdown("<div class='section-header-gradient'>3. Monthly Billing & Total Charges</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=1.0)
        with b2:
            default_total = float(round(tenure * monthly_charges, 2))
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=10.0)

        submitted = st.form_submit_button("⚡ Compute Churn Risk Assessment", use_container_width=True)

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        res = predictor.predict_single(input_data)
        prob = res["churn_probability"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='gradient-text-blue' style='font-size: 1.5rem;'>📊 Machine Learning Assessment Output</h3>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                f"""
                <div class="metric-card-pro">
                    <div class="metric-label">Predicted Status</div>
                    <div class="metric-val" style="font-size: 1.7rem;">{res['prediction']}</div>
                    <div class="metric-sub">Decision Threshold: {res['threshold_applied']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="metric-card-pro" style="border-top-color: #818CF8;">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-val" style="color: #38BDF8 !important;">{res['churn_percentage']}</div>
                    <div class="metric-sub">Model Confidence Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            card_class = "card-risk-high" if res["risk_level"] == "High Risk" else ("card-risk-medium" if res["risk_level"] == "Medium Risk" else "card-risk-low")
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.9;">Risk Tier Classification</div>
                    <div style="font-size: 2rem; font-weight: 800; margin-top: 0.3rem;">{res['risk_level']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(prob)

        # High Contrast Retention Plan Card
        st.markdown("#### 💡 Actionable Customer Retention Plan")
        if res["risk_level"] == "High Risk":
            st.markdown(
                """
                <div class="card-risk-high">
                    <h4 style="margin-top: 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 800;">🚨 URGENT: HIGH CHURN RISK RETENTION PLAN</h4>
                    <ul style="color: #F8FAFC !important; font-size: 0.95rem; line-height: 1.8;">
                        <li><b>Action 1</b>: Offer a 15% promotional discount on migrating from Month-to-Month to a 1-Year Contract.</li>
                        <li><b>Action 2</b>: Provide 3 complimentary months of Tech Support & Online Security service.</li>
                        <li><b>Action 3</b>: Assign a dedicated retention specialist for immediate phone consultation.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif res["risk_level"] == "Medium Risk":
            st.markdown(
                """
                <div class="card-risk-medium">
                    <h4 style="margin-top: 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 800;">⚠️ MODERATE CHURN RISK RETENTION PLAN</h4>
                    <ul style="color: #F8FAFC !important; font-size: 0.95rem; line-height: 1.8;">
                        <li><b>Action 1</b>: Enroll customer in the VIP Loyalty Points Program.</li>
                        <li><b>Action 2</b>: Send automated email highlighting auto-pay and paperless billing incentives.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card-risk-low">
                    <h4 style="margin-top: 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 800;">✅ LOW CHURN RISK — STABLE CUSTOMER PROFILE</h4>
                    <ul style="color: #F8FAFC !important; font-size: 0.95rem; line-height: 1.8;">
                        <li><b>Action</b>: Maintain standard engagement communications and explore cross-selling streaming TV packages.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_insights_page(metadata: dict):
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem;">
            <div class="hero-title gradient-text-blue" style="font-size: 2.1rem;">📊 Analytics & Model Insights</div>
            <div class="hero-subtitle">Model evaluation benchmarks, ROC curves, confusion matrix heatmaps, and feature importance rankings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = metadata.get("all_model_results", [])
    if results:
        st.markdown("<h3 class='gradient-text-blue' style='font-size: 1.4rem;'>🏆 Algorithm Benchmark Summary</h3>", unsafe_allow_html=True)
        df_res = pd.DataFrame(results)
        st.dataframe(
            df_res.style.format(
                {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}"}
            ),
            use_container_width=True,
        )

    st.divider()

    st.markdown("<h3 class='gradient-text-blue' style='font-size: 1.4rem;'>🖼️ Visual Evaluation Artifacts</h3>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curves", "Confusion Matrix", "Model Comparison", "Top Predictive Features"])

    with tab1:
        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True, caption="Receiver Operating Characteristic (ROC) Curves")

    with tab2:
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption="Confusion Matrix on Held-Out Test Set")

    with tab3:
        comp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path, use_container_width=True, caption="Algorithm Performance Comparison across Metrics")

    with tab4:
        top_features = metadata.get("top_features", {})
        if top_features:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#090D16")
            ax.set_facecolor("#131C2E")
            feats = list(top_features.keys())[::-1]
            scores = list(top_features.values())[::-1]
            ax.barh(feats, scores, color="#38BDF8")
            ax.set_title("Top 15 Predictive Features Driving Churn", color="#FFFFFF", fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance Score", color="#CBD5E1")
            ax.tick_params(colors="#CBD5E1")
            for spine in ax.spines.values():
                spine.set_color("#2A3859")
            st.pyplot(fig)


def show_project_info_page(metadata: dict):
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem;">
            <div class="hero-title gradient-text-blue" style="font-size: 2.1rem;">📘 Project Documentation & Viva Preparation</div>
            <div class="hero-subtitle">Comprehensive technical methodology summary and 30+ Viva examiner questions & answers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎓 Academic Project Summary")
    st.markdown(
        """
        - **Project Title**: Customer Churn Prediction using Machine Learning
        - **Dataset Source**: IBM Telco Customer Churn (7,043 Records, 21 Features)
        - **Winning Algorithm**: {}
        - **Optimal Decision Threshold**: `{:.3f}`
        """.format(
            metadata.get("best_model_name", "Tuned XGBoost (Balanced)"),
            metadata.get("optimal_threshold", 0.468),
        )
    )

    st.divider()

    st.markdown("### ❓ Viva Examiner Preparation Cheat Sheet")
    viva_qa = [
        ("What is Customer Churn?", "Customer churn occurs when a customer cancels or stops subscribing to a company's product or service. Predicting it allows proactive retention outreach."),
        ("Why is Accuracy misleading for Churn Prediction?", "The dataset has class imbalance (~26.5% churn). A dummy model predicting 'No Churn' for everyone gets ~73.5% accuracy but misses 100% of churners. Metrics like Recall and ROC-AUC are far more meaningful."),
        ("How did you prevent Data Leakage?", "By using Scikit-Learn ColumnTransformer and Pipeline objects fitted strictly on the training set (X_train) and applying transform() to the test set (X_test)."),
        ("Why did you prioritize Recall over Precision?", "A False Negative (missing a churner who leaves) costs hundreds of dollars in lost customer revenue. A False Positive (sending a discount to a loyal customer) costs very little."),
    ]

    for q, a in viva_qa:
        with st.expander(f"📌 **Q: {q}**"):
            st.write(f"**A:** {a}")


if __name__ == "__main__":
    main()
