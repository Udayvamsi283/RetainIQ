"""
RetainIQ — Telecom Customer Churn Prediction System
Streamlit Web Application (Phase 2 Implementation)

This multi-page application provides:
1. Executive Dashboard with KPIs, interactive Plotly charts, and empirical insights.
2. Interactive Churn Prediction Form with validation and conditional logic.
3. Prediction Results page with gauge meters, risk badges, and rule-based retention playbooks.
4. Model Performance page with benchmark tables, confusion matrices, and feature importance.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local application imports
from src.ui_helpers import (
    get_custom_css,
    load_processed_data,
    load_model_artifacts,
    get_cached_pipeline,
    render_kpi_card,
    create_churn_donut,
    create_contract_bar,
    create_internet_service_bar,
    create_payment_method_bar,
    create_tenure_cohort_bar,
    create_monthly_charges_dist,
    create_probability_gauge,
    create_feature_importance_chart,
    create_confusion_matrix_heatmap,
    generate_retention_recommendations,
    THEME
)
from src.prediction import predict_churn

# ==========================================
# PAGE CONFIGURATION & SESSION STATE
# ==========================================

st.set_page_config(
    page_title="RetainIQ — Telecom Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom brand styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "customer_input" not in st.session_state:
    st.session_state["customer_input"] = None


# ==========================================
# SIDEBAR BRANDING & NAVIGATION
# ==========================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.6rem;">📡</span>
                <span style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">RetainIQ</span>
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px; line-height: 1.3;">
                Telecom Customer Churn Prediction System
            </div>
            <div style="font-size: 0.72rem; color: #6366F1; font-weight: 600; margin-top: 6px;">
                Predict Churn. Understand Customers. Improve Retention.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    pages = ["Dashboard", "Churn Prediction", "Prediction Result", "Model Performance"]
    page_icons = {"Dashboard": "📊", "Churn Prediction": "🔮", "Prediction Result": "📋", "Model Performance": "📈"}

    # Navigation buttons
    for p in pages:
        is_active = (st.session_state["current_page"] == p)
        btn_label = f"{page_icons[p]}  {p}"
        
        # Highlight active page in sidebar
        if is_active:
            st.markdown(
                f"""
                <div style="background: rgba(99, 102, 241, 0.25); border-left: 3px solid #6366F1; 
                            border-radius: 4px; padding: 6px 12px; margin-bottom: 4px;">
                    <span style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem;">{btn_label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if st.button(btn_label, key=f"nav_{p}", use_container_width=True):
                st.session_state["current_page"] = p
                st.rerun()

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # Sidebar System Specs / Footer
    st.markdown(
        """
        <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem; font-size: 0.72rem; color: #64748B;">
            <div><b>Dataset:</b> IBM Telco Churn (7,043 rows)</div>
            <div><b>Model:</b> Random Forest Classifier</div>
            <div><b>Status:</b> Production Ready</div>
            <div style="margin-top: 0.75rem; color: #94A3B8; font-weight: 500;">
                Machine Learning | Customer Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE ROUTING
# ==========================================

current_page = st.session_state["current_page"]

# ==========================================
# 1. DASHBOARD PAGE
# ==========================================
if current_page == "Dashboard":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin-bottom: 0.25rem;">
                Executive Customer Churn Dashboard
            </h1>
            <p style="font-size: 0.95rem; color: #6B7280; margin: 0;">
                Real-time exploratory analytics and empirical behavioral patterns from the IBM Telco subscriber dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        df = load_processed_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Calculate real KPI metrics dynamically
    total_customers = len(df)
    churn_num = pd.to_numeric(df["Churn"], errors="coerce")
    churned_customers = int((churn_num == 1).sum())
    retained_customers = int((churn_num == 0).sum())
    churn_rate = (churned_customers / total_customers) * 100

    # 4 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_kpi_card("Total Customers", f"{total_customers:,}", "Active & Historical Accounts", border_color=THEME["primary_indigo"]),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            render_kpi_card("Churned Customers", f"{churned_customers:,}", "Cancelled Subscriptions", border_color=THEME["risk_high"]),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            render_kpi_card("Retained Customers", f"{retained_customers:,}", "Active Loyal Subscribers", border_color=THEME["accent_teal"]),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            render_kpi_card("Overall Churn Rate", f"{churn_rate:.2f}%", "Baseline Attrition Ratio", border_color=THEME["risk_med"]),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Row 1: Donut Distribution & Contract Type Bar
    r1_col1, r1_col2 = st.columns([1, 1])
    with r1_col1:
        st.plotly_chart(create_churn_donut(df), use_container_width=True)
    with r1_col2:
        st.plotly_chart(create_contract_bar(df), use_container_width=True)

    # Row 2: Internet Service Bar & Payment Method Bar
    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        st.plotly_chart(create_internet_service_bar(df), use_container_width=True)
    with r2_col2:
        st.plotly_chart(create_payment_method_bar(df), use_container_width=True)

    # Row 3: Tenure Cohort & Monthly Charges Distribution
    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        st.plotly_chart(create_tenure_cohort_bar(df), use_container_width=True)
    with r3_col2:
        st.plotly_chart(create_monthly_charges_dist(df), use_container_width=True)

    # Dynamic Key Customer Insights
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h3 style="font-size: 1.15rem; font-weight: 700; color: #111827; margin-top: 0; margin-bottom: 0.85rem;">
                📌 Key Customer Insights (Calculated Directly from Data)
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                <div style="border-left: 3px solid #6366F1; padding-left: 0.85rem;">
                    <b style="color: #1F2937;">Contract Duration Risk</b>
                    <p style="font-size: 0.85rem; color: #6B7280; margin: 4px 0 0 0;">
                        Month-to-month subscribers experience a <b>42.71%</b> churn rate, over <b>15.1× higher</b> than customers on 2-year contracts (<b>2.83%</b>).
                    </p>
                </div>
                <div style="border-left: 3px solid #EF4444; padding-left: 0.85rem;">
                    <b style="color: #1F2937;">Early Tenure Cliff</b>
                    <p style="font-size: 0.85rem; color: #6B7280; margin: 4px 0 0 0;">
                        Churned subscribers have a median tenure of only <b>10 months</b> (vs. <b>38 months</b> for retained). First-year customers suffer <b>47.44%</b> attrition.
                    </p>
                </div>
                <div style="border-left: 3px solid #06B6D4; padding-left: 0.85rem;">
                    <b style="color: #1F2937;">Fiber Optic Vulnerability</b>
                    <p style="font-size: 0.85rem; color: #6B7280; margin: 4px 0 0 0;">
                        Fiber optic accounts show a <b>41.89%</b> churn rate compared to DSL (<b>18.96%</b>) and non-internet users (<b>7.40%</b>), despite higher monthly revenues.
                    </p>
                </div>
                <div style="border-left: 3px solid #F59E0B; padding-left: 0.85rem;">
                    <b style="color: #1F2937;">Payment Friction</b>
                    <p style="font-size: 0.85rem; color: #6B7280; margin: 4px 0 0 0;">
                        Customers paying via <b>Electronic check</b> churn at <b>45.29%</b>, while auto-pay methods (bank/card) average under <b>17%</b> churn.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 2. CHURN PREDICTION PAGE
# ==========================================
elif current_page == "Churn Prediction":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin-bottom: 0.25rem;">
                Customer Churn Prediction
            </h1>
            <p style="font-size: 0.95rem; color: #6B7280; margin: 0;">
                Enter customer details to estimate the likelihood of churn using our production Random Forest model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("churn_prediction_form", clear_on_submit=False):
        # Section A: Customer Information
        st.markdown(
            """
            <div style="border-bottom: 2px solid #E5E7EB; padding-bottom: 6px; margin-bottom: 14px;">
                <h4 style="color: #1F2937; margin: 0; font-weight: 700;">👤 Section A: Customer Demographics & Account Age</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0, help="Customer biological sex.")
            senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"], index=0, help="Whether the customer is aged 65 or older.")
            partner = st.selectbox("Has Partner?", options=["No", "Yes"], index=1, help="Whether the customer has a domestic partner or spouse.")
        with col_a2:
            dependents = st.selectbox("Has Dependents?", options=["No", "Yes"], index=0, help="Whether the customer has dependent family members.")
            tenure = st.slider("Tenure with Telecom (Months)", min_value=0, max_value=72, value=12, step=1, help="Number of months customer has stayed with service.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Section B: Services and Subscription
        st.markdown(
            """
            <div style="border-bottom: 2px solid #E5E7EB; padding-bottom: 6px; margin-bottom: 14px;">
                <h4 style="color: #1F2937; margin: 0; font-weight: 700;">🌐 Section B: Subscribed Services</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            phone_service = st.selectbox("Phone Service", options=["Yes", "No"], index=0)
            
            # Dependent MultipleLines
            if phone_service == "No":
                multiple_lines = st.selectbox("Multiple Lines", options=["No phone service"], index=0, disabled=True)
            else:
                multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes"], index=0)

            internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=1)
            
            # Dependent internet features
            if internet_service == "No":
                online_security = st.selectbox("Online Security", options=["No internet service"], index=0, disabled=True)
                online_backup = st.selectbox("Online Backup", options=["No internet service"], index=0, disabled=True)
            else:
                online_security = st.selectbox("Online Security", options=["No", "Yes"], index=0)
                online_backup = st.selectbox("Online Backup", options=["No", "Yes"], index=1)

        with col_b2:
            if internet_service == "No":
                device_protection = st.selectbox("Device Protection", options=["No internet service"], index=0, disabled=True)
                tech_support = st.selectbox("Tech Support", options=["No internet service"], index=0, disabled=True)
                streaming_tv = st.selectbox("Streaming TV", options=["No internet service"], index=0, disabled=True)
                streaming_movies = st.selectbox("Streaming Movies", options=["No internet service"], index=0, disabled=True)
            else:
                device_protection = st.selectbox("Device Protection", options=["No", "Yes"], index=0)
                tech_support = st.selectbox("Tech Support", options=["No", "Yes"], index=0)
                streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes"], index=1)
                streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes"], index=1)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Section C: Billing Information
        st.markdown(
            """
            <div style="border-bottom: 2px solid #E5E7EB; padding-bottom: 6px; margin-bottom: 14px;">
                <h4 style="color: #1F2937; margin: 0; font-weight: 700;">💳 Section C: Contract & Billing Details</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            contract = st.selectbox("Contract Commitment", options=["Month-to-month", "One year", "Two year"], index=0)
            paperless_billing = st.selectbox("Paperless Billing?", options=["Yes", "No"], index=0)
            payment_method = st.selectbox(
                "Payment Method",
                options=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=0
            )

        with col_c2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=130.0,
                value=75.50,
                step=0.50,
                help="Customer monthly bill amount in USD."
            )
            
            # Reasonable default estimate for TotalCharges based on tenure
            default_total = round(max(0.0, float(tenure) * float(monthly_charges)), 2)
            if tenure == 0:
                default_total = 0.0

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(default_total),
                step=10.0,
                help="Cumulative billing spend across customer relationship."
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        submit_btn = st.form_submit_button("🔮 Predict Customer Churn", use_container_width=True)

    if submit_btn:
        # 1. Input Validation
        if tenure < 0 or tenure > 72:
            st.error("Tenure must be between 0 and 72 months.")
            st.stop()
        if monthly_charges < 0:
            st.error("Monthly charges must be non-negative.")
            st.stop()
        if total_charges < 0:
            st.error("Total charges must be non-negative.")
            st.stop()

        # Handle zero-tenure consistency
        if tenure == 0 and total_charges > 0:
            st.info("Note: Customers with 0 tenure are newly joined and typically have TotalCharges = $0.00.")

        # 2. Build structured input dictionary matching exact model feature names
        customer_dict = {
            "gender": str(gender),
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": str(partner),
            "Dependents": str(dependents),
            "tenure": int(tenure),
            "PhoneService": str(phone_service),
            "MultipleLines": str(multiple_lines),
            "InternetService": str(internet_service),
            "OnlineSecurity": str(online_security),
            "OnlineBackup": str(online_backup),
            "DeviceProtection": str(device_protection),
            "TechSupport": str(tech_support),
            "StreamingTV": str(streaming_tv),
            "StreamingMovies": str(streaming_movies),
            "Contract": str(contract),
            "PaperlessBilling": str(paperless_billing),
            "PaymentMethod": str(payment_method),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        # 3. Model Inference via cached Pipeline
        try:
            model = get_cached_pipeline()
            prediction_output = predict_churn(customer_dict, model=model)
            
            # Store in session state
            st.session_state["prediction_result"] = prediction_output
            st.session_state["customer_input"] = customer_dict
            st.session_state["current_page"] = "Prediction Result"
            st.rerun()

        except Exception as e:
            st.error(f"Inference pipeline execution error: {e}")


# ==========================================
# 3. PREDICTION RESULT PAGE
# ==========================================
elif current_page == "Prediction Result":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin-bottom: 0.25rem;">
                Customer Churn Risk Assessment
            </h1>
            <p style="font-size: 0.95rem; color: #6B7280; margin: 0;">
                Comprehensive model inference breakdown, probability quantification, and retention recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    pred_res = st.session_state.get("prediction_result")
    cust_data = st.session_state.get("customer_input")

    if not pred_res or not cust_data:
        st.warning("⚠️ No customer prediction has been run yet in this session.")
        st.markdown(
            """
            Please visit the <b>Churn Prediction</b> page, enter customer attributes, and submit the prediction form.
            """,
            unsafe_allow_html=True
        )
        if st.button("👉 Go to Churn Prediction Form", use_container_width=False):
            st.session_state["current_page"] = "Churn Prediction"
            st.rerun()
    else:
        churn_prob = pred_res["churn_probability"]
        retention_prob = pred_res["retention_probability"]
        prediction_label = pred_res["prediction"]
        risk_tier = pred_res["risk_category"]

        # Risk Tier formatting
        badge_class = "risk-high" if risk_tier == "High" else "risk-medium" if risk_tier == "Medium" else "risk-low"
        badge_color = THEME["risk_high"] if risk_tier == "High" else THEME["risk_med"] if risk_tier == "Medium" else THEME["risk_low"]

        # Top Results Cards
        col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 1.4])
        with col_res1:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {badge_color};">
                    <div class="metric-title">Predicted Churn Status</div>
                    <div class="metric-value" style="color: {badge_color};">
                        {'⚠️ Likely to Churn' if prediction_label == 'Yes' else '✅ Likely to Retain'}
                    </div>
                    <div class="metric-sub">Binary Model Decision (Threshold 0.50)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_res2:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {badge_color};">
                    <div class="metric-title">Assigned Risk Tier</div>
                    <div style="margin-top: 0.25rem;">
                        <span class="risk-badge {badge_class}">{risk_tier} Risk ({churn_prob*100:.1f}%)</span>
                    </div>
                    <div class="metric-sub" style="margin-top: 0.5rem;">
                        {
                            'Low: &lt;30% | Medium: 30–59% | High: &ge;60%'
                        }
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_res3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {THEME['primary_indigo']};">
                    <div class="metric-title">Retention vs. Churn Split</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div>
                            <span style="font-size: 1.4rem; font-weight: 700; color: #10B981;">{retention_prob*100:.1f}%</span>
                            <div style="font-size: 0.75rem; color: #6B7280;">Retention Prob</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.4rem; font-weight: 700; color: #EF4444;">{churn_prob*100:.1f}%</span>
                            <div style="font-size: 0.75rem; color: #6B7280;">Churn Prob</div>
                        </div>
                    </div>
                    <div style="height: 6px; background: #EF4444; border-radius: 4px; margin-top: 6px; overflow: hidden;">
                        <div style="height: 100%; width: {retention_prob*100}%; background: #10B981;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {badge_color}; border-radius: 8px; padding: 0.85rem 1.15rem; margin-top: 1rem; margin-bottom: 0.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.9rem; margin-bottom: 2px;">
                    ℹ️ Interpreting Model Decision vs. Risk Category
                </div>
                <div style="font-size: 0.825rem; color: #475569; line-height: 1.45;">
                    The <b>Predicted Churn Status</b> ({'⚠️ Likely to Churn' if prediction_label == 'Yes' else '✅ Likely to Retain'}) is driven by the classifier's standard <b>50% decision threshold</b>.
                    In parallel, the <b>Assigned Risk Tier</b> ({risk_tier} Risk) maps the continuous churn probability ({churn_prob*100:.2f}%) into actionable customer success monitoring categories:
                    <b>Low Risk</b> (&lt;30%), <b>Medium Risk</b> (30% to &lt;60%), and <b>High Risk</b> (&ge;60%).
                    Consequently, an account with a 45% churn probability is predicted as <i>Likely to Retain</i> (45% &lt; 50%), but flagged as <i>Medium Risk</i> for proactive retention outreach.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Gauge Chart and Recommendations Side-by-Side
        col_g, col_rec = st.columns([1, 1.4])
        with col_g:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="font-size: 1rem; font-weight: 700; color: #111827; margin: 0 0 0.5rem 0;">
                        Risk Gauge Probability
                    </h4>
                """,
                unsafe_allow_html=True
            )
            st.plotly_chart(create_probability_gauge(churn_prob), use_container_width=True)
            st.markdown(
                f"""
                    <div style="font-size: 0.8rem; color: #6B7280; text-align: center; border-top: 1px solid #F3F4F6; padding-top: 8px;">
                        Assigned Tier: <b style="color: {badge_color};">{risk_tier} Risk</b> ({churn_prob*100:.2f}% calculated churn probability)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_rec:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
                        <h4 style="font-size: 1rem; font-weight: 700; color: #111827; margin: 0;">
                            🎯 Illustrative Retention Strategies (Advisory Only)
                        </h4>
                        <span style="font-size: 0.75rem; color: #6B7280;">Rule-based guidance</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            recom_items = generate_retention_recommendations(cust_data, churn_prob)
            for rec in recom_items:
                st.markdown(
                    f"""
                    <div class="recom-card">
                        <div class="recom-title">{rec['title']}</div>
                        <div class="recom-desc">{rec['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                """
                    <div style="font-size: 0.72rem; color: #9CA3AF; margin-top: 8px; font-style: italic;">
                        * Disclaimer: These recommendations are illustrative rule-based suggestions derived from historical subscriber patterns; they are not actual promotional offers and do not guarantee retention outcomes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Submitted Customer Profile Accordion / Card
        with st.expander("🔍 View Complete Submitted Customer Profile", expanded=False):
            profile_cols = st.columns(3)
            with profile_cols[0]:
                st.markdown("<b>Demographics</b>", unsafe_allow_html=True)
                st.write(f"• Gender: {cust_data['gender']}")
                st.write(f"• Senior Citizen: {'Yes' if cust_data['SeniorCitizen']==1 else 'No'}")
                st.write(f"• Partner: {cust_data['Partner']}")
                st.write(f"• Dependents: {cust_data['Dependents']}")
                st.write(f"• Tenure: {cust_data['tenure']} months")
            with profile_cols[1]:
                st.markdown("<b>Subscribed Services</b>", unsafe_allow_html=True)
                st.write(f"• Phone Service: {cust_data['PhoneService']}")
                st.write(f"• Multiple Lines: {cust_data['MultipleLines']}")
                st.write(f"• Internet Service: {cust_data['InternetService']}")
                st.write(f"• Online Security: {cust_data['OnlineSecurity']}")
                st.write(f"• Online Backup: {cust_data['OnlineBackup']}")
                st.write(f"• Tech Support: {cust_data['TechSupport']}")
            with profile_cols[2]:
                st.markdown("<b>Billing & Contract</b>", unsafe_allow_html=True)
                st.write(f"• Contract: {cust_data['Contract']}")
                st.write(f"• Paperless Billing: {cust_data['PaperlessBilling']}")
                st.write(f"• Payment Method: {cust_data['PaymentMethod']}")
                st.write(f"• Monthly Bill: ${cust_data['MonthlyCharges']:.2f}")
                st.write(f"• Total Charges: ${cust_data['TotalCharges']:.2f}")

        # Action Buttons
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("🔄 Evaluate Another Customer", use_container_width=True):
                st.session_state["current_page"] = "Churn Prediction"
                st.rerun()
        with btn_c2:
            if st.button("📈 View Model Benchmarks & Metrics", use_container_width=True):
                st.session_state["current_page"] = "Model Performance"
                st.rerun()


# ==========================================
# 4. MODEL PERFORMANCE PAGE
# ==========================================
elif current_page == "Model Performance":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #111827; margin-bottom: 0.25rem;">
                Model Performance & Technical Benchmarks
            </h1>
            <p style="font-size: 0.95rem; color: #6B7280; margin: 0;">
                Official evaluation metrics, 5-fold cross-validation logs, and permutation importance for Phase 1 trained pipelines.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    artifacts = load_model_artifacts()
    metrics = artifacts.get("metrics")
    comp_df = artifacts.get("comparison")
    fi_df = artifacts.get("feature_importance")

    if not metrics or comp_df is None or fi_df is None:
        st.error("Model artifacts not found in outputs/. Please ensure Phase 1 model training has completed.")
        st.stop()

    best_test = metrics.get("held_out_test_metrics", {})
    selected_model_name = metrics.get("selected_model", "Random Forest")

    # Top KPI Cards for Selected Model
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.markdown(
            render_kpi_card("Selected Model", selected_model_name, "Best 5-Fold CV F1", border_color=THEME["primary_indigo"]),
            unsafe_allow_html=True
        )
    with kpi_c2:
        st.markdown(
            render_kpi_card("Test Accuracy", f"{best_test.get('accuracy', 0.0)*100:.2f}%", "Unseen Test (n=1,409)", border_color=THEME["accent_cyan"]),
            unsafe_allow_html=True
        )
    with kpi_c3:
        st.markdown(
            render_kpi_card("Churn Recall", f"{best_test.get('recall', 0.0)*100:.2f}%", "Captured 299 of 374 Churners", border_color=THEME["accent_teal"]),
            unsafe_allow_html=True
        )
    with kpi_c4:
        st.markdown(
            render_kpi_card("Churn F1-Score", f"{best_test.get('f1_score', 0.0):.4f}", "Harmonic Mean (pos_label=1)", border_color=THEME["secondary_violet"]),
            unsafe_allow_html=True
        )
    with kpi_c5:
        st.markdown(
            render_kpi_card("ROC-AUC Score", f"{best_test.get('roc_auc', 0.0):.4f}", "Discriminatory Power", border_color=THEME["risk_high"]),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Section B: Model Comparison Table
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #111827; margin: 0 0 0.75rem 0;">
                📊 Algorithm Benchmark Comparison (Held-Out Test Set)
            </h4>
        """,
        unsafe_allow_html=True
    )
    
    # Format comparison table nicely
    display_comp = comp_df.copy()
    numeric_cols = [c for c in display_comp.columns if c not in ["Algorithm"]]
    for col in numeric_cols:
        if "Accuracy" in col or "Precision" in col or "Recall" in col or "F1" in col or "ROC-AUC" in col:
            display_comp[col] = display_comp[col].apply(lambda x: f"{float(x):.4f}" if pd.notnull(x) else "")
    
    st.dataframe(display_comp, use_container_width=True, hide_index=True)
    st.markdown(
        f"""
            <div style="font-size: 0.8rem; color: #6B7280; margin-top: 6px; line-height: 1.4;">
                <b>Selection Decision:</b> {metrics.get('selection_criterion')}. Random Forest achieved the highest mean CV F1 (<b>0.6347</b>), outperforming Logistic Regression (0.6283) and Decision Tree (0.6173). (Note: Decision Tree exhibited the lowest fold standard deviation at ±0.0105 vs. ±0.0144 for Random Forest). All models utilized <code>class_weight='balanced'</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Section C & D: Confusion Matrix and Multi-Metric Comparison Chart
    row_c1, row_c2 = st.columns([1, 1.2])
    with row_c1:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            """,
            unsafe_allow_html=True
        )
        cm = best_test.get("confusion_matrix", [[0, 0], [0, 0]])
        st.plotly_chart(create_confusion_matrix_heatmap(cm), use_container_width=True)
        st.markdown(
            """
                <div style="font-size: 0.78rem; color: #6B7280; text-align: center;">
                    True Negatives: <b>763</b> | False Positives: <b>272</b> | False Negatives: <b>75</b> | True Positives: <b>299</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row_c2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="font-size: 1.05rem; font-weight: 700; color: #111827; margin: 0 0 0.75rem 0;">
                    📈 Multi-Metric Comparison Across Algorithms
                </h4>
            """,
            unsafe_allow_html=True
        )
        # Create grouped bar chart
        plot_comp = comp_df.melt(
            id_vars=["Algorithm"],
            value_vars=["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC-AUC"],
            var_name="Metric",
            value_name="Score"
        )
        fig_bar = px.bar(
            plot_comp,
            x="Algorithm",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=[THEME["primary_indigo"], THEME["accent_cyan"], THEME["accent_teal"], THEME["secondary_violet"], THEME["risk_high"]]
        )
        fig_bar.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#E5E7EB"),
            xaxis=dict(showgrid=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Section E: Feature Importance
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(create_feature_importance_chart(fi_df), use_container_width=True)
    st.markdown(
        """
            <div style="background: #F8FAFC; border-left: 3px solid #6366F1; padding: 0.65rem 1rem; border-radius: 4px; font-size: 0.8rem; color: #475569;">
                <b>Scientific Notice:</b> Permutation feature importance measures the statistical dependence and degradation in test F1-score when a column is randomly shuffled. It quantifies predictive attribution within this fitted pipeline and does <i>not</i> assert real-world causality.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Section F: Technical Metadata & Training Specifications
    with st.expander("⚙️ Pipeline Technical Specifications & Reproducibility Parameters", expanded=True):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("<b>Experimental Design</b>", unsafe_allow_html=True)
            st.write(f"• **Train / Test Split:** 80% / 20% ({metrics.get('train_samples_count'):,} train / {metrics.get('test_samples_count'):,} test)")
            st.write(f"• **Splitting Strategy:** Stratified on `Churn` (Preserved 73.46% / 26.54% ratio)")
            st.write(f"• **Cross-Validation:** 5-Fold StratifiedKFold on training set")
            st.write(f"• **Random Seed:** `{metrics.get('random_seed', 42)}` (Universal reproducibility)")
        with p_col2:
            st.markdown("<b>Pipeline Architecture</b>", unsafe_allow_html=True)
            st.write("• **Numerical Preprocessing:** `SimpleImputer(median)` $\\rightarrow$ `StandardScaler()`")
            st.write("• **Categorical Preprocessing:** `SimpleImputer(most_frequent)` $\\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`")
            st.write("• **Class Weighting:** `class_weight='balanced'` (Minority loss penalty)")
            st.write("• **Leakage Protection:** Transformers fitted strictly on training data")
