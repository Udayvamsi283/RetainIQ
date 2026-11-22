"""
RetainIQ - Telecom Customer Churn Prediction System
UI Helpers and Visualization Utilities for Streamlit Application

This module provides:
1. Custom CSS injection for the RetainIQ modern dark-navy / indigo brand theme.
2. Cached loading functions for datasets and ML evaluation artifacts.
3. KPI card HTML generators.
4. Interactive Plotly chart builders for Dashboard, Model Performance, and Predictions.
5. Contextual, rule-based retention recommendation engine.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color Palette Definitions matching project visual direction
THEME = {
    "navy_sidebar": "#0B1220",
    "dark_navy": "#111827",
    "primary_indigo": "#6366F1",
    "secondary_violet": "#8B5CF6",
    "accent_cyan": "#06B6D4",
    "accent_teal": "#10B981",
    "bg_main": "#F5F7FB",
    "card_bg": "#FFFFFF",
    "text_dark": "#1F2937",
    "text_muted": "#6B7280",
    "border": "#E5E7EB",
    "risk_low": "#10B981",      # Emerald Green
    "risk_med": "#F59E0B",      # Amber
    "risk_high": "#EF4444",     # Crimson Red
}

BASE_DIR = Path(__file__).resolve().parent.parent


def get_custom_css() -> str:
    """Returns CSS styles tailored for RetainIQ's corporate student-capstone theme."""
    return f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Main background */
    .stApp {{
        background-color: {THEME['bg_main']};
        color: {THEME['text_dark']};
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['navy_sidebar']} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: #E2E8F0 !important;
    }}

    /* Card Containers */
    .metric-card {{
        background: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.25rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .metric-card:hover {{
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }}
    .metric-title {{
        font-size: 0.825rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {THEME['text_muted']};
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {THEME['text_dark']};
        line-height: 1.2;
    }}
    .metric-sub {{
        font-size: 0.8rem;
        color: {THEME['text_muted']};
        margin-top: 0.25rem;
    }}

    /* Risk Badges */
    .risk-badge {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.025em;
    }}
    .risk-low {{
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }}
    .risk-medium {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }}
    .risk-high {{
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }}

    /* Recommendation Cards */
    .recom-card {{
        background: #FFFFFF;
        border-left: 4px solid {THEME['primary_indigo']};
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        border-top: 1px solid #F3F4F6;
        border-right: 1px solid #F3F4F6;
        border-bottom: 1px solid #F3F4F6;
    }}
    .recom-title {{
        font-weight: 600;
        color: {THEME['text_dark']};
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }}
    .recom-desc {{
        color: {THEME['text_muted']};
        font-size: 0.85rem;
        line-height: 1.4;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background-color: {THEME['primary_indigo']};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background-color 0.2s ease;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #4F46E5;
        color: white;
    }}

    /* Remove excess top padding in Streamlit */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}
    </style>
    """


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Loads cleaned dataset safely from project data directory."""
    csv_path = BASE_DIR / "data" / "processed" / "cleaned_telco_data.csv"
    if not csv_path.exists():
        raw_path = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
        if raw_path.exists():
            from src.preprocessing import clean_telco_data
            raw_df = pd.read_csv(raw_path)
            return clean_telco_data(raw_df, save_processed=True)
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{csv_path}'. Please run 'python -m src.data_loader' first."
            )
    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def load_model_artifacts() -> Dict[str, Any]:
    """Loads metrics, model comparisons, and feature importances from outputs/."""
    artifacts = {}
    metrics_path = BASE_DIR / "outputs" / "model_metrics.json"
    comp_path = BASE_DIR / "outputs" / "model_comparison.csv"
    fi_path = BASE_DIR / "outputs" / "feature_importance.csv"

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            artifacts["metrics"] = json.load(f)
    else:
        artifacts["metrics"] = None

    if comp_path.exists():
        artifacts["comparison"] = pd.read_csv(comp_path)
    else:
        artifacts["comparison"] = None

    if fi_path.exists():
        artifacts["feature_importance"] = pd.read_csv(fi_path)
    else:
        artifacts["feature_importance"] = None

    return artifacts


@st.cache_resource
def get_cached_pipeline():
    """Loads fitted scikit-learn Pipeline once using st.cache_resource."""
    from src.prediction import load_model
    model_path = BASE_DIR / "models" / "best_model.pkl"
    return load_model(str(model_path))


def render_kpi_card(title: str, value: str, subtext: str = "", border_color: Optional[str] = None) -> str:
    """Renders HTML snippet for a sleek KPI card."""
    border_style = f"border-top: 3px solid {border_color};" if border_color else ""
    return f"""
    <div class="metric-card" style="{border_style}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtext}</div>' if subtext else ''}
    </div>
    """


# ==========================================
# PLOTLY CHART GENERATORS
# ==========================================

def create_churn_donut(df: pd.DataFrame) -> go.Figure:
    """A. Donut chart showing churned vs. retained customers."""
    counts = df["Churn"].map({1: "Churned", 0: "Retained", "1": "Churned", "0": "Retained"}).value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    colors = [THEME["risk_high"] if l == "Churned" else THEME["primary_indigo"] for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, family="Inter"),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Customer Churn Distribution</b>", font=dict(size=14, color=THEME["text_dark"])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_contract_bar(df: pd.DataFrame) -> go.Figure:
    """B. Bar chart showing churn rate across contract types."""
    df_temp = df.copy()
    df_temp["Churn_Num"] = pd.to_numeric(df_temp["Churn"], errors="coerce")
    contract_rates = (df_temp.groupby("Contract")["Churn_Num"].mean() * 100).reset_index()
    contract_rates.columns = ["Contract", "Churn_Rate"]

    # Order logically
    order = ["Month-to-month", "One year", "Two year"]
    contract_rates["sort_key"] = contract_rates["Contract"].map(lambda x: order.index(x) if x in order else 99)
    contract_rates = contract_rates.sort_values("sort_key").drop(columns=["sort_key"])

    fig = px.bar(
        contract_rates,
        x="Contract",
        y="Churn_Rate",
        text=contract_rates["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
        labels={"Churn_Rate": "Churn Rate (%)", "Contract": "Contract Commitment"},
        color="Contract",
        color_discrete_sequence=[THEME["risk_high"], THEME["accent_cyan"], THEME["primary_indigo"]]
    )
    fig.update_traces(textposition="outside", textfont=dict(weight="bold", size=11))
    fig.update_layout(
        title=dict(text="<b>Churn Rate by Contract Type</b>", font=dict(size=14, color=THEME["text_dark"])),
        yaxis=dict(range=[0, max(contract_rates["Churn_Rate"]) * 1.25], showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_internet_service_bar(df: pd.DataFrame) -> go.Figure:
    """C. Bar chart showing churn rate across internet services."""
    df_temp = df.copy()
    df_temp["Churn_Num"] = pd.to_numeric(df_temp["Churn"], errors="coerce")
    rates = (df_temp.groupby("InternetService")["Churn_Num"].mean() * 100).reset_index()
    rates.columns = ["InternetService", "Churn_Rate"]

    fig = px.bar(
        rates,
        x="InternetService",
        y="Churn_Rate",
        text=rates["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
        labels={"Churn_Rate": "Churn Rate (%)", "InternetService": "Internet Technology"},
        color="InternetService",
        color_discrete_sequence=[THEME["accent_cyan"], THEME["risk_high"], THEME["accent_teal"]]
    )
    fig.update_traces(textposition="outside", textfont=dict(weight="bold", size=11))
    fig.update_layout(
        title=dict(text="<b>Churn Rate by Internet Service</b>", font=dict(size=14, color=THEME["text_dark"])),
        yaxis=dict(range=[0, max(rates["Churn_Rate"]) * 1.25], showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_payment_method_bar(df: pd.DataFrame) -> go.Figure:
    """D. Bar chart showing churn rate across payment methods."""
    df_temp = df.copy()
    df_temp["Churn_Num"] = pd.to_numeric(df_temp["Churn"], errors="coerce")
    rates = (df_temp.groupby("PaymentMethod")["Churn_Num"].mean() * 100).reset_index()
    rates.columns = ["PaymentMethod", "Churn_Rate"]
    rates = rates.sort_values(by="Churn_Rate", ascending=False)

    fig = px.bar(
        rates,
        x="Churn_Rate",
        y="PaymentMethod",
        orientation="h",
        text=rates["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
        labels={"Churn_Rate": "Churn Rate (%)", "PaymentMethod": ""},
        color="PaymentMethod",
        color_discrete_sequence=[THEME["risk_high"], THEME["primary_indigo"], THEME["secondary_violet"], THEME["accent_cyan"]]
    )
    fig.update_traces(textposition="outside", textfont=dict(weight="bold", size=11))
    fig.update_layout(
        title=dict(text="<b>Churn Rate by Payment Method</b>", font=dict(size=14, color=THEME["text_dark"])),
        xaxis=dict(range=[0, max(rates["Churn_Rate"]) * 1.25], showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(autorange="reversed", showgrid=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_tenure_cohort_bar(df: pd.DataFrame) -> go.Figure:
    """E. Show churn behavior across tenure groups: 0-12, 13-24, 25-48, 49+ mos."""
    df_temp = df.copy()
    df_temp["Churn_Num"] = pd.to_numeric(df_temp["Churn"], errors="coerce")
    bins = [-1, 12, 24, 48, 100]
    labels = ["0–12 mos", "13–24 mos", "25–48 mos", "49+ mos"]
    df_temp["Tenure_Cohort"] = pd.cut(df_temp["tenure"], bins=bins, labels=labels)

    rates = (df_temp.groupby("Tenure_Cohort", observed=False)["Churn_Num"].mean() * 100).reset_index()
    rates.columns = ["Tenure_Cohort", "Churn_Rate"]

    fig = px.bar(
        rates,
        x="Tenure_Cohort",
        y="Churn_Rate",
        text=rates["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
        labels={"Churn_Rate": "Churn Rate (%)", "Tenure_Cohort": "Tenure Cohort"},
        color="Tenure_Cohort",
        color_discrete_sequence=[THEME["risk_high"], THEME["risk_med"], THEME["accent_cyan"], THEME["primary_indigo"]]
    )
    fig.update_traces(textposition="outside", textfont=dict(weight="bold", size=11))
    fig.update_layout(
        title=dict(text="<b>Churn Rate Across Customer Tenure Cohorts</b>", font=dict(size=14, color=THEME["text_dark"])),
        yaxis=dict(range=[0, max(rates["Churn_Rate"]) * 1.25], showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_monthly_charges_dist(df: pd.DataFrame) -> go.Figure:
    """F. Compare monthly charges between churned and retained customers."""
    df_temp = df.copy()
    df_temp["Status"] = df_temp["Churn"].map({1: "Churned", 0: "Retained", "1": "Churned", "0": "Retained"})

    fig = px.box(
        df_temp,
        x="Status",
        y="MonthlyCharges",
        color="Status",
        color_discrete_map={"Retained": THEME["primary_indigo"], "Churned": THEME["risk_high"]},
        points="outliers",
        labels={"MonthlyCharges": "Monthly Bill ($)", "Status": "Customer Status"}
    )
    fig.update_layout(
        title=dict(text="<b>Monthly Charges: Churned vs. Retained</b>", font=dict(size=14, color=THEME["text_dark"])),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_probability_gauge(churn_prob: float) -> go.Figure:
    """Gauge chart indicating churn risk with color-coded risk bands."""
    pct = round(churn_prob * 100, 1)

    if churn_prob < 0.30:
        bar_color = THEME["risk_low"]
    elif churn_prob < 0.60:
        bar_color = THEME["risk_med"]
    else:
        bar_color = THEME["risk_high"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(size=36, family="Inter", weight="bold", color=bar_color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=THEME["text_muted"]),
                bar=dict(color=bar_color, thickness=0.25),
                bgcolor="white",
                borderwidth=1,
                bordercolor=THEME["border"],
                steps=[
                    dict(range=[0, 30], color="#ECFDF5"),  # Soft Green
                    dict(range=[30, 60], color="#FFFBEB"), # Soft Amber
                    dict(range=[60, 100], color="#FEF2F2"), # Soft Red
                ],
                threshold=dict(
                    line=dict(color=bar_color, width=4),
                    thickness=0.75,
                    value=pct
                )
            )
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=20, b=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter")
    )
    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing top 10 permutation features."""
    top10 = fi_df.head(10).iloc[::-1]  # Bottom-to-top

    fig = go.Figure(
        go.Bar(
            x=top10["Importance_Mean"],
            y=top10["Feature"],
            orientation="h",
            error_x=dict(type="data", array=top10["Importance_Std"], visible=True, color="#94A3B8"),
            marker=dict(color=THEME["primary_indigo"], line=dict(color="#4338CA", width=1)),
            text=[f"{v:.4f}" for v in top10["Importance_Mean"]],
            textposition="outside",
            textfont=dict(size=10, family="Inter", weight="bold")
        )
    )
    fig.update_layout(
        title=dict(text="<b>Top 10 Influential Features (Permutation Importance - F1 Metric)</b>", font=dict(size=14, color=THEME["text_dark"])),
        xaxis=dict(title="Mean Decrease in Test F1-Score when Feature is Shuffled", showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="", showgrid=False),
        margin=dict(l=20, r=40, t=40, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap with labeled counts."""
    z = np.array(cm)
    x = ["Retained (0)", "Churned (1)"]
    y = ["Retained (0)", "Churned (1)"]

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            label = "TN" if (i==0 and j==0) else "FP" if (i==0 and j==1) else "FN" if (i==1 and j==0) else "TP"
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"<b>{val}</b><br><span style='font-size:10px;'>({label})</span>",
                    showarrow=False,
                    font=dict(color="white" if val > z.max()/2 else THEME["text_dark"], size=14, family="Inter")
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            showscale=False
        )
    )
    fig.update_layout(
        title=dict(text="<b>Confusion Matrix: Selected Model (Held-Out Test Set)</b>", font=dict(size=14, color=THEME["text_dark"])),
        xaxis=dict(title="<b>Predicted Customer Status</b>", showgrid=False),
        yaxis=dict(title="<b>Actual Customer Status</b>", autorange="reversed", showgrid=False),
        annotations=annotations,
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# ==========================================
# RETENTION RECOMMENDATION ENGINE
# ==========================================

def generate_retention_recommendations(customer: Dict[str, Any], churn_prob: float) -> List[Dict[str, str]]:
    """
    Produces transparent, rule-based retention strategies tailored to the customer profile.
    Explicitly labeled as illustrative advisory suggestions, not ML-generated promises or actual company offers.
    """
    recoms = []

    # Contract Rule
    if customer.get("Contract") == "Month-to-month":
        recoms.append({
            "title": "Explore Long-Term Contract Incentives (Illustrative Advisory)",
            "desc": "Customer is on a flexible Month-to-month contract (historically associated with 42.7% churn). Consider evaluating longer-term plan options, loyalty contract incentives, or rate-lock terms to encourage commitment."
        })

    # Payment Method Rule
    if customer.get("PaymentMethod") == "Electronic check":
        recoms.append({
            "title": "Promote Automated Payment Options (Illustrative Advisory)",
            "desc": "Electronic check users experience the highest payment-related attrition (45.3%). Consider promoting convenient auto-pay options via bank transfer or credit card to minimize monthly manual payment friction."
        })

    # Cyber & Tech Support Bundling
    if customer.get("OnlineSecurity") == "No" or customer.get("TechSupport") == "No":
        recoms.append({
            "title": "Review Technical Support & Security Coverage (Illustrative Advisory)",
            "desc": "Subscribers without dedicated security or technical support show higher churn propensity. Consider reviewing available tech support packages or cybersecurity add-ons with the customer."
        })

    # High Bill / Value Optimization
    monthly_bill = float(customer.get("MonthlyCharges", 0.0))
    if monthly_bill >= 75.0:
        recoms.append({
            "title": "Conduct Value & Plan Suitability Review (Illustrative Advisory)",
            "desc": f"The customer's monthly charge of ${monthly_bill:.2f} is in the higher tier of spend. A proactive account review to ensure plan alignment and service satisfaction may help reduce bill friction."
        })

    # Early Tenure Onboarding
    tenure_val = int(customer.get("tenure", 0))
    if tenure_val <= 12:
        recoms.append({
            "title": "Early-Tenure Onboarding Check-in (Illustrative Advisory)",
            "desc": f"With {tenure_val} months of tenure, this customer belongs to the early-tenure cohort where attrition is highest. A proactive customer success check-in can address initial setup questions and improve early engagement."
        })

    # Fiber optic service friction check
    if customer.get("InternetService") == "Fiber optic" and len(recoms) < 4:
        recoms.append({
            "title": "Service Quality & Speed Assurance (Illustrative Advisory)",
            "desc": "Fiber optic accounts exhibit elevated churn in this dataset despite premium pricing. Verifying speed satisfaction, Wi-Fi coverage, and equipment reliability may help protect customer retention."
        })

    # Ensure at least 2 recommendations
    if not recoms:
        recoms.append({
            "title": "Routine Account Engagement (Illustrative Advisory)",
            "desc": "Customer exhibits stable tenure and service patterns. Standard periodic check-ins, satisfaction surveys, and loyalty acknowledgments are recommended to maintain engagement."
        })

    return recoms[:4]
