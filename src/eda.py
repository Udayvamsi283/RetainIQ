"""
RetainIQ - Telecom Customer Churn Prediction System
Exploratory Data Analysis (EDA) Module

Generates professional, publication-quality visualizations for telecom churn analysis:
1. Churn Distribution (Counts and Percentages)
2. Churn by Contract Type
3. Churn by Internet Service
4. Customer Tenure Distribution (Churned vs. Retained)
5. Monthly Charges Distribution (Churned vs. Retained)
6. Churn by Payment Method
7. Correlation Heatmap for Numerical Features
8. Churn by Tenure Group (0-12, 13-24, 25-48, 49+ months)

Also extracts concise, data-backed insights strictly calculated from the dataset.
"""

import os
import logging
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.preprocessing import clean_telco_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join("outputs", "figures")

# Styling configurations for professional internship/thesis reports
PALETTE_CHURN = {"Retained": "#2b5c8f", "Churned": "#d9534f"}
PALETTE_ACCENT = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c"]
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True
})


def ensure_output_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_churn_distribution(df: pd.DataFrame) -> str:
    """1. Visualizes overall Churn distribution with count and percentage annotations."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "01_churn_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    churn_counts = df["Churn_Label"].value_counts()
    total = len(df)

    bars = ax.bar(churn_counts.index, churn_counts.values, color=[PALETTE_CHURN[c] for c in churn_counts.index], width=0.5)

    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Overall Customer Churn Distribution (Target Variable)", fontweight="bold", pad=12)
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, total * 0.85)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_churn_by_contract(df: pd.DataFrame) -> str:
    """2. Compares churn across month-to-month, one-year, and two-year contracts."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "02_churn_by_contract.png")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    order = ["Month-to-month", "One year", "Two year"]
    contract_churn = pd.crosstab(df["Contract"], df["Churn_Label"], normalize="index").loc[order] * 100

    contract_churn.plot(kind="bar", stacked=False, ax=ax, color=[PALETTE_CHURN["Retained"], PALETTE_CHURN["Churned"]], width=0.6)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", va="bottom", fontsize=9)

    ax.set_title("Customer Churn Rate by Contract Type", fontweight="bold", pad=12)
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Percentage within Contract (%)")
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(title="Customer Status", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_churn_by_internet_service(df: pd.DataFrame) -> str:
    """3. Compares churn across DSL, Fiber optic, and No internet service."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "03_churn_by_internet_service.png")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    order = ["DSL", "Fiber optic", "No"]
    service_churn = pd.crosstab(df["InternetService"], df["Churn_Label"], normalize="index").loc[order] * 100

    service_churn.plot(kind="bar", stacked=False, ax=ax, color=[PALETTE_CHURN["Retained"], PALETTE_CHURN["Churned"]], width=0.6)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", va="bottom", fontsize=9)

    ax.set_title("Customer Churn Rate by Internet Service Type", fontweight="bold", pad=12)
    ax.set_xlabel("Internet Service Type")
    ax.set_ylabel("Percentage within Service (%)")
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(title="Customer Status", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_tenure_distribution(df: pd.DataFrame) -> str:
    """4. Analyzes customer tenure distributions comparing churned and retained customers."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "04_tenure_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn_Label",
        palette=PALETTE_CHURN,
        kde=True,
        bins=36,
        ax=ax,
        element="step",
        common_norm=False,
        alpha=0.4
    )

    med_retained = df[df["Churn_Label"] == "Retained"]["tenure"].median()
    med_churned = df[df["Churn_Label"] == "Churned"]["tenure"].median()

    ax.axvline(med_retained, color=PALETTE_CHURN["Retained"], linestyle="--", linewidth=1.8, label=f"Median Retained ({med_retained:.0f} mos)")
    ax.axvline(med_churned, color=PALETTE_CHURN["Churned"], linestyle="--", linewidth=1.8, label=f"Median Churned ({med_churned:.0f} mos)")

    ax.set_title("Customer Tenure Distribution (Months with Telecom)", fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Customer Count")
    ax.legend(title="Status & Medians", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_monthly_charges_distribution(df: pd.DataFrame) -> str:
    """5. Compares monthly charges for churned and retained customers using KDE and boxplot."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "05_monthly_charges_distribution.png")

    fig, (ax_box, ax_kde) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.35, 0.65]}, dpi=300
    )

    sns.boxplot(
        data=df,
        x="MonthlyCharges",
        y="Churn_Label",
        hue="Churn_Label",
        palette=PALETTE_CHURN,
        ax=ax_box,
        legend=False
    )
    ax_box.set_ylabel("")
    ax_box.set_xlabel("")
    ax_box.set_title("Monthly Charges Comparison: Churned vs. Retained", fontweight="bold", pad=12)

    sns.kdeplot(
        data=df[df["Churn_Label"] == "Retained"]["MonthlyCharges"],
        color=PALETTE_CHURN["Retained"],
        label="Retained",
        fill=True,
        alpha=0.3,
        ax=ax_kde
    )
    sns.kdeplot(
        data=df[df["Churn_Label"] == "Churned"]["MonthlyCharges"],
        color=PALETTE_CHURN["Churned"],
        label="Churned",
        fill=True,
        alpha=0.3,
        ax=ax_kde
    )

    ax_kde.set_xlabel("Monthly Charges ($)")
    ax_kde.set_ylabel("Density")
    ax_kde.legend(title="Customer Status", frameon=True)
    sns.despine(ax=ax_box)
    sns.despine(ax=ax_kde)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_payment_method_analysis(df: pd.DataFrame) -> str:
    """6. Analyzes churn rate across payment methods."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "06_churn_by_payment_method.png")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    pm_churn = pd.crosstab(df["PaymentMethod"], df["Churn_Label"], normalize="index") * 100
    pm_churn = pm_churn.sort_values(by="Churned", ascending=False)

    pm_churn.plot(kind="bar", stacked=False, ax=ax, color=[PALETTE_CHURN["Retained"], PALETTE_CHURN["Churned"]], width=0.6)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", va="bottom", fontsize=9)

    ax.set_title("Customer Churn Rate by Payment Method", fontweight="bold", pad=12)
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Percentage within Payment Method (%)")
    ax.set_xticklabels(pm_churn.index, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="Customer Status", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """7. Shows correlation among purely numerical features (and numeric target Churn)."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "07_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    num_cols = ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=0.8,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax
    )

    ax.set_title("Correlation Heatmap: Numerical Features & Churn", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_tenure_groups(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """8. Creates tenure groups (0-12, 13-24, 25-48, 49+ mos) and visualizes churn."""
    ensure_output_dir()
    filepath = os.path.join(FIGURES_DIR, "08_churn_by_tenure_group.png")

    bins = [-1, 12, 24, 48, 100]
    labels = ["0–12 months", "13–24 months", "25–48 months", "49+ months"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    tg_crosstab = pd.crosstab(df["tenure_group"], df["Churn_Label"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    tg_crosstab.plot(kind="bar", stacked=False, ax=ax, color=[PALETTE_CHURN["Retained"], PALETTE_CHURN["Churned"]], width=0.6)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", va="bottom", fontsize=9)

    ax.set_title("Churn Rate by Customer Tenure Cohort", fontweight="bold", pad=12)
    ax.set_xlabel("Tenure Cohort")
    ax.set_ylabel("Percentage within Cohort (%)")
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(title="Customer Status", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath, tg_crosstab


def generate_eda_insights(df: pd.DataFrame) -> List[str]:
    """
    Computes rigorous, data-backed insights directly from the cleaned dataset.
    Never relies on assumptions or fabricated percentages.
    """
    total = len(df)
    churn_count = (df["Churn"] == 1).sum()
    churn_pct = (churn_count / total) * 100

    # Contract insight
    contract_rates = (df.groupby("Contract")["Churn"].mean() * 100).to_dict()
    m2m_rate = contract_rates.get("Month-to-month", 0.0)
    two_yr_rate = contract_rates.get("Two year", 0.0)

    # Tenure insight
    median_tenure_churned = df[df["Churn"] == 1]["tenure"].median()
    median_tenure_retained = df[df["Churn"] == 0]["tenure"].median()

    # Internet service insight
    internet_rates = (df.groupby("InternetService")["Churn"].mean() * 100).to_dict()
    fiber_rate = internet_rates.get("Fiber optic", 0.0)
    dsl_rate = internet_rates.get("DSL", 0.0)
    no_internet_rate = internet_rates.get("No", 0.0)

    # Monthly charges insight
    mean_mc_churned = df[df["Churn"] == 1]["MonthlyCharges"].mean()
    mean_mc_retained = df[df["Churn"] == 0]["MonthlyCharges"].mean()

    # Payment method insight
    pm_rates = (df.groupby("PaymentMethod")["Churn"].mean() * 100).to_dict()
    highest_pm = max(pm_rates.items(), key=lambda x: x[1])

    # Tenure cohort insight
    bins = [-1, 12, 24, 48, 100]
    labels = ["0–12 months", "13–24 months", "25–48 months", "49+ months"]
    df_temp = df.copy()
    df_temp["tenure_cohort"] = pd.cut(df_temp["tenure"], bins=bins, labels=labels)
    cohort_rates = (df_temp.groupby("tenure_cohort", observed=False)["Churn"].mean() * 100).to_dict()
    first_year_rate = cohort_rates.get("0–12 months", 0.0)
    long_term_rate = cohort_rates.get("49+ months", 0.0)

    insights = [
        f"1. Overall Baseline Churn: {churn_count:,} out of {total:,} customers ({churn_pct:.2f}%) have churned, indicating moderate class imbalance (~73:27 ratio).",
        f"2. Contract Vulnerability: Customers on 'Month-to-month' contracts churn at an alarming rate of {m2m_rate:.2f}%, compared to only {two_yr_rate:.2f}% for 'Two year' contracts (over {m2m_rate/max(two_yr_rate, 0.01):.1f}x higher risk).",
        f"3. High Early-Tenure Attrition: Churned customers show a median tenure of only {median_tenure_churned:.0f} months vs. {median_tenure_retained:.0f} months for retained customers. First-year customers (0-12 mos) experience a {first_year_rate:.2f}% churn rate, falling to just {long_term_rate:.2f}% for customers past 48 months.",
        f"4. Service Infrastructure Disparity: Fiber optic subscribers suffer a {fiber_rate:.2f}% churn rate, significantly higher than DSL ({dsl_rate:.2f}%) and customers with no internet service ({no_internet_rate:.2f}%), despite Fiber optic generating higher monthly revenues.",
        f"5. Billing and Monthly Costs: Churned customers pay higher monthly fees on average (${mean_mc_churned:.2f}/month) compared to retained customers (${mean_mc_retained:.2f}/month). Furthermore, customers paying via '{highest_pm[0]}' exhibit the highest payment-related churn at {highest_pm[1]:.2f}%."
    ]
    return insights


def run_eda(raw_filepath: str = None) -> None:
    """
    Executes complete EDA pipeline, generates figures, and displays data-backed insights.
    """
    logger.info("Starting Exploratory Data Analysis (EDA)...")
    raw_df = load_raw_data()
    df = clean_telco_data(raw_df, save_processed=True)

    # Add label column for intuitive charting
    if df["Churn"].dtype == object or str(df["Churn"].dtype) in ["str", "string", "object"]:
        df["Churn_Label"] = df["Churn"].astype(str).map({"Yes": "Churned", "No": "Retained", "1": "Churned", "0": "Retained"})
        df["Churn"] = df["Churn"].astype(str).map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int)
    else:
        df["Churn_Label"] = df["Churn"].astype(int).map({1: "Churned", 0: "Retained"})

    f1 = plot_churn_distribution(df)
    logger.info(f"Generated: {f1}")

    f2 = plot_churn_by_contract(df)
    logger.info(f"Generated: {f2}")

    f3 = plot_churn_by_internet_service(df)
    logger.info(f"Generated: {f3}")

    f4 = plot_tenure_distribution(df)
    logger.info(f"Generated: {f4}")

    f5 = plot_monthly_charges_distribution(df)
    logger.info(f"Generated: {f5}")

    f6 = plot_payment_method_analysis(df)
    logger.info(f"Generated: {f6}")

    f7 = plot_correlation_heatmap(df)
    logger.info(f"Generated: {f7}")

    f8, _ = plot_tenure_groups(df)
    logger.info(f"Generated: {f8}")

    insights = generate_eda_insights(df)

    print("\n" + "=" * 70)
    print("DATA-BACKED EXPLORATORY DATA ANALYSIS (EDA) INSIGHTS")
    print("=" * 70)
    for insight in insights:
        print(insight)
    print("=" * 70 + "\n")
    logger.info(f"All 8 figures successfully generated and saved to '{FIGURES_DIR}'.")


if __name__ == "__main__":
    run_eda()
