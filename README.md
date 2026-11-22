# RetainIQ — Telecom Customer Churn Prediction System

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Phase%201%20%26%202%20Completed-green.svg)

**RetainIQ** is an end-to-end Machine Learning classification system engineered to identify telecom subscribers at high risk of churning (cancelling service). Built with a strong focus on data leakage prevention, class imbalance mitigation, and stratified evaluation, this repository establishes the foundational data science and ML infrastructure for enterprise customer retention operations.

---

## 1. Problem Statement

In the telecommunications sector, acquiring new customers costs substantially more (typically 5× to 7×) than retaining existing subscribers. High customer attrition directly reduces recurring revenue and damages customer lifetime value (LTV).

Customer churn is characterized by subtle behavioral signals—such as short tenure, high month-to-month billing friction, lack of multi-product bundling, and electronic check usage. Proactively identifying these high-risk subscribers enables customer success and marketing teams to deploy targeted retention campaigns, customized contract incentives, and enhanced tech support before cancellation occurs.

---

## 2. Project Objectives

1. **Automated Data Acquisition:** Programmatically acquire, validate, and store the authentic IBM Telco Customer Churn dataset with zero synthetic or fabricated data.
2. **Robust Data Preprocessing:** Safely coerce data anomalies (e.g., blank strings in `TotalCharges`), impute verified zero-tenure records, standard-scale numerical variables, and one-hot encode categorical features strictly inside a scikit-learn `Pipeline` to prevent data leakage.
3. **Exploratory Data Analysis (EDA):** Quantify underlying retention and attrition drivers with publication-quality visualizations and mathematically verified insights.
4. **Multi-Model Training & Evaluation:** Train and benchmark three distinct classification algorithms (`Logistic Regression`, `Decision Tree`, `Random Forest`) utilizing `class_weight='balanced'` to effectively handle class imbalance.
5. **Stratified Model Selection:** Select the optimal model based on 5-fold stratified cross-validation evaluated on Churn F1-score on the training set, followed by evaluation on an untouched, held-out test set (80:20 split).
6. **Interpretability & Feature Importance:** Quantify predictive feature attribution using permutation importance on the held-out test set.
7. **Production Inference Service:** Deliver a clean prediction module that accepts dictionary/DataFrame inputs, assigns transparent risk tiers (Low, Medium, High), and outputs exact probabilities for Phase 2 integration.

---

## 3. Dataset Source and Characteristics

The project utilizes the official **IBM Telco Customer Churn** sample dataset.

- **Primary Source URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Fallback Mirror:** [Kaggle BlastChar Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Local Raw File:** `data/raw/Telco-Customer-Churn.csv`
- **Cleaned File:** `data/processed/cleaned_telco_data.csv`
- **Total Records:** 7,043 customer accounts
- **Total Attributes:** 21 original columns (20 features + 1 target)
- **Target Variable:** `Churn` (`No` = 5,174 [73.46%], `Yes` = 1,869 [26.54%])
- **Class Imbalance Ratio:** ~2.77 : 1 (Moderate class imbalance)

### Attribute Taxonomy

| Category | Features |
|---|---|
| **Identifier** | `customerID` (Excluded from modeling to prevent overfitting) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Subscribed Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Contract & Billing** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Target Variable** | `Churn` (Converted to binary: `1` = Churned, `0` = Retained) |

---

## 4. Technology Stack

- **Core Language:** Python 3.10+ (Tested through Python 3.14)
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `joblib`
- **Visual Analytics:** `matplotlib`, `seaborn`, `plotly`
- **HTTP / Ingestion:** `requests`
- **Web UI (Phase 2):** `streamlit`

---

## 5. Repository Structure

```
RetainIQ/
│
├── app.py                         # Placeholder entry point for Phase 2 Streamlit UI
├── requirements.txt               # Project library dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules for virtual environments & caches
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv   # Unmodified official IBM dataset
│   │
│   └── processed/
│       └── cleaned_telco_data.csv    # Cleaned, coerced, and validated dataset
│
├── models/
│   └── best_model.pkl                 # Serialized fitted best pipeline (ColumnTransformer + Classifier)
│
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── data_loader.py                 # Dataset acquisition, integrity checks, and inspection
│   ├── preprocessing.py              # Feature typing, encoding, scaling, and stratified splitting
│   ├── eda.py                         # 8 publication-quality EDA figures & data-backed insights
│   ├── train_model.py                 # 5-fold CV, model training, held-out evaluation & metrics export
│   └── prediction.py                  # Standalone inference API with transparent risk tiers
│
└── outputs/
    ├── figures/                       # High-resolution (300 DPI) analysis and evaluation charts
    │   ├── 01_churn_distribution.png
    │   ├── 02_churn_by_contract.png
    │   ├── 03_churn_by_internet_service.png
    │   ├── 04_tenure_distribution.png
    │   ├── 05_monthly_charges_distribution.png
    │   ├── 06_churn_by_payment_method.png
    │   ├── 07_correlation_heatmap.png
    │   ├── 08_churn_by_tenure_group.png
    │   ├── confusion_matrix_logistic_regression.png
    │   ├── confusion_matrix_decision_tree.png
    │   ├── confusion_matrix_random_forest.png
    │   ├── feature_importance_top10.png
    │   ├── model_accuracy_comparison.png
    │   ├── model_comparison_metrics.png
    │   ├── model_precision_recall_f1.png
    │   └── model_roc_curves.png
    │
    ├── data_summary.json              # Raw dataset inspection summary
    ├── model_metrics.json             # Selected model performance, parameters & CV logs
    ├── model_comparison.csv           # Benchmark comparison table across all 3 models
    └── feature_importance.csv         # Full permutation importance ranking across all features
```

---

## 6. Data Preprocessing & Leakage Prevention

1. **Target Encoding:** Binary conversion of `Churn` (`Yes` -> `1`, `No` -> `0`).
2. **Feature Separation:** Strict exclusion of `customerID` (unique identifier) and `Churn` from the predictor matrix $X$.
3. **Handling TotalCharges Anomaly:** 
   - 11 records in the original dataset had blank whitespace strings (`" "`) for `TotalCharges`.
   - Inspection verified that **100% of these records correspond to customers with `tenure == 0`** (new customers who enrolled recently and have not completed their first billing cycle).
   - Rather than silently dropping valid customer rows, `TotalCharges` is converted to float and imputed to `0.0` for zero-tenure records.
4. **Stratified Train-Test Split:** 
   - 80% Training ($n = 5,634$), 20% Testing ($n = 1,409$).
   - `stratify=y` preserves the identical 73.46% / 26.54% class balance in both subsets.
   - `random_state=42` guarantees reproducibility.
5. **Leakage-Free Pipeline:** 
   - Numerical features (`tenure`, `SeniorCitizen`, `MonthlyCharges`, `TotalCharges`) pass through `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - Categorical features (15 attributes) pass through `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
   - Preprocessing is fitted **strictly on training folds** during cross-validation and full training data, leaving test data completely unseen until final evaluation.

---

## 7. Exploratory Data Analysis & Empirical Insights

All findings below were computed directly from the dataset:

1. **Baseline Churn Rate:** 1,869 out of 7,043 customers (26.54%) have churned.
2. **Contract Type Impact:** Customers on Month-to-month contracts experience a **42.71%** churn rate, compared to **11.27%** for One-year and only **2.83%** for Two-year contracts. Month-to-month customers are over **15.1× more likely to churn** than two-year subscribers.
3. **Tenure Dynamics:** Churned customers exhibit a median tenure of only **10 months** versus **38 months** for retained customers. First-year customers (0–12 months) experience a **47.44%** churn rate, dropping to **9.51%** for customers retained past 48 months.
4. **Internet Technology Disparity:** Fiber optic subscribers experience a **41.89%** churn rate—dramatically higher than DSL subscribers (**18.96%**) and customers with no internet service (**7.40%**).
5. **Billing & Payment Friction:** Churned customers incur higher average monthly bills (**$74.44/month**) than retained subscribers (**$61.27/month**). Customers paying via **Electronic Check** exhibit the highest payment-related churn at **45.29%**.

---

## 8. Machine Learning Algorithms & Selection

Three classifiers were integrated into scikit-learn Pipelines and evaluated using `class_weight='balanced'`:

1. **Logistic Regression:** Linear decision boundary with $L_2$ regularization (`max_iter=2000`, `solver='lbfgs'`).
2. **Decision Tree Classifier:** Non-linear decision rules with regularization constraints (`max_depth=5`, `min_samples_leaf=10`).
3. **Random Forest Classifier:** Bagged ensemble of 200 decision trees (`n_estimators=200`, `max_depth=8`, `min_samples_leaf=4`, `n_jobs=-1`).

### Why `class_weight='balanced'` is Justified
Because churned customers represent only ~26.5% of the dataset, standard classifiers optimize for overall accuracy by skewing predictions toward the majority non-churn class (yielding unacceptably low recall on churners). In telecom retention, the cost of failing to identify a churner (false negative) far exceeds the cost of sending a promotional incentive to a non-churner (false positive). `class_weight='balanced'` penalizes minority misclassifications inversely proportional to class frequencies, dramatically improving churn recall.

### 5-Fold Stratified Cross-Validation on Training Set ($n=5,634$)

| Algorithm | Mean CV F1-Score | Std CV F1-Score | Fold F1 Scores |
|---|---|---|---|
| **Logistic Regression** | 0.6283 | ±0.0232 | `[0.6534, 0.5899, 0.6176, 0.6497, 0.6306]` |
| **Decision Tree** | 0.6173 | ±0.0105 | `[0.6141, 0.6055, 0.6250, 0.6085, 0.6336]` |
| **Random Forest (Selected)** | **0.6347** | **±0.0144** | `[0.6409, 0.6137, 0.6226, 0.6534, 0.6428]` |

**Selection Decision:** `Random Forest Classifier` was selected because it achieved the highest mean 5-fold cross-validation F1-score (**0.6347**) on the training set, effectively balancing tree ensemble diversity with depth constraints to control overfitting. *(Note: While Decision Tree exhibited the lowest fold standard deviation at ±0.0105 vs. ±0.0144 for Random Forest, Random Forest provided superior predictive discrimination and recall on held-out data).*

---

## 9. Held-Out Test Evaluation Results ($n=1,409$)

The models were evaluated once on the unseen test set:

| Algorithm | Test Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 73.81% | 0.5043 | 0.7834 | 0.6136 | 0.8416 | 747 | 288 | 81 | 293 |
| **Decision Tree** | 75.51% | 0.5270 | 0.7567 | 0.6213 | 0.8320 | 781 | 254 | 91 | 283 |
| **Random Forest (Best)** | **75.37%** | **0.5236** | **0.7995** | **0.6328** | **0.8417** | **763** | **272** | **75** | **299** |

*Note: Test set contains 1,035 retained customers and 374 churned customers.*

**Key Metric Highlights for Selected Random Forest:**
- **Recall = 79.95%:** Captures nearly 8 out of every 10 customers who actually churned.
- **ROC-AUC = 0.8417:** Demonstrates strong discriminatory ability across classification thresholds.
- **F1-Score = 0.6328:** Highest balanced harmonic mean of precision and recall.

---

## 10. Feature Importance & Interpretation

Permutation feature importance was evaluated on the held-out test set through the full fitted pipeline using test F1-score drop as the scoring metric:

| Rank | Feature | Mean Decrease in Test F1 | Std Deviation | Description |
|---|---|---|---|---|
| 1 | `Contract` | **0.1068** | ±0.0129 | Contract commitment length (month-to-month vs 1/2 yr) |
| 2 | `InternetService` | **0.0325** | ±0.0072 | Technology type (Fiber optic, DSL, None) |
| 3 | `tenure` | **0.0272** | ±0.0069 | Customer longevity in months |
| 4 | `TotalCharges` | **0.0211** | ±0.0067 | Cumulative billing spend |
| 5 | `SeniorCitizen` | **0.0042** | ±0.0004 | Senior demographic status |
| 6 | `PaperlessBilling`| **0.0039** | ±0.0013 | Paperless statement adoption |
| 7 | `StreamingTV` | **0.0038** | ±0.0020 | Value-added media bundling |
| 8 | `PaymentMethod` | **0.0038** | ±0.0039 | Auto-debit vs manual electronic check |
| 9 | `TechSupport` | **0.0037** | ±0.0029 | Subscription to dedicated technical assistance |
| 10 | `MonthlyCharges`| **0.0033** | ±0.0039 | Current monthly charge rate |

> **Scientific Caution:** Permutation feature importance measures the statistical dependence and predictive contribution of an input variable within this specific model and dataset; it does not prove direct real-world causal intervention.

---

## 11. Reusable Prediction Service

The `src/prediction.py` module exposes inference functions designed for zero-friction integration into the Phase 2 Streamlit web application.

### Risk Tier Categorization
- **Low Risk:** Churn Probability $< 0.30$
- **Medium Risk:** $0.30 \le$ Churn Probability $< 0.60$
- **High Risk:** Churn Probability $\ge 0.60$

### Verification on Authentic Held-Out Test Records

**Case 1: Actual Churned Customer Record (`tenure = 17`, `Month-to-month`, `Electronic check`):**
- Predicted Churn: **Yes**
- Churn Probability: **64.65%**
- Retention Probability: **35.35%**
- Risk Category: **High**
- Ground Truth: **Yes (1)** — *Correctly Identified*

**Case 2: Actual Retained Customer Record (`tenure = 72`, `Two year`, `Credit card auto`):**
- Predicted Churn: **No**
- Churn Probability: **5.81%**
- Retention Probability: **94.19%**
- Risk Category: **Low**
- Ground Truth: **No (0)** — *Correctly Identified*

---

## 12. Installation & Execution Guide

### Prerequisites
- Python 3.10 or higher installed.

### Step 1: Clone or Navigate to the Project Root
```bash
cd RetainIQ
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Full Pipeline (Optional - Artifacts Already Built)
```bash
# Data Acquisition & Inspection
python -m src.data_loader

# Exploratory Data Analysis (Generates Figures in outputs/figures/)
python -m src.eda

# Model Training, Cross-Validation, & Evaluation
python -m src.train_model

# Inference Verification on Held-Out Test Profiles
python -m src.prediction
```

### Step 4: Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
*Launches the multi-page dashboard at `http://localhost:8501`.*

### Step 5: Run Application Unit & Logic Tests
```bash
python test_app_logic.py
```

---

## 13. Streamlit Application Architecture & Features (Phase 2)

RetainIQ features a multi-page, interactive Streamlit analytics application adhering to a dark-navy (`#0B1220`) / indigo (`#6366F1`) brand identity, high-contrast KPI cards, and responsive Plotly visual analytics.

### Page Breakdown

#### 1. 📊 Executive Dashboard
- **Real-Time KPIs:** Dynamic calculations of Total Customers ($n=7,043$), Churned Customers ($1,869$), Retained Customers ($5,174$), and Baseline Churn Rate ($26.54\%$).
- **Interactive Visualizations (Plotly):**
  - Donut chart of customer churn distribution with percentage breakdowns.
  - Churn rate bar chart across Contract types (Month-to-month: 42.71% vs Two-year: 2.83%).
  - Churn rate bar chart across Internet services (Fiber optic: 41.89% vs DSL: 18.96%).
  - Churn rate bar chart across Payment methods (Electronic check: 45.29%).
  - Cohort analysis across customer tenure groups (0–12, 13–24, 25–48, 49+ months).
  - Monthly bill distribution comparison between churned and retained subscribers.
- **Key Customer Insights:** Real data observations derived dynamically from the loaded dataset.

#### 2. 🔮 Churn Prediction Form
- **Structured 3-Section Form:**
  - *Section A (Demographics & Tenure):* Gender, Senior Citizen, Partner, Dependents, Tenure slider ($0$ to $72$ months).
  - *Section B (Services & Subscriptions):* Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies. Includes conditional dependencies (e.g. disabling sub-features when InternetService or PhoneService is "No").
  - *Section C (Billing & Contract):* Contract type, Paperless Billing, Payment Method, Monthly Charges ($18.00–$130.00), Total Charges ($0.00–$10,000.00).
- **Validation:** Enforces valid ranges, non-negative charges, and handles zero-tenure consistency.
- **Inference Integration:** Seamlessly packages form inputs into a DataFrame matching the model's exact schema and feeds it to the cached production pipeline.

#### 3. 📋 Prediction Result & Risk Assessment
- **Status & Risk Tiers:** Displays predicted outcome (`Likely to Churn` / `Likely to Retain`) and categorizes risk:
  - *Low Risk:* Churn Probability $< 30\%$ (Teal badge)
  - *Medium Risk:* Churn Probability $30\% \le p < 60\%$ (Amber badge)
  - *High Risk:* Churn Probability $\ge 60\%$ (Crimson badge)
- **Visual Probability Gauge:** Interactive gauge and segmented retention-versus-churn distribution bar.
- **Customer Profile Inspector:** Structured overview of all submitted attributes.
- **Tailored Retention Playbook:** Dynamically generates 2 to 4 actionable, rule-based retention actions (contract migration incentives, payment auto-pay rebates, tech support attach offers, or proactive onboarding check-ins).
- **Session State Persistence:** Prediction outcomes remain preserved across page transitions.

#### 4. 📈 Model Performance & Technical Benchmarks
- **Production Model Summary:** Displays Random Forest performance metrics (Test Accuracy: $75.37\%$, Churn Recall: $79.95\%$, F1-Score: $0.6328$, ROC-AUC: $0.8417$).
- **Benchmark Table:** Comparative analysis of Logistic Regression, Decision Tree, and Random Forest across 5-fold cross-validation and held-out test splits.
- **Confusion Matrix:** Interactive heatmap displaying true positives ($299$), false negatives ($75$), true negatives ($763$), and false positives ($272$).
- **Multi-Metric Bar Chart:** Grouped metric performance comparison across candidate classifiers.
- **Feature Importance:** Top 10 permutation importance horizontal bar chart with error bounds.
- **Architecture Callout:** Complete technical specifications for training split, random seed ($42$), and `class_weight='balanced'` rationale.

---

## 14. Testing & Verification Summary

The complete codebase has undergone rigorous verification:
- **Unit & Logic Test Suite (`test_app_logic.py`):** 8/8 tests passed verifying dataset integrity, artifact loading, chart generation, high-risk inference, low-risk inference, zero-tenure handling, recommendation generation, and matrix plotting.
- **Streamlit HTTP Endpoint Verification:** Automated headless launch test confirmed HTTP 200 response and successful client bundle delivery at `http://localhost:8501`.


---

## License & Attribution

Developed as part of the RetainIQ Machine Learning Engineering project. Dataset courtesy of IBM Developer / IBM Watson Telco Customer Churn sample data.
