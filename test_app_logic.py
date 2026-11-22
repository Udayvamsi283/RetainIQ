"""
RetainIQ - Comprehensive Test Suite for Streamlit Application and UI Logic
Verifies dataset integrity, artifact loading, model inference, boundary thresholds,
input validation, edge cases, chart generation, and illustrative recommendations.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.ui_helpers import (
    load_processed_data,
    load_model_artifacts,
    get_cached_pipeline,
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
)
from src.prediction import (
    predict_churn,
    get_risk_category,
    validate_input_data,
    EXPECTED_FEATURES
)


class TestRetainIQApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_processed_data()
        cls.artifacts = load_model_artifacts()
        cls.pipeline = get_cached_pipeline()

    def test_01_dataset_integrity(self):
        """Verifies that the cleaned dataset loads correctly with authentic dimensions."""
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 7043)
        self.assertIn("Churn", self.df.columns)
        self.assertIn("tenure", self.df.columns)
        self.assertIn("MonthlyCharges", self.df.columns)
        self.assertIn("TotalCharges", self.df.columns)
        self.assertEqual(self.df["TotalCharges"].isnull().sum(), 0)

    def test_02_model_artifacts_integrity(self):
        """Verifies that all Phase 1 model artifacts load with real saved values."""
        metrics = self.artifacts.get("metrics")
        comparison = self.artifacts.get("comparison")
        fi = self.artifacts.get("feature_importance")

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["selected_model"], "Random Forest")
        self.assertIn("held_out_test_metrics", metrics)
        self.assertGreater(metrics["held_out_test_metrics"]["f1_score"], 0.6)

        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison), 3)  # Logistic Regression, Decision Tree, Random Forest

        self.assertIsNotNone(fi)
        self.assertGreaterEqual(len(fi), 10)
        self.assertEqual(fi.iloc[0]["Feature"], "Contract")

    def test_03_dashboard_charts_generation(self):
        """Verifies that all 6 dashboard Plotly figures generate without exceptions."""
        fig_donut = create_churn_donut(self.df)
        self.assertIsNotNone(fig_donut)

        fig_contract = create_contract_bar(self.df)
        self.assertIsNotNone(fig_contract)

        fig_internet = create_internet_service_bar(self.df)
        self.assertIsNotNone(fig_internet)

        fig_pm = create_payment_method_bar(self.df)
        self.assertIsNotNone(fig_pm)

        fig_tenure = create_tenure_cohort_bar(self.df)
        self.assertIsNotNone(fig_tenure)

        fig_mc = create_monthly_charges_dist(self.df)
        self.assertIsNotNone(fig_mc)

    def test_04_high_risk_prediction(self):
        """Tests prediction on a typical high-risk churn profile (M2M, short tenure, electronic check)."""
        high_risk_sample = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 159.70
        }
        res = predict_churn(high_risk_sample, model=self.pipeline)
        self.assertEqual(res["prediction"], "Yes")
        self.assertGreaterEqual(res["churn_probability"], 0.50)
        self.assertEqual(res["risk_category"], "High")
        self.assertAlmostEqual(res["churn_probability"] + res["retention_probability"], 1.0, places=4)

    def test_05_low_risk_prediction(self):
        """Tests prediction on a typical low-risk loyal profile (Two-year, high tenure, auto-pay)."""
        low_risk_sample = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 68,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 64.80,
            "TotalCharges": 4406.40
        }
        res = predict_churn(low_risk_sample, model=self.pipeline)
        self.assertEqual(res["prediction"], "No")
        self.assertLess(res["churn_probability"], 0.30)
        self.assertEqual(res["risk_category"], "Low")
        self.assertAlmostEqual(res["churn_probability"] + res["retention_probability"], 1.0, places=4)

    def test_06_edge_case_zero_tenure(self):
        """Tests edge case with newly joined customer (tenure=0, TotalCharges=0)."""
        zero_tenure_sample = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 52.55,
            "TotalCharges": 0.0
        }
        res = predict_churn(zero_tenure_sample, model=self.pipeline)
        self.assertIn(res["prediction"], ["Yes", "No"])
        self.assertIn(res["risk_category"], ["Low", "Medium", "High"])
        self.assertAlmostEqual(res["churn_probability"] + res["retention_probability"], 1.0, places=4)

    def test_07_retention_recommendations(self):
        """Tests that rule-based illustrative recommendations are appropriately generated."""
        sample = {
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "MonthlyCharges": 85.0,
            "tenure": 5,
            "InternetService": "Fiber optic"
        }
        recoms = generate_retention_recommendations(sample, 0.72)
        self.assertGreaterEqual(len(recoms), 2)
        self.assertLessEqual(len(recoms), 4)
        for r in recoms:
            self.assertIn("Illustrative Advisory", r["title"])

    def test_08_model_performance_visualizations(self):
        """Tests that confusion matrix and feature importance charts generate properly."""
        metrics = self.artifacts.get("metrics")
        fi_df = self.artifacts.get("feature_importance")
        cm = metrics["held_out_test_metrics"]["confusion_matrix"]

        cm_fig = create_confusion_matrix_heatmap(cm)
        self.assertIsNotNone(cm_fig)

        fi_fig = create_feature_importance_chart(fi_df)
        self.assertIsNotNone(fi_fig)

    def test_09_risk_threshold_boundaries(self):
        """Tests exact risk threshold boundaries (<0.30 Low, 0.30-0.599 Medium, >=0.60 High)."""
        self.assertEqual(get_risk_category(0.0), "Low")
        self.assertEqual(get_risk_category(0.2999), "Low")
        self.assertEqual(get_risk_category(0.30), "Medium")
        self.assertEqual(get_risk_category(0.45), "Medium")
        self.assertEqual(get_risk_category(0.5999), "Medium")
        self.assertEqual(get_risk_category(0.60), "High")
        self.assertEqual(get_risk_category(0.85), "High")
        self.assertEqual(get_risk_category(1.0), "High")

    def test_10_negative_and_invalid_inputs(self):
        """Verifies that invalid or out-of-range numerical values raise ValueError."""
        base_sample = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 500.0
        }

        # Negative tenure
        bad_tenure = base_sample.copy()
        bad_tenure["tenure"] = -5
        with self.assertRaises(ValueError):
            validate_input_data(pd.DataFrame([bad_tenure]))

        # Tenure > 72
        over_tenure = base_sample.copy()
        over_tenure["tenure"] = 99
        with self.assertRaises(ValueError):
            validate_input_data(pd.DataFrame([over_tenure]))

        # Negative MonthlyCharges
        bad_mc = base_sample.copy()
        bad_mc["MonthlyCharges"] = -10.0
        with self.assertRaises(ValueError):
            validate_input_data(pd.DataFrame([bad_mc]))

        # Negative TotalCharges
        bad_tc = base_sample.copy()
        bad_tc["TotalCharges"] = -100.0
        with self.assertRaises(ValueError):
            validate_input_data(pd.DataFrame([bad_tc]))

    def test_11_exclusion_of_id_and_churn_columns(self):
        """Verifies that passing customerID and Churn in the input dictionary is stripped safely."""
        sample_with_id = {
            "customerID": "9999-XYZAB",
            "Churn": "Yes",
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 500.0
        }
        res = predict_churn(sample_with_id, model=self.pipeline)
        self.assertIn(res["prediction"], ["Yes", "No"])
        self.assertNotIn("customerID", res["input_features"])
        self.assertNotIn("Churn", res["input_features"])

    def test_12_conditional_services_consistency(self):
        """Verifies that conditional dependent categories (e.g. No phone service, No internet service) process cleanly."""
        no_internet_sample = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 20.0, "TotalCharges": 480.0
        }
        res = predict_churn(no_internet_sample, model=self.pipeline)
        self.assertEqual(res["prediction"], "No")
        self.assertLess(res["churn_probability"], 0.30)
        self.assertEqual(res["risk_category"], "Low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
