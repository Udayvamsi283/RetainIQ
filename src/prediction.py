"""
RetainIQ - Telecom Customer Churn Prediction System
Inference and Prediction Module

This module provides reusable prediction capabilities to be consumed by the
Phase 2 Streamlit web application:
1. Loading the serialized best model pipeline (models/best_model.pkl).
2. Accepting customer inputs as Python dictionaries or pandas DataFrames.
3. Validating feature presence and data types.
4. Performing model inference with predict() and predict_proba().
5. Classifying churn risk category into Low, Medium, or High based on transparent thresholds:
   - Low:    churn probability < 0.30
   - Medium: 0.30 <= churn probability < 0.60
   - High:   churn probability >= 0.60
6. Returning clean, structured prediction dictionaries.
"""

import os
import logging
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join("models", "best_model.pkl")
EXPECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the fitted scikit-learn Pipeline from disk.
    """
    global _CACHED_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python -m src.train_model' to generate the model first."
        )
    if _CACHED_MODEL is None:
        logger.info(f"Loading serialized model pipeline from '{model_path}'...")
        _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def validate_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that all expected features are present in the input DataFrame.
    Coerces numerical types, checks value ranges, and ensures consistent column ordering.
    Strictly excludes customerID and Churn.
    """
    # Exclude customerID and Churn if passed accidentally
    cleaned_input = input_df.copy()
    if "customerID" in cleaned_input.columns:
        cleaned_input = cleaned_input.drop(columns=["customerID"])
    if "Churn" in cleaned_input.columns:
        cleaned_input = cleaned_input.drop(columns=["Churn"])

    missing_cols = [col for col in EXPECTED_FEATURES if col not in cleaned_input.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing expected feature(s): {missing_cols}")

    df_validated = cleaned_input[EXPECTED_FEATURES].copy()

    # Ensure numerical columns are properly formatted floats/ints and check ranges
    for col in NUMERICAL_FEATURES:
        df_validated[col] = pd.to_numeric(df_validated[col], errors="coerce")
        if df_validated[col].isnull().any():
            # If invalid numeric string was passed, impute 0.0 for safety
            df_validated[col] = df_validated[col].fillna(0.0)

    # Range validations
    if (df_validated["tenure"] < 0).any() or (df_validated["tenure"] > 72).any():
        raise ValueError("tenure must be between 0 and 72 months.")
    if (df_validated["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges must be non-negative.")
    if (df_validated["TotalCharges"] < 0).any():
        raise ValueError("TotalCharges must be non-negative.")

    # Ensure categorical columns are strings
    for col in CATEGORICAL_FEATURES:
        df_validated[col] = df_validated[col].astype(str)

    return df_validated


def get_risk_category(churn_probability: float) -> str:
    """
    Maps continuous churn probability to transparent presentation risk categories:
    - Low:    churn probability < 0.30
    - Medium: 0.30 <= churn probability < 0.60
    - High:   churn probability >= 0.60
    """
    if churn_probability < 0.30:
        return "Low"
    elif churn_probability < 0.60:
        return "Medium"
    else:
        return "High"


def predict_churn(
    customer_input: Union[Dict[str, Any], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH,
    model=None
) -> Dict[str, Any]:
    """
    Predicts churn for a single customer or DataFrame record.
    Returns structured results including prediction label, exact probabilities, and risk tier.
    Guarantees positive class (1 = Churn) probability mapping and exact threshold evaluation.
    """
    if model is None:
        model = load_model(model_path)

    if isinstance(customer_input, dict):
        input_df = pd.DataFrame([customer_input])
    elif isinstance(customer_input, pd.DataFrame):
        input_df = customer_input.copy()
    else:
        raise TypeError(f"customer_input must be a dict or pd.DataFrame, got {type(customer_input)}")

    validated_df = validate_input_data(input_df)

    # Perform inference via full fitted pipeline
    raw_pred = model.predict(validated_df)[0]
    prediction_label = "Yes" if int(raw_pred) == 1 else "No"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(validated_df)[0]
        # Dynamically determine index of churn (1) and retention (0) classes
        classes = list(model.classes_)
        churn_idx = classes.index(1) if 1 in classes else 1
        retention_idx = classes.index(0) if 0 in classes else 0
        retention_prob = float(probabilities[retention_idx])
        churn_prob = float(probabilities[churn_idx])
    else:
        churn_prob = float(raw_pred)
        retention_prob = 1.0 - churn_prob

    risk_category = get_risk_category(churn_prob)

    result = {
        "prediction": prediction_label,
        "churn_probability": round(churn_prob, 4),
        "retention_probability": round(retention_prob, 4),
        "risk_category": risk_category,
        "raw_prediction": int(raw_pred),
        "decision_threshold": 0.50,
        "risk_thresholds": {
            "low": "< 0.30",
            "medium": "0.30 to < 0.60",
            "high": ">= 0.60"
        },
        "input_features": validated_df.iloc[0].to_dict()
    }
    return result


def predict_batch(
    customers_df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    model=None
) -> pd.DataFrame:
    """
    Performs batch predictions over multiple customer records.
    Returns DataFrame augmented with churn prediction, probability, and risk level.
    """
    if model is None:
        model = load_model(model_path)

    validated_df = validate_input_data(customers_df)
    preds = model.predict(validated_df)
    probs = model.predict_proba(validated_df)

    result_df = validated_df.copy()
    result_df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    result_df["Churn_Probability"] = np.round(probs[:, 1], 4)
    result_df["Retention_Probability"] = np.round(probs[:, 0], 4)
    result_df["Risk_Category"] = [get_risk_category(p) for p in probs[:, 1]]

    return result_df


def test_prediction_with_real_record():
    """
    Validates prediction functionality using an actual held-out customer record from the test set.
    """
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_telco_data, separate_features_and_target, split_data

    logger.info("Testing prediction module with an authentic held-out test customer...")
    raw_df = load_raw_data()
    cleaned_df = clean_telco_data(raw_df, save_processed=False)
    X, y = separate_features_and_target(cleaned_df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Pick sample customer records from held-out test data
    # Test record 1: Customer who actually churned
    churned_indices = y_test[y_test == 1].index
    churn_sample = X_test.loc[churned_indices[0]].to_dict()
    actual_churn_status = "Yes (1)"

    result_churn = predict_churn(churn_sample)
    print("\n" + "=" * 70)
    print("PREDICTION TEST 1: REAL HELD-OUT CHURNED CUSTOMER RECORD")
    print("=" * 70)
    print(f"Customer Input Features:")
    for k, v in churn_sample.items():
        print(f"  {k:20s}: {v}")
    print("-" * 70)
    print(f"Actual Ground Truth:     {actual_churn_status}")
    print(f"Predicted Churn Label:   {result_churn['prediction']}")
    print(f"Churn Probability:       {result_churn['churn_probability']:.4f} ({result_churn['churn_probability']*100:.2f}%)")
    print(f"Retention Probability:   {result_churn['retention_probability']:.4f} ({result_churn['retention_probability']*100:.2f}%)")
    print(f"Assigned Risk Category:  {result_churn['risk_category']}")
    print("=" * 70)

    # Test record 2: Customer who was retained
    retained_indices = y_test[y_test == 0].index
    retained_sample = X_test.loc[retained_indices[0]].to_dict()
    actual_retained_status = "No (0)"

    result_retained = predict_churn(retained_sample)
    print("\n" + "=" * 70)
    print("PREDICTION TEST 2: REAL HELD-OUT RETAINED CUSTOMER RECORD")
    print("=" * 70)
    print(f"Customer Input Features:")
    for k, v in retained_sample.items():
        print(f"  {k:20s}: {v}")
    print("-" * 70)
    print(f"Actual Ground Truth:     {actual_retained_status}")
    print(f"Predicted Churn Label:   {result_retained['prediction']}")
    print(f"Churn Probability:       {result_retained['churn_probability']:.4f} ({result_retained['churn_probability']*100:.2f}%)")
    print(f"Retention Probability:   {result_retained['retention_probability']:.4f} ({result_retained['retention_probability']*100:.2f}%)")
    print(f"Assigned Risk Category:  {result_retained['risk_category']}")
    print("=" * 70 + "\n")

    assert result_churn["risk_category"] in ["Low", "Medium", "High"], "Invalid risk category"
    assert 0.0 <= result_churn["churn_probability"] <= 1.0, "Invalid probability"
    logger.info("Prediction module validation completed successfully.")


if __name__ == "__main__":
    test_prediction_with_real_record()
