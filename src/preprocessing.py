"""
RetainIQ - Telecom Customer Churn Prediction System
Data Preprocessing and Pipeline Construction Module

This module provides reusable preprocessing functions and scikit-learn transformers:
1. Target binary encoding (Yes=1, No=0).
2. Feature separation (excluding customerID and target).
3. Explicit feature type categorization (Numerical vs. Categorical).
4. Data cleaning (coercing TotalCharges, imputing zero-tenure charges).
5. Building modular ColumnTransformer preprocessors with SimpleImputer,
   StandardScaler, and OneHotEncoder(handle_unknown='ignore').
6. Stratified train-test splitting (80:20, random_state=42).
7. Strict data leakage prevention (preprocessing is fitted strictly on training data).
"""

import os
import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_telco_data.csv")

# Explicit definition of feature types as required by specifications
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "SeniorCitizen",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def clean_telco_data(df: pd.DataFrame, save_processed: bool = True) -> pd.DataFrame:
    """
    Cleans raw Telco customer dataset:
    - Coerces TotalCharges to numeric, safely converting blank whitespace strings to NaN.
    - Imputes TotalCharges as 0.0 for customers with tenure == 0 (verified new accounts).
    - Standardizes target column 'Churn' into binary 0 and 1 (Yes=1, No=0).
    - Keeps raw records intact and saves the processed dataset to data/processed/cleaned_telco_data.csv.
    """
    cleaned_df = df.copy()

    # Safe coercion of TotalCharges
    cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # Business-justified imputation for zero tenure
    zero_tenure_mask = (cleaned_df["tenure"] == 0) & (cleaned_df["TotalCharges"].isnull())
    cleaned_df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Fallback median imputation if any remaining unexpected NaNs exist in TotalCharges
    if cleaned_df["TotalCharges"].isnull().any():
        med_val = cleaned_df["TotalCharges"].median()
        cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(med_val)
        logger.warning(f"Filled remaining nulls in TotalCharges with median ({med_val}).")

    # Encode Target: Yes -> 1, No -> 0
    if TARGET_COLUMN in cleaned_df.columns:
        unique_vals = set(cleaned_df[TARGET_COLUMN].dropna().unique())
        if "Yes" in unique_vals or "No" in unique_vals:
            cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].astype(str).map({"Yes": 1, "No": 0})
            if cleaned_df[TARGET_COLUMN].isnull().any():
                raise ValueError("Target column 'Churn' contains values other than 'Yes' and 'No'.")
            cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].astype(int)

    if save_processed:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Cleaned dataset saved to '{PROCESSED_DATA_PATH}'. Dimensions: {cleaned_df.shape}")

    return cleaned_df


def separate_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates independent variables (X) and target variable (y).
    Removes customerID from X, and ensures Churn is never included in X.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    y = df[TARGET_COLUMN].copy()
    drop_cols = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_cols.append(ID_COLUMN)

    X = df.drop(columns=drop_cols).copy()
    return X, y


def build_preprocessor(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer preprocessing pipeline:
    - Numerical: SimpleImputer(median) -> StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified training and testing sets (80:20).
    Ensures reproducibility using random_state=42.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Data split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows (stratified).")
    logger.info(f"Train target distribution: {y_train.value_counts(normalize=True).to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    logger.info("Executing preprocessing verification...")
    raw_df = load_raw_data()
    cleaned = clean_telco_data(raw_df)
    X, y = separate_features_and_target(cleaned)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    logger.info(f"Features: {X.columns.tolist()}")
    logger.info("Preprocessing module verification completed successfully.")
