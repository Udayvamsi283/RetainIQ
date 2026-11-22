"""
RetainIQ - Telecom Customer Churn Prediction System
Data Acquisition and Inspection Module

This module handles:
1. Programmatic dataset acquisition from the official IBM repository.
2. Verification and validation of the downloaded dataset.
3. Detailed inspection of data types, missing values, duplicates, and distributions.
4. Safe handling of dataset-specific anomalies (e.g. blank TotalCharges for 0-tenure customers).
5. Saving a structured data summary to outputs/data_summary.json.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OFFICIAL_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
KAGGLE_FALLBACK_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
RAW_DATA_PATH = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
DATA_SUMMARY_PATH = os.path.join("outputs", "data_summary.json")


def download_dataset(url: str = OFFICIAL_DATASET_URL, destination_path: str = RAW_DATA_PATH, force: bool = False) -> str:
    """
    Downloads the IBM Telco Customer Churn dataset programmatically.
    Ensures directories exist, avoids redundant downloads, validates CSV integrity,
    and reports clear fallbacks if official download fails.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    if os.path.exists(destination_path) and not force:
        try:
            sample_df = pd.read_csv(destination_path, nrows=5)
            if "Churn" in sample_df.columns:
                logger.info(f"Valid local dataset found at '{destination_path}'. Skipping redundant download.")
                return destination_path
            else:
                logger.warning(f"File at '{destination_path}' exists but lacks 'Churn' column. Re-downloading...")
        except Exception as e:
            logger.warning(f"Existing file check failed ({e}). Re-downloading...")

    logger.info(f"Downloading dataset from official source: {url}")
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Dataset successfully saved to '{destination_path}'.")
    except Exception as e:
        error_msg = (
            f"Failed to download dataset from official IBM URL: {url}\n"
            f"Error details: {e}\n"
            f"Fallback action: You can manually download the dataset from Kaggle at:\n"
            f"{KAGGLE_FALLBACK_URL}\n"
            f"and place the CSV at '{destination_path}'."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Validate downloaded file
    try:
        val_df = pd.read_csv(destination_path)
    except Exception as e:
        raise ValueError(f"Downloaded file at '{destination_path}' is not a readable CSV: {e}")

    if "Churn" not in val_df.columns:
        raise ValueError(f"Dataset downloaded from '{url}' does not contain expected target column 'Churn'.")

    logger.info(f"Validation successful. Dimensions: {val_df.shape[0]} rows, {val_df.shape[1]} columns.")
    logger.info(f"Columns: {list(val_df.columns)}")
    return destination_path


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw Telco Churn CSV file into a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        download_dataset(destination_path=filepath)
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data inspection on the raw dataset.
    Examines:
    - Dimensions
    - Data types
    - Null / blank values
    - Duplicate records and customer IDs
    - Target distribution
    - TotalCharges coercion & zero-tenure records analysis
    """
    print("\n" + "=" * 70)
    print("RETAIN-IQ: DATA INSPECTION & EXPLORATION SUMMARY")
    print("=" * 70)

    rows, cols = df.shape
    print(f"\n1. Dataset Dimensions: {rows} rows, {cols} columns")

    print("\n2. First 5 Records:")
    print(df.head())

    print("\n3. Column Names and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:20s}: {dtype}")

    print("\n4. Descriptive Statistics (Numerical as loaded):")
    print(df.describe(include=[np.number]))

    print("\n5. Descriptive Statistics (Categorical / Object):")
    cat_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    if len(cat_cols) > 0:
        print(df[cat_cols].describe())

    # Missing values check
    missing_counts = df.isnull().sum()
    print("\n6. Missing Values (Standard NaN):")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"   - {col}: {count} missing")
    if missing_counts.sum() == 0:
        print("   No standard NaN values detected in original CSV.")

    # Duplicate checks
    dup_rows = df.duplicated().sum()
    dup_ids = df["customerID"].duplicated().sum() if "customerID" in df.columns else 0
    print(f"\n7. Duplicate Records: {dup_rows}")
    print(f"8. Duplicate Customer IDs: {dup_ids}")

    # Target distribution
    target_dist = df["Churn"].value_counts().to_dict()
    target_pct = (df["Churn"].value_counts(normalize=True) * 100).to_dict()
    print("\n9. Target Distribution (Churn):")
    for k, v in target_dist.items():
        print(f"   - {k}: {v} ({target_pct[k]:.2f}%)")

    # TotalCharges whitespace anomaly inspection
    blank_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"\n10. TotalCharges Blank / Whitespace String Count: {blank_total_charges}")

    # Inspect records where TotalCharges is blank
    blank_tc_mask = df["TotalCharges"].astype(str).str.strip() == ""
    blank_records = df[blank_tc_mask]
    if len(blank_records) > 0:
        print("   Verification of records with blank TotalCharges:")
        print(f"   - All these records have tenure == 0: {(blank_records['tenure'] == 0).all()}")
        print(f"   - Tenure values for blank TotalCharges: {blank_records['tenure'].unique().tolist()}")
        print(f"   - Churn values for blank TotalCharges: {blank_records['Churn'].value_counts().to_dict()}")
        print("   Decision: Customers with 0 tenure joined recently and have not yet completed their first billing cycle.")
        print("   Treating these blank TotalCharges as 0.0 is business-justified and avoids discarding valuable customer records.")

    # Build structured summary dict for export
    summary = {
        "dataset_name": "IBM Telco Customer Churn",
        "raw_file_path": RAW_DATA_PATH,
        "rows": int(rows),
        "columns_count": int(cols),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(dup_rows),
        "duplicate_customer_ids": int(dup_ids),
        "target_variable": "Churn",
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
        "target_percentages": {str(k): round(float(v), 2) for k, v in target_pct.items()},
        "anomalies": {
            "total_charges_blank_strings": int(blank_total_charges),
            "total_charges_blank_records_all_zero_tenure": bool((blank_records['tenure'] == 0).all()) if len(blank_records) > 0 else True,
            "resolution": "Coerce TotalCharges to float; impute 0.0 for records where tenure == 0."
        }
    }
    return summary


def save_data_summary(summary: Dict[str, Any], output_path: str = DATA_SUMMARY_PATH) -> None:
    """
    Saves the data inspection summary dictionary into a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Data inspection summary saved to '{output_path}'.")


def main():
    logger.info("Step 1: Downloading / validating dataset...")
    filepath = download_dataset()
    logger.info("Step 2: Loading raw dataset...")
    df = load_raw_data(filepath)
    logger.info("Step 3: Performing inspection...")
    summary = inspect_data(df)
    logger.info("Step 4: Saving inspection summary...")
    save_data_summary(summary)
    print("\nData loading and inspection phase completed successfully.")


if __name__ == "__main__":
    main()
