"""
Data Ingestion, Schema Standardization, and Data Quality Auditing Module.
Loads raw real estate listings under explicit UTF-8/UTF-8-SIG configurations,
enforces schema invariants, and profiles missingness.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "city",
    "district",
    "property_type",
    "transaction_type",
    "area",
    "price",
    "description"
]

COLUMN_ALIAS_MAP = {
    # Standard English variations
    "City": "city",
    "District": "district",
    "Property_Type": "property_type",
    "PropertyType": "property_type",
    "property type": "property_type",
    "Transaction_Type": "transaction_type",
    "TransactionType": "transaction_type",
    "transaction type": "transaction_type",
    "Area": "area",
    "Price": "price",
    "Description": "description",
    "Month": "month",
    "Weekday": "weekday",
    # Arabic column header aliases
    "المدينة": "city",
    "مدينة": "city",
    "الحي": "district",
    "حي": "district",
    "نوع العقار": "property_type",
    "نوع_العقار": "property_type",
    "نوع العملية": "transaction_type",
    "نوع_العملية": "transaction_type",
    "المساحة": "area",
    "مساحة": "area",
    "السعر": "price",
    "سعر": "price",
    "الوصف": "description",
    "تفاصيل": "description",
    "الشهر": "month",
    "اليوم": "weekday"
}


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads raw listings dataset trying standard encodings (utf-8, utf-8-sig, cp1256).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Listing dataset not found at: {file_path}")

    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise ValueError(f"Failed to decode {file_path} across supported encodings: {last_err}")


def audit_data_quality(df: pd.DataFrame) -> Dict[str, object]:
    """
    Generates a foundational data-quality audit report before transformations.
    Assesses row counts, missingness ratios, and numeric anomalies.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    missing_ratios = {col: round(count / max(total_rows, 1), 4) for col, count in missing_counts.items()}

    return {
        "total_raw_rows": total_rows,
        "total_columns": len(df.columns),
        "columns_present": list(df.columns),
        "missing_counts": missing_counts,
        "missing_ratios": missing_ratios
    }


def clean_and_standardize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Aligns schema fields deterministically, validates constraints, and removes
    anomalous records (price <= 0 or area <= 0).
    """
    audit_report = audit_data_quality(df)

    # 1. Standardize column names
    clean_df = df.rename(columns=COLUMN_ALIAS_MAP)

    # 2. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in clean_df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing mandatory fields: {missing_cols}. "
            f"Found columns: {list(clean_df.columns)}"
        )

    # 3. Numeric type casting with coerced errors
    clean_df["area"] = pd.to_numeric(clean_df["area"], errors="coerce")
    clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce")

    # 4. Standardize text columns
    clean_df["city"] = clean_df["city"].astype(str).str.strip()
    clean_df["district"] = clean_df["district"].astype(str).str.strip()
    clean_df["property_type"] = clean_df["property_type"].astype(str).str.strip()
    clean_df["transaction_type"] = clean_df["transaction_type"].astype(str).str.strip()
    clean_df["description"] = clean_df["description"].astype(str).str.strip()

    # 5. Missing value & non-positive value removal
    initial_count = len(clean_df)
    clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)
    clean_df = clean_df[(clean_df["area"] > 0) & (clean_df["price"] > 0)].reset_index(drop=True)

    dropped_count = initial_count - len(clean_df)
    audit_report["cleaned_valid_rows"] = len(clean_df)
    audit_report["dropped_invalid_records"] = dropped_count

    return clean_df, audit_report
