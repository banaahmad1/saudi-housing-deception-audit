"""
Feature Engineering and Spatial District Baseline Formulation Engine.
Computes price-per-square-meter, derives localized baseline statistics,
enforces the N >= 10 minimum sample size constraint with hierarchical fallback,
and calculates statistical Z-scores and price ratios.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from src.config import MIN_DISTRICT_SAMPLE_SIZE

GROUP_FIELDS_DISTRICT = ["city", "district", "property_type", "transaction_type"]
GROUP_FIELDS_CITY = ["city", "property_type", "transaction_type"]
GROUP_FIELDS_GLOBAL = ["property_type", "transaction_type"]


def compute_price_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates unit price per square meter."""
    df = df.copy()
    df["price_per_sqm"] = df["price"] / df["area"]
    return df


def build_spatial_baselines(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Builds clean, isolated spatial baseline tables across hierarchy tiers:
    1. District-level: [city, district, property_type, transaction_type]
    2. City-level fallback: [city, property_type, transaction_type]
    3. Global property classification fallback: [property_type, transaction_type]
    """
    clean_df = compute_price_per_sqm(df)

    # 1. District level
    district_stats = (
        clean_df.groupby(GROUP_FIELDS_DISTRICT)["price_per_sqm"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    district_stats.columns = [
        "city", "district", "property_type", "transaction_type",
        "district_median_ppsqm", "district_mean_ppsqm",
        "district_std_ppsqm", "district_sample_count"
    ]

    # 2. City level fallback
    city_stats = (
        clean_df.groupby(GROUP_FIELDS_CITY)["price_per_sqm"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    city_stats.columns = [
        "city", "property_type", "transaction_type",
        "city_median_ppsqm", "city_mean_ppsqm",
        "city_std_ppsqm", "city_sample_count"
    ]

    # 3. Global fallback
    global_stats = (
        clean_df.groupby(GROUP_FIELDS_GLOBAL)["price_per_sqm"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    global_stats.columns = [
        "property_type", "transaction_type",
        "global_median_ppsqm", "global_mean_ppsqm",
        "global_std_ppsqm", "global_sample_count"
    ]

    return {
        "district": district_stats,
        "city": city_stats,
        "global": global_stats
    }


def apply_spatial_baselines(
    df: pd.DataFrame,
    baselines: Dict[str, pd.DataFrame],
    min_sample_size: int = MIN_DISTRICT_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Applies spatial baselines onto listings data, enforcing the minimum
    sample constraint (N >= min_sample_size) with automatic hierarchical fallback:
    District -> City -> Global.
    Computes Price Ratio and statistical Z-Score.
    """
    df = compute_price_per_sqm(df)

    # Merge District Tier
    merged = df.merge(baselines["district"], on=GROUP_FIELDS_DISTRICT, how="left")

    # Merge City Tier
    merged = merged.merge(baselines["city"], on=GROUP_FIELDS_CITY, how="left")

    # Merge Global Tier
    merged = merged.merge(baselines["global"], on=GROUP_FIELDS_GLOBAL, how="left")

    # Hierarchical Fallback Resolution
    # Condition: District baseline exists and has sample size >= min_sample_size
    valid_district = (
        merged["district_sample_count"].notna()
        & (merged["district_sample_count"] >= min_sample_size)
    )

    valid_city = (
        merged["city_sample_count"].notna()
        & (merged["city_sample_count"] >= min_sample_size)
    )

    # Median baseline resolution
    merged["effective_median_ppsqm"] = np.where(
        valid_district,
        merged["district_median_ppsqm"],
        np.where(valid_city, merged["city_median_ppsqm"], merged["global_median_ppsqm"])
    )

    # Mean baseline resolution
    merged["effective_mean_ppsqm"] = np.where(
        valid_district,
        merged["district_mean_ppsqm"],
        np.where(valid_city, merged["city_mean_ppsqm"], merged["global_mean_ppsqm"])
    )

    # Standard deviation baseline resolution
    merged["effective_std_ppsqm"] = np.where(
        valid_district,
        merged["district_std_ppsqm"],
        np.where(valid_city, merged["city_std_ppsqm"], merged["global_std_ppsqm"])
    )

    # Active tier indicator for auditability
    merged["baseline_tier_applied"] = np.where(
        valid_district,
        "district",
        np.where(valid_city, "city_fallback", "global_fallback")
    )

    # Handle rare cases where effective median might be NaN or zero
    global_median_all = df["price_per_sqm"].median()
    merged["effective_median_ppsqm"] = merged["effective_median_ppsqm"].fillna(global_median_all)
    merged["effective_mean_ppsqm"] = merged["effective_mean_ppsqm"].fillna(global_median_all)
    merged["effective_std_ppsqm"] = merged["effective_std_ppsqm"].fillna(0.0)

    # For paper backward compatibility, alias district_median_ppsqm to effective_median_ppsqm
    merged["district_median_ppsqm"] = merged["effective_median_ppsqm"]
    merged["district_mean_ppsqm"] = merged["effective_mean_ppsqm"]
    merged["district_std_ppsqm"] = merged["effective_std_ppsqm"]

    # Compute Anomaly Coordinates
    merged["price_ratio"] = merged["price_per_sqm"] / merged["effective_median_ppsqm"]

    merged["z_score"] = np.where(
        merged["effective_std_ppsqm"] > 0,
        (merged["price_per_sqm"] - merged["effective_mean_ppsqm"]) / merged["effective_std_ppsqm"],
        0.0
    )

    return merged
