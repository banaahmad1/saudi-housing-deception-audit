"""
Detection Rules and Multi-Modal Decision Engine.
Implements the baseline static statistical classifier and the multi-modal
semantic-enhanced dual-gate classification rule, plus sensitivity parameter grids.
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from src.config import (
    STATIC_Z_THRESHOLD,
    STATIC_RATIO_THRESHOLD,
    SEMANTIC_RATIO_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    SENSITIVITY_PRICE_RATIOS,
    SENSITIVITY_SEMANTIC_SCORES,
    SENSITIVITY_Z_SCORES
)


def apply_static_detection(
    df: pd.DataFrame,
    z_threshold: float = STATIC_Z_THRESHOLD,
    ratio_threshold: float = STATIC_RATIO_THRESHOLD
) -> pd.Series:
    """
    Static Statistical Detection Rule:
    Flags a listing as suspicious based solely on numerical deviation:
    Suspicious = (z_score >= z_threshold) OR (price_ratio >= ratio_threshold)
    """
    return (df["z_score"] >= z_threshold) | (df["price_ratio"] >= ratio_threshold)


def apply_semantic_detection(
    df: pd.DataFrame,
    ratio_threshold: float = SEMANTIC_RATIO_THRESHOLD,
    score_threshold: float = SEMANTIC_SCORE_THRESHOLD
) -> pd.Series:
    """
    Multi-Modal Semantic-Enhanced Decision Rule:
    Flags a listing if steep numerical markup is NOT justified by textual evidence:
    Suspicious = (price_ratio >= ratio_threshold) AND (semantic_score < score_threshold)
    """
    return (df["price_ratio"] >= ratio_threshold) & (df["semantic_score"] < score_threshold)


def run_sensitivity_analysis(
    df: pd.DataFrame,
    ground_truth_col: str = "is_deceptive_ground_truth"
) -> pd.DataFrame:
    """
    Runs systematic sensitivity tests across different threshold combinations:
    - Price Ratio: [1.50, 1.75, 2.00]
    - Semantic Score: [0.30, 0.40, 0.50]
    - Z-Score: [2.0, 2.5, 3.0]
    Measures impact on Detection Accuracy, Recall, Precision, FPR, and F1 Score.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        cohen_kappa_score
    )

    y_true = df[ground_truth_col].astype(int)
    results: List[Dict[str, object]] = []

    # 1. Semantic Pipeline Sensitivity Grid
    for pr in SENSITIVITY_PRICE_RATIOS:
        for ss in SENSITIVITY_SEMANTIC_SCORES:
            pred = ((df["price_ratio"] >= pr) & (df["semantic_score"] < ss)).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            qwk = cohen_kappa_score(y_true, pred, weights="quadratic")

            results.append({
                "model_type": "Semantic-Enhanced",
                "price_ratio_threshold": pr,
                "semantic_score_threshold": ss,
                "z_threshold": None,
                "accuracy": round(accuracy_score(y_true, pred), 4),
                "recall": round(recall_score(y_true, pred, zero_division=0), 4),
                "precision": round(precision_score(y_true, pred, zero_division=0), 4),
                "false_positive_rate": round(fpr, 4),
                "f1_score": round(f1_score(y_true, pred, zero_division=0), 4),
                "qwk": round(float(qwk), 4),
                "flagged_count": int(pred.sum())
            })

    # 2. Static Statistical Model Sensitivity Grid
    for pr in SENSITIVITY_PRICE_RATIOS:
        for zt in SENSITIVITY_Z_SCORES:
            pred = ((df["z_score"] >= zt) | (df["price_ratio"] >= pr)).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            qwk = cohen_kappa_score(y_true, pred, weights="quadratic")

            results.append({
                "model_type": "Static Statistical",
                "price_ratio_threshold": pr,
                "semantic_score_threshold": None,
                "z_threshold": zt,
                "accuracy": round(accuracy_score(y_true, pred), 4),
                "recall": round(recall_score(y_true, pred, zero_division=0), 4),
                "precision": round(precision_score(y_true, pred, zero_division=0), 4),
                "false_positive_rate": round(fpr, 4),
                "f1_score": round(f1_score(y_true, pred, zero_division=0), 4),
                "qwk": round(float(qwk), 4),
                "flagged_count": int(pred.sum())
            })

    return pd.DataFrame(results)
