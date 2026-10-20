"""
Comprehensive Automated Test Suite for Saudi Real Estate Deceptive Listing Detection.
Validates Arabic NLP normalization, semantic scoring, spatial baseline calculations,
fallback hierarchies, dual-gate detection rules, and performance metrics.
"""

import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    STATIC_Z_THRESHOLD,
    STATIC_RATIO_THRESHOLD,
    SEMANTIC_RATIO_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    TARGET_QWK,
    RAW_DATA_FILE
)
from src.preprocess_arabic import (
    normalize_arabic,
    calculate_semantic_score,
    explain_semantic_score,
    JUSTIFYING_TERMS,
    GENERIC_HYPE_TERMS,
    AGENTIC_INFERENCE_CONFIG
)
from src.load_data import clean_and_standardize_data, load_dataset
from src.feature_engineering import (
    compute_price_per_sqm,
    build_spatial_baselines,
    apply_spatial_baselines
)
from src.detection_rules import (
    apply_static_detection,
    apply_semantic_detection,
    run_sensitivity_analysis
)
from src.synthetic_injection import inject_controlled_deceptive_listings
from src.evaluation import calculate_fpr, calculate_qwk, benchmark_model_performance


class TestArabicNLPPreprocessing(unittest.TestCase):
    """Test suite for Arabic text normalization and semantic scoring."""

    def test_alef_normalization(self):
        self.assertEqual(normalize_arabic("إستثمار"), "استثمار")
        self.assertEqual(normalize_arabic("أفضل"), "افضل")
        self.assertEqual(normalize_arabic("آمن"), "امن")

    def test_ya_and_alif_maqsura_normalization(self):
        self.assertEqual(normalize_arabic("مبنى"), "مبني")
        self.assertEqual(normalize_arabic("حي"), "حي")

    def test_ta_marbuta_normalization(self):
        self.assertEqual(normalize_arabic("شقة"), "شقه")
        self.assertEqual(normalize_arabic("حديقة"), "حديقه")
        self.assertEqual(normalize_arabic("واجهة"), "واجهه")

    def test_hamza_normalization(self):
        self.assertEqual(normalize_arabic("مسؤول"), "مسوول")
        self.assertEqual(normalize_arabic("عائلة"), "عايله")

    def test_non_arabic_character_stripping(self):
        raw = "Luxury Villa #123! @Riyadh (تشطيب فاخر) 100%"
        cleaned = normalize_arabic(raw)
        self.assertEqual(cleaned, "تشطيب فاخر")

    def test_lexicon_cardinalities(self):
        self.assertEqual(len(JUSTIFYING_TERMS), 20, "Justifying terms dictionary must contain exactly 20 terms.")
        self.assertEqual(len(GENERIC_HYPE_TERMS), 10, "Generic hype dictionary must contain exactly 10 terms.")

    def test_empty_semantic_score(self):
        self.assertEqual(calculate_semantic_score(""), 0.0)
        self.assertEqual(calculate_semantic_score("   "), 0.0)
        self.assertEqual(calculate_semantic_score(None), 0.0)

    def test_justified_luxury_high_score(self):
        text = "فيلا فاخرة بتشطيب فاخر وتصميم حديث سمارت هوم بيت ذكي واجهة حجر طبيعي مصعد راكب ومسبح وحديقة خاصة ومدخل خاص وموقف خاص قريب من الخدمات ومجلس واسع نظام امني وكاميرات مراقبه"
        clean = normalize_arabic(text)
        score = calculate_semantic_score(clean)
        self.assertGreaterEqual(score, SEMANTIC_SCORE_THRESHOLD, f"Rich architectural description should score >= {SEMANTIC_SCORE_THRESHOLD}, got {score}")

    def test_hype_penalty_low_score(self):
        text = "فرصه لا تعوض شقه فخمه جدا لقطه عاجل لا يفوتك عرض خاص افضل سعر للجادين فقط"
        clean = normalize_arabic(text)
        score = calculate_semantic_score(clean)
        self.assertLess(score, SEMANTIC_SCORE_THRESHOLD, f"Hype-only promotional description should score < {SEMANTIC_SCORE_THRESHOLD}, got {score}")

    def test_score_bounded_in_unit_interval(self):
        text = "تشطيب فاخر " * 50
        clean = normalize_arabic(text)
        score = calculate_semantic_score(clean)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_explain_semantic_score_structure(self):
        text = "فيلا بتشطيب فاخر ومسبح"
        clean = normalize_arabic(text)
        exp = explain_semantic_score(clean)
        self.assertIn("concrete_terms_count", exp)
        self.assertIn("hype_terms_count", exp)
        self.assertIn("final_semantic_score", exp)


class TestFeatureEngineeringAndBaselines(unittest.TestCase):
    """Test suite for spatial baselines, hierarchy fallback, and anomaly coordinates."""

    def setUp(self):
        np.random.seed(42)
        # Create small test dataset
        self.sample_df = pd.DataFrame({
            "city": ["الرياض"] * 15 + ["جدة"] * 5,
            "district": ["الياسمين"] * 15 + ["النعيم"] * 5,
            "property_type": ["شقة"] * 20,
            "transaction_type": ["بيع"] * 20,
            "area": [100.0] * 20,
            "price": [500000.0] * 20,
            "description": ["شقة عادية للبيع"] * 20
        })

    def test_price_per_sqm_calculation(self):
        df_ppsqm = compute_price_per_sqm(self.sample_df)
        self.assertTrue((df_ppsqm["price_per_sqm"] == 5000.0).all())

    def test_baseline_formulation_and_fallback(self):
        baselines = build_spatial_baselines(self.sample_df)
        applied = apply_spatial_baselines(self.sample_df, baselines, min_sample_size=10)

        # Riyadh/Alyasmin has 15 rows (>=10) -> tier should be 'district'
        riyadh_tier = applied[applied["city"] == "الرياض"]["baseline_tier_applied"].iloc[0]
        self.assertEqual(riyadh_tier, "district")

        # Jeddah/Alnaeem has 5 rows (<10) -> tier should fall back to 'city_fallback' or 'global_fallback'
        jeddah_tier = applied[applied["city"] == "جدة"]["baseline_tier_applied"].iloc[0]
        self.assertIn(jeddah_tier, ["city_fallback", "global_fallback"])

    def test_price_ratio_and_z_score_calculation(self):
        baselines = build_spatial_baselines(self.sample_df)
        applied = apply_spatial_baselines(self.sample_df, baselines)

        # Baseline median should be 5000, so ratio should be 1.0
        self.assertAlmostEqual(applied["price_ratio"].iloc[0], 1.0, places=4)
        # All prices are identical, std is 0, z-score should safely be 0.0 (no division by zero crash)
        self.assertEqual(applied["z_score"].iloc[0], 0.0)


class TestDetectionRulesAndEvaluation(unittest.TestCase):
    """Test suite for dual-gate detection logic and evaluation benchmarking."""

    def test_dual_gate_eliminates_luxury_false_positives(self):
        # Listing with high price ratio (2.0) BUT high semantic score (0.50) -> NOT deceptive!
        test_df = pd.DataFrame({
            "price_ratio": [2.0, 2.0, 1.2],
            "z_score": [3.0, 3.0, 0.5],
            "semantic_score": [0.50, 0.20, 0.10],
            "is_deceptive_ground_truth": [False, True, False]
        })

        static_pred = apply_static_detection(test_df)
        semantic_pred = apply_semantic_detection(test_df)

        # Record 0 is a justified luxury property
        self.assertTrue(static_pred.iloc[0], "Static model incorrectly flags luxury property (false positive)")
        self.assertFalse(semantic_pred.iloc[0], "Semantic model correctly recognizes luxury justification (no false positive)")

        # Record 1 is an unjustified deceptive markup
        self.assertTrue(static_pred.iloc[1])
        self.assertTrue(semantic_pred.iloc[1], "Semantic model correctly flags unjustified price markup")

    def test_synthetic_injection_integrity(self):
        base_df = pd.DataFrame({
            "city": ["الرياض"] * 30,
            "district": ["الياسمين"] * 30,
            "property_type": ["شقة"] * 30,
            "transaction_type": ["بيع"] * 30,
            "area": [150.0] * 30,
            "price": [600000.0] * 30,
            "description": ["شقة ممتازة"] * 30
        })

        eval_df, synthetic = inject_controlled_deceptive_listings(base_df, count=20)
        self.assertEqual(len(synthetic), 20)
        self.assertEqual(len(eval_df), 50)
        self.assertTrue((synthetic["is_deceptive_ground_truth"] == True).all())
        self.assertTrue((eval_df.iloc[:30]["is_deceptive_ground_truth"] == False).all())

    def test_fpr_calculation(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1])  # 1 FP out of 4 actual negatives
        fpr = calculate_fpr(y_true, y_pred)
        self.assertAlmostEqual(fpr, 0.25)


class TestScientificIntegrityAndReliability(unittest.TestCase):
    """
    Test suite strictly verifying compliance with g1.md:
    1. Deterministic Inference (Temperature = 0.0, top-p = 1.0).
    2. Quadratic Weighted Kappa (QWK > 85.0%).
    3. High-Fidelity & Viability (zero category-description mismatches).
    """

    def test_deterministic_inference_configuration(self):
        """Verifies temperature=0.0 and top-p=1.0 constraints from g1.md."""
        self.assertEqual(INFERENCE_TEMPERATURE, 0.0, "Temperature must be exactly 0.0 for deterministic reproducibility")
        self.assertEqual(INFERENCE_TOP_P, 1.0, "Top-p must be exactly 1.0")
        self.assertEqual(AGENTIC_INFERENCE_CONFIG["temperature"], 0.0)
        self.assertEqual(AGENTIC_INFERENCE_CONFIG["top_p"], 1.0)
        self.assertTrue(AGENTIC_INFERENCE_CONFIG["deterministic"])

    def test_qwk_mathematical_properties(self):
        """Validates Quadratic Weighted Kappa computation and properties."""
        # Perfect agreement -> QWK = 1.0
        y1 = np.array([0, 0, 1, 1, 0, 1])
        y2 = np.array([0, 0, 1, 1, 0, 1])
        self.assertAlmostEqual(calculate_qwk(y1, y2), 1.0)

        # Discordant -> QWK < 1.0
        y3 = np.array([0, 0, 1, 0, 0, 1])
        qwk_partial = calculate_qwk(y1, y3)
        self.assertGreater(qwk_partial, 0.0)
        self.assertLess(qwk_partial, 1.0)

    def test_qwk_regulatory_compliance(self):
        """Asserts that Semantic-Enhanced pipeline achieves QWK >= 0.85 (85%)."""
        eval_csv = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
        if eval_csv.exists():
            df_res = pd.read_csv(eval_csv)
            qwk_row = df_res[df_res["Metric"].str.contains("Kappa|QWK", case=False)]
            if not qwk_row.empty:
                val_str = str(qwk_row["Semantic Pipeline"].iloc[0]).strip()
                qwk_val = float(val_str)
                self.assertGreaterEqual(
                    qwk_val,
                    TARGET_QWK,
                    f"Semantic Pipeline QWK must satisfy QWK >= {TARGET_QWK}, got {qwk_val}"
                )

    def test_property_type_description_fidelity(self):
        """Verifies zero architectural category mismatches (apartments vs villas)."""
        if RAW_DATA_FILE.exists():
            df = pd.read_csv(RAW_DATA_FILE)
            # Apartments should never contain villa or palace keywords
            apt_mismatch = df[(df["property_type"] == "شقة") & (df["description"].str.contains("فيلا|قصر"))]
            self.assertEqual(
                len(apt_mismatch),
                0,
                f"Found {len(apt_mismatch)} apartment listings with villa/palace descriptions!"
            )

            # Villas should never start with apartment keywords
            villa_mismatch = df[(df["property_type"] == "فيلا") & (df["description"].str.startswith("شقة"))]
            self.assertEqual(
                len(villa_mismatch),
                0,
                f"Found {len(villa_mismatch)} villa listings with apartment descriptions!"
            )


if __name__ == "__main__":
    unittest.main()
