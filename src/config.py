"""
Configuration module defining system thresholds, mathematical weights,
lexicon definitions, and file pathways for the Saudi Real Estate Auditing Pipeline.
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORY & PATH DEFINITIONS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = DATA_DIR / "saudi_real_estate_listings.csv"
FLAGGED_OUTPUT_FILE = OUTPUTS_DIR / "flagged_suspicious_real_estate_listings.csv"
EVALUATION_RESULTS_FILE = OUTPUTS_DIR / "evaluation_results.csv"
SENSITIVITY_RESULTS_FILE = OUTPUTS_DIR / "sensitivity_analysis.csv"
LATENCY_REPORT_FILE = OUTPUTS_DIR / "latency_report.txt"

# ==============================================================================
# 2. CORE OPERATIONAL THRESHOLDS & WEIGHTS
# ==============================================================================
RANDOM_SEED = 42

# Static Baseline Classifier Thresholds
STATIC_Z_THRESHOLD = 2.5
STATIC_RATIO_THRESHOLD = 1.75

# Multi-Modal Semantic-Enhanced Classifier Thresholds (Empirically Calibrated)
SEMANTIC_RATIO_THRESHOLD = 1.75
SEMANTIC_SCORE_THRESHOLD = 0.30

# Minimum District Sample Size for Baseline Robustness
MIN_DISTRICT_SAMPLE_SIZE = 10

# Semantic Justification Compound Index Weights
# Semantic Score = (0.65 * ConcreteScore) + (0.25 * LengthScore) - (0.30 * HypePenalty)
WEIGHT_CONCRETE = 0.65
WEIGHT_LENGTH = 0.25
WEIGHT_HYPE_PENALTY = 0.30
LENGTH_NORMALIZATION_TOKEN_CAP = 80

# Synthetic Ground-Truth Injection Parameters
SYNTHETIC_INJECTION_COUNT = 20
PRICE_MARKUP_MIN = 2.0
PRICE_MARKUP_MAX = 3.5

# Sensitivity Analysis Permutation Grids
SENSITIVITY_PRICE_RATIOS = [1.50, 1.75, 2.00]
SENSITIVITY_SEMANTIC_SCORES = [0.30, 0.40, 0.50]
SENSITIVITY_Z_SCORES = [2.0, 2.5, 3.0]

# Target Regulatory & Scientific Integrity Benchmarks (g1.md Alignment)
TARGET_ACCURACY = 0.85
TARGET_RECALL = 0.85
TARGET_FPR = 0.05
TARGET_QWK = 0.85  # Quadratic Weighted Kappa requirement: QWK > 85.0%
TARGET_MAX_LATENCY_MS = 2000.0

# Deterministic Agentic / NLP Inference Constraints (g1.md Rule)
INFERENCE_TEMPERATURE = 0.0  # Strict deterministic sampling (no stochastic randomness)
INFERENCE_TOP_P = 1.0        # Unconstrained cumulative probability
