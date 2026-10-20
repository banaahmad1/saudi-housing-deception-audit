"""
Arabic NLP Preprocessing and Semantic Justification Scoring Engine.
Standardizes orthographic variations, evaluates architectural luxury indicators,
and calculates the compound Semantic Justification Index.
"""

import re
from typing import Dict, List, Tuple
from src.config import (
    WEIGHT_CONCRETE,
    WEIGHT_LENGTH,
    WEIGHT_HYPE_PENALTY,
    LENGTH_NORMALIZATION_TOKEN_CAP,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P
)

# Deterministic Agentic NLP Inference Specification (Strictly conforms to g1.md rule)
AGENTIC_INFERENCE_CONFIG: Dict[str, object] = {
    "temperature": INFERENCE_TEMPERATURE,
    "top_p": INFERENCE_TOP_P,
    "seed": 42,
    "deterministic": True
}

# ==============================================================================
# CORE ARABIC LEXICONS (Saudi Real Estate Dialect & Market Conventions)
# ==============================================================================

# 20 Concrete Architectural & Luxury Features (Positive Verification Tokens)
RAW_JUSTIFYING_TERMS = [
    "تشطيب فاخر",
    "سمارت هوم",
    "بيت ذكي",
    "موقع مميز",
    "شارع تجاري",
    "زاوية",
    "واجهة حجر",
    "مصعد",
    "مسبح",
    "حديقة",
    "مدخل خاص",
    "دوبلكس",
    "مجلس",
    "قريب من الخدمات",
    "إطلالة",
    "موقف خاص",
    "قبو",
    "سطح خاص",
    "نظام أمني",
    "كاميرات مراقبة"
]

# 10 Generic Marketing Hype Tokens (Negative Penalty Indicators)
RAW_GENERIC_HYPE_TERMS = [
    "فرصة",
    "لقطة",
    "عاجل",
    "لا يفوتك",
    "عرض خاص",
    "سعر ممتاز",
    "مميز جدا",
    "فخم جدا",
    "أفضل سعر",
    "للجادين فقط"
]


def normalize_arabic(text: str) -> str:
    """
    Standardizes Arabic orthography variants and eliminates structural noise.
    Unifies Alef forms, Ya/Alif Maqsura, Hamzas, Ta Marbuta, and strips
    non-Arabic symbols to guarantee deterministic matching across listings.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # 1. Strip Arabic diacritics (tashkeel \u064B-\u0652) and tatweel/kashida (\u0640)
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)

    # 2. Unify Alef variations (إ, أ, آ, ا -> ا)
    text = re.sub(r"[إأآا]", "ا", text)

    # 3. Unify Alif Maqsura and Ya (ى -> ي)
    text = re.sub(r"ى", "ي", text)

    # 4. Standardize Hamza on Waw and Ya (ؤ -> و, ئ -> ي)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)

    # 5. Standardize Ta Marbuta to Ha (ة -> ه)
    text = re.sub(r"ة", "ه", text)

    # 6. Retain only Arabic Unicode range (\u0600-\u06FF) and whitespace
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # 7. Normalize extraneous whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Pre-normalize lexicon sets to ensure deterministic keyword matching
JUSTIFYING_TERMS: List[str] = [normalize_arabic(term) for term in RAW_JUSTIFYING_TERMS]
GENERIC_HYPE_TERMS: List[str] = [normalize_arabic(term) for term in RAW_GENERIC_HYPE_TERMS]


def count_terms(clean_text: str, lexicon: List[str]) -> Tuple[int, List[str]]:
    """
    Identifies matched terms from the lexicon inside the normalized text.
    Uses presence-based matching rather than raw repetition to prevent
    artificial keyword-stuffing exploits.
    """
    found_terms = [term for term in lexicon if term in clean_text]
    return len(found_terms), found_terms


def calculate_semantic_score(clean_text: str) -> float:
    """
    Calculates the multi-modal Semantic Justification Index:
    Semantic Score = (0.65 * ConcreteScore) + (0.25 * LengthScore) - (0.30 * HypePenalty)
    Bounded strictly to the interval [0.0, 1.0].
    """
    if not clean_text or clean_text.isspace():
        return 0.0

    concrete_count, _ = count_terms(clean_text, JUSTIFYING_TERMS)
    hype_count, _ = count_terms(clean_text, GENERIC_HYPE_TERMS)

    words = clean_text.split()
    text_length = max(len(words), 1)

    concrete_score = concrete_count / len(JUSTIFYING_TERMS)
    hype_penalty = hype_count / len(GENERIC_HYPE_TERMS)
    length_score = min(text_length / LENGTH_NORMALIZATION_TOKEN_CAP, 1.0)

    raw_score = (
        (WEIGHT_CONCRETE * concrete_score)
        + (WEIGHT_LENGTH * length_score)
        - (WEIGHT_HYPE_PENALTY * hype_penalty)
    )

    return float(max(0.0, min(raw_score, 1.0)))


def explain_semantic_score(clean_text: str) -> Dict[str, object]:
    """
    Provides an auditable diagnostic breakdown of the semantic scoring
    components for transparency and governance reporting.
    """
    concrete_count, concrete_found = count_terms(clean_text, JUSTIFYING_TERMS)
    hype_count, hype_found = count_terms(clean_text, GENERIC_HYPE_TERMS)

    words = clean_text.split()
    word_count = len(words)

    concrete_score = concrete_count / len(JUSTIFYING_TERMS)
    hype_penalty = hype_count / len(GENERIC_HYPE_TERMS)
    length_score = min(max(word_count, 1) / LENGTH_NORMALIZATION_TOKEN_CAP, 1.0)

    final_score = calculate_semantic_score(clean_text)

    return {
        "word_count": word_count,
        "concrete_terms_count": concrete_count,
        "concrete_terms_found": concrete_found,
        "concrete_score_component": round(WEIGHT_CONCRETE * concrete_score, 4),
        "hype_terms_count": hype_count,
        "hype_terms_found": hype_found,
        "hype_penalty_component": round(WEIGHT_HYPE_PENALTY * hype_penalty, 4),
        "length_score_component": round(WEIGHT_LENGTH * length_score, 4),
        "final_semantic_score": round(final_score, 4)
    }
