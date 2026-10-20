"""
Controlled Synthetic Deceptive Listing Injection Engine.
Injects 20 controlled ground-truth deceptive listings with price markup
and pure promotional marketing hype (without architectural justification).
Enforces zero baseline contamination.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from src.config import (
    RANDOM_SEED,
    SYNTHETIC_INJECTION_COUNT,
    PRICE_MARKUP_MIN,
    PRICE_MARKUP_MAX
)

# Canonical Ground-Truth Deceptive Descriptions Partitioned by Property Type
# Real-world deceptive listings lack concrete architectural justification while asserting inflated value
DECEPTIVE_DESCRIPTIONS_BY_TYPE = {
    "شقة": [
        "فرصة لا تعوض شقة فخمة جدا بسعر مميز وموقع استراتيجي للجادين",
        "عرض خاص شقة راقية ومميزة بسعر مغري جدا لا تفوتك",
        "لقطة عقارية شقة ممتازة في موقع رائع وفرصة استثمارية ذهبية",
        "شقة بموقع ممتاز وفرصة نادرة للمستثمرين بدون وسطاء",
        "سعر ممتاز جدا لشقة مميزة في ارقى احياء المدينة",
        "فرصة عاجلة شقة بتصميم رائع وموقع حيوي قريب من الطرق الرئيسية",
        "شقة فاخرة للبيع العاجل بسعر خاص جدا للمهتمين فقط",
        "عرض محدود شقة لقطة لا تعوض بأفضل سعر بالمنطقة",
        "شقة استثمارية نادرة تحقق عوائد ممتازة وفرصة لا تفوت",
        "شقة مميزة جدا وراقية بتصميم عصري وسعر لا يقارن"
    ],
    "فيلا": [
        "عرض خاص فيلا راقية للجادين فقط بسعر مغري وموقع مميز",
        "فيلا فخمة جدا بسعر حصري وفرصة لا تعوض للمستثمرين",
        "لقطة عقارية فيلا ممتازة في موقع رائع وتصميم فريد",
        "فيلا بموقع استراتيجي راقي وفرصة نادرة لا تتكرر",
        "سعر ممتاز جدا لفيلا فخمة في حي هادئ ومميز",
        "فرصة عاجلة فيلا مميزة جدا بسعر خاص لسرعة البيع",
        "فيلا فاخرة للبيع بسعر مغري وفرصة استثمارية واعدة",
        "عرض عقاري محدود فيلا ممتازة بأفضل سعر في السوق",
        "فيلا راقية لقطة لا تفوتك بتصميم مميز وموقع ذهبي",
        "فرصة نادرة لعشاق الفخامة فيلا مميزة جدا بسعر خاص"
    ]
}


def inject_controlled_deceptive_listings(
    clean_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    count: int = SYNTHETIC_INJECTION_COUNT,
    markup_min: float = PRICE_MARKUP_MIN,
    markup_max: float = PRICE_MARKUP_MAX
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples records from clean_df, applies synthetic markup and property-type-aware
    promotional descriptions devoid of architectural justifications, labels them with
    ground truth, and merges with clean data.
    Ensures baselines are already computed before this function is applied (Zero Data Leakage).
    """
    np.random.seed(seed)
    actual_count = min(count, len(clean_df))

    # Sample base records to maintain realistic structural attributes (city, district, area)
    synthetic_sample = clean_df.sample(actual_count, random_state=seed).copy()

    # Apply severe anomalous price inflation (2.0x to 3.5x markup)
    inflation_multipliers = np.random.uniform(markup_min, markup_max, size=actual_count)
    synthetic_sample["price"] = np.round(synthetic_sample["price"] * inflation_multipliers, -3)

    # Assign property-type congruent deceptive descriptions lacking concrete architectural justification
    deceptive_texts = []
    for idx, (_, row) in enumerate(synthetic_sample.iterrows()):
        ptype = row.get("property_type", "شقة")
        pool = DECEPTIVE_DESCRIPTIONS_BY_TYPE.get(ptype, DECEPTIVE_DESCRIPTIONS_BY_TYPE["شقة"])
        desc = pool[idx % len(pool)]
        deceptive_texts.append(desc)

    synthetic_sample["description"] = deceptive_texts

    # Explicit ground-truth labeling
    synthetic_sample["is_deceptive_ground_truth"] = True
    synthetic_sample["is_synthetic_injected"] = True

    clean_tagged = clean_df.copy()
    clean_tagged["is_deceptive_ground_truth"] = False
    clean_tagged["is_synthetic_injected"] = False

    # Combine into unified evaluation dataset
    evaluation_df = pd.concat([clean_tagged, synthetic_sample], ignore_index=True)

    return evaluation_df, synthetic_sample
