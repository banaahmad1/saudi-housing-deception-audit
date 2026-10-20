# Algorithmic Auditing of Deceptive Real Estate Listings: Dual-Gate Spatial Price and Arabic Text Validation in the Saudi Housing Market

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IC-AIRES 2026](https://img.shields.io/badge/Conference-IC--AIRES%202026-blueviolet.svg)](http://www.ic-aires.org/)
[![Springer LNNS](https://img.shields.io/badge/Proceedings-Springer%20LNNS-orange.svg)](https://www.springer.com/series/15179)
[![Agreement: QWK=0.9467](https://img.shields.io/badge/QWK-0.9467-success.svg)](https://doi.org/10.1037/h0026256)

Official open replication repository for the proceedings paper:
> **Bana Ahmad, Hani Alqaan, Saad Aldoihi, and Saeed Alzahrani**, *"Algorithmic Auditing of Deceptive Real Estate Listings: Dual-Gate Spatial Price and Arabic Text Validation in the Saudi Housing Market"*, Proceedings of the 10th International Conference on Artificial Intelligence in Renewable Energetic Systems (**IC-AIRES 2026**), Springer Lecture Notes in Networks and Systems (**LNNS**), 2026.

---

## 1. Overview

In digital real estate platforms operating under the Saudi Real Estate General Authority (REGA) 2022 Brokerage Law, property listings are susceptible to deceptive price markups uncorroborated by physical characteristics. Conventional Automated Valuation Models (AVMs) operate exclusively on tabular numerical variables (unit price and area), producing elevated false positive penalties on legitimate luxury residences while failing to detect ungrounded speculative pricing disguised by marketing hype.

This repository implements an auditable, deterministic dual-gate validation framework combining:
1. **Hierarchical Spatial Price Baselines**: Median unit pricing standardized across district, city, and global spatial lattices with an automated fallback threshold ($N_{\text{min}} = 10$).
2. **Arabic Lexical-Semantic Justification Index ($S_{\text{just}}$)**: Deterministic character normalization ($	au_{\text{strip}}, ho_{\text{ortho}}, \phi_{\text{filter}}, \psi_{\text{space}}$) coupled with domain-calibrated lexical matching against 20 concrete architectural tokens ($V_{\text{just}}$) and 10 promotional marketing hype tokens ($V_{\text{hype}}$).
3. **Deterministic Dual-Gate Decision Classification**: Disqualifies anomalous asking prices ($	heta_{\text{ratio}} = 1.75$) unless substantiated by architectural specifications ($S_{\text{just}}(T) \ge 0.30$).

---

## 2. Directory Structure

```
saudi-housing-deception-audit/
├── data/
│   ├── saudi_real_estate_listings.csv     # Baseline dataset (Kaggle Saudi Aqar repository)
│   └── held_out_expert_listings.csv       # Independent held-out expert testbed (N = 200)
├── src/
│   ├── __init__.py
│   ├── config.py                          # Thresholds, weights, and domain lexicons
│   ├── load_data.py                       # Data loading and schema validation
│   ├── preprocess_arabic.py               # Deterministic Arabic text normalization
│   ├── feature_engineering.py             # Spatial price baselines & hierarchical fallback
│   ├── detection_rules.py                 # Static AVM baseline vs. dual-gate decision logic
│   ├── synthetic_injection.py             # Controlled numerical boundary test injections
│   ├── generate_dataset.py                # Benchmark dataset generation & architectural templates
│   ├── evaluation.py                      # Cohen's Quadratic Weighted Kappa & metrics
│   ├── held_out_evaluation.py             # Benchmark evaluation on independent N = 200 cohort
│   └── main.py                            # End-to-end execution pipeline
├── scripts/
│   └── generate_charts.py                 # Publication figure generator (Figs. 3 & 4)
├── outputs/
│   ├── evaluation_results.csv             # Controlled sensitivity evaluation results
│   ├── held_out_evaluation_results.csv    # Independent expert evaluation benchmark (Table 1)
│   ├── sensitivity_analysis.csv           # 18-scenario sensitivity matrix
│   ├── latency_report.txt                 # Profiled high-resolution execution latencies
│   └── figures/
│       ├── confusion_matrices.png         # Fig. 3: Confusion matrices
│       ├── price_vs_semantic_scatter.png  # Fig. 4: Decision boundary scatter plot
│       └── metric_comparison.png          # Metric benchmark comparison
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                   # Automated test suite (21 unit tests)
├── requirements.txt                       # Minimal pinned dependencies
├── LICENSE                                # MIT open-source license
└── README.md                              # Replication instructions
```

---

## 3. Quick Start & Environment Setup

### 3.1 Prerequisites
- Python 3.10 or higher
- pip package installer

### 3.2 Installation
Clone the repository and install the verified dependencies:
```bash
git clone https://github.com/bana-ahamad/saudi-housing-deception-audit.git
cd saudi-housing-deception-audit
pip install -r requirements.txt
```

---

## 4. Replication of Empirical Results

### 4.1 Execute Full Pipeline Reproduction
To run the complete data loading, spatial baselining, lexical normalization, sensitivity sweep, and held-out benchmarking:
```bash
python src/main.py
```

### 4.2 Independent Expert Held-Out Evaluation (Table 1 Replication)
To evaluate performance strictly against the independent held-out cohort of $N = 200$ expert-annotated listings:
```bash
python src/held_out_evaluation.py
```

### 4.3 Execute Automated Test Suite
To execute the complete 21-test validation suite verifying deterministic text normalization, spatial fallbacks, relational boundaries, and QWK thresholds:
```bash
pytest tests/test_pipeline.py -v
```

### 4.4 Re-Generate Publication Figures
To re-render Figures 3 and 4 into `outputs/figures/`:
```bash
python scripts/generate_charts.py
```

---

## 5. Benchmark Performance Comparison (Table 1)

Empirical evaluation on the independent held-out expert testbed ($N = 200$ listings across 14 municipal districts, annotated with inter-rater concordance of $\kappa = 0.947$):

| Evaluation Metric | Formula / Protocol | Static AVM Baseline | Dual-Gate Bimodal Pipeline | Target Threshold | Compliance Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Detection Accuracy** | $(TP + TN) / \text{Total}$ | 75.00% | **98.00%** | $> 85.00\%$ | **PASS** |
| **Detection Recall** | $TP / (TP + FN)$ | 100.00% | **96.00%** | $> 85.00\%$ | **PASS** |
| **Model Precision** | $TP / (TP + FP)$ | 50.00% | **96.00%** | Higher is Better | **PASS** |
| **False Positive Rate (FPR)** | $FP / (FP + TN)$ | 33.33% | **1.33%** | $< 5.00\%$ | **PASS** |
| **F1 Score** | $2 \cdot (\text{Prec} \cdot \text{Rec}) / (\text{Prec} + \text{Rec})$ | 0.6667 | **0.9600** | Higher is Better | **PASS** |
| **Quadratic Weighted Kappa (QWK)** | Cohen (1968) formulation | 0.5000 | **0.9467** | $> 0.8000$ | **PASS** |
| **Decision Gate Latency** | High-Res Vectorized Evaluation | 0.0016 ms | **0.0019 ms** | $< 2000\text{ ms}$ | **PASS** |

---

## 6. Published Keyword Lexicons

### 6.1 Tangible Architectural Justification Tokens ($V_{\text{just}}$, 20 Terms)
$$\operatorname{ConcreteScore}(T) = \frac{1}{|V_{\text{just}}|} \sum_{j=1}^{20} \mathbb{I}(v_j \in \mathcal{N}(T))$$

1. `تشطيب فاخر` (luxury finishing)
2. `سمارت هوم/بيت ذكي` (smart home)
3. `موقع مميز/شارع تجاري` (prime/commercial location)
4. `زاوية` (corner)
5. `واجهة حجر` (stone facade)
6. `مصعد` (elevator)
7. `مسبح` (swimming pool)
8. `حديقة` (garden)
9. `مدخل خاص/موقف خاص/سطح خاص` (private entrance/parking/terrace)
10. `دوبلكس` (duplex)
11. `مجلس` (majlis)
12. `قريب من الخدمات` (close to services)
13. `إطلالة` (view)
14. `قبو` (basement)
15. `نظام أمني/كاميرات مراقبة` (security/CCTV)
16. `غرفة سائق` (driver room)
17. `غرفة خادمة` (maid room)
18. `مؤثثة بالكامل` (fully furnished)
19. `مكيفات راكبة` (installed ACs)
20. `مطبخ راكب` (installed kitchen)

### 6.2 Promotional Marketing Hype Tokens ($V_{\text{hype}}$, 10 Terms)
$$\operatorname{HypePenalty}(T) = \frac{1}{|V_{\text{hype}}|} \sum_{k=1}^{10} \mathbb{I}(h_k \in \mathcal{N}(T))$$

1. `فرصة` (opportunity)
2. `لقطة` (bargain)
3. `عاجل` (urgent)
4. `لا يفوتك` (do not miss)
5. `سعر ممتاز/أفضل سعر/عرض خاص` (special/best price)
6. `مميز جدا/فخم جدا` (distinguished/luxury)
7. `للجادين فقط` (serious buyers only)
8. `استثمار ذهبي` (golden investment)
9. `سعر محروق` (sacrificed price)
10. `أقل من سعر السوق` (below market price)

---

## 7. Mathematical Decision Formulation

The bimodal dual-gate decision rule is governed by:

$$C_{\text{semantic}}(x) = \begin{cases} 1 \; (\text{Deceptive}), & \text{if } (P_{\text{ratio}} \ge 1.75) \land (S_{\text{just}}(T) < 0.30) \\ 0 \; (\text{Compliant}), & \text{otherwise} \end{cases}$$

Where the compound justification index is bounded by:

$$S_{\text{just}}(T) = \max\left(0.0, \; \min\left(1.0, \; 0.65 \cdot \operatorname{ConcreteScore}(T) + 0.25 \cdot \operatorname{LengthScore}(T) - 0.30 \cdot \operatorname{HypePenalty}(T)\right)\right)$$

Concordance is validated via Cohen's Quadratic Weighted Kappa:

$$\operatorname{QWK} = 1 - \frac{\sum_{i=1}^K \sum_{j=1}^K w_{i,j} O_{i,j}}{\sum_{i=1}^K \sum_{j=1}^K w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$

---

## 8. Data Provenance & Verification

- **Baseline Spatial Corpus**: Acquired from the open Kaggle Saudi Real Estate repository (`lama122/saudi-arabia-real-estate-aqar`), comprising authentic listings from property portals in Riyadh, Jeddah, Dammam, and Al-Khobar.
- **Held-Out Expert Testbed ($N = 200$)**: Compiled across 14 municipal districts from Aqar, Bayut, and Zaahib. Independent annotations were provided by a TAQEEM-certified valuation assessor and a REGA compliance auditor (Cohen's $\kappa = 0.947$).

---

## 9. Citation

If using this codebase or replication datasets, cite the conference proceedings:

```bibtex
@inproceedings{ahmad2026algorithmic,
  author    = {Ahmad, Bana and Alqaan, Hani and Aldoihi, Saad and Alzahrani, Saeed},
  title     = {Algorithmic Auditing of Deceptive Real Estate Listings: Dual-Gate Spatial Price and Arabic Text Validation in the Saudi Housing Market},
  booktitle = {Proceedings of the 10th International Conference on Artificial Intelligence in Renewable Energetic Systems (IC-AIRES 2026)},
  series    = {Lecture Notes in Networks and Systems (LNNS)},
  publisher = {Springer},
  year      = {2026}
}
```

---

## 10. License

This repository is licensed under the [MIT License](LICENSE).
