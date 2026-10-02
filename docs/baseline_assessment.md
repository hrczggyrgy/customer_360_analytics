# Baseline Assessment: Retail Customer Intelligence Pipeline

**Project:** marketing_science (Online Retail II UCI dataset)  
**Date:** 2026-10-02  
**Assessed by:** Sisyphus (Principal Retail Data Scientist + Staff Data Engineer + ML Platform Architect)

---

## 1. Current Architecture Overview

### Repository Structure
```
marketing_science/
├── config/project.yaml            # Central config (paths, model params)
├── retail_ds/                     # Shared data science library (7 modules)
│   ├── io.py                      # Canonical ingestion (Invoice as string)
│   ├── cleaning.py                # Transaction cleaning + calendar fields
│   ├── transactions.py            # Classification + financial measures
│   ├── customer_month.py          # Customer-month dense panel
│   ├── features.py                # Point-in-time feature engine + registry
│   ├── validation.py              # Data quality gates
│   └── backtesting.py             # Temporal split utilities
├── scripts/                       # 9 pipeline scripts
│   ├── data_quality.py            # Canonical transactions + quality reports
│   ├── customer_360.py            # Descriptive + PIT features
│   ├── customer_segmentation.py   # HDBSCAN + bootstrap stability
│   ├── cohort_analysis.py         # Cohort retention + revenue
│   ├── clv_analysis.py            # Dynamic CLV Proxy (MC simulation)
│   ├── churn_next_purchase.py     # Survival + next-purchase (FIXED)
│   ├── recommendation_engine.py   # Vectorized co-purchase recommendations
│   ├── decision_engine.py         # Capacity-aware observational prioritization
│   ├── product_analytics.py       # Product metrics + roles
│   └── reactivation_model.py      # Reactivation prediction
├── streamlit_app/                 # Modular dashboard (V2)
│   ├── app.py                     # Router (~322 lines)
│   ├── app_config.py              # Config loader
│   ├── app_data.py                # ArtifactRegistry (10 module schemas)
│   ├── app_formatting.py          # 11 semantic formatters
│   ├── app_components.py          # Reusable UI components
│   ├── app_charts.py              # Consistent Plotly charts
│   └── pages/ (9 modules)
├── tests/                         # 57 tests (35 retail_ds + 22 app)
├── docs/                          # Documentation (some stale)
└── pyproject.toml / requirements.txt
```

### Pipeline Flow (Current)
```
Raw Excel (online_retail_II.xlsx)
    │
    ▼
data_quality.py ──► canonical_transactions.parquet + customer_month.parquet
    │
    ├──► customer_360.py ──► customer_360_current.parquet + features_at_*.parquet
    ├──► customer_segmentation.py ──► customer_segments.parquet + profiles
    ├──► cohort_analysis.py ──► retention matrices + scorecards
    ├──► clv_analysis.py ──► clv_customer_predictions.csv + margin scenarios
    ├──► churn_next_purchase.py ──► customer_churn_next_purchase.csv + survival
    ├──► recommendation_engine.py ──► recommendations.parquet (vectorized)
    ├──► decision_engine.py ──► customer_decision_scores.csv (capacity-aware)
    ├──► product_analytics.py ──► product_metrics.parquet + co_purchase_matrix
    └──► reactivation_model.py ──► reactivation_predictions.csv
    │
    ▼
Streamlit Dashboard (9 pages) ──► ArtifactRegistry discovers outputs
```

---

## 2. Current Known Issues (from SCIENTIFIC_REVIEW.md + Code Inspection)

### P0 — Critical Correctness
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| E01 | ~~Ingestion bug~~ **FIXED** | retail_ds/io.py | 19,494 cancellations preserved |
| E02 | Customer 360 leakage | retail_ds/features.py + scripts/customer_360.py | 40+ features use future info for historical prediction |
| E03 | ~~Churn/Next-Purchase IndexError~~ **FIXED** | scripts/churn_next_purchase.py | Now builds targets from same panel before split |

### P1 — Robustness/Validation
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| E04 | Segmentation: noise confusion | docs/SCIENTIFIC_REVIEW.md | Original review confused segment 0 (66%) with noise (16.6%) |
| E05 | Segmentation: block balancing failed | scripts/customer_segmentation.py | Segment 5 price_behavior = 23.85 dominates |
| E06 | Cohort NRR > 100% | scripts/cohort_analysis.py | 2011-05 age 6 = 328% NRR with 26% logo retention |
| E07 | CLV mislabeling | scripts/clv_analysis.py | "CLV" is actually discounted revenue forecast (no margin, no lifetime model) |
| E08 | Duplicate cleaning logic | 5 scripts | Bug fixes require 5 edits (partially mitigated by retail_ds) |
| E09 | Inconsistent feature definitions | Multiple scripts | Reconciliation difficult |

### P2 — Architecture/Performance
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| E10 | CLV MC not fully vectorized | scripts/clv_analysis.py | Python loops in state updates; ~15 min for full run |
| E11 | Dashboard hardcoded paths | streamlit_app/app_data.py | Recursive glob discovery (partially fixed by ArtifactRegistry) |
| E12 | No requirements.lock | pyproject.toml only | Environment not fully reproducible |

### P3 — Documentation Gaps
| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| E13 | Stale claims in README/IMPLEMENTATION_REPORT | docs/*.md | Claims "portfolio ready V2" but gaps remain |
| E14 | No architecture decision records | N/A | Key methodological choices undocumented |
| E15 | Model card schema inconsistent | Multiple scripts | Different fields across modules |

---

## 3. Current Artifacts (Verified Present)

### data_quality_output/
- canonical_transactions.parquet (824K rows, 100MB) ✓
- customer_month.parquet ✓
- validation_results.json, reconciliation_report.json ✓
- transaction_type_report.csv, schema_report.csv ✓

### customer_360_output/
- customer_360_current.parquet (5,878 customers × 131 features) ✓
- features_at_20100930.parquet through features_at_20110930.parquet (5 PIT snapshots) ✓
- feature_dictionary.json (112 features with point_in_time_safe flags) ✓
- feature_groups.json ✓

### online_retail_segmentation/
- customer_segments.parquet (12 clusters + noise, 83.4% coverage) ✓
- segment_profiles.csv, pca_coordinates.csv ✓
- cluster_model_candidates.csv (10 candidates with quality metrics) ✓
- run_manifest.json ✓

### cohort_analysis_output/
- matrix_logo_retention.csv, matrix_net_revenue_retention.csv ✓
- cohort_scorecard.csv, cohort_acquisition_quality.csv ✓
- retention_decay_curve.csv ✓

### clv_analysis_output/
- clv_customer_predictions.csv (4 margin scenarios) ✓
- clv_monthly_summary.csv ✓
- model_card.json ✓
- feature_importance.csv ✓

### churn_next_purchase_output/
- customer_churn_next_purchase.csv ✓
- model_card.json ✓
- customer_month_panel.parquet ✓

### recommendation_output/
- recommendations.parquet ✓
- model_card.json (temporal eval framework) ✓

### decision_engine_output/
- customer_decision_scores.csv ✓
- action_summary.csv ✓
- model_card.json ✓

### product_analytics_output/
- product_metrics.parquet ✓
- co_purchase_matrix.parquet ✓
- product_analytics_card.json ✓

### reactivation_output/
- reactivation_predictions.csv ✓
- reactivation_panel.parquet ✓
- model_card.json ✓

---

## 4. Current Tests

### Test Suite Status
- **Total collected**: 57 tests (35 retail_ds + 22 app)
- **Execution**: Blocked by Polars "Illegal instruction" (environment issue, not code)
- **Framework**: pytest with `-p no:timeout -p no:zarr` flags
- **Coverage areas**:
  - retail_ds: ingestion, cleaning, classification, customer-month, features, validation, backtesting
  - app: formatters, ArtifactRegistry, UI components, integration, data validation

### Missing Test Categories (Per Target Architecture)
- [ ] Data contract tests (schema, types, uniqueness, financial reconciliation)
- [ ] Scientific tests (no future leakage, target censoring, temporal split correctness)
- [ ] Calibration tests (CLV holdout, recommendation holdout, segment stability)
- [ ] UI/integration tests (global filters, deep linking, semantic formatting)

---

## 5. Current Analytical Limitations

### Customer 360
- **Descriptive only**: Current-state features valid for dashboard, NOT for historical ML
- **PIT features exist** but not fully integrated with downstream model training
- **Feature registry** has `point_in_time_safe` flags but unknown features default to unsafe (good)

### Segmentation
- **No temporal holdout**: Trained on full history (descriptive only)
- **Stability**: Bootstrap ARI 0.97-0.99 reported but not independently verified
- **Noise handling**: 16.6% noise correctly identified but not separated in downstream use
- **Block balancing**: Failed for price_behavior (segment 5 outlier)

### Cohort Analysis
- **NRR > 100%**: Documented but semantics unclear (retail revenue index vs contractual retention)
- **Maturity censoring**: Scorecard averages across cohorts with different max ages
- **No forward-looking**: Purely descriptive

### CLV
- **Methodology**: Dynamic Probabilistic Discounted Net-Revenue CLV Proxy
- **Not true CLV**: No margin, no BG/NBD lifetime model, no probabilistic churn process
- **Validation**: Temporal split + Platt calibration + p10/p50/p90 intervals
- **Missing**: Empirical coverage verification of uncertainty bands
- **MC simulation**: Partially vectorized (NumPy broadcasting for predictions, Python loops for state updates)

### Churn / Next-Purchase
- **Target**: Discrete-time hazard (purchase_next_month) + next-purchase 7/30/60 days
- **Fixed**: IndexError resolved by building targets from same panel
- **Calibration**: Platt calibration on validation set
- **Survival**: Recursive hazard with age-decay shrinkage (heuristic)

### Recommendations
- **Algorithm**: Vectorized co-purchase matrix + popularity fallback
- **Scoring**: `0.7 * co_purchase_score + 0.3 * popularity_score` (normalized)
- **Lift**: True association lift with log transformation
- **Issues**: Deduplication bug (33K duplicate customer-product pairs)
- **Temporal eval**: Framework exists (`--temporal-eval`) but not run/verified

### Decision Engine
- **Framing**: Explicitly "Observational prioritization" (not causal)
- **Actions**: protect_value, accelerate_purchase, reactivate, cross_sell, nurture, monitor
- **Capacity**: Hard caps per action type + global budget
- **Scores**: Heuristic combination (not calibrated probabilities)
- **Missing**: Frequency suppression rules, channel assignment, policy config file

### Reactivation
- **Framework**: Exists in customer-month panel (`is_reactivation` flag)
- **Model**: Dedicated script exists but not integrated with decision engine
- **Horizon**: 3 months default

---

## 6. Current UI Structure (Streamlit V2)

### Pages (9)
1. **Executive** — KPIs, CLV concentration, portfolio health
2. **Customer 360** — Customer lookup (deep linking `?customer_id=`), trajectory, features
3. **Segmentation** — PCA scatter, segment profiles, size/value tables
4. **Cohorts** — Retention heatmaps, decay curves, scorecard
5. **Predictive Value** — CLV distribution, decile boxplots, simulation trajectory
6. **Retention & Next Purchase** — Calibration, survival curves, feature importance
7. **Recommendations** — Product recommendations table, reason distribution
8. **Decision Engine** — Action allocation, priority scores, value proxy
9. **Methodology** — Static documentation

### Design System
- **Tokens**: `streamlit_app/ui/tokens.py` (colors, spacing, typography)
- **Components**: `streamlit_app/ui/components.py` (cards, tables, chips)
- **Charts**: `streamlit_app/ui/charts.py` (Plotly with design tokens)
- **Formatters**: 11 semantic functions in `app_formatting.py`
- **Theme**: CSS custom properties with light/dark support

### State Management
- **Global filters**: None currently (per-page only)
- **Deep linking**: Customer 360 supports `?customer_id=`
- **Cross-run consistency**: ArtifactRegistry detects mixed run_ids/data_versions

---

## 7. Migration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing artifact schemas | High | All downstream consumers | Version contracts, maintain backwards compatibility |
| PIT feature changes break models | Medium | CLV, Churn, Reactivation, Segmentation training | Explicit feature versioning, gradual migration |
| Run manifest addition changes dashboard | Medium | Streamlit ArtifactRegistry | Optional fields, graceful degradation |
| Config resolution changes script behavior | Low | All scripts | Test each script with new config loader |
| Semantic terminology changes break UI | Medium | Dashboard labels, tooltips | Centralized formatters, single source of truth |

---

## 8. Target Architecture Gap Analysis

| Target Component | Current State | Gap |
|------------------|---------------|-----|
| Canonical Ingestion | ✅ retail_ds.io | Minor: config-driven path resolution needed |
| Curated Retail Data Model | ⚠️ Partial (customer-month) | Missing: dim_customer, dim_product, dim_calendar, fct_basket, fct_customer_day, fct_customer_snapshot |
| Point-in-Time Feature Layer | ✅ retail_ds/features | Missing: leakage invariance tests, exact month windows |
| Segmentation | ✅ HDBSCAN + stability | Missing: RFM baseline, segment transitions, drift reports |
| Market Basket | ❌ Not implemented | Customer affinity exists but not true basket association |
| Recommendation Challenger | ⚠️ Single model | Missing: popularity, recency, item-item, replenishment, sequence baselines |
| CLV Benchmark | ❌ Not implemented | Missing: BG/NBD + Gamma-Gamma comparison |
| Survival/Churn | ✅ Discrete-time hazard | Missing: proper time-varying hazard, multiple horizons |
| Reactivation | ⚠️ Framework only | Missing: horizon censoring fix, dedicated model card |
| Decision Engine | ⚠️ Heuristic scores | Missing: policy config, calibrated confidence, eligibility/suppression |
| Customer 360 Snapshot | ❌ Not unified | Separate artifacts per module |
| Streamlit Global Scope | ❌ Per-page only | Missing: unified CustomerScope context |
| Artifact Registry | ✅ Config-driven | Missing: manifest-based consistency, schema versioning |
| MLflow Tracking | ❌ Not integrated | Local file-based model cards only |
| CI/CD | ❌ None | GitHub Actions needed |
| Documentation | ⚠️ Mixed freshness | Stale claims, missing authoritative docs |

---

## 9. Immediate Next Steps (Phase 0 Complete)

**Baseline established.** Proceeding to Phase 1: Canonical Data Architecture.

Key priorities:
1. Enforce single ingestion across ALL scripts (remove any residual direct Excel reads)
2. Config-driven path resolution for all scripts (`--config` flag)
3. Transaction taxonomy enforcement with tests
4. Canonical field standardization
5. Data contracts + validation layer