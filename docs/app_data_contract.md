# App Data Contract — Retail Customer Intelligence

This document defines the expected artifact contracts for the Streamlit application.
Each module produces specific output files that the app discovers via the config-driven
artifact registry (`app_data.py`).

---

## Overview

| Module | Output Directory | Primary Artifact | Status |
|--------|-----------------|------------------|--------|
| Data Quality | `data_quality_output/` | `canonical_transactions.parquet` | Required |
| Customer 360 | `customer_360_output/` | `customer_360_current.parquet` | Required |
| Segmentation | `online_retail_segmentation/` | `customer_segments.parquet` | Required |
| Cohorts | `cohort_analysis_output/` | `matrix_logo_retention.csv` | Required |
| CLV | `clv_analysis_output/` | `clv_customer_predictions.csv` | Required |
| Churn/Next Purchase | `churn_next_purchase_output/` | `customer_churn_next_purchase.csv` | Required |
| Reactivation | `reactivation_output/` | `reactivation_predictions.csv` | Required |
| Product Analytics | `product_analytics_output/` | `product_metrics.parquet` | Required |
| Recommendations | `recommendation_output/` | `recommendations.parquet` | Optional |
| Decision Engine | `decision_engine_output/` | `customer_decision_scores.csv` | Required |

---

## Artifact Contracts

### 1. Data Quality — `canonical_transactions.parquet`

**Grain:** One row per transaction line item
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Invoice | string | Invoice number |
| StockCode | string | Product code |
| Quantity | float | Quantity purchased |
| InvoiceDate | datetime | Transaction timestamp |
| Price | float | Unit price |
| Customer ID | int64 | Customer identifier |
| Country | string | Customer country |
| transaction_type | string | "sale" \| "cancellation" \| "return" |
| is_clean_sale | boolean | True for valid positive sales |
| gross_merchandise_revenue | float | Quantity × Price |

**Freshness:** Generated daily from source data
**Min Rows:** 10,000

---

### 2. Customer 360 — `customer_360_current.parquet`

**Grain:** One row per customer (current state)
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| lifetime_net_revenue | float | Net revenue (excl. returns) |
| lifetime_gross_revenue | float | Gross revenue (incl. returns) |
| lifetime_orders | int64 | Total orders placed |
| lifetime_units | float | Total units purchased |
| recency_days | int64 | Days since last order |
| active_month_count | int64 | Months with ≥1 order |
| first_order_date | datetime | First order date |
| last_order_date | datetime | Last order date |
| median_interpurchase_days | float | Median days between orders |
| unique_products | int64 | Distinct products purchased |
| lifetime_return_rate | float | Return rate (0-1) |

**Enrichment Columns (from other modules):**
| Column | Source Module | Description |
|--------|---------------|-------------|
| segment | Segmentation | Cluster assignment (-1 = noise) |
| clv_mean | CLV | Predicted future net revenue (mean) |
| clv_p10, clv_p90 | CLV | Predictive interval |
| churn_probability | Churn | Next-month inactivity risk |
| next_purchase_30d_probability | Churn | 30-day purchase propensity |
| reactivation_probability | Reactivation | Probability of reactivation |
| final_action | Decision Engine | Recommended action |

**Freshness:** Generated after all upstream modules complete
**Min Rows:** 1,000

---

### 3. Segmentation — `customer_segments.parquet`

**Grain:** One row per customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| segment | int64 | Cluster assignment (-1 = noise/low-density) |
| segment_confidence | float | Assignment confidence (0-1) |

**Freshness:** Generated after clustering completes
**Min Rows:** 1,000

---

### 3b. Segmentation — `segment_profiles.csv`

**Grain:** One row per segment
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| cluster | int64 | Segment identifier |
| customers | int64 | Segment size |
| share | float | Proportion of customer base |
| lifetime_net_revenue_mean | float | Mean net revenue |
| ... | ... | Behavioral feature means per segment |

---

### 3c. Segmentation — `pca_coordinates.csv`

**Grain:** One row per customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| PC1 | float | First principal component |
| PC2 | float | Second principal component |
| segment | int64 | Cluster assignment |

---

### 4. Cohorts — `matrix_logo_retention.csv`

**Grain:** One row per cohort per age month
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| cohort_month | string | Acquisition month (YYYY-MM) |
| age_month | int64 | Months since acquisition (0 = acquisition month) |
| logo_retention | float | Logo retention rate (0-1) |
| cohort_size | int64 | Customers in cohort |

**Maturity:** `age_month` beyond data horizon = not observable (NaN)

---

### 4b. Cohorts — `matrix_net_revenue_retention.csv`

**Grain:** One row per cohort per age month
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| cohort_month | string | Acquisition month (YYYY-MM) |
| age_month | int64 | Months since acquisition |
| net_revenue_retention | float | Net revenue retention index (can exceed 1.0) |

**Note:** NRR > 1.0 is possible because surviving customers can spend more than acquisition month.

---

### 4c. Cohorts — `retention_decay_curve.csv`

**Grain:** One row per age month (weighted across cohorts)
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| age_month | int64 | Months since acquisition |
| weighted_logo_retention | float | Population-weighted logo retention |
| weighted_net_revenue_retention | float | Population-weighted NRR |
| cohort_population_weight | float | Weight for this age month |

---

### 5. CLV — `clv_customer_predictions.csv`

**Grain:** One row per customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| clv_mean | float | Predicted future net revenue (mean) |
| clv_median | float | Predicted future net revenue (median) |
| clv_p10 | float | 10th percentile (predictive interval) |
| clv_p90 | float | 90th percentile (predictive interval) |
| horizon_months | int64 | Forecast horizon |

**Scientific Label:** "Predicted Future Net Revenue (CLV Proxy)" — no margin data available.

---

### 6. Churn/Next Purchase — `customer_churn_next_purchase.csv`

**Grain:** One row per customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| churn_probability | float | Next-month inactivity risk (0-1) |
| next_purchase_30d_probability | float | 30-day purchase propensity (0-1) |
| next_purchase_60d_probability | float | 60-day purchase propensity (0-1) |
| survival_3m | float | Model-derived 3-month survival probability |
| survival_6m | float | Model-derived 6-month survival probability |
| survival_12m | float | Model-derived 12-month survival probability |

**Scientific Labels:**
- `churn_probability` → "Next-month inactivity risk"
- `survival_*` → "Model-derived survival probability" (not Kaplan-Meier)

---

### 7. Reactivation — `reactivation_predictions.csv`

**Grain:** One row per inactive customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| reactivation_probability | float | Probability of reactivation (0-1) |
| recency_days | int64 | Days since last order |
| months_inactive | int64 | Months since last order |

---

### 8. Product Analytics — `product_metrics.parquet`

**Grain:** One row per product
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| StockCode | string | Product code |
| Description | string | Product description |
| product_role | string | "repeat" \| "one_time" \| "seasonal" \| "new" \| "declining" |
| total_revenue | float | Total revenue |
| total_units | float | Total units sold |
| unique_customers | int64 | Distinct customers |
| repeat_purchase_rate | float | Repeat purchase rate |
| avg_price | float | Average unit price |
| velocity | float | Units per month |

---

### 8b. Product Analytics — `co_purchase_matrix.parquet`

**Grain:** One row per product pair
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| product_a | string | First product code |
| product_b | string | Second product code |
| cooccurrence | int64 | Number of shared customers |

**Symmetric:** Both (A,B) and (B,A) present.

---

### 9. Recommendations — `recommendations.parquet`

**Grain:** One row per customer-product recommendation
**Required Columns (output contract):**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| recommended_product | string | Recommended product code |
| score | float | Recommendation score (0-1 after normalization) |
| reason | string | "co_purchase" \| "popularity" |
| support | int64 | Number of supporting purchased products |
| lift | float | Normalized affinity proxy (not true association-rule lift) |
| co_purchase_score | float | Raw co-purchase affinity score |
| popularity_score | float | Normalized product popularity (0-1) |

**Diagnostic Columns (enriched):**
| Column | Type | Description |
|--------|------|-------------|
| Description | string | Product description |
| product_role | string | Product role category |
| avg_price | float | Average price |
| total_revenue | float | Product total revenue |

**Generation:** Vectorized Polars joins, top-20 partners per product, popularity fallback.

---

### 10. Decision Engine — `customer_decision_scores.csv`

**Grain:** One row per customer
**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Customer ID | int64 | Unique customer identifier |
| recommended_action | string | "protect_value" \| "accelerate_purchase" \| "reactivate" \| "cross_sell" \| "nurture" \| "monitor" |
| priority_score | float | Composite priority score |
| clv_mean | float | Predicted future net revenue |
| churn_probability | float | Next-month inactivity risk |
| final_action | string | Same as recommended_action |
| capacity_capped | boolean | Whether action was capped by capacity |

**Scientific Label:** "Observational prioritization policy" — no causal uplift estimation.

---

## Cross-Module Consistency

The app validates cross-module consistency at startup:

1. **Customer ID alignment:** All modules should cover the same customer set
2. **Run ID matching:** Artifacts from the same pipeline run should share `run_id`
3. **Freshness:** Staleness warnings if artifacts exceed configured TTL

**Warning displayed when:** Modules originate from different runs or have mismatched row counts.

---

## Fallback Behavior

| Module | If Missing | App Behavior |
|--------|------------|--------------|
| Recommendations | Optional | Shows product analytics instead; "Run recommendation_engine.py" banner |
| Reactivation | Optional | Inactive customers show "Not available" for reactivation probability |
| CLV p10/p90 | Optional | Shows mean/median only; no predictive interval display |

---

## Validation Rules

Each artifact is validated on load:
- Required columns present
- Minimum row count met
- No all-NaN required columns
- Data types match expectations

Validation states: `valid` | `schema_mismatch` | `empty` | `invalid` | `missing`

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-09-28 | 1.0 | Initial contract documenting current output schemas |

---

*Generated from `app_data.py` EXPECTED_ARTIFACTS and `recommendation_engine.py` output contract.*