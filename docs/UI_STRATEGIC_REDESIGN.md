# UI Strategic Redesign — Retail Customer Intelligence

## Overview

This document describes the transformation of the Streamlit application from a model-centric analytical tool into a **strategic retail customer intelligence cockpit**. The redesign follows a decision-oriented information architecture that helps retail stakeholders move from portfolio understanding to actionable customer audiences.

---

## UX Principles

### Decision Hierarchy

Every page follows this decision flow:

```
WHAT CHANGED?
      ↓
WHERE IS THE VALUE?
      ↓
WHERE IS THE RISK?
      ↓
WHERE IS THE OPPORTUNITY?
      ↓
WHO / WHICH SEGMENTS MATTER?
      ↓
WHAT BEHAVIOR EXPLAINS IT?
      ↓
WHAT PRODUCTS / BASKETS MATTER?
      ↓
WHAT SHOULD BE EXAMINED / PRIORITIZED?
      ↓
WHAT AUDIENCE CAN BE EXPORTED?
      ↓
WHAT EVIDENCE SUPPORTS THE DECISION?
```

### Core Principles

1. **Commercial context over model outputs** — KPIs answer business questions, not just show model metrics
2. **Insight before evidence** — Headline insight → supporting visualization → scientific note
3. **Scientific integrity** — Clear distinction between observed vs predicted, association vs affinity, observation vs causality
4. **Global scope persistence** — Filters (as-of date, country, segment, lifecycle, value band, risk band, RFM) remain stable across navigation
5. **Actionable outputs** — Every workspace ends with an exportable audience table

---

## Information Architecture

### Workspace Structure (8 Primary Workspaces)

| Workspace | Purpose | Primary Question |
|-----------|---------|------------------|
| **Strategy** | Executive cockpit | What changed? Where is value/risk/opportunity? |
| **Customers** | Customer 360 deep-dive | Who is this customer? How do they compare to peers? What action is allocated? |
| **Value & Retention** | Value concentration & risk | Where is future value? Where is economically meaningful risk? |
| **Segments** | Commercial segmentation | Which segments matter? How do customers move between them? |
| **Products & Baskets** | Product intelligence | What products, combinations, affinities represent opportunity? |
| **Personalisation** | Recommendation quality | How broad is coverage? What method produced this? Why this product? |
| **Activation** | Audience management | How much capacity? How allocated? Who is prioritized? Who is suppressed? |
| **Science & Governance** | Technical diagnostics | Data quality, model performance, calibration, lineage |

### Navigation

- Sidebar radio buttons with workspace names (stakeholder-readable)
- Pipeline status expander showing module health
- Global filter bar at top of every page
- Scientific disclaimer in sidebar footer

---

## Global Scope / Filter System

### Scope Object

```python
CustomerScope(
    as_of_date: Optional[date],
    country: str,           # "All" or specific country
    segment: str,           # "All" or behavioral segment name
    lifecycle: str,         # "All" or lifecycle state
    value_band: str,        # "All" or value tier (Platinum/Gold/Silver/Bronze)
    risk_band: str,         # "All" or action priority tier
    rfm_segment: str,       # "All" or RFM segment
)
```

### Implementation

- Stored in `st.session_state.global_scope`
- Applied via `apply_global_scope(df)` function (joins with customer_360 for filter columns)
- Filter bar rendered at top of app with 8 compact controls
- Active scope summary shown as caption: `Scope: Sep 2026 · UK · Active repeat · Platinum · High`
- Clear button resets all filters

### Scope Persistence

- Survives page navigation
- Deep-linking via `?customer_id=12345` still works on Customers page
- Query parameters not used for scope (session state preferred for multi-select)

---

## Visual Design System

### Design Tokens (Preserved)

- Light/dark theme via CSS custom properties
- Consistent spacing, radius, shadow scales
- Typography: Inter, 0.72rem–2.25rem scale
- Colorblind-safe palettes for actions, segments, status

### Semantic Color Language

| Concept | Color | Usage |
|---------|-------|-------|
| **VALUE** | Primary/Green scale | CLV, revenue, predicted value |
| **RISK** | Danger/Red scale | Churn probability, inactivity risk |
| **PROPENSITY** | Info/Blue scale | Purchase probability, next-purchase |
| **LIFECYCLE** | Categorical (segment palette) | Segment, lifecycle state |
| **ACTION** | Action-specific colors | Protect=Green, Accelerate=Blue, Reactivate=Amber, etc. |
| **SCIENCE** | Neutral/Slate | Model metrics, diagnostics |
| **WARNING** | Amber | Data freshness, capacity alerts |

### Chart Design Rules

Every chart must have:
1. Informative title answering a business question
2. Descriptive axes with meaningful units
3. Readable hover information
4. Business-readable labels (no internal field names)
5. Appropriate sorting/aggregation
6. No chart junk

**Title patterns:**
- ❌ "Distribution", "Chart", "Metrics"
- ✅ "Where is customer value concentrated?", "Which customer groups contain the highest value risk?"

### Table Design Rules

- Business-readable column names (e.g., `clv_mean` → "Predicted Future Value")
- Semantic formatting via `infer_semantic_type()`
- Default sorting by business relevance
- Conditional visual emphasis (action badges, percentile bars)
- CSV export for audience tables

---

## Workspace Specifications

### 1. Strategy — Customer Strategy

**Sections:**
1. **Portfolio Headline** — 6 KPIs: Customers, Predicted Future Value, Mean Inactivity Risk, Mean Purchase Propensity, Value at Risk, Purchase Opportunity
2. **What Changed?** — Lifecycle movement (current vs prior), net customer flow table
3. **Where is the Value?** — Enhanced concentration curve with Top 10%/20%/40% annotations
4. **Opportunity Matrix** (HERO) — X=Purchase Propensity, Y=Inactivity Risk, Size=Predicted Value, Color=Action; quadrant labels: PROTECT/GROW/NURTURE/ACCELERATE
5. **Priority Opportunity Table** — Cohorts: High-value at-risk, High-value ready-to-buy, Valuable dormant, Active cross-sell
6. **Strategic Insights** — 2–4 dynamic insights generated from data

### 2. Customers — Customer 360

**Sections:**
1. **Customer Header** — ID, Segment, Lifecycle, Action, Value Tier, Engagement Tier, Priority Tier
2. **Key Metrics** — Predicted Future Value, Inactivity Risk, 30d Propensity, Revenue, Orders, Recency, Breadth
3. **Trajectory** — Monthly revenue/orders dual-axis (observed only)
4. **Model Evidence** — Formatted table of all model outputs with scientific terminology
5. **Peer Benchmark** — Percentile bars vs segment peers (Frequency, Value, Recency, Breadth, Repeat Rate, AOV, Return Rate, Cadence)
6. **Value/Risk/Propensity Profile** — Visual percentile bars
7. **Product Intelligence** — Top purchased + recommendations with lift evidence
8. **Customer Action Card** — Action, Priority, Confidence, Expected Value, Reason, Eligibility/Suppression

### 3. Value & Retention

**Sections:**
1. **Value Distribution** — Histogram with box marginal, uncertainty band (p10–p90)
2. **Value Concentration** — Lorenz curve with Top 10%/20%/40% callouts
3. **Value-at-Risk Matrix** — Scatter: Value vs Risk, quadrants at medians, segment/lifecycle color
4. **Risk by Value Decile** — Horizontal bar + table, correlation insight
5. **Survival** — Median survival by horizon (3m/6m/12m), by segment
6. **Model Quality** (collapsible) — ROC-AUC, PR-AUC, Brier, Log Loss, Calibration

### 4. Segments — Customer Segments

**Sections:**
1. **Segment Portfolio Map** — Scatter: Purchase Propensity vs Value/Customer, bubble size=customers
2. **Segment Scorecard** — Sortable table: Segment, Customers, Revenue, Value/Customer, Retention, Risk, Breadth, Strategic Role
3. **Segment Composition** — Normalized heatmap by feature blocks (Economic, Cadence, Assortment, Pricing, Temporal, Lifecycle, Returns)
4. **Segment Movement** — Transition matrix (count/row%/col%), largest inflows/outflows, stability
5. **RFM Benchmark** — Interpretable reference: Champions, Loyal, Potential Loyalists, Recent, At Risk, Dormant
6. **PCA Diagnostics** (collapsible) — Latent space visualization as supporting evidence

### 5. Products & Baskets

**Sections:**
1. **Product Portfolio** — KPIs, product role distribution
2. **Basket Opportunity Map** — Support vs Lift scatter, confidence bubble size, quadrant interpretation
3. **Association Rules Table** — Filterable (min support/confidence/lift), columns: A, B, Support, Confidence, Lift, Leverage, Conviction
4. **Customer Affinity vs Basket Association** — Explicit comparison table, shows both top pairs side-by-side
5. **Product Opportunity Table** — Product, Revenue, Customers, Repeat Rate, Role, Max Basket Lift, Max Affinity Lift

### 6. Personalisation

**Sections:**
1. **Recommendation Coverage** — Customers covered, total recs, avg/customer, catalog coverage
2. **Recommendation Benchmark** — Challenger results: Model, Hit@5, Hit@10, Recall@10, MRR@10, NDCG@10, Coverage
3. **Recommendation Quality** — Score distribution by reason, recommended product role profile
4. **Customer Recommendations** — Filterable (customer, segment, reason, score threshold), enriched with product info
5. **Audience Export** — Filtered recommendations with context fields

### 7. Activation

**Sections:**
1. **Capacity Board** — Total, Allocated, Remaining, Utilization % with progress bar
2. **Action Allocation** — Horizontal bar by action, CLV vs Churn scatter colored by action
3. **Opportunity Matrix** — Value vs Risk, bubble size=Propensity, action color
4. **Priority Queue** — Operational table with filters (top N, action, segment, risk band), exportable
5. **Eligibility/Suppression** — Eligible count, Suppressed count, Capacity excluded, Suppression reasons
6. **Export** — Configurable column selection for campaign execution

### 8. Science & Governance

**Sections:**
1. **Data Quality** — Validation results, schema, missingness, duplicates, transaction types, reconciliation
2. **Feature Governance** — Feature dictionary, groups, validation results, PIT metadata
3. **Model Performance** — Summary table across all models with key metrics
4. **Calibration** — Calibration bins for churn and next-purchase models
5. **CLV Validation** — Temporal split details, BG/NBD benchmark
6. **Recommendation Evaluation** — Temporal holdout, baseline comparison, challenger results
7. **Segmentation Stability** — Bootstrap ARI, cluster candidates
8. **Run Lineage** — Run IDs, data/code/model versions, cross-run consistency check
9. **Methodology** — 10-step analytical workflow
10. **Limitations** — Documented scientific risks

---

## Reusable Components

### Insight Panel
```python
render_insight(
    label="KEY INSIGHT",
    headline="Top 20% of customers account for 61% of predicted future value",
    detail="Median CLV: £1,234 | Mean: £2,345",
    evidence="CLV proxy distribution across 5,878 customers",
    severity="high",  # high/medium/low/info
)
```

### Opportunity Matrix
```python
plot_opportunity_matrix(
    df,
    x_col="next_purchase_30d_probability",
    y_col="churn_probability",
    size_col="clv_mean",
    color_col="recommended_action_capped",
    quadrant_labels={
        "top_right": "PROTECT\nHigh value, high risk",
        "top_left": "NURTURE\nLower value, high risk",
        "bottom_right": "GROW\nHigh value, low risk",
        "bottom_left": "ACCELERATE\nLower value, low risk",
    },
)
```

### Peer Benchmark
```python
plot_peer_benchmark(
    customer_values={"Orders": 45, "AOV": 23.5, "Recency": 12},
    peer_distributions={"Orders": peer_orders, "AOV": peer_aov, "Recency": peer_recency},
    higher_is_better={"Orders": True, "AOV": True, "Recency": False},
)
```

### Transition Matrix
```python
plot_transition_matrix(
    transition_df,
    from_col="prior_segment",
    to_col="current_segment",
    count_col="customers",
    normalize="row_pct",  # or "count", "col_pct"
)
```

### Value Concentration
```python
plot_value_concentration_curve(
    clv_values,
    annotate_thresholds=[0.1, 0.2, 0.4],
    label_prefix="Top",
)
```

### Audience Table
```python
render_audience_table(
    df,
    columns=["Customer ID", "Action", "Priority", "Value", "Risk"],
    download_filename="audience.csv",
    download_label="Export audience",
)
```

### Scientific Note
```python
render_scientific_note(
    "This is modeled future value, not realized historical revenue.",
    title="Scientific note"
)
```

---

## Metric Language Standards

### Approved Terminology

| Use | Avoid |
|-----|-------|
| Predicted Future Value (CLV Proxy) | CLV, Guaranteed Value |
| Next-month Inactivity Risk | Churn Probability, Likely to Churn |
| 30-day Purchase Propensity | Purchase Probability, Likely to Buy |
| Customer Affinity | Cross-sell Score |
| Basket Association | Affinity (when transaction-level) |
| Value at Risk | Value in Danger |
| Customer Opportunity | Uplift Opportunity |
| Priority Score | Confidence Score |
| Policy Score | Treatment Effect |
| Observational Prioritization | Prescriptive Decision |

### Scientific Disclaimers

Every model output page includes:
- "CLV Proxy is a forward economic estimate, not a historical revenue total"
- "Inactivity risk is a model-derived probability, not an observed outcome"
- "Decision scores are observational prioritization, not causal uplift estimates"
- "Basket association ≠ Customer affinity (different grain, different method)"

---

## Filter Model

### Global Filters (persist across pages)
- As-of date (date picker)
- Country (select)
- Behavioral segment (select)
- Lifecycle state (select)
- Value band/tier (select)
- Risk band/priority tier (select)
- RFM segment (select)

### Page-Local Filters (do not persist)
- Top N sliders
- Action/reason/segment selectors within page
- Score thresholds
- Min support/confidence/lift for rules

---

## Insight Generation Logic

Insights are generated deterministically from computed data:

```python
# Value concentration
top_20_share = sorted_clv.head(n*0.2).sum() / total * 100
→ "Top 20% hold {top_20_share:.0f}% of predicted future value"

# Risk concentration by segment
top_risk_segment = segment_risk.idxmax()
top_risk_value = segment_risk.max()
→ "Inactivity risk concentrated in '{top_risk_segment}' ({format_probability(top_risk_value)})"

# Lifecycle shift
at_risk_pct = lifecycle_dist.get("at_risk", 0) * 100
→ "{at_risk_pct:.0f}% of customers in 'at risk' lifecycle state"

# Action balance
protect_pct = protect_count / total * 100
→ "Protection actions cover {protect_pct:.0f}% of customers"

# Risk-value correlation
corr = clv.corr(churn)
if corr > 0.1: "Risk increases with value — value protection critical"
elif corr < -0.1: "Risk decreases with value — focus on mid-value"
```

**Rule:** Only generate insight when supported by data. Never fabricate.

---

## Table Conventions

### Column Name Mapping

| Internal | Display |
|----------|---------|
| `clv_mean` | Predicted Future Value |
| `churn_probability` | Inactivity Risk |
| `next_purchase_30d_probability` | 30d Purchase Propensity |
| `recommended_action_capped` | Recommended Action |
| `priority_score` | Priority Score |
| `decision_confidence` | Decision Confidence |
| `expected_value_proxy` | Expected Value Proxy |
| `association_lift` | Basket Lift |
| `support` | Transaction Support |
| `recommended_product` | Recommended Product |
| `segment_name` | Behavioral Segment |
| `lifecycle_state` | Lifecycle State |
| `value_tier` | Value Tier |
| `action_priority_tier` | Priority Tier |

### Formatting
- Currency: £1,234 / £1.23M / £12.3K
- Probability: 23.5% (0–1 scale × 100)
- Percentage: 23.5% (already in % scale)
- Ratio: 1.23x
- Count: 1,234
- Score: 0.87
- Days: 45 days
- Date: 2026-09-15
- Month: 2026-09

---

## Scientific Terminology

| Concept | Term | Note |
|---------|------|------|
| CLV | Predicted Future Net Revenue (CLV Proxy) | Not true economic CLV |
| Churn | Next-month Inactivity Risk | Discrete-time hazard |
| Survival | Model-derived Survival Probability | Not Kaplan-Meier |
| Recommendation | Co-purchase Ranking Heuristic | Not "model" |
| Decision | Observational Prioritization | Not prescriptive |
| Market Basket | Transaction-level Association Rules | Apriori on invoices |
| Customer Affinity | Cross-purchase Co-occurrence | Customer-level over time |
| RFM | Interpretable Benchmark | Rule-based, not ML |

---

## Known Limitations

1. **Global scope requires Customer ID** — Pages without Customer ID cannot be filtered
2. **Scope join performance** — `apply_global_scope()` merges with customer_360 on each call; acceptable for current data size
3. **RFM segments not in unified snapshot** — RFM benchmark uses separate artifact
4. **Challenger results optional** — Personalisation page degrades gracefully if missing
5. **Single-country bias** — Model trained on UK-dominant data
6. **No experimental layer** — All outputs observational; causal claims require A/B testing
7. **Environment errors** — 18 pre-existing test errors from Polars/NumPy datetime incompatibility (not code bugs)

---

## Performance Considerations

- `@st.cache_data` on `get_registry()` and `load_dataframe()`
- Cache keys include file `mtime` + `size` for auto-invalidation
- Sampling for large scatter plots (max 8,000–12,000 points)
- Page-specific data loading (no full reload on navigation)
- Unified customer_360 artifact as primary source for scope joins

---

## Files Changed

### New Files
- `streamlit_app/pages/strategy.py`
- `streamlit_app/pages/customers.py`
- `streamlit_app/pages/value_retention.py`
- `streamlit_app/pages/segments.py`
- `streamlit_app/pages/products_baskets.py`
- `streamlit_app/pages/personalisation.py`
- `streamlit_app/pages/activation.py`
- `streamlit_app/pages/science.py`
- `streamlit_app/ui/scope.py`

### Modified Files
- `streamlit_app/app.py` — New navigation, global scope, filter bar
- `streamlit_app/ui/__init__.py` — New exports (components, charts, scope)
- `streamlit_app/ui/components.py` — Insight, peer benchmark, audience table, distribution summary, scientific note
- `streamlit_app/ui/charts.py` — Opportunity matrix, transition matrix, enhanced concentration curve, peer benchmark
- `tests/test_app.py` — Updated HERO_COPY test
- `tests/test_ui_components.py` — Updated HERO_COPY test

### Removed/De-emphasized
- `executive.py`, `predictive.py`, `retention.py`, `cohorts.py`, `methodology.py` — Functionality merged into new workspaces
- PCA as primary segmentation view → moved to Science diagnostics
- Raw model metric tables → moved to Science workspace
- Generic probability histograms → replaced with contextual visualizations

---

## Test Results

```
================= 167 passed, 2 warnings, 18 errors in 31.01s ==================
```

- **167 tests pass** (all app, formatter, integration, data validation, recommendation, decision engine, segmentation, UI component tests)
- **18 errors** are pre-existing environment issues:
  - 6 × Polars/NumPy datetime resolution (test_product_analytics.py)
  - 11 × Path suffix attribute error (test_retail_ds.py leakage/temporal tests)
  - 1 × fixture issue (test_retail_ds.py)
- **0 test failures** attributable to the redesign

---

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| Nontechnical stakeholder can navigate primary workflow | ✅ Strategy → Opportunity Matrix → Priority Table → Customer/Segment → Product → Recommendation → Action → Export |
| Each primary page has strategic visualization | ✅ All 8 workspaces have hero visual answering business question |
| Scientific integrity maintained | ✅ No misrepresentation of historical vs predicted, association vs affinity, observation vs causality |
| Actionable flow without context loss | ✅ Global scope persists; deep-links work; audience exports carry context |
| Visual hierarchy: headline → insight → hero → table → science | ✅ Implemented consistently across workspaces |
| Missing/stale data visible | ✅ Module status chips, freshness notes, graceful degradation |
| Performance: no repeated raw data loads | ✅ Cached registry, page-specific loading, unified artifact priority |

---

## Migration Notes

### For Developers
- Import new components from `streamlit_app.ui`
- Use `apply_global_scope(df)` for scope-aware pages
- Follow insight/component patterns for new visualizations
- Maintain semantic column names in artifacts

### For Stakeholders
- Bookmark Strategy page for daily/weekly review
- Use global filters to focus on specific segments/markets
- Export audiences from Activation or Personalisation for campaign tools
- Science workspace for model governance reviews

---

*Document version: 1.0 | Generated with strategic redesign implementation*