# Online Retail II Customer Intelligence

## Full Scientific, Engineering, Modeling, and Product Improvement Plan

**Project root**

`/home/lptop/Documents/coding/marketing_science`

**Current state**

The project has substantial analytical foundations, but the corrected audit still identifies critical ingestion, temporal-feature, and pipeline-integration issues. The recommended strategy is to rebuild the project around a canonical transaction layer and point-in-time customer data model before adding further predictive or decisioning functionality.

---

# 1. Target End-State

The final portfolio should become a coherent retail customer intelligence platform:

```text
                         ONLINE RETAIL II
                                |
                                v
                    +-----------------------+
                    |   INGESTION LAYER     |
                    | schema + type safety  |
                    | transaction semantics |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | CANONICAL TRANSACTIONS |
                    | clean / dedup / audit |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | POINT-IN-TIME CUSTOMER |
                    | 360 / customer-month   |
                    | feature store          |
                    +-----------+-----------+
                                |
             +----------------+----------------+
             |                |                |
             v                v                v
      SEGMENTATION         COHORTS            CLV
      behavior             retention          value
             |                |                |
             +----------------+----------------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          CHURN        NEXT PURCHASE      REACTIVATION
             |                |                |
             +----------------+----------------+
                              |
                              v
                    +-----------------------+
                    | DECISION ENGINE       |
                    | next-best-action      |
                    | capacity allocation   |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | STREAMLIT CUSTOMER    |
                    | INTELLIGENCE APP      |
                    +-----------------------+
```

The current architecture already contains Customer 360, segmentation, cohorts, CLV, churn/next purchase, and a Streamlit layer, but the audit shows the ingestion layer and cross-script architecture need to be hardened before the stack is treated as production-quality.

---

# 2. Guiding Principles

The rebuild should follow seven rules.

## 2.1 One source of truth

There should be exactly one canonical implementation of:

* file loading
* schema normalization
* transaction classification
* cancellation detection
* return handling
* revenue definitions
* customer-month construction

The current project duplicates transaction cleaning across five scripts, which creates drift and makes a bug fix require multiple edits.

## 2.2 Point-in-time correctness

Every predictive feature must answer:

> "What information would have been known at prediction time?"

Customer 360 can remain a descriptive current-state mart, but it must not be treated as a generic historical ML feature store. The corrected review explicitly distinguishes these two use cases.

## 2.3 Separate descriptive analytics from predictive analytics

Maintain explicit separation between:

```text
Descriptive
    current customer state
    segmentation
    historical cohorts

Predictive
    churn
    next purchase
    future revenue
    CLV proxy

Decisioning
    opportunity prioritization
    next-best-action
```

## 2.4 Observational ≠ causal

Transaction history alone does not identify incremental treatment effects.

Therefore:

```text
propensity
risk
value
opportunity
```

must not be presented as:

```text
incremental response
uplift
causal treatment effect
```

until treatment/control data exists.

## 2.5 Time-aware validation

Any model predicting future behavior should use temporal validation.

Random train/test splitting should not be the default for customer behavior prediction.

## 2.6 Reconciliation before optimization

Do not spend time optimizing HDBSCAN, Monte Carlo, or dashboard UX while core source metrics are still inconsistent.

The order should be:

```text
correctness
→ reproducibility
→ statistical validity
→ performance
→ presentation
```

## 2.7 Every important metric must be independently reproducible

A portfolio reviewer should be able to trace:

```text
raw transaction
→ canonical transaction
→ customer-month
→ model feature
→ prediction
→ dashboard number
```

---

# 3. Phase 0 — Freeze and Preserve the Current Version

Before changing anything:

```text
git init
git add .
git commit -m "baseline before scientific remediation"
```

Create:

```text
docs/
    baseline_assessment.md
```

Record the current state exactly as audited.

Preserve:

* current outputs
* current models
* current plots
* current scientific review
* current environment

Do not overwrite the current state until a reproducible corrected run exists.

---

# 4. Phase 1 — Rebuild the Ingestion Layer

This is the highest-priority engineering task.

The corrected audit found that Polars Excel ingestion inferred `Invoice` as `Int64`, causing cancellation invoice identifiers beginning with `C` to become null. The audit identified 19,494 cancellation rows being lost.

## 4.1 Create a shared package

Create:

```text
retail_ds/
    __init__.py
    io.py
    cleaning.py
    transactions.py
    validation.py
```

Then every analytical script imports from this package.

---

## 4.2 Force source types during ingestion

The most important rule:

```python
Invoice -> Utf8 / string
```

must happen **at source ingestion**, not after Excel has already coerced the field.

Use an ingestion implementation that explicitly preserves invoice identifiers.

Required invariant:

```text
C12345 remains "C12345"
```

through the entire pipeline.

---

## 4.3 Create explicit transaction classifications

Do not define a sale as merely:

```python
Quantity > 0
Price > 0
```

Create a transaction taxonomy:

```text
sale
return
cancellation
discount
postage
fee
voucher
manual_adjustment
other
```

Then define economic measures from that taxonomy.

At minimum maintain:

```text
gross_merchandise_revenue
return_value
cancellation_value
net_merchandise_revenue
```

The corrected review specifically recommends adding transaction-type classification as a further improvement.

---

## 4.4 Deduplication policy

Do not automatically delete all duplicate rows.

Investigate:

```text
exact duplicate
same invoice + product + date
same invoice + product + quantity
cross-sheet duplicate
```

Then establish an explicit policy.

Output:

```text
data_quality/
    duplicate_analysis.csv
    duplicate_policy.md
```

The audit currently reports 6,865 exact duplicate rows, so this deserves explicit investigation rather than a silent deduplication step.

---

## 4.5 Create canonical transaction outputs

Produce:

```text
canonical_transactions.parquet
canonical_transactions.csv
```

with fields such as:

```text
invoice_id
stock_code
description
quantity
invoice_date
unit_price
customer_id
country

transaction_type
is_sale
is_return
is_cancellation

gross_value
return_value
net_value

calendar_date
calendar_month
calendar_week
year
month
weekday
hour
```

This becomes the sole upstream input for all future scripts.

---

# 5. Phase 2 — Build Automated Data Quality Gates

Create:

```text
retail_ds/validation.py
tests/test_data_quality.py
```

Every pipeline run should verify:

```text
schema valid
invoice string preserved
customer IDs valid
dates valid
quantities valid
prices valid
transaction types recognized
no impossible negative revenue
no impossible probabilities
```

Add reconciliation invariants.

## Required reconciliation tests

```text
raw row count
=
sum(sheet row counts)
```

```text
gross revenue
=
sum(valid sale merchandise values)
```

```text
net revenue
=
gross revenue
-
returns
-
cancellations
```

using the project's explicit accounting definition.

Also verify:

```text
customer total
=
sum customer-level values
```

and:

```text
cohort total
=
customer-level total
```

---

# 6. Phase 3 — Establish One Canonical Customer-Month Table

This becomes the backbone of the entire project.

Create:

```text
retail_ds/customer_month.py
```

and produce:

```text
customer_month.parquet
```

with one row per:

```text
Customer ID × Calendar Month
```

for observable periods.

Include:

```text
customer_id
calendar_month
cohort_month
age_month

orders
gross_revenue
return_value
net_revenue
units

active_flag
reactivation_flag

days_since_last_purchase
months_since_last_purchase

cumulative_orders
cumulative_gross_revenue
cumulative_net_revenue

recent_orders_1m
recent_orders_3m
recent_revenue_1m
recent_revenue_3m

cadence
cadence_volatility
assortment_breadth
product_concentration
price_behavior

return_rate
```

Crucially, this table needs a consistent observation-origin convention.

---

# 7. Phase 4 — Redesign Customer 360 as an As-Of Feature Engine

Do not delete the descriptive Customer 360.

Instead split it into two concepts.

## 7.1 Descriptive Customer 360

```text
customer_360_current.parquet
```

Purpose:

> Explain current customer state.

Valid features include:

```text
lifetime_revenue
lifetime_orders
current_recency
current_product_affinity
current_return_rate
current_segment
```

The corrected review explicitly recognizes this as valid descriptive analytics.

## 7.2 Point-in-time Customer 360

Create:

```python
features_at_date(
    prediction_date
)
```

Example:

```text
features_at_date("2011-06-30")
```

returns only information available by that date.

This should drive:

* CLV training
* churn training
* next-purchase training
* reactivation training

---

# 8. Phase 5 — Fix Customer 360 Feature Semantics

For every feature add metadata:

```text
feature
description
data_type
source
point_in_time_safe
requires_prediction_date
```

Create:

```text
docs/feature_dictionary.csv
```

Example:

| Feature            | Point-in-time safe?                   | Why                         |
| ------------------ | ------------------------------------- | --------------------------- |
| lifetime_revenue   | Yes, if calculated through as-of date | historical information only |
| current_recency    | Yes, as-of date                       | requires observation date   |
| last_purchase_date | Yes, as-of date                       | requires observation date   |
| future_12m_revenue | No                                    | outcome                     |
| product_affinity   | Yes, as-of date                       | requires cutoff             |

The current audit specifically identified the need to separate descriptive lifetime features from historical prediction features.

---

# 9. Phase 6 — Rebuild Segmentation Scientifically

The current HDBSCAN implementation is useful but needs stronger diagnostics.

The corrected audit reports:

```text
12 clusters
977 noise customers
16.6% noise
0.990 silhouette
0.086 Davies-Bouldin
```

and states that stability was not tested.

---

## 9.1 Fix behavioral block balancing

Currently the review finds that `price_behavior = 23.85` strongly influences Segment 5 and that the block-balancing implementation does not control downstream profile interpretation.

Implement explicit block normalization:

```text
economic
cadence
assortment
pricing
temporal
lifecycle
returns
```

Normalize each block independently before concatenation.

Then verify:

```text
median block variance
mean block contribution
PCA contribution
```

---

## 9.2 Add feature redundancy analysis

Calculate:

```text
correlation matrix
mutual information
near-zero variance
duplicate features
```

Remove or collapse redundant features.

---

## 9.3 Analyze PCA properly

Report:

```text
PC1 explained variance
PC2 explained variance
PC3 explained variance
cumulative variance
```

Do not claim that two components represent the full behavioral space without verifying it.

Export:

```text
pca_explained_variance.csv
pca_loadings.csv
```

---

## 9.4 Stability analysis

Run at least:

```text
10 bootstrap/subsample replications
```

Compare cluster assignments using:

```text
ARI
NMI
cluster size stability
noise stability
```

Model selection should consider both:

```text
quality
+
stability
```

rather than only a single internal metric.

The corrected audit explicitly flags stability as missing.

---

## 9.5 Temporal stability

Do not frame this as ordinary supervised train/validation.

Instead test:

```text
early-period customers
vs
later-period customers
```

and compare behavioral structures.

Also test:

```text
pre-period segmentation
→ map subsequent customers
```

where appropriate.

This answers:

> Are these segments persistent behavioral structures or artifacts of one snapshot?

---

## 9.6 Create interpretable segment labels

Do not stop at:

```text
Segment_01
Segment_02
```

Generate labels from observed behavior, for example:

```text
High-Value Frequent
High-Value Selective
Broad-Assortment Repeat
Price-Sensitive Occasional
Lapsed High-Value
```

The names must be generated from evidence, not invented independently of the profiles.

---

# 10. Phase 7 — Strengthen Cohort Analysis

The current cohort calculations were independently reconciled and are numerically correct for the defined logic.

Keep the existing cohort engine, but improve interpretation.

## 10.1 Define revenue retention correctly

Use:

```text
Net Revenue Retention Index
=
Net Revenue at Age t
/
Net Revenue at Age 0
```

Document that this is a retail revenue-retention index.

Do not cap values at 100%.

The audit verifies that the 2011-05 cohort reaches 328% at age 6 while logo retention is 26%.

---

## 10.2 Make maturity explicit

Add:

```text
max_observable_age
is_mature_at_age_3
is_mature_at_age_6
is_mature_at_age_12
```

Separate:

```text
mature cohorts
immature cohorts
right-censored cohorts
```

Do not average immature and mature cohorts indiscriminately.

---

## 10.3 Add cohort confidence intervals

For logo retention:

```text
binomial confidence interval
```

For revenue:

```text
bootstrap confidence interval
```

This makes the cohort analysis much more statistically mature.

---

## 10.4 Add cohort-level acquisition economics

For each acquisition cohort report:

```text
new customers
acquisition-month revenue
month-1 retention
month-3 retention
month-6 retention
month-12 retention

cumulative net revenue/customer
observed payback proxy
reactivation rate
```

---

# 11. Phase 8 — Upgrade CLV

The current model is best described as a dynamic probabilistic discounted net-revenue forecast / CLV proxy, not classical economic CLV. The corrected review reaches the same conclusion.

Do not throw the existing model away.

Turn it into a properly framed predictive CLV system.

---

## 11.1 Rename methodology correctly

Use:

```text
Dynamic Probabilistic Discounted Net-Revenue CLV Proxy
```

or:

```text
Predictive Contribution CLV
```

if a defensible contribution margin is available.

---

## 11.2 Separate three concepts

Report:

```text
Historical Customer Value
Predictive Future Net Revenue
Predictive Contribution CLV
```

instead of treating all three as "CLV."

---

## 11.3 Add rolling backtests

Use multiple prediction origins:

```text
2010-09
2010-12
2011-03
2011-06
2011-09
```

For every origin:

```text
train
→ predict future horizon
→ observe future
→ score
```

---

## 11.4 Validate actual monetary accuracy

Calculate:

```text
MAE
RMSE
MASE if appropriate
mean error
median error
Spearman rank correlation
top-decile lift
top-20% value capture
```

The current audit specifically notes that future realized-value bias is not formally tested.

---

## 11.5 Calibrate uncertainty

For p10/p90 intervals calculate:

```text
empirical coverage
average interval width
coverage by customer activity level
coverage by segment
```

The current uncertainty bands are not empirically calibrated.

---

## 11.6 Add contribution margin

Eventually:

```text
Expected Future Contribution
=
Expected Future Net Revenue
×
Contribution Margin
```

Then:

```text
CLV
=
Discounted Expected Future Contribution
```

If margin is unavailable from Online Retail II, make the margin an explicit parameter and run scenario analysis:

```text
10%
20%
30%
40%
```

Do not pretend the dataset contains actual margin.

---

## 11.7 Vectorize Monte Carlo

Replace Python customer × simulation × month loops with NumPy arrays.

Target:

```text
full 5,878-customer run
```

within seconds/minutes rather than ~15+ minutes.

The current audit identifies Monte Carlo loops as a material bottleneck.

---

# 12. Phase 9 — Repair Churn / Survival + Next Purchase

This is mandatory.

The current error is a direct indexing mismatch:

```text
target_panel = 92,679
X_train = 65,340
X_val = 10,392
X_test = 16,947
```

The corrected audit classifies this as P0.

---

## 12.1 Single-panel modeling architecture

Construct:

```text
customer_month_model
```

first.

Then:

```text
feature construction
→ target construction
→ split assignment
→ X/y extraction
```

from that exact table.

Never recreate target rows independently afterward.

---

## 12.2 Churn definition

Be explicit:

```text
churn = no purchase in next month
```

or another clearly defined horizon.

Do not call inactivity "churn" without defining the observation window.

---

## 12.3 Survival formulation

Build:

```text
hazard(t | customer)
```

and:

$$
S(t) =
\prod_{k=1}^{t}
(1-h_k)
$$

Then estimate:

```text
3-month survival
6-month survival
12-month survival
expected remaining active months
```

---

## 12.4 Evaluate probability quality

Use:

```text
ROC-AUC
PR-AUC
Brier
log loss
calibration curve
expected calibration error
```

A ranking metric alone is insufficient.

---

## 12.5 Next purchase

Maintain:

```text
7 day
30 day
60 day
```

predictions.

Compare against baselines:

```text
population rate
recency-only
frequency-only
simple heuristic
```

The current audit specifically identifies these baselines as missing.

---

## 12.6 Temporal backtesting

Use:

```text
TRAIN
→ VALIDATION
→ TEST
```

with all splits based on time.

No random customer-month splitting.

---

# 13. Phase 10 — Reactivation Model

Create:

```text
reactivation_model.py
```

Target:

```text
customer inactive at t
→ purchases again within N days/months
```

Features:

```text
recency
historical cadence
cadence volatility
prior reactivation count
past revenue
CLV
segment
assortment
returns
seasonality
```

Evaluate with:

```text
PR-AUC
ROC-AUC
Brier
calibration
top-decile lift
```

This should feed the eventual decision engine.

---

# 14. Phase 11 — Product Intelligence

Add:

```text
product_analytics.py
```

Build product-level metrics:

```text
revenue
units
unique customers
repeat customer rate
customer penetration
return rate
price distribution
sales velocity
```

Then classify products into behavioral roles:

```text
acquisition product
repeat product
basket builder
retention product
niche high-value product
volatile product
```

This gives the customer project a genuine product/customer dimension.

---

# 15. Phase 12 — Recommendation Engine

Create:

```text
recommendation_engine.py
```

Start with:

```text
co-purchase affinity
```

Then improve to:

```text
customer-specific product affinity
```

Potential pipeline:

```text
Product co-occurrence
        ↓
Association rules / graph
        ↓
Customer history
        ↓
Segment
        ↓
Recent basket
        ↓
Candidate generation
        ↓
Recommendation ranking
```

Produce:

```text
Customer ID
recommended_product
score
reason
support
lift
```

This becomes the product-level component of the decision engine.

---

# 16. Phase 13 — Decision Engine

Only after the predictive modules are functioning.

Create:

```text
decision_engine.py
```

Inputs:

```text
CLV
churn probability
survival probability
next purchase probability
reactivation probability
segment
product affinity
confidence
```

Output:

```text
recommended_action
priority_score
decision_confidence
expected_value_proxy
action_reason
```

Possible actions:

```text
protect_value
accelerate_purchase
reactivate
cross_sell
nurture
monitor
```

---

## 16.1 Keep the causal boundary explicit

The engine should say:

```text
observational prioritization
```

until an experiment exists.

Later, with treatment/control data:

```text
decision engine
→ uplift model
→ incremental value
```

---

# 17. Phase 14 — Capacity-Constrained Optimization

Make the decision engine commercially realistic.

Example:

```text
available CRM contacts = 1,000
```

The engine should optimize:

```text
expected opportunity
subject to contact capacity
```

and action-specific limits.

Outputs:

```text
selected_customer
action
priority
expected_value
```

This makes the project meaningfully closer to retail decision science.

---

# 18. Phase 15 — Rebuild the Streamlit App Around the Science

The current app successfully renders six main areas but has pending retention/decision sections because the upstream outputs are missing.

The final app should have:

```text
Executive
Customer 360
Segmentation
Cohorts
CLV
Retention
Next Purchase
Decision Engine
Methodology
```

---

## 18.1 Executive page

Show:

```text
customers
net revenue
predicted future value
retention
reactivation
top-value customer share
```

Include an explicit:

```text
Data health
Model health
Last successful run
```

---

## 18.2 Customer 360

Search:

```text
Customer ID
```

Then display:

```text
current segment
historical value
predictive value
risk
next purchase propensity
cohort
purchase timeline
returns
product affinity
decision recommendation
```

---

## 18.3 Segmentation

Display:

```text
PCA map
segment sizes
behavior profiles
stability
segment economics
```

Clearly distinguish:

```text
cluster
noise
```

---

## 18.4 Cohorts

Show:

```text
logo retention
revenue retention
cumulative value
maturity
reactivation
```

Use visible "not observable yet" states rather than blank NaNs.

---

## 18.5 CLV

Show:

```text
predicted value
uncertainty
calibration
observed-vs-predicted
value deciles
```

Do not show a single CLV number without contextual uncertainty.

---

## 18.6 Methodology

Explain in plain language:

```text
What was measured?
How?
Why?
What assumptions?
What cannot be inferred?
```

This is one of the strongest portfolio features you can have.

---

# 19. Phase 16 — Model Governance

Every model gets a standardized model card.

Create:

```text
model_cards/
    segmentation.json
    cohort.json
    clv.json
    churn.json
    next_purchase.json
    reactivation.json
    decision_engine.json
```

Standard fields:

```text
model_name
version
prediction_target
population
prediction_origin
features
excluded_features
training_window
validation_window
test_window

metrics
calibration
uncertainty
limitations
known_failure_modes
```

The current project already has model-card-like JSON outputs, but the audit notes that metrics are not yet independently reproducible.

---

# 20. Phase 17 — Automated Testing

Create:

```text
tests/
    test_ingestion.py
    test_transactions.py
    test_customer_360.py
    test_customer_month.py
    test_segmentation.py
    test_cohorts.py
    test_clv.py
    test_churn.py
    test_next_purchase.py
    test_decision_engine.py
```

Test:

```text
schema
shape
unique customers
revenue reconciliation
cohort identity
probability ranges
no future features
no duplicate IDs
```

---

## 20.1 Point-in-time leakage tests

This is especially important.

Create a test that selects:

```text
prediction_date = 2011-06-30
```

and asserts that every feature's source transaction date is:

```text
<= 2011-06-30
```

This turns leakage prevention from documentation into code.

---

# 21. Phase 18 — Temporal Validation Framework

Create:

```text
retail_ds/backtesting.py
```

with reusable utilities:

```python
rolling_origin_split(...)
```

For example:

```text
Origin 1 → forecast next 1–3 months
Origin 2 → forecast next 1–3 months
Origin 3 → forecast next 1–3 months
Origin 4 → forecast next 1–3 months
```

Every predictive model consumes the same backtesting framework.

That creates consistency across:

```text
CLV
churn
next purchase
reactivation
forecasting
```

---

# 22. Phase 19 — Dependency and Build Management

Add:

```text
pyproject.toml
requirements.txt
```

Pin the environment.

The corrected audit currently identifies missing environment locking and version incompatibilities in the installed scikit-learn setup.

Use a documented supported environment instead of coding against whatever happens to be installed.

---

# 23. Phase 20 — Configuration

Replace hard-coded paths with:

```text
config/
    project.yaml
```

Example:

```yaml
data:
  raw:
    online_retail_ii: data_xslx/online_retail_II.xlsx

outputs:
  customer_360: customer_360_output
  segmentation: online_retail_segmentation
  cohorts: cohort_analysis_output
  clv: clv_analysis_output
  retention: retention_output
  decision: decision_engine_output

model:
  seed: 42
```

The current dashboard uses recursive file discovery and hard-coded patterns; the audit recommends configuration-driven resolution.

---

# 24. Phase 21 — Experiment Tracking

Add MLflow after the pipeline becomes scientifically sound.

Track:

```text
experiment
run
dataset version
feature version
parameters
metrics
artifacts
model
```

Experiments:

```text
segmentation
clv
churn
next_purchase
reactivation
```

This allows you to demonstrate real model-development discipline.

---

# 25. Phase 22 — Performance Engineering

Only after correctness is established.

## Priority optimizations

### First

Vectorize CLV simulation.

### Second

Vectorize next-purchase target creation.

### Third

Reduce unnecessary Polars → pandas conversions.

### Fourth

Parallelize clustering candidate evaluation.

The audit currently identifies all four as material performance opportunities.

---

# 26. Phase 23 — Portfolio Documentation

Create:

```text
README.md
docs/
    architecture.md
    methodology.md
    feature_dictionary.md
    validation.md
    model_cards.md
    limitations.md
    adr/
```

README should tell one coherent story:

```text
Business problem
        ↓
Data
        ↓
Customer representation
        ↓
Behavioral segmentation
        ↓
Lifecycle analysis
        ↓
Predictive value
        ↓
Retention
        ↓
Decisioning
        ↓
Interactive application
```

---

# 27. Phase 24 — Scientific Narrative

The project should explicitly distinguish:

## What the data proves

Examples:

```text
customer behavioral heterogeneity
cohort retention differences
historical purchase patterns
observed value concentration
predictive associations
```

## What the models estimate

Examples:

```text
future purchase probability
future expected revenue
survival probability
customer opportunity
```

## What the data cannot prove

Examples:

```text
causal treatment effect
incremental revenue
price elasticity
optimal discount
true economic margin
```

This distinction is critical for a senior-level portfolio.

---

# 28. Phase 25 — Final Validation Run

Once all corrections are complete, delete/rebuild downstream outputs.

Do not mix old and new outputs.

Run from scratch:

```bash
python data_quality.py
python customer_360.py
python customer_segmentation.py
python cohort_analysis.py
python clv_analysis.py
python churn_next_purchase.py
python reactivation_model.py
python recommendation_engine.py
python decision_engine.py
streamlit run app.py
```

Then rerun the scientific review.

---

# 29. Final Acceptance Criteria

The project should not be called portfolio-ready until all of these are true.

## Data

```text
Invoice cancellations preserved
transaction semantics explicitly classified
customer counts reconciled
revenue reconciled
duplicate policy documented
```

## Customer 360

```text
descriptive mart separated from point-in-time features
leakage tests automated
one row/customer
feature dictionary complete
```

## Segmentation

```text
clusters reproducible
noise quantified
stability measured
feature blocks balanced
PCA diagnostics correct
business profiles interpretable
```

## Cohorts

```text
retention independently reconciled
maturity explicit
revenue-index semantics documented
confidence intervals available
```

## CLV

```text
prediction origin explicit
future value validated
multiple temporal backtests
uncertainty coverage measured
terminology scientifically correct
margin assumptions explicit
```

## Churn

```text
script executes
temporal split verified
probabilities calibrated
Brier/log loss reported
baseline comparisons available
```

## Next Purchase

```text
7/30/60-day models execute
baseline comparisons available
temporal validation complete
top-decile lift reported
```

## Decision Engine

```text
implemented
capacity-aware
confidence-aware
observationally framed
```

## Application

```text
all pages operational
dashboard numbers reconcile
filters work
customer drill-down works
empty states clear
methodology understandable
```

## Engineering

```text
shared package
tests
configuration
requirements lock
reproducible build
standard model cards
```

---

# 30. Recommended Build Order

Do not implement everything simultaneously.

Use this exact sequence:

```text
PHASE 1
Canonical ingestion + transaction semantics

        ↓

PHASE 2
Data quality gates + reconciliation

        ↓

PHASE 3
Canonical customer-month table

        ↓

PHASE 4
Point-in-time Customer 360

        ↓

PHASE 5
Repair churn / next purchase

        ↓

PHASE 6
Rebuild and validate CLV

        ↓

PHASE 7
Rebuild / validate segmentation

        ↓

PHASE 8
Strengthen cohorts

        ↓

PHASE 9
Reactivation

        ↓

PHASE 10
Decision engine

        ↓

PHASE 11
Recommendation engine

        ↓

PHASE 12
Streamlit integration

        ↓

PHASE 13
Testing / MLflow / documentation

        ↓

PHASE 14
Final scientific audit
```

This sequencing deliberately puts data correctness ahead of model sophistication.

---

# 31. What Not To Do

Do not:

```text
add more models before fixing ingestion
```

Do not:

```text
optimize the segmentation solely around silhouette
```

Do not:

```text
rename every forecast "CLV"
```

Do not:

```text
call observational targeting "uplift"
```

Do not:

```text
use Customer 360 full-history features in historical training
```

Do not:

```text
randomly split customer-month observations
```

Do not:

```text
cap retail revenue retention at 100%
```

Do not:

```text
silently impute missing outputs in the dashboard
```

Do not:

```text
mix artifacts from different model/data versions
```

---

# 32. Final Portfolio Positioning

After these changes, the project should be presented as:

> **Retail Customer Intelligence Platform**
>
> An end-to-end customer analytics system built on Online Retail II that combines point-in-time behavioral feature engineering, unsupervised customer segmentation, cohort lifecycle analysis, probabilistic future-value estimation, retention and next-purchase modeling, and capacity-aware customer decisioning.

Avoid claiming:

```text
causal uplift
true price elasticity
true profit CLV
optimal marketing treatment
```

unless the project is later extended with the necessary experimental or economic data.

---

# 33. Final Deliverable Structure

The mature repository should look like:

```text
marketing_science/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
│
├── config/
│   └── project.yaml
│
├── retail_ds/
│   ├── io.py
│   ├── cleaning.py
│   ├── transactions.py
│   ├── customer_month.py
│   ├── features.py
│   ├── validation.py
│   └── backtesting.py
│
├── customer_360.py
├── customer_segmentation.py
├── cohort_analysis.py
├── clv_analysis.py
├── churn_next_purchase.py
├── reactivation_model.py
├── recommendation_engine.py
├── decision_engine.py
├── app.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_customer_360.py
│   ├── test_segmentation.py
│   ├── test_cohorts.py
│   ├── test_clv.py
│   ├── test_churn.py
│   └── test_decision_engine.py
│
├── model_cards/
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── feature_dictionary.md
│   ├── validation.md
│   └── limitations.md
│
├── outputs/
│   ├── data_quality/
│   ├── customer_360/
│   ├── segmentation/
│   ├── cohorts/
│   ├── clv/
│   ├── retention/
│   ├── recommendations/
│   └── decision_engine/
│
└── data_xslx/
    └── online_retail_II.xlsx
```

---

# 34. Definition of Done

The project is finished when a reviewer can:

1. Run the project from a clean environment.
2. Recreate the canonical transaction dataset.
3. Verify revenue and customer counts.
4. Inspect any Customer 360 feature and understand its point-in-time status.
5. reproduce segmentation results and stability diagnostics.
6. reproduce cohort metrics.
7. evaluate CLV against realized future value.
8. evaluate churn probabilities for calibration.
9. evaluate next-purchase lift against simple baselines.
10. understand exactly how the decision engine ranks customers.
11. inspect an individual customer in the Streamlit app.
12. trace dashboard values back to model outputs.
13. understand every major assumption and limitation.

At that point the project stops being "a collection of retail analytics scripts" and becomes a coherent, reproducible retail data-science system.
