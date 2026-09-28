# AGENTS.md — Marketing Science Retail Intelligence (V2)

## Project Overview
End-to-end retail customer intelligence pipeline for the **Online Retail II** dataset (UCI). Modular, config-driven, scientifically rigorous — portfolio-ready data science artifact.

**Pipeline order (logical, not enforced):**
```
Raw Excel → data_quality.py → customer_360.py → customer_segmentation.py
        → cohort_analysis.py → clv_analysis.py → churn_next_purchase.py
        → recommendation_engine.py → decision_engine.py
        → Streamlit Dashboard (streamlit_app/app.py + 9 pages)
```

---

## Developer Commands

### Run a single pipeline script (from project root)
```bash
python -m scripts.data_quality --input ./data_xslx/online_retail_II.xlsx --output-dir ./data_quality_output
python -m scripts.customer_360 --input ./data_xslx/online_retail_II.xlsx --output-dir ./customer_360_output
python -m scripts.customer_segmentation --input ./data_xslx/online_retail_II.xlsx --output-dir ./online_retail_segmentation --stability-repeats 10
python -m scripts.cohort_analysis --input ./data_xslx/online_retail_II.xlsx --output-dir ./cohort_analysis_output
python -m scripts.clv_analysis --input ./data_xslx/online_retail_II.xlsx --output-dir ./clv_analysis_output --margin-scenarios 0.1,0.2,0.3,0.4 --simulations 50
python -m scripts.churn_next_purchase --input ./data_xslx/online_retail_II.xlsx --output-dir ./churn_next_purchase_output
python -m scripts.recommendation_engine --input ./data_xslx/online_retail_II.xlsx --output-dir ./recommendation_output --top-k 10
python -m scripts.decision_engine --input ./data_xslx/online_retail_II.xlsx --output-dir ./decision_engine_output
```

### Temporal holdout evaluation (recommendations)
```bash
python -m scripts.recommendation_engine --temporal-eval --eval-cutoff 2011-06-30 --top-k 10
# Outputs: hit_rate@5, hit_rate@10, MRR@10, recall@K, baseline comparison
```

### Launch Dashboard
```bash
streamlit run streamlit_app/app.py
```

### Run Tests
```bash
# All tests (pytest-timeout incompatible with current version)
python -m pytest -p no:timeout -p no:zarr -v

# Specific test suites
python -m pytest tests/test_app.py -p no:timeout -p no:zarr -v
python -m pytest tests/test_retail_ds.py -p no:timeout -p no:zarr -v
```

### Dependencies
```bash
pip install -r requirements.txt
# Or with dev dependencies:
pip install -e ".[dev]"
```

---

## Architecture Notes

### Repo Structure
```
marketing_science/
├── config/project.yaml            # All paths & model params (single source of truth)
├── docs/                          # Documentation (contracts, reports, reviews)
├── retail_ds/                     # Shared data science library (7 modules)
│   ├── io.py, cleaning.py, transactions.py, customer_month.py
│   ├── features.py, validation.py, backtesting.py
├── scripts/                       # 9 pipeline scripts (run with `python -m scripts.<name>`)
│   ├── data_quality.py, customer_360.py, customer_segmentation.py
│   ├── cohort_analysis.py, clv_analysis.py, churn_next_purchase.py
│   ├── recommendation_engine.py, decision_engine.py
│   ├── product_analytics.py, reactivation_model.py
├── streamlit_app/                 # Modular Streamlit dashboard (V2)
│   ├── app.py (~322 lines router)
│   ├── app_config.py, app_data.py, app_formatting.py
│   ├── app_components.py, app_charts.py
│   └── pages/ (9 page modules)
├── tests/
│   ├── test_app.py (22 Streamlit integration tests)
│   └── test_retail_ds.py (35 retail_ds unit tests)
├── pyproject.toml, requirements.txt
└── README.md
```

### Key Patterns
- **Polars-first** feature engineering; Pandas only at export/visualization boundary
- **Config-driven paths** — all resolved via `config/project.yaml` → `ArtifactRegistry` (no recursive glob discovery)
- **CLI via argparse** — all scripts accept `--input`, `--output-dir`, `--sheet`
- **Logging** — `logging.INFO` with format `%(asctime)s | %(levelname)s | %(message)s`
- **Single canonical ingestion** — `retail_ds` package eliminates duplicated cleaning logic
- **Deterministic seed** — `SEED = 42` + `np.random.seed(SEED)` everywhere
- **Semantic formatters** — 11 explicit functions (`format_currency`, `format_probability`, `format_percent`, `format_ratio`, `format_count`, `format_date`, `format_month`, `format_duration_months`, `format_score`, `auto_format`, `infer_semantic_type`) replacing heuristic `pct()`

### Scientific Terminology (V2 Corrections)
| Before | After |
|--------|-------|
| CLV | "Predicted Future Net Revenue (CLV Proxy)" |
| Churn probability | "Next-month inactivity risk" |
| Survival probability | "Model-derived survival probability" |
| Decision / Prescriptive | "Observational prioritization" |
| Recommendation model | "Co-purchase ranking heuristic" |

---

## Testing / Verification

**57 tests passing**: 35 retail_ds unit tests + 22 Streamlit integration tests

Run: `python -m pytest -p no:timeout -p no:zarr`

### Test Coverage
- **retail_ds**: Ingestion, cleaning, transaction classification, customer-month panel, point-in-time features, validation, backtesting, temporal leakage, financial reconciliation, data integrity
- **Streamlit app**: Formatters, ArtifactRegistry, UI components, integration, data validation (deduplication, reason distribution, segment noise detection)

### pytest-timeout Issue
Plugin incompatible with pytest 7.4.4. Always run with `-p no:timeout -p no:zarr`.

---

## Common Gotchas

| Issue | Resolution |
|-------|------------|
| `ModuleNotFoundError: polars` | `pip install polars fastexcel` |
| Excel read fails | Falls back to `pandas/openpyxl` automatically (logged warning) |
| HDBSCAN not available | Falls back to KMeans; install `hdbscan` for density-based clustering |
| "No clean positive sales found" | Check input has valid `Quantity > 0`, `Price > 0`, non-cancellation rows |
| Temporal split error | Dataset too small for requested `--validation-months` + `--test-months`; reduce them |
| Dashboard shows "Pending" | Run upstream scripts first to generate required output files |
| Duplicate recommendations | Fixed in V2 — uses `unique(subset=["Customer ID", "recommended_product"], maintain_order=True)` after score-desc sort |
| Product metrics join explosion | Fixed in V2 — deduplicate `product_metrics` on `StockCode` before join |

---

## Config & Artifacts

### Config: `config/project.yaml`
```yaml
inputs:
  raw_data: "./data_xslx/online_retail_II.xlsx"
outputs:
  data_quality: "./data_quality_output"
  customer_360: "./customer_360_output"
  segmentation: "./online_retail_segmentation"
  cohorts: "./cohort_analysis_output"
  clv: "./clv_analysis_output"
  churn: "./churn_next_purchase_output"
  recommendations: "./recommendation_output"
  decision_engine: "./decision_engine_output"
```

### ArtifactRegistry (`streamlit_app/app_data.py`)
- `EXPECTED_ARTIFACTS` defines 10 module schemas with required columns, min rows, supporting files
- Cache keys include file `mtime` + `size` for automatic invalidation
- Cross-run consistency detection via `run_id`, `data_version`, `code_version`

### Output Contract: `docs/app_data_contract.md`
All 10 module schemas documented with required columns and expected files.

---

## Scientific Validation Results (V2)

### Recommendation Engine (Temporal Holdout, cutoff 2011-06-30)
| Metric | Value | Baseline (Popularity) |
|--------|-------|----------------------|
| Hit Rate @5 | 13.3% | 62.1% |
| Hit Rate @10 | 19.8% | 62.1% |
| Recall @5 | 0.84% | — |
| Recall @10 | 1.41% | — |
| MRR @10 | 0.406 | — |
| Deduplication | 0 duplicates | — |

### Segmentation Stability
- 12 clusters + noise (83.4% coverage, 16.6% noise)
- Bootstrap ARI: 0.97–0.99 (10 repeats)
- Feature blocks: 7 balanced before PCA

### CLV
- Methodology: Dynamic Probabilistic Discounted Net-Revenue CLV Proxy
- Horizon: 24 months, 200 simulations
- Margin scenarios: 10%/20%/30%/40%
- Validation: Temporal split, Platt calibration, p10/p50/p90 intervals

### Churn / Next Purchase
- Target: `purchase_next_month` (discrete-time hazard)
- Temporal split: 64,915 / 10,339 / 16,866 (train/val/test)
- Metrics: ROC-AUC, PR-AUC, Brier, Log Loss, calibration curves
- Next-purchase: 7/30/60-day targets via asof join

---

## Remaining Scientific Risks (Documented)
- **NRR > 100%**: Retail revenue index semantics need clearer documentation
- **CLV uncertainty**: Empirical coverage of p10/p90 not verified
- **Reactivation model**: Framework exists (`is_reactivation` flag), dedicated model not implemented
- **Margin scenarios**: Assume constant rate; no COGS data in Online Retail II
- **Recommendation recall**: Low recall expected for sparse retail data (baseline popularity outperforms on hit rate)

---

## Conventions Worth Preserving
- **Polars expressions over Python loops** — all feature engineering uses `pl.col().over()`, `rolling_*`, `shift()`
- **Explicit temporal splits** — validation/test months are calendar-based, not random
- **Model cards** — every modeling script writes `model_card.json` with metrics, features, limitations
- **Seed = 42** everywhere for reproducibility
- **No CI/CD** — local execution model
- **pyproject.toml with pinned deps** — single source of truth for dependencies