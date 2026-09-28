# AGENTS.md — Marketing Science Retail Intelligence

## Project Overview
End-to-end retail customer intelligence pipeline for the **Online Retail II** dataset (UCI). Six analytical scripts + a Streamlit dashboard. Each script is independently runnable and produces machine-readable outputs for downstream consumption.

**Pipeline order (logical, not enforced):**
```
Data → customer_360.py → customer_segmentation.py → cohort_analysis.py → clv_analysis.py → churn_next_purchase.py → app.py (dashboard)
```

---

## Developer Commands

### Run a single pipeline script
```bash
python customer_360.py                    # uses defaults (see below)
python customer_segmentation.py --input ./data_xslx/online_retail_II.xlsx --output-dir ./my_segmentation
python cohort_analysis.py
python clv_analysis.py --horizon-months 24 --simulations 200
python churn_next_purchase.py --test-months 3 --validation-months 2
streamlit run app.py
```

### Default input / output locations
| Script | Default Input | Default Output Dir |
|--------|---------------|-------------------|
| All | `./data_xslx/online_retail_II.xlsx` | `./<script_name>_output/` |

Output directories are auto-created. Each script writes: CSVs, Parquet, JSON model cards, and PNG plots.

### Required dependencies (install once)
```bash
pip install polars fastexcel pandas matplotlib scikit-learn openpyxl plotly streamlit
# Optional for HDBSCAN in segmentation:
pip install hdbscan
```

---

## Architecture Notes

### Shared patterns (all scripts)
- **Polars-first** feature engineering; Pandas only at export/visualization boundary
- **CLI via argparse** — all scripts accept `--input`, `--output-dir`, `--sheet`
- **Logging** — `logging.INFO` with format `%(asctime)s | %(levelname)s | %(message)s`
- **Transaction cleaning** — identical logic duplicated in each script (normalize columns, cast types, identify cancellations/returns, compute `line_value`, `is_clean_sale`)
- **Column normalization** — case-insensitive alias mapping for `Invoice`, `StockCode`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- **Deterministic seed** — `SEED = 42` + `np.random.seed(SEED)`

### Data flow
1. **Raw** → Excel/CSV/Parquet in `data_xslx/`
2. **Clean** → Polars DataFrame with derived flags (`is_cancellation_invoice`, `is_clean_sale`, `return_value`, etc.)
3. **Customer-month panel** — dense calendar grid per customer (cohort month → last observed month), filled with zeros for inactive months
4. **Features** — rolling windows, entropy, lifecycle state, recency, cadence, returns, price behavior
5. **Models** — temporal train/validation/test splits; gradient boosting (HistGradientBoostingClassifier/Regressor)
6. **Outputs** — customer-level CSVs/Parquet + model_card.json + plots/

### Key outputs consumed by `app.py` (dashboard)
The dashboard discovers files recursively via `FILE_PATTERNS` dict. Expected filenames:
- `customer_360.csv/.parquet`
- `customer_segments.csv/.parquet`
- `segment_profiles.csv`
- `pca_coordinates.csv`
- `matrix_logo_retention.csv` / `matrix_net_revenue_retention.csv`
- `retention_decay_curve.csv`
- `cohort_scorecard.csv`
- `clv_customer_predictions.csv` / `customer_clv.csv`
- `churn_predictions.csv` / `*survival*.csv`
- `next_purchase_predictions.csv`
- `customer_decision_scores.csv`
- `action_summary.csv`
- `customer_month_events.csv`
- `customer_lifecycle_status.csv`

---

## Testing / Verification
**No formal test suite exists.** Verify by running a script end-to-end and checking:
- Output directory created with expected files
- No ERROR logs
- `model_card.json` written with metrics

```bash
# Quick smoke test (uses small defaults)
python customer_segmentation.py --min-customers 50 --stability-repeats 0
```

---

## Common Gotchas

| Issue | Resolution |
|-------|------------|
| `ModuleNotFoundError: polars` | `pip install polars fastexcel` |
| Excel read fails | Scripts fall back to `pandas/openpyxl` automatically (logged warning) |
| HDBSCAN not available | Falls back to KMeans; install `hdbscan` for density-based clustering |
| "No clean positive sales found" | Check input file has valid `Quantity > 0`, `Price > 0`, non-cancellation rows |
| Temporal split error | Dataset too small for requested `--validation-months` + `--test-months`; reduce them |
| Dashboard shows "Pending" | Run upstream scripts first to generate the required output files |

---

## File Layout (source of truth)
```
marketing_science/
├── app.py                      # Streamlit dashboard (entry point for UI)
├── customer_360.py             # Customer 360 feature mart
├── customer_segmentation.py    # Behavioral segmentation (PCA + HDBSCAN/KMeans)
├── cohort_analysis.py          # Cohort retention, revenue, reactivation
├── clv_analysis.py             # Dynamic CLV (Monte Carlo, Empirical Bayes)
├── churn_next_purchase.py      # Discrete-time survival + next-purchase models
├── data_xslx/
│   └── online_retail_II.xlsx   # Source data (not committed in some setups)
├── *_output/                   # Per-script outputs (gitignored typically)
└── AGENTS.md                   # This file
```

---

## Conventions Worth Preserving
- **No `requirements.txt` / `pyproject.toml`** — deps documented in script docstrings only
- **No CI/CD** — local execution model
- **Polars expressions over Python loops** — all feature engineering uses `pl.col().over()`, `rolling_*`, `shift()`
- **Explicit temporal splits** — validation/test months are calendar-based, not random
- **Model cards** — every modeling script writes `model_card.json` with metrics, features, limitations
- **Seed = 42** everywhere for reproducibility