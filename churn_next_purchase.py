#!/usr/bin/env python3
"""
Retail Customer Retention + Next-Purchase Intelligence
======================================================

End-to-end modeling pipeline for Online Retail II.

The script deliberately goes beyond a generic "churn model":

1. Loads and validates Online Retail II.
2. Creates a customer-month behavioral panel in Polars.
3. Optionally enriches customers with outputs from:
      - customer_360.py
      - customer_segmentation.py
      - cohort_analysis.py
      - clv_analysis.py
4. Trains a discrete-time survival / hazard model.
5. Converts predicted hazards into 3/6/12-month survival probabilities.
6. Trains separate calibrated next-purchase models for 7/30/60 days.
7. Uses strict temporal train / validation / test splits.
8. Reports ROC-AUC, PR-AUC, Brier score, log loss, calibration,
   lift by decile, and temporal drift diagnostics.
9. Produces customer-level retention and next-purchase predictions.
10. Produces management-ready visuals.

Default data location:
    /home/lptop/Documents/coding/marketing_science/data_xslx/online_retail_II.xlsx

Default output location:
    /home/lptop/Documents/coding/marketing_science/
    churn_next_purchase_output

Recommended environment:
    pip install polars fastexcel pandas scikit-learn matplotlib openpyxl

Optional:
    pip install xgboost

The default implementation uses scikit-learn HistGradientBoostingClassifier
and calibration through a held-out temporal validation set, so it does not
require XGBoost.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import QuantileTransformer


SEED = 42
np.random.seed(SEED)


# ============================================================================
# LOGGING
# ============================================================================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("retention_next_purchase")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
        )
        logger.addHandler(handler)

    return logger


LOGGER = setup_logger()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Customer churn/survival and next-purchase modeling "
            "for Online Retail II."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "/home/lptop/Documents/coding/marketing_science/"
            "data_xslx/online_retail_II.xlsx"
        ),
        help="Raw Online Retail II .xlsx/.csv/.parquet file.",
    )

    parser.add_argument(
        "--customer-360",
        default=None,
        help="Optional customer_360 output CSV/Parquet to enrich predictions.",
    )

    parser.add_argument(
        "--segmentation",
        default=None,
        help="Optional customer_segments.csv from customer_segmentation.py.",
    )

    parser.add_argument(
        "--clv",
        default=None,
        help="Optional customer-level CLV CSV/Parquet output.",
    )

    parser.add_argument(
        "--cohort",
        default=None,
        help="Optional customer_acquisition_cohorts.csv from cohort_analysis.py.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/home/lptop/Documents/coding/marketing_science/"
            "churn_next_purchase_output"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name.",
    )

    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Final calendar months reserved for test.",
    )

    parser.add_argument(
        "--validation-months",
        type=int,
        default=2,
        help="Months before test reserved for validation.",
    )

    parser.add_argument(
        "--horizon-months",
        type=int,
        default=12,
        help="Maximum monthly survival horizon written to outputs.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2,
        help="Number of temporal bootstrap evaluations for stability diagnostics.",
    )

    parser.add_argument(
        "--prediction-sample",
        type=int,
        default=15000,
        help="Maximum points used in scatter/diagnostic visuals.",
    )

    return parser.parse_args()


# ============================================================================
# IO
# ============================================================================


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    aliases = {
        "invoiceno": "Invoice",
        "invoice": "Invoice",
        "invoice no": "Invoice",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "invoice date": "InvoiceDate",
        "unitprice": "Price",
        "price": "Price",
        "customerid": "Customer ID",
        "customer id": "Customer ID",
        "country": "Country",
    }

    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        rename_map[column] = aliases.get(normalized, column)

    df = df.rename(rename_map)

    required = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {df.columns}"
        )

    return df


def load_table(path: Path, sheet: str | None = None) -> pl.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)

    if suffix == ".csv":
        return pl.read_csv(
            path,
            infer_schema_length=20000,
            try_parse_dates=True,
            ignore_errors=True,
        )

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            if sheet:
                return pl.read_excel(path, sheet_name=sheet)

            loaded = pl.read_excel(path, sheet_id=0)
            if isinstance(loaded, dict):
                frames = [
                    x for x in loaded.values()
                    if isinstance(x, pl.DataFrame) and x.height > 0
                ]
                if not frames:
                    raise ValueError("Workbook contains no non-empty sheets.")
                return pl.concat(frames, how="diagonal_relaxed")
            return loaded
        except Exception as exc:
            LOGGER.warning(
                "Polars Excel reader failed (%s); using pandas/openpyxl.",
                exc,
            )
            loaded = pd.read_excel(
                path,
                sheet_name=sheet if sheet else None,
            )
            if isinstance(loaded, dict):
                loaded = pd.concat(loaded.values(), ignore_index=True)
            return pl.from_pandas(loaded)

    raise ValueError(f"Unsupported file format: {suffix}")


def load_optional_table(path_string: str | None) -> Optional[pl.DataFrame]:
    if not path_string:
        return None
    path = Path(path_string).expanduser().resolve()
    if not path.exists():
        LOGGER.warning("Optional enrichment file not found: %s", path)
        return None
    try:
        return normalize_optional_columns(load_table(path))
    except Exception as exc:
        LOGGER.warning("Could not load enrichment %s: %s", path, exc)
        return None


def normalize_optional_columns(df: pl.DataFrame) -> pl.DataFrame:
    aliases = {
        "customer_id": "Customer ID",
        "customerid": "Customer ID",
        "customer id": "Customer ID",
    }

    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("-", "_")
        rename_map[column] = aliases.get(normalized, column)

    return df.rename(rename_map)


# ============================================================================
# RAW TRANSACTIONS
# ============================================================================


def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(
        [
            pl.col("Invoice").cast(pl.Utf8).str.strip_chars(),
            pl.col("StockCode").cast(pl.Utf8).str.strip_chars(),
            pl.col("Quantity").cast(pl.Float64, strict=False),
            pl.col("Price").cast(pl.Float64, strict=False),
            pl.col("Customer ID")
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False),
        ]
    )

    if out.schema["InvoiceDate"] not in {pl.Date, pl.Datetime}:
        out = out.with_columns(
            pl.col("InvoiceDate")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False, exact=False)
        )
    else:
        out = out.with_columns(
            pl.col("InvoiceDate").cast(pl.Datetime, strict=False)
        )

    out = out.filter(
        pl.col("Invoice").is_not_null()
        & pl.col("Customer ID").is_not_null()
        & pl.col("InvoiceDate").is_not_null()
        & pl.col("Quantity").is_not_null()
        & pl.col("Price").is_not_null()
    )

    out = out.with_columns(
        [
            pl.col("Invoice")
            .str.to_uppercase()
            .str.starts_with("C")
            .alias("is_cancellation_invoice"),
            (pl.col("Quantity") > 0).alias("is_positive_quantity"),
            (pl.col("Quantity") < 0).alias("is_return_quantity"),
            (pl.col("Price") > 0).alias("is_positive_price"),
        ]
    )

    out = out.with_columns(
        [
            (
                pl.col("is_positive_quantity")
                & (~pl.col("is_cancellation_invoice"))
                & pl.col("is_positive_price")
            ).alias("is_clean_sale"),
            (
                pl.col("Quantity") * pl.col("Price")
            ).alias("line_value"),
        ]
    )

    return out


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def build_customer_month_panel(
    tx: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pd.Timestamp]:
    """
    Build a complete customer-month panel from first purchase month through the
    last observed sale month. Feature values at month t only use information up
    to and including t, while targets are defined from t+1 onward.
    """

    sales = tx.filter(pl.col("is_clean_sale"))

    if sales.height == 0:
        raise ValueError("No positive sales found.")

    sales = sales.with_columns(
        [
            pl.col("InvoiceDate").dt.truncate("1mo").alias("calendar_month"),
            (pl.col("Quantity") * pl.col("Price")).alias("line_revenue"),
        ]
    )

    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("calendar_month").min().alias("calendar_month"),
                pl.col("line_revenue").sum().alias("invoice_revenue"),
                pl.col("Quantity").sum().alias("invoice_units"),
                pl.col("StockCode").n_unique().alias("invoice_products"),
                pl.col("Price").median().alias("invoice_median_price"),
            ]
        )
        .sort(["Customer ID", "invoice_date"])
    )

    first_purchase = (
        invoice.group_by("Customer ID")
        .agg(pl.col("invoice_date").min().alias("first_purchase_date"))
        .with_columns(
            pl.col("first_purchase_date")
            .dt.truncate("1mo")
            .alias("cohort_month")
        )
    )

    customer_month_observed = (
        invoice.group_by(["Customer ID", "calendar_month"])
        .agg(
            [
                pl.len().alias("orders_month"),
                pl.col("invoice_revenue").sum().alias("revenue_month"),
                pl.col("invoice_units").sum().alias("units_month"),
                pl.col("invoice_products").sum().alias("product_lines_month"),
                pl.col("invoice_revenue").mean().alias("aov_month"),
                pl.col("invoice_revenue").median().alias("median_order_value_month"),
                pl.col("invoice_median_price").median().alias("median_price_month"),
                pl.col("invoice_date").max().alias("last_invoice_date_month"),
            ]
        )
    )

    min_month = sales.select(pl.col("calendar_month").min()).item()
    max_month = sales.select(pl.col("calendar_month").max()).item()

    customers = first_purchase.select("Customer ID", "cohort_month")

    # Build a complete calendar once. It is small relative to transactions.
    calendar = pl.DataFrame(
        {
            "calendar_month": pd.date_range(
                pd.Timestamp(min_month),
                pd.Timestamp(max_month),
                freq="MS",
            )
        }
    ).with_columns(
        pl.col("calendar_month").cast(pl.Datetime)
    )

    panel = (
        customers.join(calendar, how="cross")
        .filter(pl.col("calendar_month") >= pl.col("cohort_month"))
        .join(
            customer_month_observed,
            on=["Customer ID", "calendar_month"],
            how="left",
        )
        .with_columns(
            [
                pl.col("orders_month").fill_null(0),
                pl.col("revenue_month").fill_null(0.0),
                pl.col("units_month").fill_null(0.0),
                pl.col("product_lines_month").fill_null(0.0),
                pl.col("aov_month").fill_null(0.0),
                pl.col("median_order_value_month").fill_null(0.0),
                pl.col("median_price_month").fill_null(0.0),
                pl.col("last_invoice_date_month"),
                (pl.col("orders_month") > 0).cast(pl.Int8).alias("active_month"),
            ]
        )
        .sort(["Customer ID", "calendar_month"])
    )

    panel = panel.with_columns(
        (
            (pl.col("calendar_month").dt.year() - pl.col("cohort_month").dt.year()) * 12
            + (pl.col("calendar_month").dt.month() - pl.col("cohort_month").dt.month())
        ).alias("age_month")
    )

    # ------------------------------------------------------------------------
    # Rolling behavioral state
    # ------------------------------------------------------------------------
    # Lagged/rolling features are computed from t and earlier. Target is a
    # future event, so no target leakage enters the feature matrix.
    # ------------------------------------------------------------------------

    panel = panel.with_columns(
        [
            pl.col("active_month")
            .shift(1)
            .over("Customer ID")
            .fill_null(0)
            .alias("active_prev_month"),

            pl.col("active_month")
            .rolling_sum(window_size=3, min_samples=1)
            .over("Customer ID")
            .alias("active_months_3m"),

            pl.col("active_month")
            .rolling_sum(window_size=6, min_samples=1)
            .over("Customer ID")
            .alias("active_months_6m"),

            pl.col("orders_month")
            .rolling_sum(window_size=3, min_samples=1)
            .over("Customer ID")
            .alias("orders_3m"),

            pl.col("orders_month")
            .rolling_sum(window_size=6, min_samples=1)
            .over("Customer ID")
            .alias("orders_6m"),

            pl.col("revenue_month")
            .rolling_sum(window_size=3, min_samples=1)
            .over("Customer ID")
            .alias("revenue_3m"),

            pl.col("revenue_month")
            .rolling_sum(window_size=6, min_samples=1)
            .over("Customer ID")
            .alias("revenue_6m"),

            pl.col("units_month")
            .rolling_sum(window_size=3, min_samples=1)
            .over("Customer ID")
            .alias("units_3m"),

            pl.col("units_month")
            .rolling_sum(window_size=6, min_samples=1)
            .over("Customer ID")
            .alias("units_6m"),
        ]
    )

    panel = panel.with_columns(
        [
            pl.col("revenue_month")
            .shift(1)
            .over("Customer ID")
            .fill_null(0.0)
            .alias("revenue_prev_month"),

            pl.col("orders_month")
            .shift(1)
            .over("Customer ID")
            .fill_null(0)
            .alias("orders_prev_month"),

            pl.col("active_month")
            .shift(1)
            .over("Customer ID")
            .fill_null(0)
            .alias("active_prev_month_2"),
        ]
    )

    # Days since last purchase as observed at t.
    panel = panel.with_columns(
        pl.col("last_invoice_date_month")
        .forward_fill()
        .over("Customer ID")
        .alias("last_purchase_date_asof_month")
    )

    panel = panel.with_columns(
        [
            (
                pl.col("calendar_month")
                + pl.duration(days=32)
            )
            .dt.truncate("1mo")
            .alias("month_end_proxy"),

            (
                pl.col("age_month")
                + 1
            ).alias("next_age_month"),
        ]
    )

    panel = panel.with_columns(
        [
            (
                (
                    pl.col("calendar_month")
                    - pl.col("last_purchase_date_asof_month")
                ).dt.total_days()
            )
            .fill_null(9999.0)
            .alias("days_since_purchase_month_start"),

            (
                pl.col("revenue_month")
                / pl.col("orders_month").clip(lower_bound=1)
            ).alias("aov_safe"),

            (
                pl.col("orders_3m")
                / 3.0
            ).alias("monthly_order_rate_3m"),

            (
                pl.col("orders_6m")
                / 6.0
            ).alias("monthly_order_rate_6m"),

            (
                pl.col("revenue_3m")
                / 3.0
            ).alias("monthly_revenue_rate_3m"),

            (
                pl.col("revenue_6m")
                / 6.0
            ).alias("monthly_revenue_rate_6m"),

            (
                pl.col("active_months_3m")
                / 3.0
            ).alias("activity_ratio_3m"),

            (
                pl.col("active_months_6m")
                / 6.0
            ).alias("activity_ratio_6m"),
        ]
    )

    # Consecutive inactivity length.
    panel = panel.with_columns(
        pl.when(pl.col("active_month") == 1)
        .then(0)
        .otherwise(1)
        .alias("inactive_indicator")
    )

    # A rolling maximum/minimum based method is more stable in Polars than
    # hand-written Python loops for this dataset.
    panel = panel.with_columns(
        pl.col("inactive_indicator")
        .cum_sum()
        .over("Customer ID")
        .alias("inactive_run_counter")
    )

    panel = panel.with_columns(
        pl.when(pl.col("active_month") == 1)
        .then(pl.col("inactive_run_counter"))
        .forward_fill()
        .over("Customer ID")
        .alias("last_active_counter")
    )

    panel = panel.with_columns(
        (
            pl.col("inactive_run_counter")
            - pl.col("last_active_counter").fill_null(0)
        ).alias("consecutive_inactive_months")
    )

    # ------------------------------------------------------------------------
    # Forward target construction
    # ------------------------------------------------------------------------

    panel = panel.with_columns(
        [
            pl.col("active_month")
            .shift(-1)
            .over("Customer ID")
            .alias("purchase_next_month"),

            pl.col("orders_month")
            .shift(-1)
            .over("Customer ID")
            .fill_null(0)
            .alias("orders_next_month"),

            pl.col("revenue_month")
            .shift(-1)
            .over("Customer ID")
            .fill_null(0.0)
            .alias("revenue_next_month"),
        ]
    )

    # We only model months for which the next calendar month is observable.
    last_month = pd.Timestamp(max_month)
    panel_model = panel.filter(
        pl.col("calendar_month") < last_month
    ).with_columns(
        pl.col("purchase_next_month")
        .fill_null(0)
        .cast(pl.Int8)
        .alias("purchase_next_month")
    )

    panel_model = panel_model.with_columns(
        [
            (
                pl.col("purchase_next_month") == 1
            ).cast(pl.Int8).alias("survival_event_next_month"),

            (
                pl.col("purchase_next_month") == 0
            ).cast(pl.Int8).alias("failure_next_month"),
        ]
    )

    return panel, first_purchase, pd.Timestamp(last_month)


# ============================================================================
# CUSTOMER-LEVEL SNAPSHOT
# ============================================================================


def build_current_customer_snapshot(
    panel: pl.DataFrame,
    latest_observation_month: pd.Timestamp,
) -> pl.DataFrame:
    latest = panel.filter(
        pl.col("calendar_month") == pl.lit(latest_observation_month.to_pydatetime())
    )

    if latest.height == 0:
        raise ValueError("No customer snapshots exist at the latest month.")

    return latest


# ============================================================================
# ENRICHMENT
# ============================================================================


def join_optional_customer_data(
    snapshot: pl.DataFrame,
    customer_360: Optional[pl.DataFrame],
    segmentation: Optional[pl.DataFrame],
    clv: Optional[pl.DataFrame],
    cohort: Optional[pl.DataFrame],
) -> pl.DataFrame:
    out = snapshot

    sources = [
        ("customer_360", customer_360),
        ("segmentation", segmentation),
        ("clv", clv),
        ("cohort", cohort),
    ]

    for source_name, table in sources:
        if table is None or "Customer ID" not in table.columns:
            continue

        table = table.unique(subset=["Customer ID"], keep="last")
        columns = [
            c for c in table.columns
            if c != "Customer ID"
            and c not in out.columns
        ]

        if not columns:
            continue

        LOGGER.info(
            "Enriching current customer snapshot with %s (%d fields).",
            source_name,
            len(columns),
        )

        out = out.join(
            table.select(["Customer ID"] + columns),
            on="Customer ID",
            how="left",
        )

    return out


# ============================================================================
# FEATURES
# ============================================================================


BASE_FEATURES = [
    "age_month",
    "active_month",
    "active_prev_month",
    "active_months_3m",
    "active_months_6m",
    "orders_month",
    "orders_prev_month",
    "orders_3m",
    "orders_6m",
    "revenue_month",
    "revenue_prev_month",
    "revenue_3m",
    "revenue_6m",
    "units_month",
    "units_3m",
    "units_6m",
    "aov_month",
    "aov_safe",
    "median_order_value_month",
    "median_price_month",
    "monthly_order_rate_3m",
    "monthly_order_rate_6m",
    "monthly_revenue_rate_3m",
    "monthly_revenue_rate_6m",
    "activity_ratio_3m",
    "activity_ratio_6m",
    "consecutive_inactive_months",
    "days_since_purchase_month_start",
]


DERIVED_OPTIONAL_NAMES = [
    "segment",
    "segment_confidence",
    "clv",
    "clv_point_estimate",
    "clv_lower",
    "clv_upper",
    "customer_clv",
    "lifetime_value",
    "cohort_month",
    "tenure_days",
    "revenue",
    "invoice_count",
    "orders",
    "return_value",
    "value_return_rate",
    "purchase_frequency",
    "days_since_last_purchase",
]


def select_model_features(
    df: pl.DataFrame,
) -> List[str]:
    candidates = BASE_FEATURES + DERIVED_OPTIONAL_NAMES

    features = []
    for column in candidates:
        if column in df.columns and column != "Customer ID":
            dtype = df.schema[column]
            if dtype in {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            }:
                features.append(column)

    # Remove features with too much missingness or no information.
    valid = []
    for column in features:
        null_fraction = df.select(
            pl.col(column).is_null().mean()
        ).item()

        if null_fraction > 0.50:
            continue

        variance = df.select(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .var()
        ).item()

        if variance is None or not np.isfinite(float(variance)) or float(variance) <= 1e-12:
            continue

        valid.append(column)

    return valid


def prepare_matrix(
    df: pl.DataFrame,
    features: List[str],
) -> Tuple[np.ndarray, QuantileTransformer]:
    X = (
        df.select(features)
        .to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    # Percentile clipping controls the effect of very large baskets/customers.
    low = np.nanpercentile(X, 1.0, axis=0)
    high = np.nanpercentile(X, 99.0, axis=0)
    X = np.clip(X, low, high)

    transformer = QuantileTransformer(
        n_quantiles=min(1000, max(50, X.shape[0] // 20)),
        output_distribution="normal",
        subsample=10000,
        random_state=SEED,
    )

    X_t = transformer.fit_transform(X)
    return X_t, transformer


# ============================================================================
# TEMPORAL SPLITS
# ============================================================================


@dataclass
class TemporalSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_temporal_split(
    panel_model: pl.DataFrame,
    validation_months: int,
    test_months: int,
) -> TemporalSplit:
    months = sorted(
        pd.Timestamp(x)
        for x in panel_model
        .select("calendar_month")
        .unique()
        .to_series()
        .to_list()
    )

    if len(months) < validation_months + test_months + 4:
        raise ValueError(
            "Not enough monthly observations for requested temporal split."
        )

    test_month_values = months[-test_months:]
    validation_month_values = months[
        -(test_months + validation_months):-test_months
    ]
    train_month_values = months[:-(test_months + validation_months)]

    return TemporalSplit(
        train_end=train_month_values[-1],
        validation_start=validation_month_values[0],
        validation_end=validation_month_values[-1],
        test_start=test_month_values[0],
        test_end=test_month_values[-1],
    )


# ============================================================================
# METRICS
# ============================================================================


def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def lift_table(
    y: np.ndarray,
    p: np.ndarray,
    deciles: int = 10,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "target": y,
            "probability": p,
        }
    ).sort_values("probability", ascending=False)

    n = len(data)
    groups = np.ceil(np.arange(1, n + 1) / max(1, n / deciles)).astype(int)
    groups = np.clip(groups, 1, deciles)
    data["decile"] = groups

    overall = data["target"].mean()

    out = (
        data.groupby("decile", as_index=False)
        .agg(
            observations=("target", "size"),
            positives=("target", "sum"),
            response_rate=("target", "mean"),
            mean_predicted_probability=("probability", "mean"),
        )
        .sort_values("decile")
    )

    out["lift_vs_population"] = np.where(
        overall > 0,
        out["response_rate"] / overall,
        np.nan,
    )

    out["cumulative_positive_capture"] = (
        out["positives"].cumsum()
        / max(1, out["positives"].sum())
    )

    return out


def evaluate_predictions(
    y: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float]:
    return {
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "brier": safe_brier(y, p),
        "log_loss": safe_logloss(y, p),
        "base_rate": float(y.mean()),
    }


# ============================================================================
# MODELING
# ============================================================================


def make_model(
    learning_rate: float = 0.06,
    max_iter: int = 300,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
    min_samples_leaf: int = 40,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
        early_stopping=False,
    )


def tune_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[HistGradientBoostingClassifier, pd.DataFrame]:
    candidates = list(
        ParameterGrid(
            {
                "learning_rate": [0.06],
                "max_leaf_nodes": [31],
                "min_samples_leaf": [40],
                "l2_regularization": [1.0],
            }
        )
    )

    records = []
    best_model = None
    best_score = -np.inf

    for i, params in enumerate(candidates, start=1):
        model = make_model(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
        )

        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(y_val, p_val)
        # PR-AUC gets the largest weight because future purchase is usually
        # class-imbalanced; Brier is included to reward useful probabilities.
        score = (
            0.55 * (metrics["pr_auc"] if np.isfinite(metrics["pr_auc"]) else 0.0)
            + 0.25 * (metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else 0.0)
            + 0.20 * (1.0 - metrics["brier"])
        )

        records.append(
            {
                **params,
                **metrics,
                "selection_score": score,
            }
        )

        if score > best_score:
            best_score = score
            best_model = model

        LOGGER.info(
            "Tune %d/%d | PR-AUC=%.4f | ROC-AUC=%.4f | Brier=%.5f",
            i,
            len(candidates),
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["brier"],
        )

    if best_model is None:
        raise RuntimeError("No model candidate completed.")

    return best_model, pd.DataFrame(records).sort_values(
        "selection_score", ascending=False
    )


def fit_calibrated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[
    HistGradientBoostingClassifier,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
]:
    model, tuning = tune_on_validation(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    p_val_raw = model.predict_proba(X_val)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------------
    # Platt calibration using the untouched validation period.
    # ------------------------------------------------------------------------
    # Rather than using random CV, calibrate on a later temporal period.
    # This preserves the direction of time and limits leakage.
    # ------------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )

    calibrator.fit(
        p_val_raw.reshape(-1, 1),
        y_val,
    )

    p_val = calibrator.predict_proba(
        p_val_raw.reshape(-1, 1)
    )[:, 1]

    p_test = calibrator.predict_proba(
        p_test_raw.reshape(-1, 1)
    )[:, 1]

    metrics = evaluate_predictions(
        y_test,
        p_test,
    )

    # ------------------------------------------------------------------------
    # Permutation feature importance on validation data.
    # ------------------------------------------------------------------------
    importance = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=3,
        scoring="average_precision",
        random_state=SEED,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    return (
        model,
        p_val,
        p_test,
        p_test_raw,
        tuning,
        importance_df,
        metrics,
    )


# ============================================================================
# SURVIVAL FORECASTING
# ============================================================================


def build_survival_curve(
    hazard: float,
    horizon: int,
) -> Dict[str, float]:
    hazard = float(np.clip(hazard, 1e-6, 1 - 1e-6))
    output = {}
    survival = 1.0

    for month in range(1, horizon + 1):
        survival *= 1.0 - hazard
        output[f"survival_month_{month}"] = survival

    return output


def forecast_customer_survival(
    current_X: np.ndarray,
    current_frame: pl.DataFrame,
    survival_model: HistGradientBoostingClassifier,
    horizon: int,
) -> pd.DataFrame:
    """
    Approximate future survival recursively from the customer-specific hazard.

    The base hazard is modulated by current behavioral state and decays with
    age through the trained discrete-time hazard model. Because we do not have
    future covariates, the current-state hazard is carried forward using a
    modest age-decay prior. This is intentionally conservative rather than
    pretending future purchases are known.
    """
    base_hazard = survival_model.predict_proba(current_X)[:, 1]

    # A light structural prior: as a customer gets further from acquisition,
    # hazard predictions often become less stable due to sparse evidence. We
    # therefore use a bounded shrinkage toward each customer's base hazard.
    age = (
        current_frame
        .select("age_month")
        .to_pandas()["age_month"]
        .to_numpy(dtype=float)
    )

    output = {
        "Customer ID": (
            current_frame
            .select("Customer ID")
            .to_pandas()["Customer ID"]
            .to_numpy()
        )
    }

    survival = np.ones_like(base_hazard, dtype=float)

    for month in range(1, horizon + 1):
        # Shrink long-horizon hazards moderately toward the customer population
        # average rather than letting a one-step model explode indefinitely.
        population_hazard = float(np.mean(base_hazard))
        shrink = 1.0 - np.exp(-month / 8.0)
        hazard_t = (
            (1.0 - 0.25 * shrink) * base_hazard
            + (0.25 * shrink) * population_hazard
        )

        # Age-aware damping, bounded so it remains a diagnostic rather than a
        # fake structural law.
        age_adjustment = 1.0 / np.sqrt(
            1.0 + np.maximum(age, 0.0) / 24.0
        )
        hazard_t = np.clip(
            0.02
            + (hazard_t - 0.02) * age_adjustment,
            1e-4,
            0.98,
        )

        survival *= 1.0 - hazard_t
        output[
            f"survival_month_{month}"
        ] = survival.copy()

    return pd.DataFrame(output)


# ============================================================================
# VISUALS
# ============================================================================


def save_plots(
    output_dir: Path,
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    lift_test: pd.DataFrame,
    importance: pd.DataFrame,
    survival_curve: pd.DataFrame,
    next_purchase_30: pd.DataFrame,
    prediction_sample: int,
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------------
    frac_pos_val, mean_pred_val = calibration_curve(
        y_val,
        p_val,
        n_bins=10,
        strategy="quantile",
    )

    frac_pos_test, mean_pred_test = calibration_curve(
        y_test,
        p_test,
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )
    ax.plot(
        mean_pred_val,
        frac_pos_val,
        marker="o",
        linewidth=2,
        label="Validation",
    )
    ax.plot(
        mean_pred_test,
        frac_pos_test,
        marker="o",
        linewidth=2,
        label="Test",
    )
    ax.set_title("Next-Purchase Probability Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        plots / "01_next_purchase_calibration.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Lift
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        lift_test["decile"],
        lift_test["cumulative_positive_capture"],
        marker="o",
        linewidth=2,
    )
    ax.set_title("Cumulative Positive Capture by Probability Decile")
    ax.set_xlabel("Probability decile (1 = highest score)")
    ax.set_ylabel("Share of future purchasers captured")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(
        plots / "02_cumulative_capture.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------------
    top_imp = importance.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_imp["feature"],
        top_imp["importance_mean"],
        xerr=top_imp["importance_std"],
    )
    ax.set_title("Next-Purchase Model — Permutation Importance")
    ax.set_xlabel("Average precision decrease after permutation")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(
        plots / "03_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Survival curve
    # ------------------------------------------------------------------------
    survival_cols = [
        c for c in survival_curve.columns
        if c.startswith("survival_month_")
    ]
    survival_values = survival_curve[survival_cols].mean(axis=0).to_numpy()
    months = np.arange(1, len(survival_values) + 1)

    q10 = survival_curve[survival_cols].quantile(0.10, axis=0).to_numpy()
    q90 = survival_curve[survival_cols].quantile(0.90, axis=0).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        months,
        survival_values,
        marker="o",
        linewidth=2,
        label="Mean predicted survival",
    )
    ax.fill_between(
        months,
        q10,
        q90,
        alpha=0.20,
        label="Customer prediction band (10th–90th pct.)",
    )
    ax.set_title("Predicted Customer Survival Curve")
    ax.set_xlabel("Months after forecast origin")
    ax.set_ylabel("Probability customer remains commercially active")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        plots / "04_survival_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Risk vs opportunity
    # ------------------------------------------------------------------------
    if not next_purchase_30.empty:
        n = len(next_purchase_30)
        rng = np.random.default_rng(SEED)
        idx = np.arange(n)
        if n > prediction_sample:
            idx = rng.choice(idx, size=prediction_sample, replace=False)

        plot_df = next_purchase_30.iloc[idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            np.log1p(plot_df["clv_or_revenue_proxy"]),
            plot_df["next_purchase_30d_probability"],
            s=16,
            alpha=0.35,
        )
        ax.set_title("Customer Economic Value vs Next-Purchase Probability")
        ax.set_xlabel("log(1 + CLV / observed revenue proxy)")
        ax.set_ylabel("30-day next-purchase probability")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(
            plots / "05_value_vs_next_purchase_probability.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------------
    # Score distribution
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        next_purchase_30[
            "next_purchase_30d_probability"
        ],
        bins=30,
        alpha=0.80,
    )
    ax.set_title("Distribution of 30-Day Next-Purchase Propensity")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(
        plots / "06_next_purchase_probability_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================================
# MODEL CARD
# ============================================================================


def create_model_card(
    output_dir: Path,
    feature_names: List[str],
    split: TemporalSplit,
    metrics_by_horizon: Dict,
    survival_metadata: Dict,
) -> None:
    model_card = {
        "project": "Retail Customer Retention + Next Purchase",
        "modeling_framework": "Discrete-time hazard + calibrated gradient boosting",
        "temporal_split": {
            "train_end": str(split.train_end),
            "validation_start": str(split.validation_start),
            "validation_end": str(split.validation_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
        },
        "features": feature_names,
        "next_purchase_metrics": metrics_by_horizon,
        "survival": survival_metadata,
        "known_limitations": [
            "Future covariates are unavailable, so multi-month survival is a conditional scenario forecast rather than an oracle forecast.",
            "Online Retail II is observational transaction data and does not identify treatment effects from marketing interventions.",
            "Calendar months with no purchase are treated as inactivity, not necessarily absolute churn, because the underlying business relationship is unobserved.",
            "Returns/cancellations are retained as behavioral signals but positive-purchase targets are based on clean positive sales.",
        ],
    }

    with open(
        output_dir / "model_card.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(model_card, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Load + clean
    # ------------------------------------------------------------------------
    LOGGER.info("Loading raw transaction data...")
    raw = normalize_columns(
        load_table(input_path, args.sheet)
    )
    tx = clean_transactions(raw)
    del raw

    LOGGER.info(
        "Clean transaction rows: %s",
        f"{tx.height:,}",
    )

    # ------------------------------------------------------------------------
    # 2. Customer-month panel
    # ------------------------------------------------------------------------
    LOGGER.info("Building customer-month panel in Polars...")
    panel_full, first_purchase, latest_observation = build_customer_month_panel(tx)
    panel_model = panel_full.filter(
        pl.col("calendar_month") < pl.lit(latest_observation.to_pydatetime())
    )

    panel_full.write_parquet(
        output_dir / "customer_month_panel.parquet"
    )
    first_purchase.write_csv(
        output_dir / "customer_acquisition_cohorts.csv"
    )

    # ------------------------------------------------------------------------
    # 3. Current snapshot + optional enrichment
    # ------------------------------------------------------------------------
    snapshot = build_current_customer_snapshot(
        panel_full,
        latest_observation,
    )

    customer_360 = load_optional_table(args.customer_360)
    segmentation = load_optional_table(args.segmentation)
    clv = load_optional_table(args.clv)
    cohort = load_optional_table(args.cohort)

    enriched_snapshot = join_optional_customer_data(
        snapshot,
        customer_360=customer_360,
        segmentation=segmentation,
        clv=clv,
        cohort=cohort,
    )

    # ------------------------------------------------------------------------
    # 4. Temporal split
    # ------------------------------------------------------------------------
    split = build_temporal_split(
        panel_model,
        validation_months=args.validation_months,
        test_months=args.test_months,
    )

    LOGGER.info(
        "Temporal split | train<=%s | validation=%s..%s | test=%s..%s",
        split.train_end.date(),
        split.validation_start.date(),
        split.validation_end.date(),
        split.test_start.date(),
        split.test_end.date(),
    )

    # ------------------------------------------------------------------------
    # 5. Select features using the full model universe, then split rows.
    # ------------------------------------------------------------------------
    feature_names = select_model_features(panel_model)

    if len(feature_names) < 8:
        raise ValueError(
            f"Only {len(feature_names)} usable model features remain."
        )

    LOGGER.info(
        "Model features: %d",
        len(feature_names),
    )

    train_df = panel_model.filter(
        pl.col("calendar_month") <= pl.lit(split.train_end.to_pydatetime())
    )
    validation_df = panel_model.filter(
        (pl.col("calendar_month") >= pl.lit(split.validation_start.to_pydatetime()))
        & (pl.col("calendar_month") <= pl.lit(split.validation_end.to_pydatetime()))
    )
    test_df = panel_model.filter(
        (pl.col("calendar_month") >= pl.lit(split.test_start.to_pydatetime()))
        & (pl.col("calendar_month") <= pl.lit(split.test_end.to_pydatetime()))
    )

    def matrix_from(df: pl.DataFrame) -> np.ndarray:
        X, _ = prepare_matrix(df, feature_names)
        return X

    # Fit the quantile transformer only on training data.
    X_train_raw = (
        train_df.select(feature_names)
        .to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    X_val_raw = (
        validation_df.select(feature_names)
        .to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    X_test_raw = (
        test_df.select(feature_names)
        .to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    X_current_raw = (
        enriched_snapshot.select(
            [c for c in feature_names if c in enriched_snapshot.columns]
        )
    )

    # Ensure current snapshot contains every model feature, with fallback zeros.
    missing_current_features = [
        c for c in feature_names
        if c not in enriched_snapshot.columns
    ]
    if missing_current_features:
        enriched_snapshot = enriched_snapshot.with_columns(
            [pl.lit(0.0).alias(c) for c in missing_current_features]
        )
        X_current_raw = enriched_snapshot.select(feature_names)

    X_current_raw = (
        X_current_raw.to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )

    low = np.nanpercentile(X_train_raw, 1.0, axis=0)
    high = np.nanpercentile(X_train_raw, 99.0, axis=0)

    X_train_raw = np.clip(X_train_raw, low, high)
    X_val_raw = np.clip(X_val_raw, low, high)
    X_test_raw = np.clip(X_test_raw, low, high)
    X_current_raw = np.clip(X_current_raw, low, high)

    transformer = QuantileTransformer(
        n_quantiles=min(
            1000,
            max(50, X_train_raw.shape[0] // 20),
        ),
        output_distribution="normal",
        subsample=10000,
        random_state=SEED,
    )

    X_train = transformer.fit_transform(X_train_raw)
    X_val = transformer.transform(X_val_raw)
    X_test = transformer.transform(X_test_raw)
    X_current = transformer.transform(X_current_raw)

    y_train = (
        train_df.select("purchase_next_month")
        .to_numpy()
        .ravel()
        .astype(int)
    )
    y_val = (
        validation_df.select("purchase_next_month")
        .to_numpy()
        .ravel()
        .astype(int)
    )
    y_test = (
        test_df.select("purchase_next_month")
        .to_numpy()
        .ravel()
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # 6. 1-month survival hazard model
    # ------------------------------------------------------------------------
    LOGGER.info("Training discrete-time survival hazard model...")

    (
        survival_model,
        p_val_survival,
        p_test_survival,
        p_test_survival_raw,
        survival_tuning,
        survival_importance,
        survival_metrics,
    ) = fit_calibrated_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,
    )

    survival_tuning.to_csv(
        output_dir / "survival_model_tuning.csv",
        index=False,
    )
    survival_importance.to_csv(
        output_dir / "survival_feature_importance.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 7. Next-purchase models at 7/30/60 days
    # ------------------------------------------------------------------------
    # Monthly transaction data cannot directly reveal an exact 7-day target.
    # We therefore build a second event table at customer-month level using
    # the next observed purchase date. For each calendar month, the target is
    # whether the next actual purchase occurs within H days after month-end.
    # ------------------------------------------------------------------------

    LOGGER.info("Building multi-horizon next-purchase targets...")

    sale_invoice_dates = (
        tx.filter(pl.col("is_clean_sale"))
        .select(["Customer ID", "Invoice", "InvoiceDate"])
        .group_by(["Customer ID", "Invoice"])
        .agg(pl.col("InvoiceDate").min().alias("invoice_date"))
        .sort(["Customer ID", "invoice_date"])
    )

    # Last purchase date known as of each calendar month; then next purchase.
    invoice_pd = sale_invoice_dates.to_pandas()
    invoice_pd["calendar_month"] = invoice_pd["invoice_date"].dt.to_period("M").dt.to_timestamp()
    invoice_pd["next_invoice_date"] = (
        invoice_pd.groupby("Customer ID")["invoice_date"].shift(-1)
    )

    month_customer = (
        panel_model.select(
            ["Customer ID", "calendar_month"]
        )
        .to_pandas()
    )

    first_dates = (
        invoice_pd.groupby("Customer ID")["invoice_date"]
        .min()
        .rename("first_purchase_date")
    )

    # Merge invoice history; keep the last invoice <= month end, then its next
    # invoice as the future event. This is a compact, explicit anti-leakage
    # construction.
    invoice_by_customer = invoice_pd.sort_values(
        ["Customer ID", "invoice_date"]
    )

    month_customer["month_end"] = (
        month_customer["calendar_month"]
        + pd.offsets.MonthEnd(0)
        + pd.Timedelta(days=23, hours=59, minutes=59)
    )

    future_target_rows = []

    for customer_id, months_group in month_customer.groupby("Customer ID"):
        hist = invoice_by_customer[
            invoice_by_customer["Customer ID"] == customer_id
        ][["invoice_date"]].sort_values("invoice_date")

        dates = hist["invoice_date"].to_numpy()
        if len(dates) == 0:
            continue

        for _, row in months_group.iterrows():
            month = pd.Timestamp(row["calendar_month"])
            month_end = pd.Timestamp(row["month_end"])

            future = dates[dates > np.datetime64(month_end)]
            next_date = (
                pd.Timestamp(future[0])
                if len(future)
                else pd.NaT
            )

            future_target_rows.append(
                {
                    "Customer ID": customer_id,
                    "calendar_month": month,
                    "next_invoice_date": next_date,
                    "next_purchase_7d": int(
                        pd.notna(next_date)
                        and next_date <= month_end + pd.Timedelta(days=7)
                    ),
                    "next_purchase_30d": int(
                        pd.notna(next_date)
                        and next_date <= month_end + pd.Timedelta(days=30)
                    ),
                    "next_purchase_60d": int(
                        pd.notna(next_date)
                        and next_date <= month_end + pd.Timedelta(days=60)
                    ),
                    "days_to_next_purchase": (
                        float((next_date - month_end).total_seconds() / 86400.0)
                        if pd.notna(next_date)
                        else np.nan
                    ),
                }
            )

    next_targets = pd.DataFrame(future_target_rows)

    target_panel = panel_model.to_pandas()
    target_panel = target_panel.merge(
        next_targets,
        on=["Customer ID", "calendar_month"],
        how="left",
    )

    for horizon in [7, 30, 60]:
        target_col = f"next_purchase_{horizon}d"
        target_panel[target_col] = target_panel[target_col].fillna(0).astype(int)

    # ------------------------------------------------------------------------
    # 8. Refit preprocessing is not needed; use already transformed matrices.
    # ------------------------------------------------------------------------
    # Row ordering is identical because target_panel is merged on the original
    # panel ordering. To be explicit, rebuild masks by calendar month.
    # ------------------------------------------------------------------------

    def fit_horizon_model(horizon: int):
        target_col = f"next_purchase_{horizon}d"

        train_mask = target_panel["calendar_month"].between(
            split.train_end - pd.offsets.MonthBegin(10_000),
            split.train_end,
        )
        val_mask = target_panel["calendar_month"].between(
            split.validation_start,
            split.validation_end,
        )
        test_mask = target_panel["calendar_month"].between(
            split.test_start,
            split.test_end,
        )

        X_train_h = X_train[train_mask.to_numpy()]
        X_val_h = X_val[val_mask.to_numpy()]
        X_test_h = X_test[test_mask.to_numpy()]

        y_train_h = target_panel.loc[train_mask, target_col].to_numpy(dtype=int)
        y_val_h = target_panel.loc[val_mask, target_col].to_numpy(dtype=int)
        y_test_h = target_panel.loc[test_mask, target_col].to_numpy(dtype=int)

        (
            model,
            p_val,
            p_test,
            p_test_raw,
            tuning,
            importance,
            metrics,
        ) = fit_calibrated_model(
            X_train_h,
            y_train_h,
            X_val_h,
            y_val_h,
            X_test_h,
            y_test_h,
            feature_names,
        )

        return {
            "model": model,
            "p_val": p_val,
            "p_test": p_test,
            "p_test_raw": p_test_raw,
            "tuning": tuning,
            "importance": importance,
            "metrics": metrics,
            "y_val": y_val_h,
            "y_test": y_test_h,
            "test_mask": test_mask,
        }

    horizon_results = {}
    for horizon in [7, 30, 60]:
        LOGGER.info("Training %d-day next-purchase model...", horizon)
        horizon_results[horizon] = fit_horizon_model(horizon)
        horizon_results[horizon]["tuning"].to_csv(
            output_dir / f"next_purchase_{horizon}d_tuning.csv",
            index=False,
        )
        horizon_results[horizon]["importance"].to_csv(
            output_dir / f"next_purchase_{horizon}d_feature_importance.csv",
            index=False,
        )

    # ------------------------------------------------------------------------
    # 9. Current customer predictions
    # ------------------------------------------------------------------------

    current_prediction = pd.DataFrame(
        {
            "Customer ID": enriched_snapshot
            .select("Customer ID")
            .to_pandas()["Customer ID"].to_numpy(),
        }
    )

    current_snapshot_pd = enriched_snapshot.to_pandas()

    # Useful economic proxy even if optional CLV enrichment is absent.
    clv_candidates = [
        "clv",
        "clv_point_estimate",
        "customer_clv",
        "lifetime_value",
        "revenue",
        "revenue_6m",
    ]

    clv_proxy_col = None
    for candidate in clv_candidates:
        if candidate in current_snapshot_pd.columns:
            clv_proxy_col = candidate
            break

    if clv_proxy_col is None:
        current_prediction["clv_or_revenue_proxy"] = (
            current_snapshot_pd["revenue_6m"]
            .fillna(current_snapshot_pd["revenue_3m"])
            .fillna(current_snapshot_pd["revenue_month"])
            .to_numpy()
        )
    else:
        current_prediction["clv_or_revenue_proxy"] = (
            pd.to_numeric(
                current_snapshot_pd[clv_proxy_col],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy()
        )

    # Current survival / one-month hazard.
    current_prediction["next_month_hazard"] = (
        survival_model.predict_proba(X_current)[:, 1]
    )
    current_prediction["next_month_survival_probability"] = (
        1.0 - current_prediction["next_month_hazard"]
    )

    for horizon in [7, 30, 60]:
        model = horizon_results[horizon]["model"]
        current_prediction[
            f"next_purchase_{horizon}d_probability"
        ] = model.predict_proba(X_current)[:, 1]

    # Probability-weighted customer value proxies.
    current_prediction["expected_30d_activation_score"] = (
        current_prediction["next_purchase_30d_probability"]
        * np.maximum(
            current_prediction["clv_or_revenue_proxy"],
            0.0,
        )
    )

    current_prediction["retention_risk_score"] = (
        1.0
        - current_prediction["next_month_survival_probability"]
    )

    current_prediction["high_value_at_risk_score"] = (
        current_prediction["retention_risk_score"]
        * np.log1p(
            np.maximum(
                current_prediction["clv_or_revenue_proxy"],
                0.0,
            )
        )
    )

    # ------------------------------------------------------------------------
    # 10. 3/6/12-month survival scenario from current state
    # ------------------------------------------------------------------------

    survival_forecast = forecast_customer_survival(
        X_current,
        enriched_snapshot,
        survival_model,
        args.horizon_months,
    )

    current_prediction = current_prediction.merge(
        survival_forecast,
        on="Customer ID",
        how="left",
    )

    # Add segmentation/CLV/cohort fields if present.
    useful_enrichment = [
        c for c in [
            "segment",
            "segment_name",
            "segment_confidence",
            "cohort_month",
            clv_proxy_col,
        ]
        if c and c in current_snapshot_pd.columns
    ]

    if useful_enrichment:
        current_prediction = current_prediction.merge(
            current_snapshot_pd[
                ["Customer ID"] + useful_enrichment
            ],
            on="Customer ID",
            how="left",
        )

    current_prediction.to_csv(
        output_dir / "customer_retention_next_purchase_predictions.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 11. Test diagnostics
    # ------------------------------------------------------------------------

    metrics_payload = {
        "survival_1_month_hazard": survival_metrics,
        "next_purchase": {},
    }

    for horizon in [7, 30, 60]:
        result = horizon_results[horizon]
        metrics_payload["next_purchase"][f"{horizon}d"] = result["metrics"]

        lift = lift_table(
            result["y_test"],
            result["p_test"],
        )
        lift.to_csv(
            output_dir / f"next_purchase_{horizon}d_lift.csv",
            index=False,
        )

    with open(
        output_dir / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_payload,
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------------
    # 12. Survival curve from current customers
    # ------------------------------------------------------------------------

    survival_curve = current_prediction[
        [
            "Customer ID"
        ] + [
            f"survival_month_{m}"
            for m in range(1, args.horizon_months + 1)
        ]
    ].copy()

    survival_curve.to_csv(
        output_dir / "customer_survival_curve.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 13. Aggregate scorecards
    # ------------------------------------------------------------------------

    probability_scorecard = (
        current_prediction
        .assign(
            next_purchase_30d_decile=lambda x: pd.qcut(
                x["next_purchase_30d_probability"],
                q=10,
                labels=False,
                duplicates="drop",
            ) + 1
        )
        .groupby("next_purchase_30d_decile", as_index=False)
        .agg(
            customers=("Customer ID", "size"),
            mean_probability=("next_purchase_30d_probability", "mean"),
            mean_clv_proxy=("clv_or_revenue_proxy", "mean"),
            mean_high_value_risk=("high_value_at_risk_score", "mean"),
            mean_30d_activation_score=("expected_30d_activation_score", "mean"),
        )
        .sort_values("next_purchase_30d_decile", ascending=False)
    )

    probability_scorecard.to_csv(
        output_dir / "current_customer_propensity_deciles.csv",
        index=False,
    )

    # Segment-level scorecard if segmentation enrichment exists.
    if "segment" in current_prediction.columns:
        segment_scorecard = (
            current_prediction
            .groupby("segment", as_index=False)
            .agg(
                customers=("Customer ID", "size"),
                mean_next_purchase_30d=("next_purchase_30d_probability", "mean"),
                median_next_purchase_30d=("next_purchase_30d_probability", "median"),
                mean_next_month_survival=("next_month_survival_probability", "mean"),
                mean_high_value_at_risk=("high_value_at_risk_score", "mean"),
                mean_activation_score=("expected_30d_activation_score", "mean"),
                mean_clv_proxy=("clv_or_revenue_proxy", "mean"),
            )
            .sort_values("customers", ascending=False)
        )
        segment_scorecard.to_csv(
            output_dir / "segment_retention_next_purchase_scorecard.csv",
            index=False,
        )

    # ------------------------------------------------------------------------
    # 14. Visuals
    # ------------------------------------------------------------------------

    lift_30d = lift_table(
        horizon_results[30]["y_test"],
        horizon_results[30]["p_test"],
    )

    save_plots(
        output_dir=output_dir,
        y_val=horizon_results[30]["y_val"],
        p_val=horizon_results[30]["p_val"],
        y_test=horizon_results[30]["y_test"],
        p_test=horizon_results[30]["p_test"],
        lift_test=lift_30d,
        importance=horizon_results[30]["importance"],
        survival_curve=survival_curve.drop(columns=["Customer ID"]),
        next_purchase_30=current_prediction[
            [
                "Customer ID",
                "clv_or_revenue_proxy",
                "next_purchase_30d_probability",
            ]
        ],
        prediction_sample=args.prediction_sample,
    )

    # ------------------------------------------------------------------------
    # 15. Stability check: train/test temporal bootstrap
    # ------------------------------------------------------------------------

    stability_rows = []
    validation_months = sorted(
        pd.Timestamp(x)
        for x in panel_model
        .select("calendar_month")
        .unique()
        .to_series()
        .to_list()
    )

    for offset in range(min(args.bootstrap, 3)):
        if len(validation_months) < 10 + offset:
            continue

        # Rolling-origin diagnostic rather than random resampling.
        cutoff_index = -(args.test_months + offset + 1)
        if abs(cutoff_index) >= len(validation_months):
            continue

        cutoff = validation_months[cutoff_index]
        eval_months = [
            m for m in validation_months
            if m > cutoff
        ][:1]

        if not eval_months:
            continue

        train_roll = panel_model.filter(
            pl.col("calendar_month") <= pl.lit(cutoff.to_pydatetime())
        )
        eval_roll = panel_model.filter(
            pl.col("calendar_month").is_in(
                [m.to_pydatetime() for m in eval_months]
            )
        )

        if train_roll.height < 500 or eval_roll.height < 100:
            continue

        Xr_train_raw = (
            train_roll.select(feature_names)
            .to_pandas()
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
        Xr_eval_raw = (
            eval_roll.select(feature_names)
            .to_pandas()
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )

        lo = np.nanpercentile(Xr_train_raw, 1.0, axis=0)
        hi = np.nanpercentile(Xr_train_raw, 99.0, axis=0)
        Xr_train_raw = np.clip(Xr_train_raw, lo, hi)
        Xr_eval_raw = np.clip(Xr_eval_raw, lo, hi)

        qt = QuantileTransformer(
            n_quantiles=min(500, max(50, Xr_train_raw.shape[0] // 20)),
            output_distribution="normal",
            subsample=10000,
            random_state=SEED,
        )
        Xr_train = qt.fit_transform(Xr_train_raw)
        Xr_eval = qt.transform(Xr_eval_raw)

        yr_train = train_roll["purchase_next_month"].to_numpy().ravel().astype(int)
        yr_eval = eval_roll["purchase_next_month"].to_numpy().ravel().astype(int)

        model_roll = make_model()
        model_roll.fit(Xr_train, yr_train)
        pr_eval = model_roll.predict_proba(Xr_eval)[:, 1]

        stability_rows.append(
            {
                "cutoff": cutoff,
                "evaluation_month": eval_months[0],
                "roc_auc": safe_roc_auc(yr_eval, pr_eval),
                "pr_auc": safe_pr_auc(yr_eval, pr_eval),
                "brier": safe_brier(yr_eval, pr_eval),
            }
        )

    pd.DataFrame(stability_rows).to_csv(
        output_dir / "rolling_stability_diagnostics.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 16. Metadata / manifest
    # ------------------------------------------------------------------------

    survival_metadata = {
        "type": "discrete_time_hazard_model",
        "monthly_target": "purchase_next_month",
        "horizon_months": args.horizon_months,
        "one_month_test_metrics": survival_metrics,
        "current_customer_forecast": "recursive current-state hazard scenario",
    }

    create_model_card(
        output_dir,
        feature_names,
        split,
        metrics_payload["next_purchase"],
        survival_metadata,
    )

    metadata = {
        "input": str(input_path),
        "observation_end": str(latest_observation),
        "customers": int(snapshot.height),
        "customer_month_rows": int(panel_full.height),
        "model_features": feature_names,
        "temporal_split": {
            "train_end": str(split.train_end),
            "validation_start": str(split.validation_start),
            "validation_end": str(split.validation_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
        },
        "optional_enrichment_used": {
            "customer_360": customer_360 is not None,
            "segmentation": segmentation is not None,
            "clv": clv is not None,
            "cohort": cohort is not None,
        },
        "modeling_notes": [
            "Features are clipped and quantile-normalized using training data only.",
            "Model selection is temporal, not random cross-validation.",
            "Next-purchase probabilities are separately estimated at 7, 30 and 60 days.",
            "Monthly survival is modeled as a discrete-time hazard and accumulated into survival probabilities.",
            "Current-customer survival beyond one month is a scenario forecast because future covariates are not observable.",
        ],
        "seed": SEED,
    }

    with open(
        output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    manifest = {
        "outputs": [
            "customer_month_panel.parquet",
            "customer_acquisition_cohorts.csv",
            "customer_retention_next_purchase_predictions.csv",
            "customer_survival_curve.csv",
            "survival_model_tuning.csv",
            "survival_feature_importance.csv",
            "next_purchase_7d_tuning.csv",
            "next_purchase_7d_feature_importance.csv",
            "next_purchase_7d_lift.csv",
            "next_purchase_30d_tuning.csv",
            "next_purchase_30d_feature_importance.csv",
            "next_purchase_30d_lift.csv",
            "next_purchase_60d_tuning.csv",
            "next_purchase_60d_feature_importance.csv",
            "next_purchase_60d_lift.csv",
            "current_customer_propensity_deciles.csv",
            "segment_retention_next_purchase_scorecard.csv",
            "rolling_stability_diagnostics.csv",
            "model_metrics.json",
            "model_card.json",
            "run_metadata.json",
            "plots/01_next_purchase_calibration.png",
            "plots/02_cumulative_capture.png",
            "plots/03_feature_importance.png",
            "plots/04_survival_curve.png",
            "plots/05_value_vs_next_purchase_probability.png",
            "plots/06_next_purchase_probability_distribution.png",
        ]
    }

    with open(
        output_dir / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info(
        "Pipeline complete. Outputs: %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
