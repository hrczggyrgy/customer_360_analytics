#!/usr/bin/env python3
"""
Online Retail II — modern, uncertainty-aware Customer Lifetime Value (CLV).

Design philosophy
-----------------
This is a dynamic CLV engine rather than a static RFM monetization exercise.
It combines:

1. Polars transaction engineering.
2. A leakage-safe next-month purchase model using non-linear tabular ML.
3. A conditional spend model for net monthly value when a customer is active.
4. Empirical-Bayes propensity shrinkage to stabilize sparse customers.
5. Monte-Carlo future-state simulation with dynamic recency/cadence updates.
6. Return-aware net value rather than gross sales only.
7. Time-discounted CLV and uncertainty bands.
8. Optional integration with the project's customer segmentation output.
9. Optional integration with the project's cohort acquisition output.
10. Model calibration, feature importance, CLV deciles, cohort and segment views.

Important interpretation
------------------------
Online Retail II has revenue/price data but no COGS or gross-margin field.
Therefore the default CLV metric is forecasted discounted NET REVENUE, not economic
profit. Use --margin-rate to convert revenue CLV into a contribution-style proxy.

Example:
    python clv_analysis.py \
        --input /home/lptop/Documents/coding/marketing_science/data_xslx/online_retail_II.xlsx \
        --output-dir /home/lptop/Documents/coding/marketing_science/clv_analysis_output

Dependencies:
    pip install polars fastexcel pandas matplotlib scikit-learn openpyxl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


SEED = 42
np.random.seed(SEED)


# =============================================================================
# LOGGING
# =============================================================================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("clv_analysis")
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


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Modern dynamic CLV analysis for Online Retail II."
    )

    parser.add_argument(
        "--input",
        default=str(
            project_dir
            / "data_xslx"
            / "online_retail_II.xlsx"
        ),
        help="Input .xlsx/.xls/.csv/.parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            project_dir
            / "clv_analysis_output"
        ),
        help="Directory where CLV outputs are written.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. If omitted, all non-empty sheets are read.",
    )

    parser.add_argument(
        "--horizon-months",
        type=int,
        default=24,
        help="Future CLV simulation horizon.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="Monte-Carlo paths per customer.",
    )

    parser.add_argument(
        "--annual-discount-rate",
        type=float,
        default=0.10,
        help="Annual discount rate applied to future revenue.",
    )

    parser.add_argument(
        "--margin-rate",
        type=float,
        default=1.0,
        help="Optional contribution margin proxy. 1.0 leaves CLV as revenue CLV.",
    )

    parser.add_argument(
        "--validation-months",
        type=int,
        default=3,
        help="Trailing calendar months reserved as a time-based validation set.",
    )

    parser.add_argument(
        "--min-customers",
        type=int,
        default=100,
        help="Minimum customer count required to fit the model.",
    )

    parser.add_argument(
        "--min-active-history-months",
        type=int,
        default=2,
        help="Minimum active history used for ML training rows.",
    )

    parser.add_argument(
        "--random-customer-cap",
        type=int,
        default=0,
        help="Optional customer cap for development runs. 0 means all customers.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    return parser.parse_args()


# =============================================================================
# INPUT
# =============================================================================


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
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("_", " ")
        )
        rename_map[column] = aliases.get(
            normalized,
            column,
        )

    df = df.rename(rename_map)

    required = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}; found {df.columns}"
        )

    return df


def load_input(path: Path, sheet: Optional[str]) -> pl.DataFrame:
    suffix = path.suffix.lower()
    LOGGER.info("Loading %s", path)

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            if sheet:
                df = pl.read_excel(
                    path,
                    sheet_name=sheet,
                )
            else:
                loaded = pl.read_excel(
                    path,
                    sheet_id=0,
                )

                if isinstance(loaded, dict):
                    frames = [
                        frame
                        for frame in loaded.values()
                        if isinstance(frame, pl.DataFrame)
                        and frame.height > 0
                    ]
                    if not frames:
                        raise ValueError(
                            "Workbook contains no non-empty sheets."
                        )
                    df = pl.concat(
                        frames,
                        how="diagonal_relaxed",
                    )
                else:
                    df = loaded

        except Exception as exc:
            LOGGER.warning(
                "Polars Excel reader failed (%s); using pandas/openpyxl fallback.",
                exc,
            )

            loaded = pd.read_excel(
                path,
                sheet_name=sheet if sheet else None,
            )

            if isinstance(loaded, dict):
                loaded = pd.concat(
                    loaded.values(),
                    ignore_index=True,
                )

            df = pl.from_pandas(loaded)

    elif suffix == ".csv":
        df = pl.read_csv(
            path,
            try_parse_dates=True,
            infer_schema_length=20_000,
            ignore_errors=True,
        )

    elif suffix in {".parquet", ".pq"}:
        df = pl.read_parquet(path)

    else:
        raise ValueError(
            f"Unsupported input format: {suffix}"
        )

    return normalize_columns(df)


# =============================================================================
# CLEANING
# =============================================================================


def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(
        [
            pl.col("Invoice")
            .cast(pl.Utf8)
            .str.strip_chars(),

            pl.col("StockCode")
            .cast(pl.Utf8)
            .str.strip_chars(),

            pl.col("Quantity")
            .cast(pl.Float64, strict=False),

            pl.col("Price")
            .cast(pl.Float64, strict=False),

            pl.col("Customer ID")
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False),
        ]
    )

    if out.schema["InvoiceDate"] not in {
        pl.Date,
        pl.Datetime,
    }:
        out = out.with_columns(
            pl.col("InvoiceDate")
            .cast(pl.Utf8)
            .str.strptime(
                pl.Datetime,
                strict=False,
                exact=False,
            )
        )
    else:
        out = out.with_columns(
            pl.col("InvoiceDate")
            .cast(pl.Datetime, strict=False)
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

            (pl.col("Quantity") > 0).alias("positive_quantity"),
            (pl.col("Quantity") < 0).alias("negative_quantity"),

            (
                pl.col("Quantity") * pl.col("Price")
            ).alias("line_value"),
        ]
    )

    out = out.with_columns(
        [
            (
                pl.col("positive_quantity")
                & (~pl.col("is_cancellation_invoice"))
                & (pl.col("Price") > 0)
            ).alias("is_sale"),

            pl.when(
                pl.col("positive_quantity")
                & (~pl.col("is_cancellation_invoice"))
                & (pl.col("Price") > 0)
            )
            .then(pl.col("line_value"))
            .otherwise(0.0)
            .alias("gross_sale_value"),

            pl.when(
                pl.col("negative_quantity")
                | pl.col("is_cancellation_invoice")
            )
            .then(pl.col("line_value").abs())
            .otherwise(0.0)
            .alias("return_value"),
        ]
    )

    LOGGER.info(
        "Cleaned transaction rows: %s",
        f"{out.height:,}",
    )

    return out


# =============================================================================
# PROJECT INTEGRATION
# =============================================================================


def discover_project_file(
    explicit: Optional[str],
    project_dir: Path,
    candidates: List[Path],
) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None

    search_paths = candidates + [
        project_dir / candidate
        for candidate in candidates
    ]

    seen = set()
    for path in search_paths:
        path = Path(path)
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists():
            return path

    return None


def load_optional_segments(project_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        Path.cwd() / "online_retail_segmentation" / "customer_segments.csv",
        project_dir / "online_retail_segmentation" / "customer_segments.csv",
        Path.cwd() / "customer_segments.csv",
        project_dir / "customer_segments.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                seg = pd.read_csv(candidate)
                if "Customer ID" in seg.columns:
                    keep = ["Customer ID"]
                    for column in ["segment", "segment_name", "segment_confidence"]:
                        if column in seg.columns:
                            keep.append(column)
                    seg = seg[keep].copy()
                    seg["Customer ID"] = pd.to_numeric(
                        seg["Customer ID"], errors="coerce"
                    ).astype("Int64")
                    seg = seg.drop_duplicates("Customer ID")
                    LOGGER.info(
                        "Integrated segmentation output: %s",
                        candidate,
                    )
                    return seg
            except Exception as exc:
                LOGGER.warning(
                    "Could not read segmentation output %s: %s",
                    candidate,
                    exc,
                )

    return None


def load_optional_cohorts(project_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        Path.cwd() / "cohort_analysis_output" / "customer_acquisition_cohorts.csv",
        project_dir / "cohort_analysis_output" / "customer_acquisition_cohorts.csv",
        Path.cwd() / "customer_acquisition_cohorts.csv",
        project_dir / "customer_acquisition_cohorts.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                cohort = pd.read_csv(candidate)
                required = {"Customer ID", "cohort_month"}
                if required.issubset(cohort.columns):
                    cohort = cohort[
                        ["Customer ID", "cohort_month"]
                    ].copy()
                    cohort["Customer ID"] = pd.to_numeric(
                        cohort["Customer ID"], errors="coerce"
                    ).astype("Int64")
                    cohort["cohort_month"] = pd.to_datetime(
                        cohort["cohort_month"], errors="coerce"
                    )
                    cohort = cohort.drop_duplicates("Customer ID")
                    LOGGER.info(
                        "Integrated cohort output: %s",
                        candidate,
                    )
                    return cohort
            except Exception as exc:
                LOGGER.warning(
                    "Could not read cohort output %s: %s",
                    candidate,
                    exc,
                )

    return None


# =============================================================================
# CUSTOMER-MONTH FEATURE ENGINEERING
# =============================================================================


def build_customer_month_panel(
    tx: pl.DataFrame,
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pd.Timestamp,
    pd.Timestamp,
]:
    sales = tx.filter(
        pl.col("is_sale")
    )

    if sales.height == 0:
        raise ValueError("No valid positive-sale rows were found.")

    invoice_month = (
        sales
        .group_by(
            ["Customer ID", "Invoice"]
        )
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_sale_value").sum().alias("gross_revenue"),
                pl.col("Quantity").sum().alias("units"),
                pl.col("StockCode").n_unique().alias("unique_products"),
            ]
        )
        .with_columns(
            pl.col("invoice_date")
            .dt.truncate("1mo")
            .alias("calendar_month")
        )
    )

    monthly_sales = (
        invoice_month
        .group_by(
            ["Customer ID", "calendar_month"]
        )
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("unique_products").sum().alias("product_lines"),
            ]
        )
    )

    returns = tx.filter(
        pl.col("negative_quantity")
        | pl.col("is_cancellation_invoice")
    )

    if returns.height:
        monthly_returns = (
            returns
            .with_columns(
                pl.col("InvoiceDate")
                .dt.truncate("1mo")
                .alias("calendar_month")
            )
            .group_by(
                ["Customer ID", "calendar_month"]
            )
            .agg(
                pl.col("return_value").sum().alias("return_value")
            )
        )
    else:
        monthly_returns = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "calendar_month": pl.Datetime,
                "return_value": pl.Float64,
            }
        )

    customer_first = (
        monthly_sales
        .group_by("Customer ID")
        .agg(
            pl.col("calendar_month")
            .min()
            .alias("cohort_month")
        )
    )

    observation_start = (
        monthly_sales
        .select(pl.col("calendar_month").min())
        .item()
    )

    observation_end = (
        monthly_sales
        .select(pl.col("calendar_month").max())
        .item()
    )

    # Global calendar.
    month_count = (
        (
            observation_end.year - observation_start.year
        )
        * 12
        + observation_end.month
        - observation_start.month
        + 1
    )

    calendar_rows = [
        {
            "calendar_month": (
                observation_start
                + pd.DateOffset(months=i)
            )
        }
        for i in range(month_count)
    ]

    calendar = pl.from_pandas(
        pd.DataFrame(calendar_rows)
        .assign(
            calendar_month=lambda x: pd.to_datetime(
                x["calendar_month"]
            )
        )
    ).with_columns(
        pl.col("calendar_month").cast(pl.Datetime("us"))
    )

    customers = customer_first.select(
        "Customer ID"
    ).unique()

    # Dense customer x calendar panel.
    dense = (
        customers
        .join(
            calendar,
            how="cross",
        )
        .join(
            customer_first,
            on="Customer ID",
            how="left",
        )
        .filter(
            pl.col("calendar_month")
            >= pl.col("cohort_month")
        )
        .join(
            monthly_sales,
            on=[
                "Customer ID",
                "calendar_month",
            ],
            how="left",
        )
        .join(
            monthly_returns,
            on=[
                "Customer ID",
                "calendar_month",
            ],
            how="left",
        )
        .with_columns(
            [
                pl.col("orders").fill_null(0).cast(pl.Float64),
                pl.col("gross_revenue").fill_null(0.0),
                pl.col("units").fill_null(0.0),
                pl.col("product_lines").fill_null(0.0),
                pl.col("return_value").fill_null(0.0),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("gross_revenue")
                    - pl.col("return_value")
                ).alias("net_revenue"),

                (
                    pl.col("orders") > 0
                ).cast(pl.Int8).alias("active"),
            ]
        )
        .sort(
            ["Customer ID", "calendar_month"]
        )
    )

    dense = dense.with_columns(
        [
            (
                (
                    pl.col("calendar_month").dt.year()
                    * 12
                    + pl.col("calendar_month").dt.month()
                )
                - (
                    pl.col("cohort_month").dt.year()
                    * 12
                    + pl.col("cohort_month").dt.month()
                )
            ).alias("age_month"),

            (
                pl.col("calendar_month").dt.month().cast(pl.Float64)
                / 12.0
            ).alias("calendar_month_fraction"),
        ]
    )

    # Cumulative features up to current month.
    dense = dense.with_columns(
        [
            pl.col("orders")
            .cum_sum()
            .over("Customer ID")
            .alias("lifetime_orders"),

            pl.col("gross_revenue")
            .cum_sum()
            .over("Customer ID")
            .alias("lifetime_gross_revenue"),

            pl.col("net_revenue")
            .cum_sum()
            .over("Customer ID")
            .alias("lifetime_net_revenue"),

            pl.col("active")
            .cum_sum()
            .over("Customer ID")
            .alias("lifetime_active_months"),

            pl.col("return_value")
            .cum_sum()
            .over("Customer ID")
            .alias("lifetime_return_value"),
        ]
    )

    # Lagged three-month rolling features.
    for lag in (1, 2, 3):
        dense = dense.with_columns(
            [
                pl.col("orders")
                .shift(lag)
                .over("Customer ID")
                .fill_null(0.0)
                .alias(f"orders_lag_{lag}"),

                pl.col("net_revenue")
                .shift(lag)
                .over("Customer ID")
                .fill_null(0.0)
                .alias(f"net_revenue_lag_{lag}"),
            ]
        )

    dense = dense.with_columns(
        [
            (
                pl.col("orders")
                + pl.col("orders_lag_1")
                + pl.col("orders_lag_2")
            ).alias("orders_last_3m"),

            (
                pl.col("net_revenue")
                + pl.col("net_revenue_lag_1")
                + pl.col("net_revenue_lag_2")
            ).alias("net_revenue_last_3m"),

            (
                pl.col("orders_lag_1")
                + pl.col("orders_lag_2")
                + pl.col("orders_lag_3")
            ).alias("orders_prev_3m"),

            (
                pl.col("net_revenue_lag_1")
                + pl.col("net_revenue_lag_2")
                + pl.col("net_revenue_lag_3")
            ).alias("net_revenue_prev_3m"),
        ]
    )

    # Rolling order / value moments.
    dense = dense.with_columns(
        [
            pl.col("net_revenue")
            .rolling_mean(3)
            .over("Customer ID")
            .fill_null(0.0)
            .alias("net_revenue_3m_mean"),

            pl.col("orders")
            .rolling_mean(3)
            .over("Customer ID")
            .fill_null(0.0)
            .alias("orders_3m_mean"),
        ]
    )

    # Last active month ID via cumulative maximum of active month IDs.
    dense = dense.with_columns(
        [
            (
                pl.col("calendar_month").dt.year() * 12
                + pl.col("calendar_month").dt.month()
            ).alias("calendar_month_id"),
        ]
    )

    dense = dense.with_columns(
        pl.when(pl.col("active") == 1)
        .then(pl.col("calendar_month_id"))
        .otherwise(None)
        .alias("active_month_id_candidate")
    )

    dense = dense.with_columns(
        pl.col("active_month_id_candidate")
        .forward_fill()
        .over("Customer ID")
        .alias("last_active_month_id")
    )

    dense = dense.with_columns(
        [
            (
                pl.col("calendar_month_id")
                - pl.col("last_active_month_id")
            )
            .fill_null(pl.col("age_month"))
            .clip(lower_bound=0)
            .alias("recency_months"),

            (
                pl.col("lifetime_orders")
                / pl.col("age_month").add(1)
            ).alias("lifetime_order_rate_per_month"),

            (
                pl.col("lifetime_active_months")
                / pl.col("age_month").add(1)
            ).alias("lifetime_active_month_share"),

            (
                pl.col("net_revenue_last_3m")
                / (
                    pl.col("net_revenue_prev_3m").abs()
                    + 10.0
                )
            ).alias("recent_value_momentum"),

            (
                pl.col("orders_last_3m")
                / (
                    pl.col("age_month").clip(lower_bound=1)
                    + 0.0
                )
            ).alias("recent_orders_intensity"),

            (
                pl.col("lifetime_net_revenue")
                / pl.col("lifetime_orders").clip(lower_bound=1)
            ).alias("historical_value_per_order"),

            (
                pl.col("lifetime_return_value")
                / (
                    pl.col("lifetime_gross_revenue")
                    + 1e-9
                )
            ).alias("historical_return_ratio"),

            (
                pl.col("calendar_month").dt.month().cast(pl.Float64)
            ).alias("calendar_month_number"),
        ]
    )

    # Seasonal harmonics.
    dense = dense.with_columns(
        [
            (
                2
                * math.pi
                * pl.col("calendar_month_number")
                / 12.0
            ).sin().alias("month_sin"),

            (
                2
                * math.pi
                * pl.col("calendar_month_number")
                / 12.0
            ).cos().alias("month_cos"),
        ]
    )

    # Target is next-month activity / next-month net revenue.
    dense = dense.with_columns(
        [
            pl.col("active")
            .shift(-1)
            .over("Customer ID")
            .alias("next_active"),

            pl.col("net_revenue")
            .shift(-1)
            .over("Customer ID")
            .alias("next_net_revenue"),

            pl.col("calendar_month")
            .shift(-1)
            .over("Customer ID")
            .alias("next_calendar_month"),
        ]
    )

    feature_columns = [
        "age_month",
        "recency_months",
        "lifetime_orders",
        "lifetime_active_months",
        "lifetime_order_rate_per_month",
        "lifetime_active_month_share",
        "orders_last_3m",
        "orders_prev_3m",
        "orders_3m_mean",
        "recent_orders_intensity",
        "net_revenue_last_3m",
        "net_revenue_prev_3m",
        "net_revenue_3m_mean",
        "recent_value_momentum",
        "lifetime_net_revenue",
        "historical_value_per_order",
        "historical_return_ratio",
        "month_sin",
        "month_cos",
    ]

    dense = dense.with_columns(
        [
            pl.col(column)
            .cast(pl.Float64)
            .fill_null(0.0)
            .fill_nan(0.0)
            for column in feature_columns
        ]
    )

    LOGGER.info(
        "Customer-month modeling panel: %s rows | %s customers",
        f"{dense.height:,}",
        f"{dense.select('Customer ID').n_unique():,}",
    )

    return (
        dense,
        pl.DataFrame(
            {
                "Customer ID": customer_first["Customer ID"],
                "cohort_month": customer_first["cohort_month"],
            }
        ),
        pd.Timestamp(observation_start),
        pd.Timestamp(observation_end),
    )


# =============================================================================
# MODEL FEATURES
# =============================================================================


FEATURE_COLUMNS = [
    "age_month",
    "recency_months",
    "lifetime_orders",
    "lifetime_active_months",
    "lifetime_order_rate_per_month",
    "lifetime_active_month_share",
    "orders_last_3m",
    "orders_prev_3m",
    "orders_3m_mean",
    "recent_orders_intensity",
    "net_revenue_last_3m",
    "net_revenue_prev_3m",
    "net_revenue_3m_mean",
    "recent_value_momentum",
    "lifetime_net_revenue",
    "historical_value_per_order",
    "historical_return_ratio",
    "month_sin",
    "month_cos",
]


def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))


def sample_weights_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    positives = max(1, int(y.sum()))
    negatives = max(1, int((1 - y).sum()))
    weights = np.ones_like(y, dtype=float)
    weights[y == 1] = 0.5 * len(y) / positives
    weights[y == 0] = 0.5 * len(y) / negatives
    return weights


def train_models(
    panel: pl.DataFrame,
    validation_months: int,
    min_active_history_months: int,
    seed: int,
) -> Tuple[
    object,
    object,
    Dict,
    pd.DataFrame,
    Dict[str, np.ndarray],
]:
    pdf = panel.to_pandas()

    pdf["calendar_month"] = pd.to_datetime(
        pdf["calendar_month"]
    )

    unique_months = sorted(
        pdf["calendar_month"].dropna().unique()
    )

    if len(unique_months) < validation_months + 6:
        validation_months = max(
            1,
            min(
                validation_months,
                len(unique_months) // 4,
            ),
        )

    validation_start = pd.Timestamp(
        unique_months[-validation_months]
    )

    target_available = pdf["next_active"].notna()

    train_mask = (
        (pdf["calendar_month"] < validation_start)
        & target_available
    )

    validation_mask = (
        (pdf["calendar_month"] >= validation_start)
        & target_available
    )

    # Prevent very immature rows from dominating the purchase model.
    train_mask &= (
        pdf["lifetime_active_months"]
        >= min_active_history_months
    )

    X_train = pdf.loc[
        train_mask,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=float)

    y_train = pdf.loc[
        train_mask,
        "next_active",
    ].astype(int).to_numpy()

    X_valid = pdf.loc[
        validation_mask,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=float)

    y_valid = pdf.loc[
        validation_mask,
        "next_active",
    ].astype(int).to_numpy()

    if len(X_train) == 0 or len(X_valid) == 0:
        raise ValueError(
            "Time-based validation split produced no training or validation rows."
        )

    if np.unique(y_train).size < 2:
        raise ValueError(
            "Purchase target has fewer than two classes in training data."
        )

    purchase_model = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=seed,
    )

    purchase_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_binary(y_train),
    )

    purchase_valid_prob = purchase_model.predict_proba(
        X_valid
    )[:, 1]

    purchase_metrics = {
        "roc_auc": (
            float(roc_auc_score(y_valid, purchase_valid_prob))
            if np.unique(y_valid).size > 1
            else None
        ),
        "average_precision": (
            float(
                average_precision_score(
                    y_valid,
                    purchase_valid_prob,
                )
            )
            if np.unique(y_valid).size > 1
            else None
        ),
        "brier_score": float(
            brier_score_loss(
                y_valid,
                purchase_valid_prob,
            )
        ),
        "validation_rows": int(len(y_valid)),
        "validation_positive_rate": float(y_valid.mean()),
        "validation_predicted_rate": float(
            purchase_valid_prob.mean()
        ),
    }

    # -------------------------------------------------------------------------
    # Spend model: only next-month active rows.
    # -------------------------------------------------------------------------

    spend_train_mask = (
        train_mask
        & (pdf["next_active"] == 1)
        & np.isfinite(
            pdf["next_net_revenue"]
        )
    )

    spend_valid_mask = (
        validation_mask
        & (pdf["next_active"] == 1)
        & np.isfinite(
            pdf["next_net_revenue"]
        )
    )

    X_spend_train = pdf.loc[
        spend_train_mask,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=float)

    y_spend_train = signed_log1p(
        pdf.loc[
            spend_train_mask,
            "next_net_revenue",
        ].to_numpy(dtype=float)
    )

    X_spend_valid = pdf.loc[
        spend_valid_mask,
        FEATURE_COLUMNS,
    ].to_numpy(dtype=float)

    y_spend_valid_raw = pdf.loc[
        spend_valid_mask,
        "next_net_revenue",
    ].to_numpy(dtype=float)

    if len(X_spend_train) < 100:
        raise ValueError(
            "Too few active next-month observations for a robust spend model."
        )

    spend_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=seed,
    )

    spend_model.fit(
        X_spend_train,
        y_spend_train,
    )

    spend_valid_pred_raw = signed_expm1(
        spend_model.predict(
            X_spend_valid
        )
    )

    spend_metrics = {
        "mae": float(
            mean_absolute_error(
                y_spend_valid_raw,
                spend_valid_pred_raw,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_spend_valid_raw,
                    spend_valid_pred_raw,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_spend_valid_raw,
                spend_valid_pred_raw,
            )
        )
        if len(y_spend_valid_raw) > 2
        and np.var(y_spend_valid_raw) > 1e-12
        else None,
        "validation_active_rows": int(
            len(y_spend_valid_raw)
        ),
        "validation_actual_mean": float(
            np.mean(y_spend_valid_raw)
        )
        if len(y_spend_valid_raw)
        else None,
        "validation_predicted_mean": float(
            np.mean(spend_valid_pred_raw)
        )
        if len(spend_valid_pred_raw)
        else None,
    }

    spend_residuals = (
        y_spend_valid_raw
        - spend_valid_pred_raw
    )

    residual_scale = float(
        np.median(
            np.abs(
                spend_residuals
                - np.median(spend_residuals)
            )
        )
        * 1.4826
    )

    # Residuals in transformed space are more appropriate for future simulation.
    if len(X_spend_valid):
        spend_valid_pred_transformed = spend_model.predict(
            X_spend_valid
        )
        transformed_residuals = (
            signed_log1p(y_spend_valid_raw)
            - spend_valid_pred_transformed
        )
    else:
        transformed_residuals = np.array(
            [0.5],
            dtype=float,
        )

    transformed_residuals = transformed_residuals[
        np.isfinite(transformed_residuals)
    ]

    transformed_sigma = float(
        np.median(
            np.abs(
                transformed_residuals
                - np.median(transformed_residuals)
            )
        )
        * 1.4826
    )

    transformed_sigma = max(
        transformed_sigma,
        0.15,
    )

    # -------------------------------------------------------------------------
    # Permutation importance on a bounded validation sample.
    # -------------------------------------------------------------------------

    importance_rows = []

    rng = np.random.default_rng(seed)

    if len(X_valid) > 5000:
        idx = rng.choice(
            len(X_valid),
            size=5000,
            replace=False,
        )
        X_imp = X_valid[idx]
        y_imp = y_valid[idx]
    else:
        X_imp = X_valid
        y_imp = y_valid

    if len(X_imp) and np.unique(y_imp).size > 1:
        perm = permutation_importance(
            purchase_model,
            X_imp,
            y_imp,
            scoring="roc_auc",
            n_repeats=3,
            random_state=seed,
        )

        for feature, importance in zip(
            FEATURE_COLUMNS,
            perm.importances_mean,
        ):
            importance_rows.append(
                {
                    "model": "purchase_probability",
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    if len(X_spend_valid):
        n_spend_imp = min(
            5000,
            len(X_spend_valid),
        )
        idx = rng.choice(
            len(X_spend_valid),
            size=n_spend_imp,
            replace=False,
        ) if len(X_spend_valid) > n_spend_imp else np.arange(
            len(X_spend_valid)
        )

        if len(idx) > 30:
            y_spend_valid_transformed = signed_log1p(
                y_spend_valid_raw[idx]
            )

            perm = permutation_importance(
                spend_model,
                X_spend_valid[idx],
                y_spend_valid_transformed,
                scoring="neg_mean_absolute_error",
                n_repeats=3,
                random_state=seed,
            )

            for feature, importance in zip(
                FEATURE_COLUMNS,
                perm.importances_mean,
            ):
                importance_rows.append(
                    {
                        "model": "conditional_spend",
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    metrics = {
        "purchase_model": purchase_metrics,
        "conditional_spend_model": spend_metrics,
        "validation": {
            "validation_start": str(validation_start),
            "validation_months": int(validation_months),
        },
        "spend_noise": {
            "raw_residual_robust_scale": residual_scale,
            "transformed_residual_sigma": transformed_sigma,
        },
    }

    # Holdout predictions are returned for calibration/diagnostics.
    validation_predictions = {
        "calendar_month": pdf.loc[validation_mask, "calendar_month"].to_numpy(),
        "actual_purchase": y_valid,
        "predicted_purchase_probability": purchase_valid_prob,
        "actual_spend": y_spend_valid_raw,
        "predicted_spend": spend_valid_pred_raw,
    }

    return (
        purchase_model,
        spend_model,
        metrics,
        pd.DataFrame(importance_rows),
        validation_predictions,
        transformed_sigma,
    )


# =============================================================================
# EMPIRICAL-BAYES PRIOR
# =============================================================================


def estimate_beta_prior(
    panel: pl.DataFrame,
) -> Tuple[float, float]:
    pdf = panel.select(
        [
            "active",
            "age_month",
        ]
    ).to_pandas()

    successes = float(
        pdf["active"].sum()
    )

    exposure = float(
        (pdf["age_month"] + 1).sum()
    )

    global_rate = min(
        max(
            successes
            / max(exposure, 1.0),
            0.01,
        ),
        0.80,
    )

    # Moderately informative prior; customer evidence can still dominate.
    prior_strength = 12.0
    alpha = max(
        0.25,
        global_rate * prior_strength,
    )
    beta = max(
        0.25,
        (1.0 - global_rate) * prior_strength,
    )

    return alpha, beta


# =============================================================================
# FUTURE STATE INITIALIZATION
# =============================================================================


def build_customer_snapshot(
    panel: pl.DataFrame,
    tx: pl.DataFrame,
) -> pd.DataFrame:
    # Last modeling row per customer gives state immediately before forecasting.
    snapshot = (
        panel
        .sort(
            ["Customer ID", "calendar_month"]
        )
        .group_by("Customer ID", maintain_order=True)
        .agg(
            [
                pl.col("calendar_month").last().alias("last_model_month"),
                pl.col("cohort_month").last().alias("cohort_month"),
                pl.col("age_month").last().alias("age_month"),
                pl.col("recency_months").last().alias("recency_months"),
                pl.col("lifetime_orders").last().alias("lifetime_orders"),
                pl.col("lifetime_active_months").last().alias("lifetime_active_months"),
                pl.col("lifetime_net_revenue").last().alias("lifetime_net_revenue"),
                pl.col("lifetime_gross_revenue").last().alias("lifetime_gross_revenue"),
                pl.col("lifetime_return_value").last().alias("lifetime_return_value"),
                pl.col("orders_last_3m").last().alias("orders_last_3m"),
                pl.col("orders_prev_3m").last().alias("orders_prev_3m"),
                pl.col("net_revenue_last_3m").last().alias("net_revenue_last_3m"),
                pl.col("net_revenue_prev_3m").last().alias("net_revenue_prev_3m"),
                pl.col("net_revenue_3m_mean").last().alias("net_revenue_3m_mean"),
                pl.col("orders_3m_mean").last().alias("orders_3m_mean"),
                pl.col("historical_value_per_order").last().alias("historical_value_per_order"),
                pl.col("historical_return_ratio").last().alias("historical_return_ratio"),
                pl.col("active").last().alias("last_month_active"),
                pl.col("next_net_revenue").last().alias("unused"),
            ]
        )
        .drop("unused")
        .to_pandas()
    )

    # Need the latest four observed months for simulation. The fourth month
    # is required because the trained feature `*_prev_3m` excludes the
    # current month and therefore looks three months backward.
    monthly = (
        panel
        .select(
            [
                "Customer ID",
                "calendar_month",
                "orders",
                "net_revenue",
            ]
        )
        .to_pandas()
    )

    monthly["calendar_month"] = pd.to_datetime(
        monthly["calendar_month"]
    )

    recent_rows = []
    for customer_id, group in monthly.groupby(
        "Customer ID",
        sort=False,
    ):
        group = group.sort_values(
            "calendar_month"
        ).tail(4)
        row = {
            "Customer ID": customer_id,
            "orders_t0": 0.0,
            "orders_t1": 0.0,
            "orders_t2": 0.0,
            "orders_t3": 0.0,
            "revenue_t0": 0.0,
            "revenue_t1": 0.0,
            "revenue_t2": 0.0,
            "revenue_t3": 0.0,
        }

        values = group["orders"].to_numpy()
        revenues = group["net_revenue"].to_numpy()

        start = 4 - len(values)
        for i, value in enumerate(values):
            row[f"orders_t{start+i}"] = float(value)
        for i, value in enumerate(revenues):
            row[f"revenue_t{start+i}"] = float(value)

        recent_rows.append(row)

    recent = pd.DataFrame(recent_rows)

    snapshot = snapshot.merge(
        recent,
        on="Customer ID",
        how="left",
    )

    return snapshot


# =============================================================================
# SEGMENT / COHORT ENRICHMENT
# =============================================================================


def enrich_snapshot(
    snapshot: pd.DataFrame,
    segments: Optional[pd.DataFrame],
    cohorts: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = snapshot.copy()

    if cohorts is not None:
        cohort = cohorts.copy()
        cohort["Customer ID"] = pd.to_numeric(
            cohort["Customer ID"],
            errors="coerce",
        )
        out["Customer ID"] = pd.to_numeric(
            out["Customer ID"],
            errors="coerce",
        )
        out = out.merge(
            cohort,
            on="Customer ID",
            how="left",
            suffixes=("", "_external"),
        )
        if "cohort_month_external" in out.columns:
            out["cohort_month"] = out[
                "cohort_month_external"
            ].fillna(
                out["cohort_month"]
            )
            out = out.drop(
                columns=[
                    "cohort_month_external"
                ]
            )

    if segments is not None:
        seg = segments.copy()
        seg["Customer ID"] = pd.to_numeric(
            seg["Customer ID"],
            errors="coerce",
        )
        out = out.merge(
            seg,
            on="Customer ID",
            how="left",
            suffixes=("", "_segment")
        )

        for column in [
            "segment",
            "segment_name",
            "segment_confidence",
        ]:
            segment_column = f"{column}_segment"
            if segment_column in out.columns:
                if column not in out.columns:
                    out[column] = out[
                        segment_column
                    ]
                else:
                    out[column] = out[column].fillna(
                        out[segment_column]
                    )
                out = out.drop(
                    columns=[segment_column]
                )

    if "segment_name" not in out.columns:
        if "segment" in out.columns:
            out["segment_name"] = out[
                "segment"
            ].apply(
                lambda x: (
                    "Noise / low-density"
                    if pd.isna(x) or x == -1
                    else f"Segment_{int(x):02d}"
                )
            )
        else:
            out["segment_name"] = "Not integrated"

    out["cohort_month"] = pd.to_datetime(
        out["cohort_month"],
        errors="coerce",
    )

    return out


# =============================================================================
# FUTURE SIMULATION
# =============================================================================


def feature_frame_from_state(
    state: Dict[str, np.ndarray],
    month_index: int,
    reference_month_number: np.ndarray,
) -> np.ndarray:
    age = state["age_month"]
    recency = state["recency_months"]
    lifetime_orders = state["lifetime_orders"]
    lifetime_active = state["lifetime_active_months"]
    lifetime_net = state["lifetime_net_revenue"]
    lifetime_gross = state["lifetime_gross_revenue"]
    lifetime_returns = state["lifetime_return_value"]
    orders_last_3 = state["orders_last_3m"]
    orders_prev_3 = state["orders_prev_3m"]
    orders_3m_mean = state["orders_3m_mean"]
    net_last_3 = state["net_revenue_last_3m"]
    net_prev_3 = state["net_revenue_prev_3m"]
    net_3m_mean = state["net_revenue_3m_mean"]

    order_rate = lifetime_orders / np.maximum(age + 1.0, 1.0)
    active_share = lifetime_active / np.maximum(age + 1.0, 1.0)
    recent_order_intensity = orders_last_3 / np.maximum(age, 1.0)
    recent_value_momentum = net_last_3 / (np.abs(net_prev_3) + 10.0)
    historical_value_per_order = lifetime_net / np.maximum(lifetime_orders, 1.0)
    historical_return_ratio = lifetime_returns / np.maximum(
        lifetime_gross,
        1e-9,
    )

    month_number = (
        reference_month_number
        + month_index
    )

    radians = 2.0 * math.pi * month_number / 12.0

    return np.column_stack(
        [
            age,
            recency,
            lifetime_orders,
            lifetime_active,
            order_rate,
            active_share,
            orders_last_3,
            orders_prev_3,
            orders_3m_mean,
            recent_order_intensity,
            net_last_3,
            net_prev_3,
            net_3m_mean,
            recent_value_momentum,
            lifetime_net,
            historical_value_per_order,
            historical_return_ratio,
            np.sin(radians),
            np.cos(radians),
        ]
    )


def simulate_clv(
    snapshot: pd.DataFrame,
    purchase_model: object,
    spend_model: object,
    prior_alpha: float,
    prior_beta: float,
    transformed_sigma: float,
    horizon_months: int,
    simulations: int,
    annual_discount_rate: float,
    margin_rate: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    if simulations < 20:
        raise ValueError("simulations must be >= 20")

    rng = np.random.default_rng(seed)

    n_customers = len(snapshot)
    n_paths = simulations

    customer_ids = snapshot["Customer ID"].to_numpy()

    # Replicate customer state across paths.
    def tiled(column: str) -> np.ndarray:
        values = snapshot[column].fillna(0.0).to_numpy(dtype=float)
        return np.repeat(
            values,
            n_paths,
        )

    age = tiled("age_month")
    recency = tiled("recency_months")
    lifetime_orders = tiled("lifetime_orders")
    lifetime_active = tiled("lifetime_active_months")
    lifetime_net = tiled("lifetime_net_revenue")
    lifetime_gross = tiled("lifetime_gross_revenue")
    lifetime_returns = tiled("lifetime_return_value")
    orders_last_3 = tiled("orders_last_3m")
    orders_prev_3 = tiled("orders_prev_3m")
    net_last_3 = tiled("net_revenue_last_3m")
    net_prev_3 = tiled("net_revenue_prev_3m")
    net_3m_mean = tiled("net_revenue_3m_mean")
    orders_3m_mean = tiled("orders_3m_mean")

    # Recent ring: current, t-1, t-2.
    orders_r0 = tiled("orders_t0")
    orders_r1 = tiled("orders_t1")
    orders_r2 = tiled("orders_t2")
    orders_r3 = tiled("orders_t3")
    revenue_r0 = tiled("revenue_t0")
    revenue_r1 = tiled("revenue_t1")
    revenue_r2 = tiled("revenue_t2")
    revenue_r3 = tiled("revenue_t3")

    customer_history_exposure = np.maximum(
        snapshot["age_month"].fillna(0).to_numpy(dtype=float) + 1.0,
        1.0,
    )

    customer_history_active = np.maximum(
        snapshot["lifetime_active_months"].fillna(0).to_numpy(dtype=float),
        0.0,
    )

    # Repeat each history statistic for simulation paths.
    alpha = np.repeat(
        prior_alpha + customer_history_active,
        n_paths,
    )
    beta = np.repeat(
        prior_beta
        + np.maximum(
            customer_history_exposure
            - customer_history_active,
            0.0,
        ),
        n_paths,
    )

    reference_month_number = np.repeat(
        snapshot["last_model_month"].dt.month.to_numpy(dtype=float),
        n_paths,
    )

    annual_discount_rate = max(
        0.0,
        annual_discount_rate,
    )

    monthly_discount_rate = (
        (1.0 + annual_discount_rate) ** (1.0 / 12.0)
        - 1.0
    )

    # Simulated discounted net revenue.
    path_clv = np.zeros(
        len(snapshot) * n_paths,
        dtype=float,
    )

    monthly_mean_rows = []

    state = {
        "age_month": age,
        "recency_months": recency,
        "lifetime_orders": lifetime_orders,
        "lifetime_active_months": lifetime_active,
        "lifetime_net_revenue": lifetime_net,
        "lifetime_gross_revenue": lifetime_gross,
        "lifetime_return_value": lifetime_returns,
        "orders_last_3m": orders_last_3,
        "orders_prev_3m": orders_prev_3,
        "net_revenue_last_3m": net_last_3,
        "net_revenue_prev_3m": net_prev_3,
        "net_revenue_3m_mean": net_3m_mean,
        "orders_3m_mean": orders_3m_mean,
        "historical_value_per_order": lifetime_net / np.maximum(lifetime_orders, 1.0),
    }

    # CLV snapshots at common decision horizons.
    requested_checkpoints = sorted(
        set(
            [
                3,
                6,
                12,
                24,
                horizon_months,
            ]
        )
    )
    requested_checkpoints = [
        x
        for x in requested_checkpoints
        if x <= horizon_months
    ]

    checkpoint_samples = {}

    for future_month in range(
        1,
        horizon_months + 1,
    ):
        # Predict the state at the current horizon month.
        X_future = feature_frame_from_state(
            state,
            month_index=future_month,
            reference_month_number=reference_month_number,
        )

        p_ml = purchase_model.predict_proba(
            X_future
        )[:, 1]

        history_weight = np.clip(
            state["lifetime_active_months"]
            / (
                state["lifetime_active_months"]
                + 3.0
            ),
            0.15,
            0.90,
        )

        p_eb = alpha / np.maximum(
            alpha + beta,
            1e-12,
        )

        # Dynamic shrinkage: sparse customers lean more heavily on the
        # empirical-Bayes prior; data-rich customers lean toward ML propensity.
        p = (
            history_weight * p_ml
            + (1.0 - history_weight) * p_eb
        )

        p = np.clip(
            p,
            0.002,
            0.995,
        )

        active_draw = (
            rng.random(
                len(p)
            )
            < p
        )

        spend_pred_transformed = spend_model.predict(
            X_future
        )

        spend_noise = rng.normal(
            loc=0.0,
            scale=transformed_sigma,
            size=len(spend_pred_transformed),
        )

        spend_realized = signed_expm1(
            spend_pred_transformed
            + spend_noise
        )

        # Conservative tail clipping at approximately customer-history level.
        # It limits pathological simulation explosions while retaining asymmetric
        # net revenue / return outcomes.
        history_aov = np.maximum(
            np.abs(
                state["historical_value_per_order"]
            ),
            1.0,
        )

        spend_cap = np.maximum(
            12.0 * history_aov,
            500.0,
        )

        spend_realized = np.clip(
            spend_realized,
            -spend_cap,
            spend_cap,
        )

        realized_net = np.where(
            active_draw,
            spend_realized,
            0.0,
        )

        discount_factor = (
            1.0
            / (
                1.0
                + monthly_discount_rate
            ) ** future_month
        )

        path_clv += (
            realized_net
            * margin_rate
            * discount_factor
        )

        # Update dynamic state.
        previous_active = active_draw.astype(float)

        state["age_month"] += 1.0
        state["recency_months"] = np.where(
            active_draw,
            0.0,
            state["recency_months"] + 1.0,
        )

        state["lifetime_orders"] += previous_active
        state["lifetime_active_months"] += previous_active
        state["lifetime_net_revenue"] += realized_net
        state["lifetime_gross_revenue"] += np.maximum(
            realized_net,
            0.0,
        )
        state["lifetime_return_value"] += np.maximum(
            -realized_net,
            0.0,
        )

        # Advance four-month rings so the simulated state matches the
        # training-time definitions of current-3m vs previous-3m behavior.
        orders_r3 = orders_r2.copy()
        orders_r2 = orders_r1.copy()
        orders_r1 = orders_r0.copy()
        orders_r0 = previous_active

        revenue_r3 = revenue_r2.copy()
        revenue_r2 = revenue_r1.copy()
        revenue_r1 = revenue_r0.copy()
        revenue_r0 = realized_net

        state["orders_last_3m"] = (
            orders_r0
            + orders_r1
            + orders_r2
        )
        state["orders_prev_3m"] = (
            orders_r1
            + orders_r2
            + orders_r3
        )
        state["net_revenue_last_3m"] = (
            revenue_r0
            + revenue_r1
            + revenue_r2
        )
        state["net_revenue_prev_3m"] = (
            revenue_r1
            + revenue_r2
            + revenue_r3
        )
        state["orders_3m_mean"] = (
            state["orders_last_3m"]
            / 3.0
        )
        state["net_revenue_3m_mean"] = (
            state["net_revenue_last_3m"]
            / 3.0
        )

        # Posterior update per simulated path.
        alpha += previous_active
        beta += 1.0 - previous_active

        # Monthly population expected / quantile path distribution.
        monthly_mean_rows.append(
            {
                "future_month": future_month,
                "expected_monthly_net_revenue": float(
                    np.mean(realized_net)
                    * margin_rate
                    * discount_factor
                ),
                "purchase_probability_mean": float(
                    np.mean(p)
                ),
            }
        )

        if future_month in requested_checkpoints:
            # Store one CLV path vector for checkpoint.
            checkpoint_samples[
                future_month
            ] = path_clv.copy()

    # Aggregate simulation distribution by customer.
    result_rows = []

    for i, customer_id in enumerate(customer_ids):
        start = i * n_paths
        end = (i + 1) * n_paths

        values = path_clv[
            start:end
        ]

        row = {
            "Customer ID": customer_id,
            "clv_mean": float(np.mean(values)),
            "clv_p10": float(np.quantile(values, 0.10)),
            "clv_median": float(np.quantile(values, 0.50)),
            "clv_p90": float(np.quantile(values, 0.90)),
            "clv_std": float(np.std(values)),
            "probability_positive_clv": float(np.mean(values > 0)),
        }

        for checkpoint, samples in checkpoint_samples.items():
            cvalues = samples[start:end]
            row[f"clv_{checkpoint}m_mean"] = float(
                np.mean(cvalues)
            )
            row[f"clv_{checkpoint}m_p10"] = float(
                np.quantile(cvalues, 0.10)
            )
            row[f"clv_{checkpoint}m_p90"] = float(
                np.quantile(cvalues, 0.90)
            )

        result_rows.append(row)

    clv_df = pd.DataFrame(
        result_rows
    )

    monthly_df = pd.DataFrame(
        monthly_mean_rows
    )

    return (
        clv_df,
        monthly_df,
    )


# =============================================================================
# BENCHMARKS
# =============================================================================


def add_clv_benchmarks(
    clv: pd.DataFrame,
) -> pd.DataFrame:
    out = clv.copy()

    # CLV deciles: 1 is lowest, 10 is highest.
    out["clv_decile"] = pd.qcut(
        out["clv_mean"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    out["clv_decile"] = (
        out["clv_decile"]
        .astype("float")
        + 1
    )

    out["clv_rank_percentile"] = (
        out["clv_mean"]
        .rank(
            method="average",
            pct=True,
        )
    )

    return out


def segment_summary(
    clv: pd.DataFrame,
) -> pd.DataFrame:
    if "segment_name" not in clv.columns:
        return pd.DataFrame()

    grouped = (
        clv
        .groupby(
            "segment_name",
            dropna=False,
        )
        .agg(
            customers=("Customer ID", "nunique"),
            mean_clv=("clv_mean", "mean"),
            median_clv=("clv_median", "median"),
            p10_clv=("clv_p10", "mean"),
            p90_clv=("clv_p90", "mean"),
            positive_clv_share=("probability_positive_clv", "mean"),
            current_lifetime_value=("lifetime_net_revenue", "mean"),
            lifetime_orders=("lifetime_orders", "mean"),
            recency_months=("recency_months", "mean"),
        )
        .reset_index()
        .sort_values(
            "mean_clv",
            ascending=False,
        )
    )

    grouped["customer_share"] = (
        grouped["customers"]
        / grouped["customers"].sum()
    )

    return grouped


def cohort_summary(
    clv: pd.DataFrame,
) -> pd.DataFrame:
    if "cohort_month" not in clv.columns:
        return pd.DataFrame()

    grouped = (
        clv
        .groupby(
            "cohort_month",
            dropna=False,
        )
        .agg(
            customers=("Customer ID", "nunique"),
            mean_clv=("clv_mean", "mean"),
            median_clv=("clv_median", "median"),
            p10_clv=("clv_p10", "mean"),
            p90_clv=("clv_p90", "mean"),
            positive_clv_share=("probability_positive_clv", "mean"),
            mean_lifetime_value=("lifetime_net_revenue", "mean"),
            mean_recency_months=("recency_months", "mean"),
            mean_lifetime_orders=("lifetime_orders", "mean"),
        )
        .reset_index()
        .sort_values(
            "cohort_month"
        )
    )

    grouped["cohort_share"] = (
        grouped["customers"]
        / grouped["customers"].sum()
    )

    return grouped


# =============================================================================
# DIAGNOSTIC VISUALS
# =============================================================================


def save_diagnostic_plots(
    output_dir: Path,
    clv: pd.DataFrame,
    monthly: pd.DataFrame,
    purchase_validation: Dict[str, np.ndarray],
    importance: pd.DataFrame,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 01. CLV distribution.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    values = clv["clv_mean"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    ax.hist(
        values,
        bins=60,
        alpha=0.82,
    )

    ax.axvline(
        values.median(),
        linestyle="--",
        linewidth=2,
        label=f"Median = {values.median():,.0f}",
    )

    ax.set_title(
        "Forecasted Discounted Net-CLV Distribution"
    )
    ax.set_xlabel("Expected customer CLV")
    ax.set_ylabel("Customers")
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()
    fig.savefig(
        plots_dir / "01_clv_distribution.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 02. Segment CLV boxplot.
    # -------------------------------------------------------------------------
    if "segment_name" in clv.columns:
        groups = []
        labels = []

        segment_order = (
            clv.groupby("segment_name")["clv_mean"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        for name in segment_order:
            values = clv.loc[
                clv["segment_name"] == name,
                "clv_mean",
            ].dropna().to_numpy()
            if len(values):
                groups.append(values)
                labels.append(name)

        if groups:
            fig, ax = plt.subplots(
                figsize=(11, 7)
            )
            ax.boxplot(
                groups,
                labels=labels,
                showfliers=False,
            )
            ax.set_title(
                "Forecasted CLV by Existing Behavioral Segment"
            )
            ax.set_ylabel("Expected CLV")
            ax.tick_params(
                axis="x",
                rotation=35,
            )
            ax.grid(
                axis="y",
                alpha=0.15,
            )
            fig.tight_layout()
            fig.savefig(
                plots_dir / "02_clv_by_segment.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # 03. Cohort x CLV heatmap.
    # -------------------------------------------------------------------------
    if "cohort_month" in clv.columns:
        cohort = (
            clv.groupby("cohort_month")["clv_mean"]
            .median()
            .sort_index()
        )

        if len(cohort):
            fig, ax = plt.subplots(
                figsize=(11, max(5, len(cohort) * 0.32))
            )

            arr = cohort.to_numpy(dtype=float).reshape(-1, 1)
            image = ax.imshow(
                arr,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_yticks(
                np.arange(len(cohort))
            )
            ax.set_yticklabels(
                [
                    pd.Timestamp(x).strftime("%Y-%m")
                    for x in cohort.index
                ]
            )
            ax.set_xticks([0])
            ax.set_xticklabels(["Median 24M CLV"])
            ax.set_title(
                "Median Forecasted CLV by Acquisition Cohort"
            )

            fig.colorbar(
                image,
                ax=ax,
                shrink=0.8,
                label="Forecasted CLV",
            )

            fig.tight_layout()
            fig.savefig(
                plots_dir / "03_clv_by_cohort.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # 04. Future revenue curve.
    # -------------------------------------------------------------------------
    if not monthly.empty:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        cumulative = monthly[
            "expected_monthly_net_revenue"
        ].cumsum()

        ax.plot(
            monthly["future_month"],
            cumulative,
            linewidth=2.5,
        )

        ax.set_title(
            "Projected Cumulative Discounted Net Revenue"
        )
        ax.set_xlabel("Future month")
        ax.set_ylabel("Projected discounted net revenue")
        ax.grid(alpha=0.15)

        fig.tight_layout()
        fig.savefig(
            plots_dir / "04_projected_cumulative_revenue.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 05. Purchase-model calibration.
    # -------------------------------------------------------------------------
    actual = purchase_validation[
        "actual_purchase"
    ]
    prob = purchase_validation[
        "predicted_purchase_probability"
    ]

    if len(actual):
        bins = np.linspace(
            0.0,
            1.0,
            11,
        )

        centers = []
        observed = []
        counts = []

        for left, right in zip(
            bins[:-1],
            bins[1:],
        ):
            mask = (
                (prob >= left)
                & (prob < right)
            )

            if mask.sum() == 0:
                continue

            centers.append(
                float(prob[mask].mean())
            )
            observed.append(
                float(actual[mask].mean())
            )
            counts.append(
                int(mask.sum())
            )

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Perfect calibration",
        )

        ax.plot(
            centers,
            observed,
            marker="o",
            linewidth=2,
            label="Observed",
        )

        ax.set_xlabel("Predicted next-month purchase probability")
        ax.set_ylabel("Observed purchase rate")
        ax.set_title(
            "Next-Month Purchase Probability Calibration"
        )
        ax.legend()
        ax.grid(alpha=0.15)

        fig.tight_layout()
        fig.savefig(
            plots_dir / "05_purchase_probability_calibration.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 06. Feature importance.
    # -------------------------------------------------------------------------
    if not importance.empty:
        for model_name in importance["model"].unique():
            sub = (
                importance[
                    importance["model"] == model_name
                ]
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .head(12)
                .sort_values(
                    "importance"
                )
            )

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            ax.barh(
                sub["feature"],
                sub["importance"],
            )

            ax.set_title(
                f"Top Predictive Features — {model_name.replace('_', ' ').title()}"
            )
            ax.set_xlabel("Permutation importance")
            ax.grid(
                axis="x",
                alpha=0.15,
            )

            fig.tight_layout()
            fig.savefig(
                plots_dir
                / f"06_importance_{model_name}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # 07. CLV uncertainty.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    temp = clv.sort_values(
        "clv_mean"
    ).reset_index(drop=True)

    x = np.arange(
        len(temp)
    )

    # Plot a sampled percentile envelope to avoid giant dense rendering.
    if len(temp) > 1500:
        sample = np.linspace(
            0,
            len(temp) - 1,
            1500,
        ).astype(int)
        temp = temp.iloc[
            sample
        ].copy()
        x = np.arange(
            len(temp)
        )

    ax.fill_between(
        x,
        temp["clv_p10"],
        temp["clv_p90"],
        alpha=0.22,
        label="P10–P90",
    )

    ax.plot(
        x,
        temp["clv_median"],
        linewidth=2,
        label="Median",
    )

    ax.plot(
        x,
        temp["clv_mean"],
        linewidth=1.5,
        linestyle="--",
        label="Mean",
    )

    ax.set_title(
        "Customer-Level CLV Uncertainty Envelope"
    )
    ax.set_xlabel("Customers sorted by expected CLV")
    ax.set_ylabel("Forecasted discounted net CLV")
    ax.legend()
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "07_clv_uncertainty_envelope.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 08. CLV vs recency.
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)

    if len(clv) > 8000:
        idx = rng.choice(
            len(clv),
            8000,
            replace=False,
        )
        sample = clv.iloc[idx]
    else:
        sample = clv

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    scatter_kwargs = {
        "s": 14,
        "alpha": 0.28,
    }

    if "segment" in sample.columns:
        scatter_kwargs["c"] = sample["segment"].fillna(-1)

    transformed_clv = (
        np.sign(sample["clv_mean"])
        * np.log1p(np.abs(sample["clv_mean"]))
    )

    ax.scatter(
        transformed_clv,
        sample["recency_months"],
        **scatter_kwargs,
    )

    ax.set_xlabel("Transformed expected CLV")
    ax.set_ylabel("Observed months since last purchase")
    ax.set_title(
        "Forecasted CLV vs Current Recency State"
    )
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "08_clv_vs_recency.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# EXECUTIVE PRIORITIZATION OUTPUT
# =============================================================================


def create_action_bands(
    clv: pd.DataFrame,
) -> pd.DataFrame:
    out = clv.copy()

    # These are descriptive operating bands, not statistical claims.
    # They are deliberately based on CLV uncertainty and current recency.
    out["value_recency_band"] = np.select(
        [
            (
                out["clv_rank_percentile"] >= 0.75
            )
            & (
                out["recency_months"] <= 2
            ),
            (
                out["clv_rank_percentile"] >= 0.75
            )
            & (
                out["recency_months"] > 2
            ),
            (
                out["clv_rank_percentile"] >= 0.50
            )
            & (
                out["recency_months"] <= 3
            ),
        ],
        [
            "high_value_currently_active",
            "high_value_at_risk_by_recency",
            "mid_value_currently_active",
        ],
        default="lower_or_lapsed_value_state",
    )

    out["uncertainty_band_width"] = (
        out["clv_p90"]
        - out["clv_p10"]
    )

    out["relative_uncertainty"] = (
        out["uncertainty_band_width"]
        / (
            np.abs(out["clv_mean"])
            + 1.0
        )
    )

    return out


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    input_path = (
        Path(args.input)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    project_dir = Path(__file__).resolve().parent

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not 0.0 <= args.margin_rate <= 1.0:
        raise ValueError(
            "margin-rate must be between 0 and 1."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir / "plots").mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------------------

    raw = load_input(
        input_path,
        args.sheet,
    )

    tx = clean_transactions(
        raw
    )

    raw = None

    # -------------------------------------------------------------------------
    # 2. CUSTOMER-MONTH FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    (
        panel,
        derived_cohorts,
        observation_start,
        observation_end,
    ) = build_customer_month_panel(
        tx
    )

    customer_count = panel.select(
        "Customer ID"
    ).n_unique()

    if customer_count < args.min_customers:
        raise ValueError(
            f"Only {customer_count:,} customers found; need at least "
            f"{args.min_customers:,}."
        )

    # -------------------------------------------------------------------------
    # 3. OPTIONAL INTEGRATION WITH PREVIOUS PROJECTS
    # -------------------------------------------------------------------------

    segments = load_optional_segments(
        project_dir
    )

    external_cohorts = load_optional_cohorts(
        project_dir
    )

    # -------------------------------------------------------------------------
    # 4. LEAKAGE-SAFE ML MODELING
    # -------------------------------------------------------------------------

    (
        purchase_model,
        spend_model,
        model_metrics,
        importance,
        validation_predictions,
        transformed_sigma,
    ) = train_models(
        panel,
        validation_months=args.validation_months,
        min_active_history_months=args.min_active_history_months,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # 5. EMPIRICAL-BAYES PROPENSITY PRIOR
    # -------------------------------------------------------------------------

    prior_alpha, prior_beta = estimate_beta_prior(
        panel
    )

    LOGGER.info(
        "Empirical-Bayes purchase prior: alpha=%.4f | beta=%.4f | mean=%.4f",
        prior_alpha,
        prior_beta,
        prior_alpha / (prior_alpha + prior_beta),
    )

    # -------------------------------------------------------------------------
    # 6. SNAPSHOT STATE AT FORECAST ORIGIN
    # -------------------------------------------------------------------------

    snapshot = build_customer_snapshot(
        panel,
        tx,
    )

    snapshot = enrich_snapshot(
        snapshot,
        segments,
        external_cohorts,
    )

    # Development cap, deterministic with seed.
    if (
        args.random_customer_cap > 0
        and len(snapshot) > args.random_customer_cap
    ):
        snapshot = (
            snapshot
            .sample(
                args.random_customer_cap,
                random_state=args.seed,
            )
            .reset_index(drop=True)
        )
        LOGGER.info(
            "Development customer cap active: %s customers",
            f"{len(snapshot):,}",
        )

    # -------------------------------------------------------------------------
    # 7. MONTE-CARLO DYNAMIC CLV
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Simulating %s months x %s paths for %s customers...",
        args.horizon_months,
        f"{args.simulations:,}",
        f"{len(snapshot):,}",
    )

    (
        clv,
        monthly_projection,
    ) = simulate_clv(
        snapshot=snapshot,
        purchase_model=purchase_model,
        spend_model=spend_model,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        transformed_sigma=transformed_sigma,
        horizon_months=args.horizon_months,
        simulations=args.simulations,
        annual_discount_rate=args.annual_discount_rate,
        margin_rate=args.margin_rate,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # 8. ENRICH CLV OUTPUT WITH OBSERVED CUSTOMER STATE
    # -------------------------------------------------------------------------

    base_columns = [
        "Customer ID",
        "cohort_month",
        "age_month",
        "recency_months",
        "lifetime_orders",
        "lifetime_active_months",
        "lifetime_net_revenue",
        "lifetime_gross_revenue",
        "lifetime_return_value",
        "historical_value_per_order",
        "historical_return_ratio",
        "orders_last_3m",
        "orders_prev_3m",
        "net_revenue_last_3m",
        "net_revenue_prev_3m",
        "last_month_active",
        "segment",
        "segment_name",
        "segment_confidence",
    ]

    base_columns = [
        c
        for c in base_columns
        if c in snapshot.columns
    ]

    clv = (
        snapshot[base_columns]
        .merge(
            clv,
            on="Customer ID",
            how="inner",
        )
    )

    clv = add_clv_benchmarks(
        clv
    )

    clv = create_action_bands(
        clv
    )

    # -------------------------------------------------------------------------
    # 9. SUMMARIES
    # -------------------------------------------------------------------------

    segment = segment_summary(
        clv
    )

    cohort = cohort_summary(
        clv
    )

    # -------------------------------------------------------------------------
    # 10. EXPORT CORE TABLES
    # -------------------------------------------------------------------------

    clv.to_csv(
        output_dir / "customer_clv.csv",
        index=False,
    )

    try:
        pl.from_pandas(
            clv,
        ).write_parquet(
            output_dir / "customer_clv.parquet"
        )
    except Exception as exc:
        LOGGER.warning(
            "Parquet CLV export failed: %s",
            exc,
        )

    segment.to_csv(
        output_dir / "clv_by_segment.csv",
        index=False,
    )

    cohort.to_csv(
        output_dir / "clv_by_cohort.csv",
        index=False,
    )

    monthly_projection.to_csv(
        output_dir / "future_monthly_clv_projection.csv",
        index=False,
    )

    importance.to_csv(
        output_dir / "model_feature_importance.csv",
        index=False,
    )

    # Customer-month modeling data is useful for auditability / downstream BI.
    panel.write_csv(
        output_dir / "customer_month_clv_model_panel.csv"
    )

    # -------------------------------------------------------------------------
    # 11. VALIDATION PREDICTIONS
    # -------------------------------------------------------------------------

    validation_df = pd.DataFrame(
        {
            "calendar_month": validation_predictions[
                "calendar_month"
            ],
            "actual_purchase": validation_predictions[
                "actual_purchase"
            ],
            "predicted_purchase_probability": validation_predictions[
                "predicted_purchase_probability"
            ],
        }
    )

    validation_df.to_csv(
        output_dir / "purchase_model_validation_predictions.csv",
        index=False,
    )

    spend_actual = validation_predictions[
        "actual_spend"
    ]
    spend_pred = validation_predictions[
        "predicted_spend"
    ]

    if len(spend_actual):
        pd.DataFrame(
            {
                "actual_next_month_net_revenue": spend_actual,
                "predicted_next_month_net_revenue": spend_pred,
            }
        ).to_csv(
            output_dir / "spend_model_validation_predictions.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # 12. VISUALS
    # -------------------------------------------------------------------------

    save_diagnostic_plots(
        output_dir,
        clv,
        monthly_projection,
        validation_predictions,
        importance,
    )

    # -------------------------------------------------------------------------
    # 13. SUMMARY STATISTICS
    # -------------------------------------------------------------------------

    horizon_column = (
        f"clv_{args.horizon_months}m_mean"
    )

    total_expected_clv = float(
        clv["clv_mean"].sum()
    )

    summary = {
        "input_file": str(input_path),
        "observation_start": str(observation_start),
        "observation_end": str(observation_end),
        "customers_scored": int(len(clv)),
        "clv_definition": (
            "discounted future NET REVENUE; margin-rate converts it to a contribution proxy"
        ),
        "horizon_months": int(args.horizon_months),
        "simulations_per_customer": int(args.simulations),
        "annual_discount_rate": float(args.annual_discount_rate),
        "margin_rate": float(args.margin_rate),
        "empirical_bayes_prior": {
            "alpha": float(prior_alpha),
            "beta": float(prior_beta),
            "prior_mean_purchase_rate": float(
                prior_alpha
                / (prior_alpha + prior_beta)
            ),
        },
        "model_metrics": model_metrics,
        "clv_distribution": {
            "mean": float(clv["clv_mean"].mean()),
            "median": float(clv["clv_median"].median()),
            "p10_mean": float(clv["clv_p10"].mean()),
            "p90_mean": float(clv["clv_p90"].mean()),
            "positive_clv_probability_mean": float(
                clv["probability_positive_clv"].mean()
            ),
            "total_expected_customer_clv": total_expected_clv,
        },
        "top_1_percent_expected_clv": float(
            clv["clv_mean"].quantile(0.99)
        ),
        "top_10_percent_expected_clv": float(
            clv["clv_mean"].quantile(0.90)
        ),
        "customer_value_state_counts": (
            clv["value_recency_band"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "integrations": {
            "segmentation_integrated": segments is not None,
            "cohort_analysis_integrated": external_cohorts is not None,
        },
        "horizon_column": horizon_column if horizon_column in clv.columns else None,
        "seed": int(args.seed),
    }

    with open(
        output_dir / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------------------
    # 14. MODEL / RUN MANIFEST
    # -------------------------------------------------------------------------

    manifest = {
        "outputs": [
            "customer_clv.csv",
            "customer_clv.parquet",
            "clv_by_segment.csv",
            "clv_by_cohort.csv",
            "future_monthly_clv_projection.csv",
            "customer_month_clv_model_panel.csv",
            "model_feature_importance.csv",
            "purchase_model_validation_predictions.csv",
            "spend_model_validation_predictions.csv",
            "run_summary.json",
            "plots/01_clv_distribution.png",
            "plots/02_clv_by_segment.png",
            "plots/03_clv_by_cohort.png",
            "plots/04_projected_cumulative_revenue.png",
            "plots/05_purchase_probability_calibration.png",
            "plots/06_importance_purchase_probability.png",
            "plots/06_importance_conditional_spend.png",
            "plots/07_clv_uncertainty_envelope.png",
            "plots/08_clv_vs_recency.png",
        ]
    }

    with open(
        output_dir / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    LOGGER.info(
        "CLV analysis complete: %s",
        output_dir,
    )

    LOGGER.info(
        "Expected customer portfolio CLV: %,.2f",
        total_expected_clv,
    )


if __name__ == "__main__":
    main()
