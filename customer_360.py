#!/usr/bin/env python3
"""
Online Retail II — production-style Customer 360 feature mart.

Purpose
-------
Create a reusable, auditable customer-level data product that becomes the
single customer state table for the project's downstream:

    customer segmentation -> cohort analysis -> CLV -> retention -> decisioning

The script is intentionally more than a feature dump. It provides:

1. Polars-first transaction cleaning and feature engineering.
2. Transaction, invoice, customer-month, product-affinity, temporal,
   lifecycle, return and country behavior features.
3. Trailing-window dynamics (1/3/6/12 month) and acceleration signals.
4. Recency/cadence state and deterministic lifecycle-state labeling.
5. Optional enrichment from the sibling segmentation/cohort/CLV pipelines.
6. Data-quality assertions and coverage diagnostics.
7. CSV + Parquet + JSON outputs suitable for BI and ML.
8. Visual diagnostics for portfolio / executive presentation.

This file is designed to live next to:
    customer_segmentation.py
    cohort_analysis.py
    clv_analysis.py

Example:
    python customer_360.py

Or explicitly:
    python customer_360.py \
        --input /home/lptop/Documents/coding/marketing_science/data_xslx/online_retail_II.xlsx \
        --output-dir /home/lptop/Documents/coding/marketing_science/customer_360_output

Dependencies:
    pip install polars fastexcel pandas matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl


SEED = 42
np.random.seed(SEED)


# =============================================================================
# LOGGING
# =============================================================================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("customer_360")
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
# CONFIG / CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Build a reusable Online Retail II Customer 360 feature mart."
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
            / "customer_360_output"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. If omitted, all non-empty sheets are read.",
    )

    parser.add_argument(
        "--windows",
        default="1,3,6,12",
        help="Comma-separated trailing windows in months.",
    )

    parser.add_argument(
        "--min-orders-for-dynamics",
        type=int,
        default=2,
        help="Minimum historical orders for some dynamic state metrics.",
    )

    parser.add_argument(
        "--stale-after-months",
        type=int,
        default=6,
        help="Inactive months after which an active customer is labeled stale.",
    )

    parser.add_argument(
        "--dormant-after-months",
        type=int,
        default=12,
        help="Inactive months after which a customer is labeled dormant.",
    )

    parser.add_argument(
        "--reactivation-gap-months",
        type=int,
        default=1,
        help="Number of inactive months needed before activity qualifies as reactivation.",
    )

    parser.add_argument(
        "--plot-customer-cap",
        type=int,
        default=10000,
        help="Maximum customers sampled for dense customer scatter plots.",
    )

    parser.add_argument(
        "--top-products",
        type=int,
        default=10,
        help="Number of top customer-product affinity features to materialize.",
    )

    return parser.parse_args()


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def safe_divide(
    numerator: pl.Expr,
    denominator: pl.Expr,
    floor: float = 1e-9,
) -> pl.Expr:
    return (
        pl.when(denominator.abs() > floor)
        .then(numerator / denominator)
        .otherwise(0.0)
    )


def month_id_expr(column: str) -> pl.Expr:
    return (
        pl.col(column).dt.year() * 12
        + pl.col(column).dt.month()
    )


def entropy_from_counts(
    df: pl.DataFrame,
    key: str,
    bucket: str,
    output_name: str,
    normalization: float,
) -> pl.DataFrame:
    counts = (
        df.group_by([key, bucket])
        .len("n")
        .with_columns(
            (
                pl.col("n")
                / pl.col("n").sum().over(key)
            ).alias("p")
        )
    )

    return (
        counts
        .group_by(key)
        .agg(
            (
                -(
                    pl.col("p")
                    * pl.col("p").log()
                ).sum()
                / math.log(normalization)
            ).alias(output_name)
        )
    )


def ensure_float_columns(
    df: pl.DataFrame,
    columns: Sequence[str],
) -> pl.DataFrame:
    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing:
        return df

    return df.with_columns(
        [
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias(column)
            for column in existing
        ]
    )


# =============================================================================
# INPUT
# =============================================================================


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    aliases = {
        "invoiceno": "Invoice",
        "invoice": "Invoice",
        "invoice no": "Invoice",
        "stockcode": "StockCode",
        "stock code": "StockCode",
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
            f"Missing required columns: {missing}. Found: {df.columns}"
        )

    if "Description" not in df.columns:
        df = df.with_columns(
            pl.lit("").alias("Description")
        )

    if "Country" not in df.columns:
        df = df.with_columns(
            pl.lit("Unknown").alias("Country")
        )

    return df


def load_input(
    path: Path,
    sheet: Optional[str] = None,
) -> pl.DataFrame:
    LOGGER.info(
        "Loading %s",
        path,
    )

    suffix = path.suffix.lower()

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
                "Polars Excel reader failed (%s). Falling back to pandas/openpyxl.",
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
            infer_schema_length=20_000,
            try_parse_dates=True,
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
# TRANSACTION CLEANING
# =============================================================================


def clean_transactions(
    df: pl.DataFrame,
) -> pl.DataFrame:
    raw_rows = df.height

    out = df.with_columns(
        [
            pl.col("Invoice")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("Invoice"),

            pl.col("StockCode")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("StockCode"),

            pl.col("Description")
            .cast(pl.Utf8)
            .fill_null("")
            .str.strip_chars()
            .alias("Description"),

            pl.col("Country")
            .cast(pl.Utf8)
            .fill_null("Unknown")
            .str.strip_chars()
            .alias("Country"),

            pl.col("Quantity")
            .cast(pl.Float64, strict=False)
            .alias("Quantity"),

            pl.col("Price")
            .cast(pl.Float64, strict=False)
            .alias("Price"),

            pl.col("Customer ID")
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias("Customer ID"),
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
            .alias("InvoiceDate")
        )
    else:
        out = out.with_columns(
            pl.col("InvoiceDate")
            .cast(pl.Datetime, strict=False)
            .alias("InvoiceDate")
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
            (pl.col("Quantity") < 0).alias("is_negative_quantity"),
            (pl.col("Price") > 0).alias("is_positive_price"),

            (
                pl.col("Quantity")
                * pl.col("Price")
            ).alias("line_value"),
        ]
    )

    out = out.with_columns(
        [
            (
                pl.col("is_positive_quantity")
                & ~pl.col("is_cancellation_invoice")
                & pl.col("is_positive_price")
            ).alias("is_clean_sale"),

            (
                pl.col("is_negative_quantity")
                | pl.col("is_cancellation_invoice")
            ).alias("is_return_or_cancellation"),
        ]
    )

    out = out.with_columns(
        [
            pl.when(pl.col("is_clean_sale"))
            .then(pl.col("line_value"))
            .otherwise(0.0)
            .alias("gross_sale_value"),

            pl.when(pl.col("is_return_or_cancellation"))
            .then(pl.col("line_value").abs())
            .otherwise(0.0)
            .alias("return_value"),

            pl.when(pl.col("is_return_or_cancellation"))
            .then(pl.col("Quantity").abs())
            .otherwise(0.0)
            .alias("return_units"),
        ]
    )

    LOGGER.info(
        "Raw rows=%s | rows after base cleaning=%s | clean sale lines=%s",
        f"{raw_rows:,}",
        f"{out.height:,}",
        f"{out.filter(pl.col('is_clean_sale')).height:,}",
    )

    return out


# =============================================================================
# PRODUCT-LEVEL PRECOMPUTATION
# =============================================================================


def build_global_product_stats(
    sales: pl.DataFrame,
    top_products: int,
) -> Tuple[pl.DataFrame, List[str]]:
    product = (
        sales
        .group_by("StockCode")
        .agg(
            [
                pl.col("gross_sale_value")
                .sum()
                .alias("product_revenue"),

                pl.col("Customer ID")
                .n_unique()
                .alias("product_customers"),

                pl.col("Invoice")
                .n_unique()
                .alias("product_invoices"),

                pl.col("Quantity")
                .sum()
                .alias("product_units"),
            ]
        )
        .sort(
            "product_revenue",
            descending=True,
        )
    )

    top_codes = (
        product
        .head(top_products)
        .get_column("StockCode")
        .to_list()
    )

    product = product.with_columns(
        safe_divide(
            pl.col("product_revenue"),
            pl.col("product_customers"),
        ).alias("product_revenue_per_customer"),

        safe_divide(
            pl.col("product_revenue"),
            pl.col("product_invoices"),
        ).alias("product_revenue_per_invoice"),
    )

    return product, top_codes


# =============================================================================
# CUSTOMER-MONTH EVENT LAYER
# =============================================================================


def build_customer_month(
    tx: pl.DataFrame,
    reactivation_gap_months: int = 1,
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    Dict[str, str],
]:
    sales = tx.filter(
        pl.col("is_clean_sale")
    )

    returns = tx.filter(
        pl.col("is_return_or_cancellation")
    )

    if sales.height == 0:
        raise ValueError(
            "No clean positive sales found."
        )

    # -------------------------------------------------------------------------
    # Invoice layer
    # -------------------------------------------------------------------------
    invoice = (
        sales
        .group_by(
            ["Customer ID", "Invoice"]
        )
        .agg(
            [
                pl.col("InvoiceDate")
                .min()
                .alias("invoice_date"),

                pl.col("gross_sale_value")
                .sum()
                .alias("invoice_revenue"),

                pl.col("Quantity")
                .sum()
                .alias("invoice_units"),

                pl.len()
                .alias("invoice_lines"),

                pl.col("StockCode")
                .n_unique()
                .alias("invoice_unique_products"),
            ]
        )
        .with_columns(
            pl.col("invoice_date")
            .dt.truncate("1mo")
            .alias("calendar_month"),
        )
    )

    # -------------------------------------------------------------------------
    # Sales month
    # -------------------------------------------------------------------------
    monthly_sales = (
        invoice
        .group_by(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
        .agg(
            [
                pl.len().alias("orders"),

                pl.col("invoice_revenue")
                .sum()
                .alias("gross_revenue"),

                pl.col("invoice_units")
                .sum()
                .alias("units"),

                pl.col("invoice_lines")
                .sum()
                .alias("lines"),

                pl.col("invoice_unique_products")
                .sum()
                .alias("product_breadth_events"),

                pl.col("invoice_revenue")
                .median()
                .alias("median_order_value"),
            ]
        )
        .with_columns(
            pl.lit(0.0).alias("return_value"),
            pl.lit(0.0).alias("return_units"),
            pl.lit(1).alias("active"),
        )
    )

    # -------------------------------------------------------------------------
    # Returns month. Return-only months are retained.
    # -------------------------------------------------------------------------
    if returns.height:
        monthly_returns = (
            returns
            .with_columns(
                pl.col("InvoiceDate")
                .dt.truncate("1mo")
                .alias("calendar_month"),
            )
            .group_by(
                [
                    "Customer ID",
                    "calendar_month",
                ]
            )
            .agg(
                [
                    pl.col("return_value")
                    .sum()
                    .alias("return_value"),

                    pl.col("return_units")
                    .sum()
                    .alias("return_units"),
                ]
            )
            .with_columns(
                pl.lit(0).alias("orders"),
                pl.lit(0.0).alias("gross_revenue"),
                pl.lit(0.0).alias("units"),
                pl.lit(0.0).alias("lines"),
                pl.lit(0.0).alias("product_breadth_events"),
                pl.lit(0.0).alias("median_order_value"),
                pl.lit(0).alias("active"),
            )
        )
    else:
        monthly_returns = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "calendar_month": pl.Datetime,
                "return_value": pl.Float64,
                "return_units": pl.Float64,
                "orders": pl.Int64,
                "gross_revenue": pl.Float64,
                "units": pl.Float64,
                "lines": pl.Float64,
                "product_breadth_events": pl.Float64,
                "median_order_value": pl.Float64,
                "active": pl.Int64,
            }
        )

    customer_month = (
        pl.concat(
            [
                monthly_sales,
                monthly_returns,
            ],
            how="diagonal_relaxed",
        )
        .group_by(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
        .agg(
            [
                pl.col("orders")
                .sum()
                .alias("orders"),

                pl.col("gross_revenue")
                .sum()
                .alias("gross_revenue"),

                pl.col("units")
                .sum()
                .alias("units"),

                pl.col("lines")
                .sum()
                .alias("lines"),

                pl.col("product_breadth_events")
                .sum()
                .alias("product_breadth_events"),

                pl.col("median_order_value")
                .mean()
                .fill_null(0.0)
                .alias("median_order_value"),

                pl.col("return_value")
                .sum()
                .alias("return_value"),

                pl.col("return_units")
                .sum()
                .alias("return_units"),

                pl.col("active")
                .max()
                .alias("active"),
            ]
        )
        .sort(
            ["Customer ID", "calendar_month"]
        )
    )

    customer_month = customer_month.with_columns(
        [
            (
                pl.col("gross_revenue")
                - pl.col("return_value")
            ).alias("net_revenue"),

            safe_divide(
                pl.col("return_value"),
                pl.col("gross_revenue"),
            ).alias("return_to_gross_ratio"),

            safe_divide(
                pl.col("gross_revenue"),
                pl.col("orders"),
            ).alias("aov"),
        ]
    )

    # -------------------------------------------------------------------------
    # First observed purchase and cohort.
    # -------------------------------------------------------------------------
    first_purchase = (
        customer_month
        .filter(
            pl.col("active") == 1
        )
        .group_by("Customer ID")
        .agg(
            pl.col("calendar_month")
            .min()
            .alias("first_purchase_month")
        )
        .with_columns(
            pl.col("first_purchase_month")
            .alias("cohort_month")
        )
    )

    observation_start = (
        customer_month
        .select(
            pl.col("calendar_month").min()
        )
        .item()
    )

    observation_end = (
        customer_month
        .select(
            pl.col("calendar_month").max()
        )
        .item()
    )

    # -------------------------------------------------------------------------
    # Dense monthly panel: customer x calendar month after acquisition.
    # This is intentionally dense so inactive months become explicit states.
    # -------------------------------------------------------------------------
    calendar_count = (
        (
            observation_end.year
            - observation_start.year
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
        for i in range(calendar_count)
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

    customers = (
        first_purchase
        .select(
            [
                "Customer ID",
                "cohort_month",
            ]
        )
        .unique()
    )

    dense = (
        customers
        .join(
            calendar,
            how="cross",
        )
        .filter(
            pl.col("calendar_month")
            >= pl.col("cohort_month")
        )
        .join(
            customer_month,
            on=[
                "Customer ID",
                "calendar_month",
            ],
            how="left",
        )
        .with_columns(
            [
                pl.col("orders")
                .fill_null(0)
                .cast(pl.Float64),

                pl.col("gross_revenue")
                .fill_null(0.0),

                pl.col("net_revenue")
                .fill_null(0.0),

                pl.col("units")
                .fill_null(0.0),

                pl.col("lines")
                .fill_null(0.0),

                pl.col("product_breadth_events")
                .fill_null(0.0),

                pl.col("median_order_value")
                .fill_null(0.0),

                pl.col("return_value")
                .fill_null(0.0),

                pl.col("return_units")
                .fill_null(0.0),

                pl.col("active")
                .fill_null(0)
                .cast(pl.Int8),
            ]
        )
        .with_columns(
            [
                month_id_expr("calendar_month")
                .alias("calendar_month_id"),

                month_id_expr("cohort_month")
                .alias("cohort_month_id"),
            ]
        )
        .with_columns(
            (
                pl.col("calendar_month_id")
                - pl.col("cohort_month_id")
            ).alias("age_month"),
        )
        .sort(
            ["Customer ID", "calendar_month"]
        )
    )

    # -------------------------------------------------------------------------
    # Monthly state transitions.
    # -------------------------------------------------------------------------
    dense = dense.with_columns(
        [
            pl.col("active")
            .shift(1)
            .over("Customer ID")
            .fill_null(0)
            .alias("previous_month_active"),

            pl.col("active")
            .shift(2)
            .over("Customer ID")
            .fill_null(0)
            .alias("two_months_ago_active"),
        ]
    ).with_columns(
        [
            pl.when(
                (pl.col("active") == 1)
                & (pl.col("previous_month_active") == 0)
                & (pl.col("age_month") > reactivation_gap_months)
            )
            .then(1)
            .otherwise(0)
            .alias("reactivation_event"),

            pl.when(
                (pl.col("active") == 0)
                & (pl.col("previous_month_active") == 1)
            )
            .then(1)
            .otherwise(0)
            .alias("churn_transition"),
        ]
    )

    return (
        customer_month,
        dense,
        {
            "observation_start": str(observation_start),
            "observation_end": str(observation_end),
            "calendar_months": str(calendar_count),
        },
    )


# =============================================================================
# CUSTOMER-LEVEL BEHAVIORAL FEATURE ENGINEERING
# =============================================================================


def build_customer_features(
    tx: pl.DataFrame,
    dense: pl.DataFrame,
    top_product_codes: Sequence[str],
    windows: Sequence[int],
    stale_after_months: int,
    dormant_after_months: int,
    min_orders_for_dynamics: int,
) -> Tuple[
    pl.DataFrame,
    Dict[str, List[str]],
]:
    sales = tx.filter(
        pl.col("is_clean_sale")
    )

    observation_end = (
        dense
        .select(
            pl.col("calendar_month").max()
        )
        .item()
    )

    # -------------------------------------------------------------------------
    # Overall transaction / invoice behavior.
    # -------------------------------------------------------------------------
    invoice = (
        sales
        .group_by(
            ["Customer ID", "Invoice"]
        )
        .agg(
            [
                pl.col("InvoiceDate")
                .min()
                .alias("invoice_date"),

                pl.col("gross_sale_value")
                .sum()
                .alias("invoice_revenue"),

                pl.col("Quantity")
                .sum()
                .alias("invoice_units"),

                pl.len()
                .alias("invoice_lines"),

                pl.col("StockCode")
                .n_unique()
                .alias("invoice_unique_products"),
            ]
        )
        .sort(
            ["Customer ID", "invoice_date"]
        )
    )

    invoice = invoice.with_columns(
        [
            pl.col("invoice_date")
            .diff()
            .over("Customer ID")
            .dt.total_seconds()
            .fill_null(0.0)
            .alias("interpurchase_days"),

            pl.col("invoice_date")
            .dt.weekday()
            .alias("invoice_weekday"),

            pl.col("invoice_date")
            .dt.hour()
            .alias("invoice_hour"),
        ]
    )

    customer_base = (
        invoice
        .group_by("Customer ID")
        .agg(
            [
                pl.len().alias("lifetime_orders"),

                pl.col("invoice_date")
                .min()
                .alias("first_purchase_date"),

                pl.col("invoice_date")
                .max()
                .alias("last_purchase_date"),

                pl.col("invoice_revenue")
                .sum()
                .alias("lifetime_gross_revenue"),

                pl.col("invoice_revenue")
                .mean()
                .alias("historical_avg_order_value"),

                pl.col("invoice_revenue")
                .median()
                .alias("historical_median_order_value"),

                pl.col("invoice_revenue")
                .std()
                .fill_null(0.0)
                .alias("historical_order_value_std"),

                pl.col("invoice_units")
                .sum()
                .alias("lifetime_units"),

                pl.col("invoice_lines")
                .sum()
                .alias("lifetime_invoice_lines"),

                pl.col("invoice_unique_products")
                .sum()
                .alias("lifetime_product_line_events"),

                pl.col("interpurchase_days")
                .filter(
                    pl.col("interpurchase_days") > 0
                )
                .median()
                .alias("median_interpurchase_days"),

                pl.col("interpurchase_days")
                .filter(
                    pl.col("interpurchase_days") > 0
                )
                .mean()
                .alias("mean_interpurchase_days"),

                pl.col("interpurchase_days")
                .filter(
                    pl.col("interpurchase_days") > 0
                )
                .std()
                .fill_null(0.0)
                .alias("std_interpurchase_days"),

                pl.col("invoice_unique_products")
                .mean()
                .alias("avg_products_per_order"),

                pl.col("invoice_lines")
                .mean()
                .alias("avg_lines_per_order"),

                pl.col("invoice_units")
                .mean()
                .alias("avg_units_per_order"),

                (
                    pl.col("invoice_weekday") >= 6
                )
                .mean()
                .alias("weekend_order_share"),

                (
                    (pl.col("invoice_hour") >= 9)
                    & (pl.col("invoice_hour") < 18)
                )
                .mean()
                .alias("business_hour_order_share"),
            ]
        )
        .with_columns(
            [
                safe_divide(
                    pl.col("historical_order_value_std"),
                    pl.col("historical_avg_order_value"),
                ).alias("order_value_cv"),

                safe_divide(
                    pl.col("std_interpurchase_days"),
                    pl.col("mean_interpurchase_days"),
                ).alias("interpurchase_cv"),

                (
                    pl.col("last_purchase_date")
                    - pl.col("first_purchase_date")
                )
                .dt.total_days()
                .fill_null(0.0)
                .alias("tenure_days"),

                (
                    pl.lit(
                        observation_end
                    )
                    - pl.col("last_purchase_date")
                )
                .dt.total_days()
                .fill_null(0.0)
                .alias("recency_days"),

                pl.col("first_purchase_date")
                .dt.truncate("1mo")
                .alias("cohort_month"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("recency_days")
                    / 30.4375
                ).alias("recency_months"),

                safe_divide(
                    pl.col("lifetime_orders"),
                    (
                        pl.col("tenure_days") / 30.4375
                    ).clip(lower_bound=1.0),
                ).alias("orders_per_active_tenure_month"),

                safe_divide(
                    pl.col("lifetime_gross_revenue"),
                    pl.col("lifetime_orders"),
                ).alias("aov_check"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Returns.
    # -------------------------------------------------------------------------
    returns = tx.filter(
        pl.col("is_return_or_cancellation")
    )

    if returns.height:
        return_features = (
            returns
            .group_by("Customer ID")
            .agg(
                [
                    pl.col("return_value")
                    .sum()
                    .alias("lifetime_return_value"),

                    pl.col("return_units")
                    .sum()
                    .alias("lifetime_return_units"),

                    pl.col("Invoice")
                    .n_unique()
                    .alias("return_invoice_count"),

                    pl.len()
                    .alias("return_line_count"),
                ]
            )
        )
    else:
        return_features = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "lifetime_return_value": pl.Float64,
                "lifetime_return_units": pl.Float64,
                "return_invoice_count": pl.Float64,
                "return_line_count": pl.Float64,
            }
        )

    # -------------------------------------------------------------------------
    # Product breadth and concentration.
    # -------------------------------------------------------------------------
    customer_product = (
        sales
        .group_by(
            ["Customer ID", "StockCode"]
        )
        .agg(
            [
                pl.col("gross_sale_value")
                .sum()
                .alias("product_revenue"),

                pl.col("Quantity")
                .sum()
                .alias("product_units"),

                pl.col("Invoice")
                .n_unique()
                .alias("product_order_count"),
            ]
        )
        .with_columns(
            safe_divide(
                pl.col("product_revenue"),
                pl.col("product_revenue")
                .sum()
                .over("Customer ID"),
            ).alias("product_revenue_share")
        )
    )

    product_features = (
        customer_product
        .group_by("Customer ID")
        .agg(
            [
                pl.col("StockCode")
                .n_unique()
                .alias("unique_products"),

                pl.col("product_revenue_share")
                .pow(2)
                .sum()
                .alias("product_revenue_hhi"),

                pl.col("product_order_count")
                .filter(
                    pl.col("product_order_count") > 1
                )
                .count()
                .alias("repeat_product_count"),

                pl.col("product_revenue")
                .max()
                .alias("top_product_revenue"),
            ]
        )
        .with_columns(
            safe_divide(
                pl.col("repeat_product_count"),
                pl.col("unique_products"),
            ).alias("repeat_product_ratio")
        )
    )

    # -------------------------------------------------------------------------
    # Price behavior.
    # -------------------------------------------------------------------------
    price_features = (
        sales
        .group_by("Customer ID")
        .agg(
            [
                pl.col("Price")
                .mean()
                .alias("mean_unit_price"),

                pl.col("Price")
                .median()
                .alias("median_unit_price"),

                (
                    pl.col("Price").quantile(0.75)
                    - pl.col("Price").quantile(0.25)
                )
                .alias("unit_price_iqr"),

                pl.col("Price")
                .std()
                .fill_null(0.0)
                .alias("unit_price_std"),

                pl.col("Price")
                .quantile(0.90)
                .alias("unit_price_p90"),

                (
                    pl.col("Price") <= 0
                )
                .mean()
                .alias("nonpositive_price_share"),
            ]
        )
        .with_columns(
            safe_divide(
                pl.col("unit_price_std"),
                pl.col("mean_unit_price"),
            ).alias("unit_price_cv")
        )
    )

    global_price_p75 = (
        sales
        .select(
            pl.col("Price")
            .quantile(0.75)
        )
        .item()
    )

    premium_features = (
        sales
        .group_by("Customer ID")
        .agg(
            (
                pl.col("Price")
                > global_price_p75
            )
            .mean()
            .alias("premium_price_line_share")
        )
    )

    # -------------------------------------------------------------------------
    # Country behavior.
    # -------------------------------------------------------------------------
    country_features = (
        sales
        .group_by("Customer ID")
        .agg(
            [
                pl.col("Country")
                .n_unique()
                .alias("unique_countries"),

                pl.col("Country")
                .mode()
                .first()
                .alias("primary_country"),

                pl.col("Country")
                .first()
                .alias("first_country"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Temporal entropy / preferences.
    # -------------------------------------------------------------------------
    invoice_temporal = invoice.select(
        [
            "Customer ID",
            "invoice_date",
            "invoice_weekday",
            "invoice_hour",
        ]
    )

    temporal = (
        invoice_temporal
        .group_by("Customer ID")
        .agg(
            [
                pl.col("invoice_weekday")
                .mean()
                .alias("avg_order_weekday"),

                pl.col("invoice_hour")
                .mean()
                .alias("avg_order_hour"),

                pl.col("invoice_date")
                .dt.month()
                .n_unique()
                .alias("purchase_month_count"),
            ]
        )
    )

    temporal = temporal.join(
        entropy_from_counts(
            invoice_temporal.with_columns(
                pl.col("invoice_hour")
                .cast(pl.Int64)
                .alias("hour_bucket")
            ),
            "Customer ID",
            "hour_bucket",
            "hour_entropy",
            24.0,
        ),
        on="Customer ID",
        how="left",
    )

    temporal = temporal.join(
        entropy_from_counts(
            invoice_temporal.with_columns(
                pl.col("invoice_weekday")
                .cast(pl.Int64)
                .alias("weekday_bucket")
            ),
            "Customer ID",
            "weekday_bucket",
            "weekday_entropy",
            7.0,
        ),
        on="Customer ID",
        how="left",
    )

    temporal = temporal.join(
        entropy_from_counts(
            invoice_temporal.with_columns(
                pl.col("invoice_date")
                .dt.month()
                .cast(pl.Int64)
                .alias("month_bucket")
            ),
            "Customer ID",
            "month_bucket",
            "month_entropy",
            12.0,
        ),
        on="Customer ID",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Windowed customer dynamics.
    # -------------------------------------------------------------------------
    dynamic = build_dynamic_features(
        dense=dense,
        windows=windows,
        min_orders_for_dynamics=min_orders_for_dynamics,
    )

    # -------------------------------------------------------------------------
    # Reactivation / lifecycle events.
    # -------------------------------------------------------------------------
    transitions = (
        dense
        .group_by("Customer ID")
        .agg(
            [
                pl.col("reactivation_event")
                .sum()
                .alias("reactivation_count"),

                pl.col("churn_transition")
                .sum()
                .alias("churn_transition_count"),

                pl.col("active")
                .sum()
                .alias("active_month_count"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Top product affinity features.
    # -------------------------------------------------------------------------
    if top_product_codes:
        top_affinity = build_top_product_affinity(
            sales=sales,
            top_product_codes=top_product_codes,
        )
    else:
        top_affinity = pl.DataFrame(
            schema={"Customer ID": pl.Int64}
        )

    # -------------------------------------------------------------------------
    # Combine the feature views.
    # -------------------------------------------------------------------------
    customer = customer_base

    for right in [
        return_features,
        product_features,
        price_features,
        premium_features,
        country_features,
        temporal,
        dynamic,
        transitions,
        top_affinity,
    ]:
        customer = customer.join(
            right,
            on="Customer ID",
            how="left",
        )

    customer = customer.with_columns(
        [
            (
                pl.col("lifetime_gross_revenue")
                - pl.col("lifetime_return_value")
            ).alias("lifetime_net_revenue"),

            safe_divide(
                pl.col("lifetime_return_value"),
                pl.col("lifetime_gross_revenue"),
            ).alias("lifetime_return_value_rate"),

            safe_divide(
                pl.col("lifetime_return_units"),
                pl.col("lifetime_units") + pl.col("lifetime_return_units"),
            ).alias("lifetime_return_unit_rate"),

            safe_divide(
                pl.col("lifetime_return_value"),
                pl.col("lifetime_gross_revenue") + pl.col("lifetime_return_value"),
            ).alias("return_share_of_total_value_flow"),

            safe_divide(
                pl.col("active_month_count"),
                (
                    pl.col("tenure_days") / 30.4375
                ).clip(lower_bound=1.0),
            ).alias("active_month_density"),

            safe_divide(
                pl.col("lifetime_gross_revenue"),
                pl.col("unique_products"),
            ).alias("revenue_per_unique_product"),

            safe_divide(
                pl.col("lifetime_gross_revenue"),
                pl.col("lifetime_units"),
            ).alias("revenue_per_unit"),

            (
                pl.col("recency_months")
                <= stale_after_months
            ).alias("within_stale_threshold"),
        ]
    )

    # -------------------------------------------------------------------------
    # Lifecycle state.
    # -------------------------------------------------------------------------
    customer = customer.with_columns(
        pl.when(
            pl.col("lifetime_orders") == 1
        )
        .then(
            pl.lit("new_single_order")
        )
        .when(
            pl.col("recency_months") <= 1
            & (pl.col("lifetime_orders") >= 2)
        )
        .then(
            pl.lit("active_repeat")
        )
        .when(
            (pl.col("recency_months") > 1)
            & (pl.col("recency_months") <= stale_after_months)
        )
        .then(
            pl.lit("at_risk")
        )
        .when(
            (pl.col("recency_months") > stale_after_months)
            & (pl.col("recency_months") <= dormant_after_months)
        )
        .then(
            pl.lit("stale")
        )
        .otherwise(
            pl.lit("dormant")
        )
        .alias("lifecycle_state")
    )

    # Re-activated customers get a separate state label without replacing
    # long-term churn/dormancy information.
    customer = customer.with_columns(
        pl.when(
            pl.col("reactivation_count") > 0
        )
        .then(
            pl.lit(True)
        )
        .otherwise(
            pl.lit(False)
        )
        .alias("has_reactivated"),

        pl.when(
            pl.col("lifetime_orders") >= min_orders_for_dynamics
            & (pl.col("interpurchase_cv") > 1.0)
        )
        .then(
            pl.lit("volatile_cadence")
        )
        .otherwise(
            pl.lit("regular_cadence")
        )
        .alias("cadence_regime"),
    )

    customer = customer.with_columns(
        pl.when(
            pl.col("has_reactivated")
            & (pl.col("recency_months") <= 3)
        )
        .then(
            pl.lit("reactivated_recently")
        )
        .otherwise(
            pl.col("lifecycle_state")
        )
        .alias("customer_state")
    )

    # -------------------------------------------------------------------------
    # Feature families for downstream documentation / model governance.
    # -------------------------------------------------------------------------
    numeric_columns = [
        column
        for column, dtype in customer.schema.items()
        if dtype.is_numeric()
    ]

    customer = ensure_float_columns(
        customer,
        [
            column
            for column in numeric_columns
            if column != "Customer ID"
        ],
    )

    feature_groups = {
        "lifecycle": [
            "first_purchase_date",
            "last_purchase_date",
            "tenure_days",
            "recency_days",
            "recency_months",
            "active_month_count",
            "active_month_density",
            "purchase_month_count",
            "lifecycle_state",
            "customer_state",
        ],
        "economic_value": [
            "lifetime_gross_revenue",
            "lifetime_net_revenue",
            "historical_avg_order_value",
            "historical_median_order_value",
            "historical_order_value_std",
            "order_value_cv",
            "revenue_per_unique_product",
            "revenue_per_unit",
        ],
        "purchase_cadence": [
            "lifetime_orders",
            "median_interpurchase_days",
            "mean_interpurchase_days",
            "std_interpurchase_days",
            "interpurchase_cv",
            "orders_per_active_tenure_month",
            "cadence_regime",
        ],
        "assortment": [
            "unique_products",
            "product_revenue_hhi",
            "repeat_product_count",
            "repeat_product_ratio",
            "avg_products_per_order",
            "lifetime_product_line_events",
        ],
        "pricing": [
            "mean_unit_price",
            "median_unit_price",
            "unit_price_iqr",
            "unit_price_std",
            "unit_price_p90",
            "unit_price_cv",
            "premium_price_line_share",
        ],
        "returns": [
            "lifetime_return_value",
            "lifetime_return_units",
            "return_invoice_count",
            "return_line_count",
            "lifetime_return_value_rate",
            "lifetime_return_unit_rate",
            "return_share_of_total_value_flow",
        ],
        "temporal_behavior": [
            "weekend_order_share",
            "business_hour_order_share",
            "avg_order_weekday",
            "avg_order_hour",
            "hour_entropy",
            "weekday_entropy",
            "month_entropy",
        ],
        "dynamics": [
            column
            for column in customer.columns
            if (
                column.startswith("orders_last_")
                or column.startswith("revenue_last_")
                or column.startswith("aov_last_")
                or column.startswith("active_months_last_")
                or column.startswith("return_rate_last_")
                or column.startswith("revenue_share_last_")
                or column.startswith("orders_acceleration_")
                or column.startswith("revenue_acceleration_")
                or column.startswith("aov_acceleration_")
            )
        ],
        "reactivation": [
            "reactivation_count",
            "churn_transition_count",
            "has_reactivated",
        ],
        "geography": [
            "primary_country",
            "first_country",
            "unique_countries",
        ],
        "product_affinity": [
            column
            for column in customer.columns
            if column.startswith("share_top_product_pool_")
        ],
    }

    # Use explicit observation date in the mart.
    customer = customer.with_columns(
        pl.lit(observation_end).alias(
            "snapshot_date"
        ),
    )

    # Sort for deterministic downstream joins.
    customer = customer.sort(
        "Customer ID"
    )

    # Ensure Customer ID stays integer.
    customer = customer.with_columns(
        pl.col("Customer ID")
        .cast(pl.Int64)
    )

    return customer, feature_groups


# =============================================================================
# DYNAMIC WINDOWS
# =============================================================================


def build_dynamic_features(
    dense: pl.DataFrame,
    windows: Sequence[int],
    min_orders_for_dynamics: int,
) -> pl.DataFrame:
    snapshot_month = (
        dense
        .select(
            pl.col("calendar_month").max()
        )
        .item()
    )

    output = dense.select(
        [
            "Customer ID",
            "calendar_month",
            "orders",
            "gross_revenue",
            "net_revenue",
            "active",
            "return_value",
        ]
    )

    expressions = []

    for window in sorted(set(windows)):
        start_month_id = (
            snapshot_month.year * 12
            + snapshot_month.month
            - window
            + 1
        )

        expressions.extend(
            [
                pl.when(
                    month_id_expr("calendar_month") >= start_month_id
                )
                .then(pl.col("orders"))
                .otherwise(0.0)
                .sum()
                .over("Customer ID")
                .alias(
                    f"orders_last_{window}m"
                ),

                pl.when(
                    month_id_expr("calendar_month") >= start_month_id
                )
                .then(pl.col("gross_revenue"))
                .otherwise(0.0)
                .sum()
                .over("Customer ID")
                .alias(
                    f"revenue_last_{window}m"
                ),

                pl.when(
                    month_id_expr("calendar_month") >= start_month_id
                )
                .then(pl.col("net_revenue"))
                .otherwise(0.0)
                .sum()
                .over("Customer ID")
                .alias(
                    f"net_revenue_last_{window}m"
                ),

                pl.when(
                    month_id_expr("calendar_month") >= start_month_id
                )
                .then(pl.col("active"))
                .otherwise(0)
                .sum()
                .over("Customer ID")
                .alias(
                    f"active_months_last_{window}m"
                ),

                pl.when(
                    month_id_expr("calendar_month") >= start_month_id
                )
                .then(pl.col("return_value"))
                .otherwise(0.0)
                .sum()
                .over("Customer ID")
                .alias(
                    f"return_value_last_{window}m"
                ),
            ]
        )

    # We only need the final snapshot row per customer after the window
    # calculations. Keeping the entire dense panel until here makes the logic
    # easy to audit and avoids fragile joins for every window.
    enriched = output.with_columns(
        expressions
    )

    final = enriched.filter(
        pl.col("calendar_month") == snapshot_month
    )

    final = final.select(
        [
            "Customer ID",
            *[
                column
                for column in final.columns
                if (
                    column.startswith("orders_last_")
                    or column.startswith("revenue_last_")
                    or column.startswith("net_revenue_last_")
                    or column.startswith("active_months_last_")
                    or column.startswith("return_value_last_")
                )
            ],
        ]
    )

    additions = []

    for window in sorted(set(windows)):
        additions.extend(
            [
                safe_divide(
                    pl.col(
                        f"revenue_last_{window}m"
                    ),
                    pl.col(
                        f"orders_last_{window}m"
                    ),
                ).alias(
                    f"aov_last_{window}m"
                ),

                safe_divide(
                    pl.col(
                        f"return_value_last_{window}m"
                    ),
                    pl.col(
                        f"revenue_last_{window}m"
                    ),
                ).alias(
                    f"return_rate_last_{window}m"
                ),

                safe_divide(
                    pl.col(
                        f"revenue_last_{window}m"
                    ),
                    pl.col(
                        f"net_revenue_last_{window}m"
                    ),
                ).alias(
                    f"gross_to_net_multiplier_last_{window}m"
                ),
            ]
        )

    final = final.with_columns(
        additions
    )

    # -------------------------------------------------------------------------
    # Momentum / acceleration across adjacent windows.
    # -------------------------------------------------------------------------
    window_set = sorted(set(windows))

    if 3 in window_set and 6 in window_set:
        final = final.with_columns(
            [
                safe_divide(
                    pl.col("orders_last_3m"),
                    pl.col("orders_last_6m") / 2.0,
                ).alias("orders_acceleration_3v6"),

                safe_divide(
                    pl.col("revenue_last_3m"),
                    pl.col("revenue_last_6m") / 2.0,
                ).alias("revenue_acceleration_3v6"),

                safe_divide(
                    pl.col("aov_last_3m"),
                    pl.col("aov_last_6m"),
                ).alias("aov_acceleration_3v6"),
            ]
        )

    if 1 in window_set and 3 in window_set:
        final = final.with_columns(
            [
                safe_divide(
                    pl.col("orders_last_1m"),
                    pl.col("orders_last_3m") / 3.0,
                ).alias("orders_acceleration_1v3"),

                safe_divide(
                    pl.col("revenue_last_1m"),
                    pl.col("revenue_last_3m") / 3.0,
                ).alias("revenue_acceleration_1v3"),

                safe_divide(
                    pl.col("aov_last_1m"),
                    pl.col("aov_last_3m"),
                ).alias("aov_acceleration_1v3"),
            ]
        )

    # Number of active windows, useful as a compact consistency indicator.
    active_window_cols = [
        f"active_months_last_{window}m"
        for window in window_set
        if f"active_months_last_{window}m" in final.columns
    ]

    if active_window_cols:
        final = final.with_columns(
            sum(
                [
                    (
                        pl.col(column) > 0
                    ).cast(pl.Int8)
                    for column in active_window_cols
                ]
            ).alias(
                "number_of_active_recent_windows"
            )
        )

    # Preserve parameter in feature output metadata without adding a fake
    # feature. This also prevents an unused CLI parameter from silently being
    # ignored during data generation.
    if min_orders_for_dynamics < 1:
        raise ValueError(
            "--min-orders-for-dynamics must be >= 1"
        )

    return final


# =============================================================================
# TOP PRODUCT AFFINITY
# =============================================================================


def build_top_product_affinity(
    sales: pl.DataFrame,
    top_product_codes: Sequence[str],
) -> pl.DataFrame:
    selected = sales.filter(
        pl.col("StockCode")
        .is_in(list(top_product_codes))
    )

    if selected.height == 0:
        return pl.DataFrame(
            schema={
                "Customer ID": pl.Int64
            }
        )

    by_product = (
        selected
        .group_by(
            [
                "Customer ID",
                "StockCode",
            ]
        )
        .agg(
            pl.col("gross_sale_value")
            .sum()
            .alias("product_revenue")
        )
        .with_columns(
            safe_divide(
                pl.col("product_revenue"),
                pl.col("product_revenue")
                .sum()
                .over("Customer ID"),
            ).alias("within_top_product_share")
        )
    )

    # Materialize wide features. Product codes are normalized so they are
    # valid and deterministic column names.
    customer_ids = (
        by_product
        .select("Customer ID")
        .unique()
    )

    pieces = [customer_ids]

    for code in top_product_codes:
        one = (
            by_product
            .filter(
                pl.col("StockCode") == code
            )
            .select(
                [
                    "Customer ID",
                    "product_revenue",
                    "within_top_product_share",
                ]
            )
            .rename(
                {
                    "product_revenue": f"revenue_product_{safe_column_token(code)}",
                    "within_top_product_share": f"share_top_product_pool_{safe_column_token(code)}",
                }
            )
        )
        pieces.append(one)

    output = pieces[0]

    for piece in pieces[1:]:
        output = output.join(
            piece,
            on="Customer ID",
            how="left",
        )

    return output


def safe_column_token(value: str) -> str:
    token = "".join(
        character
        if character.isalnum()
        else "_"
        for character in str(value)
    )
    token = token.strip("_")
    return token[:40] or "unknown"


# =============================================================================
# OPTIONAL INTEGRATION WITH THE OTHER THREE PIPELINES
# =============================================================================


def read_optional_csv(
    candidates: Sequence[Path],
    customer_id_column: str = "Customer ID",
) -> Optional[pd.DataFrame]:
    for candidate in candidates:
        if not candidate.exists():
            continue

        try:
            df = pd.read_csv(
                candidate
            )

            if customer_id_column not in df.columns:
                continue

            df[customer_id_column] = pd.to_numeric(
                df[customer_id_column],
                errors="coerce",
            )

            df = df.dropna(
                subset=[customer_id_column]
            )

            df[customer_id_column] = (
                df[customer_id_column]
                .round()
                .astype("int64")
            )

            df = df.drop_duplicates(
                subset=[customer_id_column]
            )

            LOGGER.info(
                "Optional enrichment loaded: %s",
                candidate,
            )

            return df

        except Exception as exc:
            LOGGER.warning(
                "Could not load optional output %s: %s",
                candidate,
                exc,
            )

    return None


def load_project_enrichments(
    project_dir: Path,
) -> Dict[str, Optional[pd.DataFrame]]:
    segmentation = read_optional_csv(
        [
            project_dir
            / "online_retail_segmentation"
            / "customer_segments.csv",
            project_dir
            / "customer_segmentation_output"
            / "customer_segments.csv",
            project_dir
            / "segmentation_output"
            / "customer_segments.csv",
            project_dir
            / "customer_segments.csv",
        ]
    )

    cohorts = read_optional_csv(
        [
            project_dir
            / "cohort_analysis_output"
            / "customer_acquisition_cohorts.csv",
            project_dir
            / "cohort_output"
            / "customer_acquisition_cohorts.csv",
            project_dir
            / "customer_acquisition_cohorts.csv",
        ]
    )

    clv = read_optional_csv(
        [
            project_dir
            / "clv_analysis_output"
            / "customer_clv.csv",
            project_dir
            / "customer_clv"
            / "customer_clv.csv",
            project_dir
            / "customer_clv.csv",
        ]
    )

    return {
        "segmentation": segmentation,
        "cohorts": cohorts,
        "clv": clv,
    }


def select_enrichment_columns(
    df: pd.DataFrame,
    include: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "Customer ID"
    ] + [
        column
        for column in include
        if column in df.columns
    ]

    return (
        df[columns]
        .drop_duplicates("Customer ID")
    )


def enrich_customer_360(
    customer: pl.DataFrame,
    enrichments: Dict[str, Optional[pd.DataFrame]],
) -> Tuple[
    pl.DataFrame,
    Dict[str, bool],
]:
    pd_customer = customer.to_pandas()

    availability = {}

    # Segmentation.
    segmentation = enrichments.get(
        "segmentation"
    )

    if segmentation is not None:
        segmentation = select_enrichment_columns(
            segmentation,
            [
                "segment",
                "segment_name",
                "segment_confidence",
            ],
        )
        pd_customer = pd_customer.merge(
            segmentation,
            on="Customer ID",
            how="left",
            suffixes=("", "_segmentation"),
        )
        availability["segmentation"] = True
    else:
        availability["segmentation"] = False

    # Cohort.
    cohorts = enrichments.get(
        "cohorts"
    )

    if cohorts is not None:
        cohorts = select_enrichment_columns(
            cohorts,
            [
                "cohort_month",
                "first_purchase_date",
            ],
        )

        rename = {}
        if "cohort_month" in pd_customer.columns:
            rename["cohort_month"] = "cohort_month_external"
        if "first_purchase_date" in pd_customer.columns:
            rename["first_purchase_date"] = "first_purchase_date_external"

        if rename:
            cohorts = cohorts.rename(
                columns=rename
            )

        pd_customer = pd_customer.merge(
            cohorts,
            on="Customer ID",
            how="left",
            suffixes=("", "_cohort"),
        )
        availability["cohorts"] = True
    else:
        availability["cohorts"] = False

    # CLV.
    clv = enrichments.get(
        "clv"
    )

    if clv is not None:
        clv = select_enrichment_columns(
            clv,
            [
                "clv_mean",
                "clv_median",
                "clv_p10",
                "clv_p90",
                "probability_positive_clv",
                "clv_rank_percentile",
                "value_recency_band",
                "action_band",
            ],
        )

        pd_customer = pd_customer.merge(
            clv,
            on="Customer ID",
            how="left",
            suffixes=("", "_clv"),
        )
        availability["clv"] = True
    else:
        availability["clv"] = False

    output = pl.from_pandas(
        pd_customer
    )

    return output, availability


# =============================================================================
# DATA QUALITY
# =============================================================================


def run_quality_checks(
    customer: pl.DataFrame,
    dense: pl.DataFrame,
    tx: pl.DataFrame,
) -> Dict[str, object]:
    numeric_columns = [
        column
        for column, dtype in customer.schema.items()
        if dtype.is_numeric()
        and column != "Customer ID"
    ]

    null_rates = {}

    for column in customer.columns:
        null_count = customer.select(
            pl.col(column).null_count()
        ).item()
        null_rates[column] = float(
            null_count / max(1, customer.height)
        )

    id_unique = (
        customer
        .select(
            pl.col("Customer ID")
            .n_unique()
        )
        .item()
    )

    duplicate_ids = (
        customer.height
        - id_unique
    )

    invalid_probability_columns = {}
    for column in [
        "segment_confidence",
        "probability_positive_clv",
    ]:
        if column in customer.columns:
            bad = customer.filter(
                ~pl.col(column)
                .is_between(0.0, 1.0)
            ).height
            invalid_probability_columns[column] = int(bad)

    negative_value_columns = {}
    for column in [
        "lifetime_gross_revenue",
        "lifetime_orders",
        "lifetime_units",
        "unique_products",
        "active_month_count",
    ]:
        if column in customer.columns:
            negative_value_columns[column] = int(
                customer.filter(
                    pl.col(column) < 0
                ).height
            )

    checks = {
        "customer_count": int(customer.height),
        "customer_month_rows": int(dense.height),
        "transaction_rows_after_cleaning": int(tx.height),
        "customer_id_unique": bool(
            id_unique == customer.height
        ),
        "duplicate_customer_ids": int(
            duplicate_ids
        ),
        "invalid_probability_values": invalid_probability_columns,
        "negative_core_metrics": negative_value_columns,
        "max_null_rate": float(
            max(null_rates.values())
            if null_rates
            else 0.0
        ),
        "null_rates": null_rates,
        "finite_numeric_values": True,
    }

    # Explicit finite-value audit. Polars does not provide a universal
    # isfinite aggregation across all dtypes, so inspect numeric columns.
    for column in numeric_columns:
        bad = customer.select(
            pl.col(column)
            .is_nan()
            .sum()
        ).item()
        if bad:
            checks["finite_numeric_values"] = False
            break

    if not checks["customer_id_unique"]:
        raise AssertionError(
            "Customer 360 must contain exactly one row per Customer ID."
        )

    if not checks["finite_numeric_values"]:
        raise AssertionError(
            "Customer 360 contains NaN values in numeric features."
        )

    return checks


# =============================================================================
# PROFILE TABLES
# =============================================================================


def build_360_summary(
    customer: pl.DataFrame,
    feature_groups: Dict[str, List[str]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pd_customer = customer.to_pandas()

    # Numeric distribution summary.
    feature_rows = []

    for group_name, columns in feature_groups.items():
        for column in columns:
            if column not in pd_customer.columns:
                continue

            series = pd.to_numeric(
                pd_customer[column],
                errors="coerce",
            )

            if series.notna().sum() == 0:
                continue

            # Skip boolean columns (0/1) as quantiles are not meaningful
            if series.dropna().isin([0, 1]).all() and series.nunique() <= 2:
                feature_rows.append(
                    {
                        "feature_group": group_name,
                        "feature": column,
                        "non_null_share": float(series.notna().mean()),
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "p10": None,
                        "p25": None,
                        "p75": None,
                        "p90": None,
                    }
                )
                continue

            feature_rows.append(
                {
                    "feature_group": group_name,
                    "feature": column,
                    "non_null_share": float(series.notna().mean()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "p10": float(series.quantile(0.10)),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                    "p90": float(series.quantile(0.90)),
                }
            )

    feature_summary = pd.DataFrame(
        feature_rows
    )

    # Compact portfolio KPIs.
    kpis = {
        "customers": len(pd_customer),
        "lifetime_gross_revenue": float(
            pd_customer["lifetime_gross_revenue"].sum()
        ),
        "lifetime_orders": float(
            pd_customer["lifetime_orders"].sum()
        ),
        "lifetime_return_value": float(
            pd_customer.get(
                "lifetime_return_value",
                pd.Series(dtype=float),
            ).sum()
        ),
        "median_lifetime_revenue": float(
            pd_customer["lifetime_gross_revenue"].median()
        ),
        "median_orders": float(
            pd_customer["lifetime_orders"].median()
        ),
        "median_recency_months": float(
            pd_customer["recency_months"].median()
        ),
        "reactivated_customer_share": float(
            pd_customer["has_reactivated"].mean()
            if "has_reactivated" in pd_customer.columns
            else 0.0
        ),
    }

    # If CLV has been joined, include its portfolio metrics.
    if "clv_mean" in pd_customer.columns:
        kpis["expected_clv_total"] = float(
            pd.to_numeric(
                pd_customer["clv_mean"],
                errors="coerce",
            ).fillna(0).sum()
        )
        kpis["median_clv"] = float(
            pd.to_numeric(
                pd_customer["clv_mean"],
                errors="coerce",
            ).median()
        )

    if "segment" in pd_customer.columns:
        kpis["segment_count"] = int(
            pd_customer["segment"]
            .dropna()
            .nunique()
        )

    kpi_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in kpis.items()
        ]
    )

    return feature_summary, kpi_df


# =============================================================================
# VISUALS
# =============================================================================


def sample_indices(
    n: int,
    cap: int,
) -> np.ndarray:
    if n <= cap:
        return np.arange(n)

    rng = np.random.default_rng(
        SEED
    )

    return rng.choice(
        n,
        size=cap,
        replace=False,
    )


def save_visuals(
    output_dir: Path,
    customer: pl.DataFrame,
    dense: pl.DataFrame,
    plot_customer_cap: int,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = customer.to_pandas()
    idx = sample_indices(
        len(df),
        plot_customer_cap,
    )

    # -------------------------------------------------------------------------
    # 01. Customer value vs recency
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    if "segment" in df.columns:
        c = pd.to_numeric(
            df.loc[idx, "segment"],
            errors="coerce",
        )
    else:
        c = None

    ax.scatter(
        df.loc[idx, "recency_months"],
        np.log1p(
            df.loc[idx, "lifetime_gross_revenue"]
        ),
        c=c,
        s=18,
        alpha=0.40,
    )

    ax.set_xlabel(
        "Recency (months)"
    )
    ax.set_ylabel(
        "log(1 + lifetime gross revenue)"
    )
    ax.set_title(
        "Customer 360 — Value vs Recency"
    )
    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()
    fig.savefig(
        plots_dir / "01_value_vs_recency.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 02. Cadence vs basket value
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    x = pd.to_numeric(
        df.loc[idx, "median_interpurchase_days"],
        errors="coerce",
    ).fillna(0)

    y = np.log1p(
        pd.to_numeric(
            df.loc[idx, "historical_avg_order_value"],
            errors="coerce",
        ).fillna(0)
    )

    ax.scatter(
        x,
        y,
        c=(
            pd.to_numeric(
                df.loc[idx, "lifetime_orders"],
                errors="coerce",
            ).fillna(0)
        ),
        s=18,
        alpha=0.40,
    )

    ax.set_xlabel(
        "Median interpurchase days"
    )
    ax.set_ylabel(
        "log(1 + historical AOV)"
    )
    ax.set_title(
        "Customer 360 — Cadence vs Basket Economics"
    )
    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()
    fig.savefig(
        plots_dir / "02_cadence_vs_basket.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 03. Lifecycle state distribution
    # -------------------------------------------------------------------------
    if "customer_state" in df.columns:
        counts = (
            df["customer_state"]
            .value_counts()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            counts.index.astype(str),
            counts.values,
        )

        ax.set_title(
            "Customer 360 — Lifecycle State Distribution"
        )
        ax.set_ylabel(
            "Customers"
        )
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
            plots_dir / "03_lifecycle_state_distribution.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 04. Dynamics: recent revenue vs lifetime revenue
    # -------------------------------------------------------------------------
    if "revenue_last_3m" in df.columns:
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.scatter(
            np.log1p(
                pd.to_numeric(
                    df.loc[idx, "lifetime_gross_revenue"],
                    errors="coerce",
                ).fillna(0)
            ),
            np.log1p(
                pd.to_numeric(
                    df.loc[idx, "revenue_last_3m"],
                    errors="coerce",
                ).fillna(0)
            ),
            c=pd.to_numeric(
                df.loc[idx, "revenue_acceleration_3v6"]
                if "revenue_acceleration_3v6" in df.columns
                else pd.Series(
                    1.0,
                    index=df.loc[idx].index,
                ),
                errors="coerce",
            ).fillna(0),
            s=18,
            alpha=0.42,
        )

        ax.set_xlabel(
            "log(1 + lifetime gross revenue)"
        )
        ax.set_ylabel(
            "log(1 + revenue, last 3 months)"
        )
        ax.set_title(
            "Customer 360 — Long-Term Value vs Recent Momentum"
        )
        ax.grid(
            alpha=0.15
        )

        fig.tight_layout()
        fig.savefig(
            plots_dir / "04_lifetime_vs_recent_revenue.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 05. Returns vs value
    # -------------------------------------------------------------------------
    if "lifetime_return_value_rate" in df.columns:
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.scatter(
            np.log1p(
                pd.to_numeric(
                    df.loc[idx, "lifetime_gross_revenue"],
                    errors="coerce",
                ).fillna(0)
            ),
            pd.to_numeric(
                df.loc[idx, "lifetime_return_value_rate"],
                errors="coerce",
            ).fillna(0),
            s=18,
            alpha=0.42,
        )

        ax.set_xlabel(
            "log(1 + lifetime gross revenue)"
        )
        ax.set_ylabel(
            "Return value rate"
        )
        ax.set_title(
            "Customer 360 — Commercial Value vs Return Friction"
        )
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(1.0)
        )
        ax.grid(
            alpha=0.15
        )

        fig.tight_layout()
        fig.savefig(
            plots_dir / "05_value_vs_returns.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 06. Segment x lifecycle if segmentation is present
    # -------------------------------------------------------------------------
    if "segment_name" in df.columns:
        segment_state = pd.crosstab(
            df["segment_name"].fillna("Unassigned"),
            df["customer_state"].fillna("Unknown"),
            normalize="index",
        )

        if not segment_state.empty:
            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            image = ax.imshow(
                segment_state.to_numpy(),
                aspect="auto",
            )

            ax.set_yticks(
                np.arange(
                    len(segment_state.index)
                )
            )
            ax.set_yticklabels(
                segment_state.index.astype(str)
            )
            ax.set_xticks(
                np.arange(
                    len(segment_state.columns)
                )
            )
            ax.set_xticklabels(
                segment_state.columns.astype(str),
                rotation=30,
                ha="right",
            )
            ax.set_title(
                "Customer 360 — Segment x Lifecycle-State Mix"
            )

            for i in range(
                segment_state.shape[0]
            ):
                for j in range(
                    segment_state.shape[1]
                ):
                    value = segment_state.iat[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            fig.colorbar(
                image,
                ax=ax,
                label="Share of segment",
                shrink=0.80,
            )

            fig.tight_layout()
            fig.savefig(
                plots_dir / "06_segment_lifecycle_heatmap.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # 07. Monthly customer population state
    # -------------------------------------------------------------------------
    monthly_state = (
        dense
        .group_by(
            "calendar_month"
        )
        .agg(
            [
                pl.col("Customer ID")
                .n_unique()
                .alias("customer_population"),

                pl.col("active")
                .sum()
                .alias("active_customer_months"),

                pl.col("reactivation_event")
                .sum()
                .alias("reactivations"),

                pl.col("churn_transition")
                .sum()
                .alias("churn_transitions"),
            ]
        )
        .sort(
            "calendar_month"
        )
        .to_pandas()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = monthly_state[
        "calendar_month"
    ]

    ax.plot(
        x,
        monthly_state[
            "customer_population"
        ],
        label="Customer population",
        linewidth=2,
    )

    ax.plot(
        x,
        monthly_state[
            "active_customer_months"
        ],
        label="Active customer-months",
        linewidth=2,
    )

    ax.set_title(
        "Customer 360 — Monthly Customer Base Dynamics"
    )
    ax.set_xlabel(
        "Calendar month"
    )
    ax.set_ylabel(
        "Customers / customer-months"
    )
    ax.legend(
        loc="best"
    )
    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()
    fig.savefig(
        plots_dir / "07_customer_base_dynamics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 08. CLV if available
    # -------------------------------------------------------------------------
    if "clv_mean" in df.columns:
        valid = df.loc[idx].copy()
        clv = pd.to_numeric(
            valid["clv_mean"],
            errors="coerce",
        ).fillna(0)
        revenue = pd.to_numeric(
            valid["lifetime_gross_revenue"],
            errors="coerce",
        ).fillna(0)

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.scatter(
            np.log1p(revenue),
            np.log1p(clv),
            s=18,
            alpha=0.42,
        )

        ax.set_xlabel(
            "log(1 + observed lifetime gross revenue)"
        )
        ax.set_ylabel(
            "log(1 + predicted CLV)"
        )
        ax.set_title(
            "Customer 360 — Observed Value vs Forward CLV"
        )
        ax.grid(
            alpha=0.15,
        )

        fig.tight_layout()
        fig.savefig(
            plots_dir / "08_observed_value_vs_clv.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# FEATURE DICTIONARY
# =============================================================================


def make_feature_dictionary(
    customer: pl.DataFrame,
    feature_groups: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []

    for group, columns in feature_groups.items():
        for column in columns:
            if column not in customer.columns:
                continue

            dtype = str(
                customer.schema[column]
            )

            if column in {
                "first_purchase_date",
                "last_purchase_date",
                "snapshot_date",
            }:
                description = "Customer lifecycle date."
            elif "revenue" in column:
                description = "Revenue / net value behavior."
            elif "order" in column:
                description = "Order frequency, basket or cadence behavior."
            elif "product" in column:
                description = "Assortment or product affinity behavior."
            elif "return" in column:
                description = "Returns / reversal behavior."
            elif "price" in column:
                description = "Observed transaction price behavior."
            elif "entropy" in column:
                description = "Temporal concentration / diversity measure."
            elif "acceleration" in column:
                description = "Recent momentum relative to a longer baseline."
            elif "recency" in column:
                description = "Time since latest observed purchase."
            elif "cohort" in column:
                description = "Acquisition cohort state."
            elif "country" in column:
                description = "Customer geographic behavior."
            elif "segment" in column:
                description = "Downstream clustering enrichment."
            elif "clv" in column:
                description = "Forward-looking CLV enrichment."
            else:
                description = "Customer-level behavioral feature."

            rows.append(
                {
                    "feature_group": group,
                    "feature": column,
                    "dtype": dtype,
                    "description": description,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# JSON-SAFE CONVERSION
# =============================================================================


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return value


def write_json(
    payload: Dict,
    path: Path,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=json_safe,
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

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

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    try:
        windows = sorted(
            {
                int(value.strip())
                for value in args.windows.split(",")
                if value.strip()
            }
        )
    except ValueError as exc:
        raise ValueError(
            "--windows must be a comma-separated list of positive integers."
        ) from exc

    if not windows or any(window <= 0 for window in windows):
        raise ValueError(
            "--windows must contain positive integers."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "plots").mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # 1. INGESTION + CLEANING
    # =========================================================================
    raw = load_input(
        input_path,
        args.sheet,
    )

    tx = clean_transactions(
        raw
    )

    del raw

    # =========================================================================
    # 2. GLOBAL PRODUCT FEATURES
    # =========================================================================
    sales = tx.filter(
        pl.col("is_clean_sale")
    )

    product_stats, top_product_codes = build_global_product_stats(
        sales,
        args.top_products,
    )

    # =========================================================================
    # 3. CUSTOMER-MONTH LAYER
    # =========================================================================
    (
        customer_month,
        dense,
        observation_metadata,
    ) = build_customer_month(
        tx,
        reactivation_gap_months=args.reactivation_gap_months,
    )

    # =========================================================================
    # 4. CUSTOMER FEATURE MART
    # =========================================================================
    customer, feature_groups = build_customer_features(
        tx=tx,
        dense=dense,
        top_product_codes=top_product_codes,
        windows=windows,
        stale_after_months=args.stale_after_months,
        dormant_after_months=args.dormant_after_months,
        min_orders_for_dynamics=args.min_orders_for_dynamics,
    )

    # =========================================================================
    # 5. OPTIONAL ENRICHMENT FROM SEGMENTATION / COHORT / CLV
    # =========================================================================
    project_dir = Path(__file__).resolve().parent

    enrichments = load_project_enrichments(
        project_dir
    )

    customer, enrichment_availability = enrich_customer_360(
        customer,
        enrichments,
    )

    # =========================================================================
    # 6. DATA QUALITY
    # =========================================================================
    quality = run_quality_checks(
        customer=customer,
        dense=dense,
        tx=tx,
    )

    # =========================================================================
    # 7. PROFILE / DOCUMENTATION TABLES
    # =========================================================================
    feature_summary, kpi_summary = build_360_summary(
        customer,
        feature_groups,
    )

    feature_dictionary = make_feature_dictionary(
        customer,
        feature_groups,
    )

    # =========================================================================
    # 8. EXPORT CORE DATA MART
    # =========================================================================
    customer.write_csv(
        output_dir / "customer_360.csv"
    )

    customer.write_parquet(
        output_dir / "customer_360.parquet"
    )

    customer_month.write_csv(
        output_dir / "customer_month_events.csv"
    )

    customer_month.write_parquet(
        output_dir / "customer_month_events.parquet"
    )

    dense.write_parquet(
        output_dir / "customer_month_dense.parquet"
    )

    product_stats.write_csv(
        output_dir / "product_global_statistics.csv"
    )

    feature_summary.to_csv(
        output_dir / "feature_summary.csv",
        index=False,
    )

    kpi_summary.to_csv(
        output_dir / "portfolio_kpis.csv",
        index=False,
    )

    feature_dictionary.to_csv(
        output_dir / "feature_dictionary.csv",
        index=False,
    )

    # =========================================================================
    # 9. VISUALS
    # =========================================================================
    save_visuals(
        output_dir=output_dir,
        customer=customer,
        dense=dense,
        plot_customer_cap=args.plot_customer_cap,
    )

    # =========================================================================
    # 10. RUN METADATA
    # =========================================================================
    metadata = {
        "script": "customer_360.py",
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "rows_after_cleaning": int(tx.height),
        "customers_in_customer_360": int(customer.height),
        "customer_month_rows": int(customer_month.height),
        "dense_customer_month_rows": int(dense.height),
        "top_product_codes": [str(code) for code in top_product_codes],
        "windows_months": windows,
        "observation": observation_metadata,
        "enrichment_integrated": enrichment_availability,
        "feature_groups": feature_groups,
        "data_quality": quality,
        "parameters": {
            "stale_after_months": args.stale_after_months,
            "dormant_after_months": args.dormant_after_months,
            "min_orders_for_dynamics": args.min_orders_for_dynamics,
            "reactivation_gap_months": args.reactivation_gap_months,
            "plot_customer_cap": args.plot_customer_cap,
            "top_products": args.top_products,
            "seed": SEED,
        },
    }

    write_json(
        metadata,
        output_dir / "customer_360_metadata.json",
    )

    # =========================================================================
    # 11. MANIFEST
    # =========================================================================
    manifest = {
        "outputs": [
            "customer_360.csv",
            "customer_360.parquet",
            "customer_month_events.csv",
            "customer_month_events.parquet",
            "customer_month_dense.parquet",
            "product_global_statistics.csv",
            "feature_summary.csv",
            "portfolio_kpis.csv",
            "feature_dictionary.csv",
            "customer_360_metadata.json",
            "plots/01_value_vs_recency.png",
            "plots/02_cadence_vs_basket.png",
            "plots/03_lifecycle_state_distribution.png",
            "plots/04_lifetime_vs_recent_revenue.png",
            "plots/05_value_vs_returns.png",
            "plots/06_segment_lifecycle_heatmap.png",
            "plots/07_customer_base_dynamics.png",
            "plots/08_observed_value_vs_clv.png",
        ],
        "primary_artifact": "customer_360.parquet",
        "downstream_consumers": [
            "customer_segmentation.py",
            "cohort_analysis.py",
            "clv_analysis.py",
            "future retention / next-purchase models",
            "future decision engine / next-best-action",
        ],
    }

    write_json(
        manifest,
        output_dir / "run_manifest.json",
    )

    # =========================================================================
    # 12. RUN SUMMARY
    # =========================================================================
    LOGGER.info(
        "Customer 360 complete | customers=%s | features=%s | output=%s",
        f"{customer.height:,}",
        f"{len(customer.columns):,}",
        output_dir,
    )

    LOGGER.info(
        "Integrated outputs | segmentation=%s | cohorts=%s | CLV=%s",
        enrichment_availability["segmentation"],
        enrichment_availability["cohorts"],
        enrichment_availability["clv"],
    )


if __name__ == "__main__":
    main()
