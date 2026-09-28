#!/usr/bin/env python3
"""
Online Retail II — end-to-end cohort intelligence pipeline.

This is deliberately beyond a basic cohort retention heatmap.

It builds a customer-month event model and produces:
- acquisition cohorts based on first positive purchase month
- classic logo retention by cohort age
- gross and net revenue retention
- orders/customer and units/customer
- cumulative observed customer value
- reactivation diagnostics
- return / reversal behavior
- acquisition-month quality diagnostics
- maturity-aware aggregate retention curves
- observed customer lifecycle states
- wide cohort matrices for BI / modeling
- machine-readable outputs
- publication-ready visualizations
- run metadata and diagnostics

Feature engineering is performed in Polars.
Pandas is used only at the presentation / export boundary.

Example:
    python cohort_analysis.py \
        --input /home/lptop/Documents/coding/marketing_science/data_xslx/online_retail_II.xlsx \
        --output-dir /home/lptop/Documents/coding/marketing_science/cohort_analysis_output

Dependencies:
    pip install polars fastexcel pandas matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    logger = logging.getLogger("cohort_analysis")
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
    parser = argparse.ArgumentParser(
        description="Professional cohort analysis for Online Retail II."
    )

    parser.add_argument(
        "--input",
        default=(
            "/home/lptop/Documents/coding/marketing_science/"
            "data_xslx/online_retail_II.xlsx"
        ),
        help="Input .xlsx/.xls/.csv/.parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/home/lptop/Documents/coding/marketing_science/"
            "cohort_analysis_output"
        ),
        help="Directory where outputs are written.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. If omitted, all non-empty sheets are read.",
    )

    parser.add_argument(
        "--min-cohort-size",
        type=int,
        default=25,
        help="Minimum acquisition cohort size used in scorecards.",
    )

    parser.add_argument(
        "--max-age-months",
        type=int,
        default=None,
        help="Optional maximum cohort age to report.",
    )

    parser.add_argument(
        "--early-retention-months",
        type=int,
        default=3,
        help="Early retention checkpoint.",
    )

    parser.add_argument(
        "--mature-retention-months",
        type=int,
        default=6,
        help="Mature retention checkpoint.",
    )

    parser.add_argument(
        "--heatmap-limit",
        type=int,
        default=36,
        help="Maximum age columns shown in heatmaps.",
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
            f"Missing required columns: {missing}. "
            f"Found: {df.columns}"
        )

    return df


def load_input(
    path: Path,
    sheet: str | None = None,
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

                if isinstance(
                    loaded,
                    dict,
                ):

                    frames = [
                        frame
                        for frame in loaded.values()
                        if (
                            isinstance(
                                frame,
                                pl.DataFrame,
                            )
                            and frame.height > 0
                        )
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
                "Polars Excel reader failed (%s). "
                "Falling back to pandas/openpyxl.",
                exc,
            )

            loaded = pd.read_excel(
                path,
                sheet_name=(
                    sheet
                    if sheet
                    else None
                ),
            )

            if isinstance(
                loaded,
                dict,
            ):

                loaded = pd.concat(
                    loaded.values(),
                    ignore_index=True,
                )

            df = pl.from_pandas(
                loaded
            )

    elif suffix == ".csv":

        df = pl.read_csv(
            path,
            infer_schema_length=20_000,
            try_parse_dates=True,
            ignore_errors=True,
        )

    elif suffix in {
        ".parquet",
        ".pq",
    }:

        df = pl.read_parquet(
            path
        )

    else:

        raise ValueError(
            f"Unsupported input format: {suffix}"
        )

    return normalize_columns(
        df
    )


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
            .str.strip_chars(),

            pl.col("StockCode")
            .cast(pl.Utf8)
            .str.strip_chars(),

            pl.col("Quantity")
            .cast(
                pl.Float64,
                strict=False,
            ),

            pl.col("Price")
            .cast(
                pl.Float64,
                strict=False,
            ),

            pl.col("Customer ID")
            .cast(
                pl.Float64,
                strict=False,
            )
            .round(0)
            .cast(
                pl.Int64,
                strict=False,
            ),
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
            .cast(
                pl.Datetime,
                strict=False,
            )
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
            .alias(
                "is_cancellation_invoice"
            ),

            (
                pl.col("Quantity") > 0
            ).alias(
                "is_positive_sale"
            ),

            (
                pl.col("Quantity") < 0
            ).alias(
                "is_return_line"
            ),

            (
                pl.col("Price") <= 0
            ).alias(
                "is_zero_or_negative_price"
            ),

            (
                pl.col("Quantity")
                * pl.col("Price")
            ).alias(
                "line_value"
            ),
        ]
    )

    # Conservative definition of realized positive sales:
    # positive units, non-cancellation invoice, positive unit price.
    out = out.with_columns(
        [
            (
                pl.col("is_positive_sale")
                & (
                    ~pl.col(
                        "is_cancellation_invoice"
                    )
                )
                & (
                    pl.col("Price") > 0
                )
            ).alias(
                "is_clean_sale_line"
            ),

            pl.when(
                pl.col("is_positive_sale")
                & (
                    ~pl.col(
                        "is_cancellation_invoice"
                    )
                )
                & (
                    pl.col("Price") > 0
                )
            )
            .then(
                pl.col("line_value")
            )
            .otherwise(
                0.0
            )
            .alias(
                "gross_sale_value"
            ),

            pl.when(
                pl.col("is_return_line")
                | pl.col(
                    "is_cancellation_invoice"
                )
            )
            .then(
                pl.col(
                    "line_value"
                ).abs()
            )
            .otherwise(
                0.0
            )
            .alias(
                "return_value"
            ),
        ]
    )

    LOGGER.info(
        "Raw rows=%s | retained transaction rows=%s",
        f"{raw_rows:,}",
        f"{out.height:,}",
    )

    return out


# =============================================================================
# CUSTOMER-MONTH EVENT MODEL
# =============================================================================

def build_customer_month_events(
    tx: pl.DataFrame,
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    Dict,
]:

    sales = tx.filter(
        pl.col(
            "is_clean_sale_line"
        )
    )

    returns = tx.filter(
        pl.col("is_return_line")
        | pl.col(
            "is_cancellation_invoice"
        )
    )

    if sales.height == 0:
        raise ValueError(
            "No clean positive sales were found."
        )

    # -------------------------------------------------------------------------
    # Invoice-level aggregation
    # -------------------------------------------------------------------------

    sale_invoices = (
        sales
        .group_by(
            [
                "Customer ID",
                "Invoice",
            ]
        )
        .agg(
            [
                pl.col("InvoiceDate")
                .min()
                .alias(
                    "invoice_date"
                ),

                pl.col("gross_sale_value")
                .sum()
                .alias(
                    "gross_revenue"
                ),

                pl.col("Quantity")
                .sum()
                .alias(
                    "units"
                ),

                pl.col("StockCode")
                .n_unique()
                .alias(
                    "unique_products"
                ),

                pl.col("Price")
                .median()
                .alias(
                    "median_line_price"
                ),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # First purchase / acquisition cohort
    # -------------------------------------------------------------------------

    first_purchase = (
        sale_invoices
        .group_by(
            "Customer ID"
        )
        .agg(
            pl.col("invoice_date")
            .min()
            .alias(
                "first_purchase_date"
            )
        )
        .with_columns(
            pl.col(
                "first_purchase_date"
            )
            .dt.truncate("1mo")
            .alias(
                "cohort_month"
            )
        )
    )

    # -------------------------------------------------------------------------
    # Customer-month sales events
    # -------------------------------------------------------------------------

    sales_month = (
        sale_invoices
        .with_columns(
            pl.col(
                "invoice_date"
            )
            .dt.truncate("1mo")
            .alias(
                "calendar_month"
            )
        )
        .group_by(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
        .agg(
            [
                pl.len()
                .alias(
                    "orders"
                ),

                pl.col("gross_revenue")
                .sum()
                .alias(
                    "gross_revenue"
                ),

                pl.col("units")
                .sum()
                .alias(
                    "units"
                ),

                pl.col(
                    "unique_products"
                )
                .sum()
                .alias(
                    "product_lines"
                ),
            ]
        )
        .with_columns(
            [
                pl.lit(
                    0.0
                ).alias(
                    "return_value"
                ),

                pl.lit(
                    0.0
                ).alias(
                    "return_units"
                ),

                pl.lit(
                    1
                ).alias(
                    "is_active_purchase_month"
                ),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Customer-month return events
    #
    # Return-only months are retained intentionally. Otherwise, returns can
    # disappear from the cohort economics.
    # -------------------------------------------------------------------------

    if returns.height:

        return_month = (
            returns
            .filter(
                pl.col(
                    "Customer ID"
                ).is_not_null()
            )
            .with_columns(
                pl.col("InvoiceDate")
                .dt.truncate("1mo")
                .alias(
                    "calendar_month"
                )
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
                    .alias(
                        "return_value"
                    ),

                    pl.col("Quantity")
                    .filter(
                        pl.col(
                            "Quantity"
                        ) < 0
                    )
                    .abs()
                    .sum()
                    .fill_null(0.0)
                    .alias(
                        "return_units"
                    ),
                ]
            )
            .with_columns(
                [
                    pl.lit(0)
                    .alias(
                        "orders"
                    ),

                    pl.lit(0.0)
                    .alias(
                        "gross_revenue"
                    ),

                    pl.lit(0.0)
                    .alias(
                        "units"
                    ),

                    pl.lit(0.0)
                    .alias(
                        "product_lines"
                    ),

                    pl.lit(0)
                    .alias(
                        "is_active_purchase_month"
                    ),
                ]
            )
        )

    else:

        return_month = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "calendar_month": pl.Datetime,
                "return_value": pl.Float64,
                "return_units": pl.Float64,
                "orders": pl.Int64,
                "gross_revenue": pl.Float64,
                "units": pl.Float64,
                "product_lines": pl.Float64,
                "is_active_purchase_month": pl.Int64,
            }
        )

    # -------------------------------------------------------------------------
    # Combine sales + returns into a single customer-month table
    # -------------------------------------------------------------------------

    customer_month = (
        pl.concat(
            [
                sales_month,
                return_month,
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
                .alias(
                    "orders"
                ),

                pl.col("gross_revenue")
                .sum()
                .alias(
                    "gross_revenue"
                ),

                pl.col("units")
                .sum()
                .alias(
                    "units"
                ),

                pl.col("product_lines")
                .sum()
                .alias(
                    "product_lines"
                ),

                pl.col("return_value")
                .sum()
                .alias(
                    "return_value"
                ),

                pl.col("return_units")
                .sum()
                .alias(
                    "return_units"
                ),

                pl.col(
                    "is_active_purchase_month"
                )
                .max()
                .alias(
                    "is_active_purchase_month"
                ),
            ]
        )
        .join(
            first_purchase,
            on="Customer ID",
            how="left",
        )
        .with_columns(
            [
                (
                    (
                        pl.col(
                            "calendar_month"
                        ).dt.year()
                        -
                        pl.col(
                            "cohort_month"
                        ).dt.year()
                    )
                    * 12
                    +
                    (
                        pl.col(
                            "calendar_month"
                        ).dt.month()
                        -
                        pl.col(
                            "cohort_month"
                        ).dt.month()
                    )
                ).alias(
                    "age_month"
                ),

                (
                    pl.col(
                        "calendar_month"
                    ).dt.year()
                    * 12
                    +
                    pl.col(
                        "calendar_month"
                    ).dt.month()
                ).alias(
                    "calendar_month_id"
                ),

                (
                    pl.col(
                        "gross_revenue"
                    )
                    -
                    pl.col(
                        "return_value"
                    )
                ).alias(
                    "net_revenue"
                ),

                (
                    pl.col(
                        "return_value"
                    )
                    /
                    (
                        pl.col(
                            "gross_revenue"
                        )
                        + 1e-9
                    )
                ).alias(
                    "return_to_sales_ratio"
                ),
            ]
        )
        .sort(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
    )

    # Exclude anomalous return activity that predates the first observed sale.
    customer_month = customer_month.filter(
        pl.col("age_month") >= 0
    )

    # -------------------------------------------------------------------------
    # Reactivation
    #
    # A reactivation occurs when a customer has an active purchase month after
    # at least one completely inactive month.
    # -------------------------------------------------------------------------

    active_months = (
        customer_month
        .filter(
            pl.col(
                "is_active_purchase_month"
            ) == 1
        )
        .select(
            [
                "Customer ID",
                "calendar_month",
                "age_month",
            ]
        )
        .sort(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
        .with_columns(
            pl.col("age_month")
            .shift(1)
            .over("Customer ID")
            .alias(
                "previous_active_age_month"
            )
        )
        .with_columns(
            [
                pl.when(
                    pl.col(
                        "previous_active_age_month"
                    ).is_null()
                )
                .then(
                    pl.lit(0)
                )
                .when(
                    (
                        pl.col(
                            "age_month"
                        )
                        -
                        pl.col(
                            "previous_active_age_month"
                        )
                        > 1
                    )
                )
                .then(
                    pl.lit(1)
                )
                .otherwise(
                    pl.lit(0)
                )
                .alias(
                    "is_reactivation"
                ),

                (
                    pl.col(
                        "age_month"
                    )
                    -
                    pl.col(
                        "previous_active_age_month"
                    )
                ).alias(
                    "active_month_gap"
                ),
            ]
        )
    )

    customer_month = (
        customer_month
        .join(
            active_months.select(
                [
                    "Customer ID",
                    "calendar_month",
                    "is_reactivation",
                    "active_month_gap",
                ]
            ),
            on=[
                "Customer ID",
                "calendar_month",
            ],
            how="left",
        )
        .with_columns(
            [
                pl.col(
                    "is_reactivation"
                )
                .fill_null(0),

                pl.col(
                    "active_month_gap"
                )
                .fill_null(0),
            ]
        )
        .sort(
            [
                "Customer ID",
                "calendar_month",
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Observation window
    # -------------------------------------------------------------------------

    last_observation_month = (
        sales
        .select(
            pl.col(
                "InvoiceDate"
            )
            .max()
            .dt.truncate("1mo")
        )
        .item()
    )

    first_observation_month = (
        sales
        .select(
            pl.col(
                "InvoiceDate"
            )
            .min()
            .dt.truncate("1mo")
        )
        .item()
    )

    months_observable = (
        (
            last_observation_month.year
            -
            first_observation_month.year
        )
        * 12
        +
        (
            last_observation_month.month
            -
            first_observation_month.month
        )
        + 1
    )

    data_quality = {
        "raw_rows": int(
            tx.height
        ),
        "clean_sale_lines": int(
            sales.height
        ),
        "customer_month_rows": int(
            customer_month.height
        ),
        "unique_customers": int(
            first_purchase.height
        ),
        "observation_start": str(
            first_observation_month
        ),
        "observation_end": str(
            last_observation_month
        ),
        "observable_calendar_months": int(
            months_observable
        ),
        "return_rows": int(
            returns.height
        ),
        "return_only_customer_months": int(
            customer_month
            .filter(
                (
                    pl.col(
                        "is_active_purchase_month"
                    ) == 0
                )
                &
                (
                    pl.col(
                        "return_value"
                    ) > 0
                )
            )
            .height
        ),
    }

    return (
        customer_month,
        first_purchase,
        data_quality,
    )


# =============================================================================
# COHORT GRID
# =============================================================================

def dense_age_calendar(
    customer_month: pl.DataFrame,
    last_observation_month,
) -> pl.DataFrame:

    cohort_calendar = (
        customer_month
        .select(
            "cohort_month"
        )
        .unique()
        .sort(
            "cohort_month"
        )
    )

    rows = []

    for cohort in cohort_calendar[
        "cohort_month"
    ].to_list():

        max_age = (
            (
                last_observation_month.year
                - cohort.year
            )
            * 12
            +
            (
                last_observation_month.month
                - cohort.month
            )
        )

        for age in range(
            max_age + 1
        ):

            rows.append(
                {
                    "cohort_month": cohort,
                    "age_month": age,
                }
            )

    return pl.DataFrame(
        rows
    )


def build_cohort_tables(
    customer_month: pl.DataFrame,
) -> Dict[str, pl.DataFrame]:

    last_observation_month = (
        customer_month
        .select(
            pl.col(
                "calendar_month"
            ).max()
        )
        .item()
    )

    cohort_sizes = (
        customer_month
        .filter(
            pl.col(
                "age_month"
            ) == 0
        )
        .group_by(
            "cohort_month"
        )
        .agg(
            pl.col(
                "Customer ID"
            )
            .n_unique()
            .alias(
                "cohort_customers"
            )
        )
    )

    customer_age = customer_month.select(
        [
            "Customer ID",
            "cohort_month",
            "age_month",
            "calendar_month",
            "orders",
            "gross_revenue",
            "net_revenue",
            "units",
            "return_value",
            "is_reactivation",
        ]
    )

    # -------------------------------------------------------------------------
    # Cohort x age metrics
    # -------------------------------------------------------------------------

    cohort_age = (
        customer_age
        .group_by(
            [
                "cohort_month",
                "age_month",
            ]
        )
        .agg(
            [
                pl.col(
                    "Customer ID"
                )
                .filter(
                    pl.col(
                        "orders"
                    ) > 0
                )
                .n_unique()
                .alias(
                    "active_customers"
                ),

                pl.col(
                    "orders"
                )
                .sum()
                .alias(
                    "orders"
                ),

                pl.col(
                    "gross_revenue"
                )
                .sum()
                .alias(
                    "gross_revenue"
                ),

                pl.col(
                    "net_revenue"
                )
                .sum()
                .alias(
                    "net_revenue"
                ),

                pl.col(
                    "units"
                )
                .sum()
                .alias(
                    "units"
                ),

                pl.col(
                    "return_value"
                )
                .sum()
                .alias(
                    "return_value"
                ),

                pl.col(
                    "is_reactivation"
                )
                .sum()
                .alias(
                    "reactivated_customer_months"
                ),
            ]
        )
        .join(
            cohort_sizes,
            on="cohort_month",
            how="left",
        )
        .with_columns(
            [
                (
                    pl.col(
                        "active_customers"
                    )
                    /
                    pl.col(
                        "cohort_customers"
                    )
                ).alias(
                    "logo_retention"
                ),

                (
                    pl.col(
                        "orders"
                    )
                    /
                    pl.col(
                        "cohort_customers"
                    )
                ).alias(
                    "orders_per_acquired_customer"
                ),

                (
                    pl.col(
                        "gross_revenue"
                    )
                    /
                    pl.col(
                        "cohort_customers"
                    )
                ).alias(
                    "gross_revenue_per_acquired_customer"
                ),

                (
                    pl.col(
                        "net_revenue"
                    )
                    /
                    pl.col(
                        "cohort_customers"
                    )
                ).alias(
                    "net_revenue_per_acquired_customer"
                ),

                (
                    pl.col(
                        "units"
                    )
                    /
                    pl.col(
                        "cohort_customers"
                    )
                ).alias(
                    "units_per_acquired_customer"
                ),

                (
                    pl.col(
                        "reactivated_customer_months"
                    )
                    /
                    pl.col(
                        "active_customers"
                    ).clip(
                        lower_bound=1
                    )
                ).alias(
                    "reactivation_share_of_active"
                ),
            ]
        )
        .sort(
            [
                "cohort_month",
                "age_month",
            ]
        )
    )

    # Safe overwrite in case active_customers = 0.
    cohort_age = cohort_age.with_columns(
        pl.when(
            pl.col(
                "active_customers"
            ) > 0
        )
        .then(
            pl.col(
                "reactivated_customer_months"
            )
            /
            pl.col(
                "active_customers"
            )
        )
        .otherwise(
            0.0
        )
        .alias(
            "reactivation_share_of_active"
        )
    )

    # -------------------------------------------------------------------------
    # Cohort baseline
    # -------------------------------------------------------------------------

    cohort_baseline = (
        cohort_age
        .filter(
            pl.col(
                "age_month"
            ) == 0
        )
        .select(
            [
                "cohort_month",

                pl.col(
                    "gross_revenue"
                ).alias(
                    "baseline_gross_revenue"
                ),

                pl.col(
                    "net_revenue"
                ).alias(
                    "baseline_net_revenue"
                ),

                pl.col(
                    "orders"
                ).alias(
                    "baseline_orders"
                ),

                pl.col(
                    "units"
                ).alias(
                    "baseline_units"
                ),
            ]
        )
    )

    cohort_age = (
        cohort_age
        .join(
            cohort_baseline,
            on="cohort_month",
            how="left",
        )
        .with_columns(
            [
                (
                    pl.col(
                        "gross_revenue"
                    )
                    /
                    (
                        pl.col(
                            "baseline_gross_revenue"
                        )
                        + 1e-9
                    )
                ).alias(
                    "gross_revenue_retention"
                ),

                (
                    pl.col(
                        "net_revenue"
                    )
                    /
                    (
                        pl.col(
                            "baseline_net_revenue"
                        )
                        + 1e-9
                    )
                ).alias(
                    "net_revenue_retention"
                ),

                (
                    pl.col(
                        "orders"
                    )
                    /
                    (
                        pl.col(
                            "baseline_orders"
                        )
                        + 1e-9
                    )
                ).alias(
                    "order_retention"
                ),

                (
                    pl.col(
                        "units"
                    )
                    /
                    (
                        pl.col(
                            "baseline_units"
                        )
                        + 1e-9
                    )
                ).alias(
                    "unit_retention"
                ),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Dense maturity-aware grid
    # -------------------------------------------------------------------------

    dense = dense_age_calendar(
        customer_month,
        last_observation_month,
    )

    dense = (
        dense
        .join(
            cohort_age,
            on=[
                "cohort_month",
                "age_month",
            ],
            how="left",
        )
        .join(
            cohort_sizes,
            on="cohort_month",
            how="left",
        )
        .with_columns(
            pl.lit(
                True
            ).alias(
                "age_observable"
            )
        )
        .sort(
            [
                "cohort_month",
                "age_month",
            ]
        )
    )

    numeric_defaults = [
        "active_customers",
        "orders",
        "gross_revenue",
        "net_revenue",
        "units",
        "return_value",
        "reactivated_customer_months",
        "logo_retention",
        "orders_per_acquired_customer",
        "gross_revenue_per_acquired_customer",
        "net_revenue_per_acquired_customer",
        "units_per_acquired_customer",
        "reactivation_share_of_active",
        "baseline_gross_revenue",
        "baseline_net_revenue",
        "baseline_orders",
        "baseline_units",
        "gross_revenue_retention",
        "net_revenue_retention",
        "order_retention",
        "unit_retention",
    ]

    dense = dense.with_columns(
        [
            pl.col(column)
            .fill_null(0.0)
            for column in numeric_defaults
            if column in dense.columns
        ]
    )

    # -------------------------------------------------------------------------
    # Calendar-month performance
    # -------------------------------------------------------------------------

    calendar = (
        customer_age
        .group_by(
            "calendar_month"
        )
        .agg(
            [
                pl.col(
                    "Customer ID"
                )
                .filter(
                    pl.col(
                        "orders"
                    ) > 0
                )
                .n_unique()
                .alias(
                    "active_customers"
                ),

                pl.col(
                    "orders"
                )
                .sum()
                .alias(
                    "orders"
                ),

                pl.col(
                    "gross_revenue"
                )
                .sum()
                .alias(
                    "gross_revenue"
                ),

                pl.col(
                    "net_revenue"
                )
                .sum()
                .alias(
                    "net_revenue"
                ),

                pl.col(
                    "units"
                )
                .sum()
                .alias(
                    "units"
                ),

                pl.col(
                    "return_value"
                )
                .sum()
                .alias(
                    "return_value"
                ),

                pl.col(
                    "is_reactivation"
                )
                .sum()
                .alias(
                    "reactivated_customer_months"
                ),
            ]
        )
        .sort(
            "calendar_month"
        )
        .with_columns(
            (
                pl.col(
                    "return_value"
                )
                /
                (
                    pl.col(
                        "gross_revenue"
                    )
                    + 1e-9
                )
            ).alias(
                "return_to_gross_sales_ratio"
            )
        )
    )

    return {
        "cohort_age": cohort_age,
        "dense_cohort_age": dense,
        "calendar_month": calendar,
        "cohort_sizes": cohort_sizes,
    }


# =============================================================================
# COHORT ACQUISITION QUALITY
# =============================================================================

def build_cohort_quality(
    tx: pl.DataFrame,
    customer_month: pl.DataFrame,
) -> pl.DataFrame:

    first_month = customer_month.filter(
        pl.col(
            "age_month"
        ) == 0
    )

    quality = (
        first_month
        .group_by(
            "cohort_month"
        )
        .agg(
            [
                pl.col(
                    "Customer ID"
                )
                .n_unique()
                .alias(
                    "new_customers"
                ),

                pl.col(
                    "gross_revenue"
                )
                .sum()
                .alias(
                    "acquisition_month_gross_revenue"
                ),

                pl.col(
                    "net_revenue"
                )
                .sum()
                .alias(
                    "acquisition_month_net_revenue"
                ),

                pl.col(
                    "orders"
                )
                .sum()
                .alias(
                    "acquisition_month_orders"
                ),

                pl.col(
                    "units"
                )
                .sum()
                .alias(
                    "acquisition_month_units"
                ),
            ]
        )
        .with_columns(
            [
                (
                    pl.col(
                        "acquisition_month_gross_revenue"
                    )
                    /
                    pl.col(
                        "new_customers"
                    ).clip(
                        lower_bound=1
                    )
                ).alias(
                    "gross_revenue_per_new_customer"
                ),

                (
                    pl.col(
                        "acquisition_month_net_revenue"
                    )
                    /
                    pl.col(
                        "new_customers"
                    ).clip(
                        lower_bound=1
                    )
                ).alias(
                    "net_revenue_per_new_customer"
                ),

                (
                    pl.col(
                        "acquisition_month_orders"
                    )
                    /
                    pl.col(
                        "new_customers"
                    ).clip(
                        lower_bound=1
                    )
                ).alias(
                    "orders_per_new_customer"
                ),

                (
                    pl.col(
                        "acquisition_month_units"
                    )
                    /
                    pl.col(
                        "new_customers"
                    ).clip(
                        lower_bound=1
                    )
                ).alias(
                    "units_per_new_customer"
                ),
            ]
        )
        .sort(
            "cohort_month"
        )
    )

    # Customer-level acquisition IDs.
    acquisition_customer_ids = first_month.select(
        [
            "Customer ID",
            "cohort_month",
        ]
    )

    acquisition_lines = (
        tx
        .filter(
            pl.col(
                "is_clean_sale_line"
            )
        )
        .with_columns(
            pl.col(
                "InvoiceDate"
            )
            .dt.truncate("1mo")
            .alias(
                "calendar_month"
            )
        )
        .join(
            acquisition_customer_ids,
            on="Customer ID",
            how="inner",
        )
        .filter(
            pl.col(
                "calendar_month"
            )
            ==
            pl.col(
                "cohort_month"
            )
        )
    )

    acquisition_behavior = (
        acquisition_lines
        .group_by(
            "cohort_month"
        )
        .agg(
            [
                pl.col(
                    "StockCode"
                )
                .n_unique()
                .alias(
                    "unique_products_bought_on_acquisition_month"
                ),

                pl.col(
                    "Price"
                )
                .median()
                .alias(
                    "median_unit_price_acquisition_month"
                ),

                pl.col(
                    "Price"
                )
                .mean()
                .alias(
                    "mean_unit_price_acquisition_month"
                ),

                (
                    pl.col(
                        "Price"
                    )
                    >
                    pl.col(
                        "Price"
                    ).median()
                )
                .mean()
                .alias(
                    "above_cohort_median_price_line_share"
                ),
            ]
        )
    )

    return quality.join(
        acquisition_behavior,
        on="cohort_month",
        how="left",
    )


# =============================================================================
# COHORT SCORECARD
# =============================================================================

def cohort_snapshot_scorecard(
    dense: pl.DataFrame,
    min_cohort_size: int,
    early_age: int,
    mature_age: int,
) -> pl.DataFrame:

    base = dense.filter(
        pl.col(
            "cohort_customers"
        )
        >= min_cohort_size
    )

    observations = []

    cohort_values = (
        base
        .select(
            "cohort_month"
        )
        .unique()
        .sort(
            "cohort_month"
        )[
            "cohort_month"
        ]
        .to_list()
    )

    for cohort in cohort_values:

        row = {
            "cohort_month": cohort
        }

        subset = base.filter(
            pl.col(
                "cohort_month"
            )
            == cohort
        )

        for age, prefix in [
            (0, "m0"),
            (
                early_age,
                f"m{early_age}",
            ),
            (
                mature_age,
                f"m{mature_age}",
            ),
        ]:

            age_row = subset.filter(
                pl.col(
                    "age_month"
                )
                == age
            )

            if age_row.height == 0:

                row[
                    f"{prefix}_observable"
                ] = False

                row[
                    f"{prefix}_logo_retention"
                ] = np.nan

                row[
                    f"{prefix}_net_revenue_retention"
                ] = np.nan

                row[
                    f"{prefix}_net_revenue_per_acquired_customer"
                ] = np.nan

                continue

            row[
                f"{prefix}_observable"
            ] = True

            record = age_row.row(
                0,
                named=True,
            )

            row[
                f"{prefix}_logo_retention"
            ] = float(
                record[
                    "logo_retention"
                ]
            )

            row[
                f"{prefix}_net_revenue_retention"
            ] = float(
                record[
                    "net_revenue_retention"
                ]
            )

            row[
                f"{prefix}_net_revenue_per_acquired_customer"
            ] = float(
                record[
                    "net_revenue_per_acquired_customer"
                ]
            )

        row[
            "cumulative_net_revenue_per_acquired_customer_observed"
        ] = float(
            subset
            .select(
                pl.col(
                    "net_revenue_per_acquired_customer"
                )
                .sum()
            )
            .item()
        )

        row[
            "max_observed_age_month"
        ] = int(
            subset
            .select(
                pl.col(
                    "age_month"
                ).max()
            )
            .item()
        )

        observations.append(
            row
        )

    return (
        pl.DataFrame(
            observations
        )
        .sort(
            "cohort_month"
        )
    )


# =============================================================================
# MATURITY-AWARE RETENTION CURVES
# =============================================================================

def retention_decay_table(
    dense: pl.DataFrame,
) -> pl.DataFrame:

    return (
        dense
        .group_by(
            "age_month"
        )
        .agg(
            [
                pl.col(
                    "cohort_customers"
                )
                .sum()
                .alias(
                    "cohort_population_weight"
                ),

                (
                    pl.col(
                        "active_customers"
                    ).sum()
                    /
                    pl.col(
                        "cohort_customers"
                    ).sum()
                    .clip(
                        lower_bound=1
                    )
                ).alias(
                    "weighted_logo_retention"
                ),

                (
                    pl.col(
                        "net_revenue"
                    ).sum()
                    /
                    pl.col(
                        "baseline_net_revenue"
                    ).sum()
                    .clip(
                        lower_bound=1e-9
                    )
                ).alias(
                    "weighted_net_revenue_retention"
                ),

                (
                    pl.col(
                        "orders"
                    ).sum()
                    /
                    pl.col(
                        "cohort_customers"
                    ).sum()
                    .clip(
                        lower_bound=1
                    )
                ).alias(
                    "orders_per_acquired_customer"
                ),

                (
                    pl.col(
                        "net_revenue"
                    ).sum()
                    /
                    pl.col(
                        "cohort_customers"
                    ).sum()
                    .clip(
                        lower_bound=1
                    )
                ).alias(
                    "net_revenue_per_acquired_customer"
                ),
            ]
        )
        .sort(
            "age_month"
        )
    )


# =============================================================================
# DECAY DIAGNOSTICS
# =============================================================================

def linear_decay_slope(
    ages: np.ndarray,
    retention: np.ndarray,
) -> float:

    mask = (
        np.isfinite(
            ages
        )
        &
        np.isfinite(
            retention
        )
        &
        (
            retention > 0
        )
        &
        (
            ages >= 0
        )
    )

    if mask.sum() < 3:
        return float("nan")

    x = ages[
        mask
    ]

    y = np.log(
        retention[
            mask
        ]
    )

    if np.unique(
        x
    ).size < 3:

        return float("nan")

    slope, _ = np.polyfit(
        x,
        y,
        1,
    )

    return float(
        slope
    )


def estimate_half_life(
    decay: pd.DataFrame,
) -> float:

    slope = linear_decay_slope(
        decay[
            "age_month"
        ].to_numpy(
            dtype=float
        ),
        decay[
            "weighted_logo_retention"
        ].to_numpy(
            dtype=float
        ),
    )

    if (
        not np.isfinite(
            slope
        )
        or slope >= 0
    ):
        return float("nan")

    return float(
        math.log(
            0.5
        ) / slope
    )


# =============================================================================
# VISUAL HELPERS
# =============================================================================

def heatmap(
    table: pd.DataFrame,
    title: str,
    cbar_label: str,
    output_path: Path,
    value_format: str = ".0%",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:

    if table.empty:
        return

    arr = table.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                0.43
                * table.shape[1]
                + 3,
            ),
            max(
                5,
                0.35
                * table.shape[0]
                + 2.5,
            ),
        )
    )

    image = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Months since acquisition"
    )

    ax.set_ylabel(
        "Acquisition cohort"
    )

    ax.set_xticks(
        np.arange(
            table.shape[1]
        )
    )

    ax.set_xticklabels(
        table.columns.tolist()
    )

    ax.set_yticks(
        np.arange(
            table.shape[0]
        )
    )

    ax.set_yticklabels(
        [
            pd.Timestamp(
                x
            ).strftime(
                "%Y-%m"
            )
            for x in table.index
        ]
    )

    if (
        table.shape[0] <= 25
        and table.shape[1] <= 24
    ):

        for i in range(
            table.shape[0]
        ):

            for j in range(
                table.shape[1]
            ):

                value = arr[
                    i,
                    j,
                ]

                if not np.isfinite(
                    value
                ):
                    continue

                if (
                    value_format
                    == ".0%"
                ):

                    text = (
                        f"{value:.0%}"
                    )

                elif (
                    value_format
                    == ".1f"
                ):

                    text = (
                        f"{value:.1f}"
                    )

                else:

                    text = (
                        f"{value:.2f}"
                    )

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.82,
        label=cbar_label,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def line_chart(
    x: Iterable,
    y: Iterable,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    percent: bool = False,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        alpha=0.15
    )

    if percent:
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(
                1.0
            )
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# VISUAL OUTPUT
# =============================================================================

def save_visuals(
    output_dir: Path,
    dense: pl.DataFrame,
    scorecard: pl.DataFrame,
    decay: pl.DataFrame,
    cohort_quality: pl.DataFrame,
    heatmap_limit: int,
) -> None:

    plots_dir = (
        output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dense_pd = (
        dense.to_pandas()
    )

    # -------------------------------------------------------------------------
    # 01. Logo retention
    # -------------------------------------------------------------------------

    logo_matrix = (
        dense_pd
        .pivot(
            index="cohort_month",
            columns="age_month",
            values="logo_retention",
        )
        .sort_index()
    )

    logo_matrix = logo_matrix.iloc[
        :,
        :heatmap_limit,
    ]

    heatmap(
        logo_matrix,
        "Logo Retention by Acquisition Cohort",
        "Customer retention",
        plots_dir
        / "01_logo_retention_heatmap.png",
        value_format=".0%",
        vmin=0,
        vmax=1,
    )

    # -------------------------------------------------------------------------
    # 02. Net revenue retention
    # -------------------------------------------------------------------------

    revenue_matrix = (
        dense_pd
        .pivot(
            index="cohort_month",
            columns="age_month",
            values="net_revenue_retention",
        )
        .sort_index()
    )

    revenue_matrix = revenue_matrix.iloc[
        :,
        :heatmap_limit,
    ]

    vmax = (
        max(
            1.0,
            float(
                np.nanpercentile(
                    revenue_matrix.to_numpy(),
                    95,
                )
            ),
        )
        if not revenue_matrix.empty
        else 1.0
    )

    heatmap(
        revenue_matrix,
        "Net Revenue Retention by Acquisition Cohort",
        "Revenue at age / acquisition-month revenue",
        plots_dir
        / "02_net_revenue_retention_heatmap.png",
        value_format=".0%",
        vmin=0,
        vmax=vmax,
    )

    # -------------------------------------------------------------------------
    # 03. Net revenue per acquired customer
    # -------------------------------------------------------------------------

    ltv_matrix = (
        dense_pd
        .pivot(
            index="cohort_month",
            columns="age_month",
            values="net_revenue_per_acquired_customer",
        )
        .sort_index()
    )

    ltv_matrix = ltv_matrix.iloc[
        :,
        :heatmap_limit,
    ]

    heatmap(
        ltv_matrix,
        "Monthly Net Revenue per Acquired Customer",
        "Net revenue",
        plots_dir
        / "03_revenue_per_acquired_customer_heatmap.png",
        value_format=".1f",
        vmin=0,
    )

    # -------------------------------------------------------------------------
    # 04. Logo retention maturity curve
    # -------------------------------------------------------------------------

    line_chart(
        decay[
            "age_month"
        ].to_numpy(),
        decay[
            "weighted_logo_retention"
        ].to_numpy(),
        "Maturity-Aware Logo Retention Curve",
        "Months since acquisition",
        "Weighted logo retention",
        plots_dir
        / "04_weighted_logo_retention_curve.png",
        percent=True,
    )

    # -------------------------------------------------------------------------
    # 05. Revenue retention maturity curve
    # -------------------------------------------------------------------------

    line_chart(
        decay[
            "age_month"
        ].to_numpy(),
        decay[
            "weighted_net_revenue_retention"
        ].to_numpy(),
        "Maturity-Aware Net Revenue Retention Curve",
        "Months since acquisition",
        "Weighted net revenue retention",
        plots_dir
        / "05_weighted_revenue_retention_curve.png",
        percent=True,
    )

    # -------------------------------------------------------------------------
    # 06. Acquisition quality vs observed cohort value
    # -------------------------------------------------------------------------

    quality_pd = (
        cohort_quality.to_pandas()
    )

    score_pd = (
        scorecard.to_pandas()
    )

    merged = quality_pd.merge(
        score_pd,
        on="cohort_month",
        how="left",
    )

    if not merged.empty:

        fig, ax = plt.subplots(
            figsize=(
                11,
                7,
            )
        )

        ax.scatter(
            merged[
                "gross_revenue_per_new_customer"
            ],
            merged[
                "cumulative_net_revenue_per_acquired_customer_observed"
            ],
            s=np.clip(
                merged[
                    "new_customers"
                ],
                25,
                500,
            ),
            alpha=0.55,
        )

        for _, row in merged.iterrows():

            ax.annotate(
                pd.Timestamp(
                    row[
                        "cohort_month"
                    ]
                ).strftime(
                    "%Y-%m"
                ),
                (
                    row[
                        "gross_revenue_per_new_customer"
                    ],
                    row[
                        "cumulative_net_revenue_per_acquired_customer_observed"
                    ],
                ),
                fontsize=7,
                alpha=0.8,
            )

        ax.set_title(
            "Acquisition-Month Economics vs Observed Cohort Value"
        )

        ax.set_xlabel(
            "Acquisition-month gross revenue / new customer"
        )

        ax.set_ylabel(
            "Observed cumulative net revenue / acquired customer"
        )

        ax.grid(
            alpha=0.15
        )

        fig.tight_layout()

        fig.savefig(
            plots_dir
            / "06_acquisition_quality_vs_cohort_value.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # -------------------------------------------------------------------------
    # 07. Reactivation curve
    # -------------------------------------------------------------------------

    reactivation = (
        dense_pd
        .groupby(
            "age_month",
            as_index=False,
        )
        .agg(
            active_customers=(
                "active_customers",
                "sum",
            ),
            reactivated=(
                "reactivated_customer_months",
                "sum",
            ),
        )
    )

    reactivation[
        "reactivation_share"
    ] = np.where(
        reactivation[
            "active_customers"
        ] > 0,
        reactivation[
            "reactivated"
        ]
        /
        reactivation[
            "active_customers"
        ],
        0.0,
    )

    line_chart(
        reactivation[
            "age_month"
        ],
        reactivation[
            "reactivation_share"
        ],
        "Reactivation Share of Active Customer-Months",
        "Months since acquisition",
        "Reactivation share",
        plots_dir
        / "07_reactivation_curve.png",
        percent=True,
    )

    # -------------------------------------------------------------------------
    # 08. Early retention diagnostic
    # -------------------------------------------------------------------------

    snapshot = (
        scorecard.to_pandas()
    )

    if not snapshot.empty:

        observable = snapshot[
            snapshot[
                "m3_observable"
            ]
        ].copy()

        if not observable.empty:

            fig, ax = plt.subplots(
                figsize=(
                    10,
                    6,
                )
            )

            ax.scatter(
                observable[
                    "m0_logo_retention"
                ],
                observable[
                    "m3_logo_retention"
                ],
                s=np.clip(
                    observable[
                        "m0_logo_retention"
                    ]
                    * 1000,
                    30,
                    500,
                ),
                alpha=0.55,
            )

            for _, row in observable.iterrows():

                ax.annotate(
                    pd.Timestamp(
                        row[
                            "cohort_month"
                        ]
                    ).strftime(
                        "%Y-%m"
                    ),
                    (
                        row[
                            "m0_logo_retention"
                        ],
                        row[
                            "m3_logo_retention"
                        ],
                    ),
                    fontsize=7,
                )

            ax.set_xlabel(
                "Month-0 retention"
            )

            ax.set_ylabel(
                "Month-3 retention"
            )

            ax.set_title(
                "Acquisition Cohort Early Retention Diagnostic"
            )

            ax.grid(
                alpha=0.15
            )

            ax.xaxis.set_major_formatter(
                mticker.PercentFormatter(
                    1.0
                )
            )

            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(
                    1.0
                )
            )

            fig.tight_layout()

            fig.savefig(
                plots_dir
                / "08_early_retention_diagnostic.png",
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )


# =============================================================================
# EXPORT
# =============================================================================

def export_csv(
    df: pl.DataFrame,
    path: Path,
) -> None:

    df.write_csv(
        path
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    input_path = (
        Path(
            args.input
        )
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "plots"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # 1. INGESTION
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
    # 2. CUSTOMER-MONTH EVENT MODEL
    # =========================================================================

    (
        customer_month,
        first_purchase,
        data_quality,
    ) = build_customer_month_events(
        tx
    )

    LOGGER.info(
        "Customer-month model: customers=%s | "
        "customer-month rows=%s",
        f"{data_quality['unique_customers']:,}",
        f"{data_quality['customer_month_rows']:,}",
    )

    # =========================================================================
    # 3. COHORT METRICS
    # =========================================================================

    tables = build_cohort_tables(
        customer_month
    )

    cohort_age = tables[
        "cohort_age"
    ]

    dense = tables[
        "dense_cohort_age"
    ]

    calendar = tables[
        "calendar_month"
    ]

    cohort_sizes = tables[
        "cohort_sizes"
    ]

    # =========================================================================
    # 4. COHORT ACQUISITION QUALITY
    # =========================================================================

    cohort_quality = build_cohort_quality(
        tx,
        customer_month,
    )

    # =========================================================================
    # 5. SCORECARD + DECAY CURVES
    # =========================================================================

    scorecard = cohort_snapshot_scorecard(
        dense,
        min_cohort_size=args.min_cohort_size,
        early_age=args.early_retention_months,
        mature_age=args.mature_retention_months,
    )

    decay = retention_decay_table(
        dense
    )

    # =========================================================================
    # 6. OPTIONAL AGE LIMIT
    # =========================================================================

    if args.max_age_months is not None:

        dense = dense.filter(
            pl.col(
                "age_month"
            )
            <= args.max_age_months
        )

        cohort_age = cohort_age.filter(
            pl.col(
                "age_month"
            )
            <= args.max_age_months
        )

        decay = decay.filter(
            pl.col(
                "age_month"
            )
            <= args.max_age_months
        )

    # =========================================================================
    # 7. CUSTOMER LIFECYCLE STATUS
    # =========================================================================

    observation_end_dt = pd.Timestamp(
        data_quality[
            "observation_end"
        ]
    )

    customer_lifecycle = (
        customer_month
        .group_by(
            [
                "Customer ID",
                "cohort_month",
            ]
        )
        .agg(
            [
                pl.col(
                    "calendar_month"
                )
                .filter(
                    pl.col(
                        "orders"
                    ) > 0
                )
                .max()
                .alias(
                    "last_active_month"
                ),

                pl.col(
                    "calendar_month"
                )
                .filter(
                    pl.col(
                        "orders"
                    ) > 0
                )
                .n_unique()
                .alias(
                    "active_calendar_months"
                ),

                pl.col(
                    "is_reactivation"
                )
                .sum()
                .alias(
                    "reactivation_count"
                ),

                pl.col(
                    "net_revenue"
                )
                .sum()
                .alias(
                    "lifetime_observed_net_revenue"
                ),

                pl.col(
                    "gross_revenue"
                )
                .sum()
                .alias(
                    "lifetime_observed_gross_revenue"
                ),

                pl.col(
                    "orders"
                )
                .sum()
                .alias(
                    "lifetime_orders"
                ),

                pl.col(
                    "return_value"
                )
                .sum()
                .alias(
                    "lifetime_return_value"
                ),
            ]
        )
        .with_columns(
            (
                (
                    pl.lit(
                        observation_end_dt.to_pydatetime()
                    )
                    -
                    pl.col(
                        "last_active_month"
                    )
                )
                .dt.total_days()
                /
                30.4375
            ).alias(
                "months_since_last_observed_activity"
            ),

            (
                pl.col(
                    "active_calendar_months"
                ) > 1
            ).alias(
                "has_repeat_active_month"
            ),

            (
                pl.col(
                    "reactivation_count"
                ) > 0
            ).alias(
                "has_reactivated"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "has_reactivated"
                )
            )
            .then(
                pl.lit(
                    "reactivated_at_least_once"
                )
            )
            .when(
                pl.col(
                    "has_repeat_active_month"
                )
            )
            .then(
                pl.lit(
                    "repeat_active"
                )
            )
            .otherwise(
                pl.lit(
                    "single_active_month"
                )
            )
            .alias(
                "observed_lifecycle_status"
            )
        )
    )

    # =========================================================================
    # 8. EXPORT TRANSACTION-DERIVED TABLES
    # =========================================================================

    export_csv(
        customer_month,
        output_dir
        / "customer_month_events.csv",
    )

    export_csv(
        first_purchase,
        output_dir
        / "customer_acquisition_cohorts.csv",
    )

    export_csv(
        cohort_age,
        output_dir
        / "cohort_age_metrics.csv",
    )

    export_csv(
        dense,
        output_dir
        / "cohort_age_metrics_dense.csv",
    )

    export_csv(
        cohort_sizes,
        output_dir
        / "cohort_sizes.csv",
    )

    export_csv(
        calendar,
        output_dir
        / "calendar_month_performance.csv",
    )

    export_csv(
        cohort_quality,
        output_dir
        / "cohort_acquisition_quality.csv",
    )

    export_csv(
        scorecard,
        output_dir
        / "cohort_scorecard.csv",
    )

    export_csv(
        decay,
        output_dir
        / "retention_decay_curve.csv",
    )

    export_csv(
        customer_lifecycle,
        output_dir
        / "customer_lifecycle_status.csv",
    )

    # =========================================================================
    # 9. BI-FRIENDLY WIDE MATRICES
    # =========================================================================

    dense_pd = dense.to_pandas()

    matrix_definitions = [
        (
            "logo_retention",
            "matrix_logo_retention.csv",
        ),
        (
            "gross_revenue_retention",
            "matrix_gross_revenue_retention.csv",
        ),
        (
            "net_revenue_retention",
            "matrix_net_revenue_retention.csv",
        ),
        (
            "orders_per_acquired_customer",
            "matrix_orders_per_acquired_customer.csv",
        ),
        (
            "net_revenue_per_acquired_customer",
            "matrix_net_revenue_per_acquired_customer.csv",
        ),
        (
            "reactivation_share_of_active",
            "matrix_reactivation_share.csv",
        ),
    ]

    for metric, filename in matrix_definitions:

        matrix = (
            dense_pd
            .pivot(
                index="cohort_month",
                columns="age_month",
                values=metric,
            )
            .sort_index()
        )

        matrix.index = matrix.index.map(
            lambda x: pd.Timestamp(
                x
            ).strftime(
                "%Y-%m"
            )
        )

        matrix.to_csv(
            output_dir
            / filename
        )

    # =========================================================================
    # 10. VISUALS
    # =========================================================================

    save_visuals(
        output_dir=output_dir,
        dense=dense,
        scorecard=scorecard,
        decay=decay,
        cohort_quality=cohort_quality,
        heatmap_limit=args.heatmap_limit,
    )

    # =========================================================================
    # 11. SUMMARY DIAGNOSTICS
    # =========================================================================

    decay_pd = decay.to_pandas()

    half_life = estimate_half_life(
        decay_pd
    )

    def metric_at_age(
        age: int,
        metric: str,
    ):
        subset = decay.filter(
            pl.col(
                "age_month"
            ) == age
        )

        if subset.height == 0:
            return None

        return float(
            subset
            .select(
                metric
            )
            .item()
        )

    scorecard_pd = (
        scorecard.to_pandas()
    )

    if (
        not scorecard_pd.empty
        and "m3_observable" in scorecard_pd.columns
    ):

        observed_m3 = scorecard_pd.loc[
            scorecard_pd[
                "m3_observable"
            ].eq(True),
            "m3_logo_retention",
        ]

    else:

        observed_m3 = pd.Series(
            dtype=float
        )

    summary = {
        **data_quality,

        "input_file": str(
            input_path
        ),

        "output_directory": str(
            output_dir
        ),

        "cohort_count": int(
            cohort_sizes.height
        ),

        "cohort_size_median": float(
            cohort_sizes
            .select(
                pl.col(
                    "cohort_customers"
                ).median()
            )
            .item()
        ),

        "overall_observed_gross_revenue": float(
            customer_month
            .select(
                pl.col(
                    "gross_revenue"
                ).sum()
            )
            .item()
        ),

        "overall_observed_net_revenue": float(
            customer_month
            .select(
                pl.col(
                    "net_revenue"
                ).sum()
            )
            .item()
        ),

        "overall_return_value": float(
            customer_month
            .select(
                pl.col(
                    "return_value"
                ).sum()
            )
            .item()
        ),

        "weighted_logo_retention_age_1": metric_at_age(
            1,
            "weighted_logo_retention",
        ),

        "weighted_logo_retention_age_3": metric_at_age(
            3,
            "weighted_logo_retention",
        ),

        "weighted_logo_retention_age_6": metric_at_age(
            6,
            "weighted_logo_retention",
        ),

        "weighted_net_revenue_retention_age_1": metric_at_age(
            1,
            "weighted_net_revenue_retention",
        ),

        "weighted_net_revenue_retention_age_3": metric_at_age(
            3,
            "weighted_net_revenue_retention",
        ),

        "weighted_net_revenue_retention_age_6": metric_at_age(
            6,
            "weighted_net_revenue_retention",
        ),

        "estimated_logo_retention_half_life_months": (
            None
            if not np.isfinite(
                half_life
            )
            else float(
                half_life
            )
        ),

        "observable_cohorts_with_m3": int(
            observed_m3.notna().sum()
        ),

        "parameters": {
            "min_cohort_size": args.min_cohort_size,
            "max_age_months": args.max_age_months,
            "early_retention_months": (
                args.early_retention_months
            ),
            "mature_retention_months": (
                args.mature_retention_months
            ),
            "heatmap_limit": args.heatmap_limit,
            "seed": SEED,
        },
    }

    with open(
        output_dir
        / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    # =========================================================================
    # 12. MANIFEST
    # =========================================================================

    manifest = {
        "outputs": [
            "customer_month_events.csv",
            "customer_acquisition_cohorts.csv",
            "cohort_age_metrics.csv",
            "cohort_age_metrics_dense.csv",
            "cohort_sizes.csv",
            "calendar_month_performance.csv",
            "cohort_acquisition_quality.csv",
            "cohort_scorecard.csv",
            "retention_decay_curve.csv",
            "customer_lifecycle_status.csv",
            "matrix_logo_retention.csv",
            "matrix_gross_revenue_retention.csv",
            "matrix_net_revenue_retention.csv",
            "matrix_orders_per_acquired_customer.csv",
            "matrix_net_revenue_per_acquired_customer.csv",
            "matrix_reactivation_share.csv",
            "run_summary.json",
            "plots/01_logo_retention_heatmap.png",
            "plots/02_net_revenue_retention_heatmap.png",
            "plots/03_revenue_per_acquired_customer_heatmap.png",
            "plots/04_weighted_logo_retention_curve.png",
            "plots/05_weighted_revenue_retention_curve.png",
            "plots/06_acquisition_quality_vs_cohort_value.png",
            "plots/07_reactivation_curve.png",
            "plots/08_early_retention_diagnostic.png",
        ]
    }

    with open(
        output_dir
        / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    LOGGER.info(
        "Cohort analysis complete: %s",
        output_dir,
    )


if __name__ == "__main__":
    main()