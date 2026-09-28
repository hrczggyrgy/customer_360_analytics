#!/usr/bin/env python3
"""
Online Retail II — behavioral customer segmentation.

End-to-end pipeline:
1) Read/validate Online Retail II.
2) Feature-engineer a multi-view customer "behavioral signature" in Polars.
3) Winsorize -> Yeo-Johnson -> RobustScaler -> balanced behavioral blocks.
4) PCA for denoising/compression.
5) HDBSCAN model selection + bootstrap stability diagnostic.
6) KMeans fallback only if HDBSCAN is unavailable.
7) Export customer scores, segment profiles, model diagnostics, loadings,
   feature dictionary, PCA coordinates, and decision-ready visuals.

This is deliberately broader than RFM. It models:
- economic intensity
- order-value concentration
- purchase cadence
- assortment breadth and concentration
- repeat-product behavior
- price behavior
- temporal behavior
- customer lifecycle
- cancellation / reversal behavior

Example:
    python customer_segmentation.py \
        --input /home/lptop/Documents/coding/marketing_science/data_xslx/online_retail_II.xlsx \
        --output-dir ./online_retail_segmentation

Recommended environment:
    pip install polars fastexcel pandas scikit-learn matplotlib openpyxl

Optional if scikit-learn does not provide HDBSCAN:
    pip install hdbscan
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import PowerTransformer, RobustScaler

try:
    from sklearn.cluster import HDBSCAN as SKHDBSCAN
except Exception:
    SKHDBSCAN = None

try:
    import hdbscan as EXTERNAL_HDBSCAN
except Exception:
    EXTERNAL_HDBSCAN = None


SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Behavioral feature architecture
# ---------------------------------------------------------------------

FEATURE_GROUPS = {
    "economic_intensity": [
        "revenue",
        "log_revenue",
        "avg_order_value",
        "median_order_value",
        "revenue_per_active_day",
        "units_per_invoice",
        "lines_per_invoice",
        "order_value_cv",
        "order_value_hhi",
        "invoice_intensity_per_month",
    ],
    "purchase_cadence": [
        "invoice_count",
        "active_purchase_days",
        "purchase_day_share",
        "median_interpurchase_days",
        "mean_interpurchase_days",
        "interpurchase_cv",
        "burstiness",
        "invoice_count_per_active_day",
    ],
    "assortment_behavior": [
        "unique_products",
        "products_per_invoice",
        "repeat_product_ratio",
        "product_hhi",
        "single_product_invoice_share",
        "multi_product_invoice_share",
    ],
    "price_behavior": [
        "weighted_avg_unit_price",
        "median_unit_price",
        "price_iqr",
        "line_price_cv",
        "premium_line_share",
        "zero_price_line_share",
    ],
    "temporal_signature": [
        "weekend_purchase_share",
        "business_hour_share",
        "hour_entropy",
        "weekday_entropy",
        "month_entropy",
    ],
    "lifecycle": [
        "tenure_days",
        "days_since_last_purchase",
        "active_span_ratio",
        "purchase_months",
        "months_since_last_purchase",
    ],
    "return_friction": [
        "return_units",
        "return_value",
        "unit_return_rate",
        "value_return_rate",
        "return_invoice_rate",
        "return_line_share",
    ],
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("customer_segmentation")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)

    return logger


LOGGER = setup_logger()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral customer segmentation for Online Retail II."
    )

    parser.add_argument(
        "--input",
        default=(
            "/home/lptop/Documents/coding/marketing_science/"
            "data_xslx/online_retail_II.xlsx"
        ),
        help="Input .xlsx/.csv/.parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        default="./online_retail_segmentation",
        help="Output directory.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name. Default: all sheets.",
    )

    parser.add_argument(
        "--min-customers",
        type=int,
        default=150,
        help="Minimum number of customers required to cluster.",
    )

    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.90,
        help="Target cumulative PCA variance for clustering.",
    )

    parser.add_argument(
        "--max-pca-components",
        type=int,
        default=12,
        help="Maximum number of PCA components used for clustering.",
    )

    parser.add_argument(
        "--sample-for-plots",
        type=int,
        default=10000,
        help="Maximum observations used in scatter plots.",
    )

    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=2,
        help="Bootstrap repeats used for cluster stability diagnostics.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

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

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {df.columns}"
        )

    return df


def load_input(
    path: Path,
    sheet: str | None = None,
) -> pl.DataFrame:

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
                "Polars Excel reader failed (%s). "
                "Falling back to pandas/openpyxl.",
                exc,
            )

            pd_obj = pd.read_excel(
                path,
                sheet_name=sheet if sheet else None,
            )

            if isinstance(pd_obj, dict):
                pd_df = pd.concat(
                    pd_obj.values(),
                    ignore_index=True,
                )
            else:
                pd_df = pd_obj

            df = pl.from_pandas(pd_df)

    elif suffix == ".csv":

        df = pl.read_csv(
            path,
            try_parse_dates=True,
            infer_schema_length=10000,
            ignore_errors=True,
        )

    elif suffix in {".parquet", ".pq"}:

        df = pl.read_parquet(path)

    else:

        raise ValueError(
            f"Unsupported input format: {suffix}"
        )

    return normalize_columns(df)


# ---------------------------------------------------------------------
# Transaction cleaning
# ---------------------------------------------------------------------

def clean_transactions(
    df: pl.DataFrame,
) -> pl.DataFrame:

    LOGGER.info(
        "Raw rows: %s",
        f"{df.height:,}",
    )

    out = df

    out = out.with_columns(
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

    invoice_date_dtype = out.schema["InvoiceDate"]

    if invoice_date_dtype not in (
        pl.Date,
        pl.Datetime,
        pl.Time,
    ):

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

    out = out.with_columns(
        [
            pl.col("Invoice")
            .str.to_uppercase()
            .str.starts_with("C")
            .alias("is_cancellation_invoice"),

            (pl.col("Quantity") < 0)
            .alias("is_negative_quantity"),

            (
                pl.col("Quantity")
                * pl.col("Price")
            ).alias("line_value"),
        ]
    )

    out = out.filter(
        pl.col("Customer ID").is_not_null()
        & pl.col("InvoiceDate").is_not_null()
        & pl.col("Invoice").is_not_null()
        & pl.col("StockCode").is_not_null()
        & pl.col("Quantity").is_not_null()
        & pl.col("Price").is_not_null()
    )

    out = out.with_columns(
        [
            pl.when(pl.col("line_value") > 0)
            .then(pl.col("line_value"))
            .otherwise(0.0)
            .alias("positive_line_value"),

            pl.when(
                pl.col("is_cancellation_invoice")
                | (pl.col("Quantity") < 0)
            )
            .then(pl.col("Quantity").abs())
            .otherwise(0.0)
            .alias("return_units_line"),

            pl.when(
                pl.col("is_cancellation_invoice")
                | (pl.col("Quantity") < 0)
            )
            .then(pl.col("line_value").abs())
            .otherwise(0.0)
            .alias("return_value_line"),

            pl.when(
                (~pl.col("is_cancellation_invoice"))
                & (pl.col("Quantity") > 0)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_sale_line"),
        ]
    )

    LOGGER.info(
        "Rows after cleaning: %s",
        f"{out.height:,}",
    )

    return out


# ---------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------

def entropy_feature(
    df: pl.DataFrame,
    date_part: str,
    denominator: float,
    alias: str,
    date_column: str = "invoice_date",
) -> pl.DataFrame:

    if date_part == "hour":

        values = df.with_columns(
            pl.col(date_column)
            .dt.hour()
            .alias("bucket")
        )

    elif date_part == "weekday":

        values = df.with_columns(
            pl.col(date_column)
            .dt.weekday()
            .alias("bucket")
        )

    elif date_part == "month":

        values = df.with_columns(
            pl.col(date_column)
            .dt.month()
            .alias("bucket")
        )

    else:
        raise ValueError(
            f"Unsupported temporal component: {date_part}"
        )

    counts = (
        values
        .group_by(["Customer ID", "bucket"])
        .len("n")
    )

    counts = counts.with_columns(
        (
            pl.col("n")
            / pl.col("n").sum().over("Customer ID")
        ).alias("p")
    )

    entropy = (
        counts
        .group_by("Customer ID")
        .agg(
            (
                -(
                    pl.col("p")
                    * pl.col("p").log()
                ).sum()
                / math.log(denominator)
            ).alias(alias)
        )
    )

    return entropy


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def build_customer_features(
    tx: pl.DataFrame,
) -> Tuple[
    pl.DataFrame,
    Dict[str, List[str]],
]:

    LOGGER.info(
        "Engineering customer behavioral features with Polars..."
    )

    sales = tx.filter(
        (pl.col("is_sale_line") == 1)
        & (pl.col("Quantity") > 0)
    )

    all_customers = (
        tx
        .select("Customer ID")
        .unique()
    )

    if sales.height == 0:
        raise ValueError(
            "No positive purchase lines found."
        )

    # -------------------------------------------------------------
    # Invoice-level behavioral table
    # -------------------------------------------------------------

    invoices = (
        sales
        .group_by(
            ["Customer ID", "Invoice"]
        )
        .agg(
            [
                pl.col("InvoiceDate")
                .min()
                .alias("invoice_date"),

                pl.col("positive_line_value")
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
        .with_columns(
            [
                pl.col("invoice_date")
                .dt.weekday()
                .alias("invoice_weekday"),

                pl.col("invoice_date")
                .dt.hour()
                .alias("invoice_hour"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Inter-purchase cadence
    # -------------------------------------------------------------

    invoice_dates = (
        invoices
        .select(
            [
                "Customer ID",
                "invoice_date",
            ]
        )
        .with_columns(
            (
                pl.col("invoice_date")
                .diff()
                .over("Customer ID")
                .dt.total_seconds()
                / 86400
            ).alias("interpurchase_days")
        )
    )

    cadence = (
        invoice_dates
        .group_by("Customer ID")
        .agg(
            [
                pl.col("interpurchase_days")
                .median()
                .alias(
                    "median_interpurchase_days"
                ),

                pl.col("interpurchase_days")
                .mean()
                .alias(
                    "mean_interpurchase_days"
                ),

                pl.col("interpurchase_days")
                .std()
                .alias(
                    "std_interpurchase_days"
                ),

                pl.col("interpurchase_days")
                .count()
                .alias(
                    "interpurchase_count"
                ),

                (
                    (
                        pl.col("interpurchase_days")
                        > 0
                    ).sum()
                    / pl.col(
                        "interpurchase_days"
                    ).count()
                ).alias(
                    "positive_interpurchase_share"
                ),
            ]
        )
        .with_columns(
            [
                (
                    pl.col(
                        "std_interpurchase_days"
                    )
                    / (
                        pl.col(
                            "mean_interpurchase_days"
                        ).abs()
                        + 1e-9
                    )
                )
                .fill_null(0.0)
                .alias(
                    "interpurchase_cv"
                ),

                (
                    (
                        pl.col(
                            "std_interpurchase_days"
                        )
                        - pl.col(
                            "mean_interpurchase_days"
                        )
                    )
                    / (
                        pl.col(
                            "std_interpurchase_days"
                        )
                        + pl.col(
                            "mean_interpurchase_days"
                        )
                        + 1e-9
                    )
                )
                .fill_null(0.0)
                .alias(
                    "burstiness"
                ),
            ]
        )
    )

    # -------------------------------------------------------------
    # Order value concentration
    # -------------------------------------------------------------

    invoice_value_shape = (
        invoices
        .with_columns(
            (
                pl.col("invoice_revenue")
                / pl.col(
                    "invoice_revenue"
                )
                .sum()
                .over("Customer ID")
                .clip(lower_bound=1e-9)
            ).alias(
                "invoice_revenue_share"
            )
        )
        .group_by("Customer ID")
        .agg(
            (
                pl.col(
                    "invoice_revenue_share"
                ) ** 2
            )
            .sum()
            .alias("order_value_hhi")
        )
    )

    # -------------------------------------------------------------
    # Customer-level invoice features
    # -------------------------------------------------------------

    invoice_base = (
        invoices
        .group_by("Customer ID")
        .agg(
            [
                pl.len()
                .alias("invoice_count"),

                pl.col("invoice_date")
                .n_unique()
                .alias(
                    "active_purchase_days"
                ),

                pl.col("invoice_revenue")
                .sum()
                .alias("revenue"),

                pl.col("invoice_revenue")
                .mean()
                .alias("avg_order_value"),

                pl.col("invoice_revenue")
                .median()
                .alias(
                    "median_order_value"
                ),

                pl.col("invoice_revenue")
                .std()
                .fill_null(0.0)
                .alias("order_value_std"),

                (
                    pl.col("invoice_revenue")
                    .std()
                    .fill_null(0.0)
                    / (
                        pl.col("invoice_revenue")
                        .mean()
                        .abs()
                        + 1e-9
                    )
                )
                .fill_null(0.0)
                .alias("order_value_cv"),

                pl.col("invoice_units")
                .sum()
                .alias("sales_units"),

                pl.col("invoice_lines")
                .sum()
                .alias("sales_lines"),

                pl.col(
                    "invoice_unique_products"
                )
                .sum()
                .alias(
                    "invoice_product_count_sum"
                ),

                pl.col(
                    "invoice_unique_products"
                )
                .filter(
                    pl.col(
                        "invoice_unique_products"
                    ) == 1
                )
                .count()
                .alias(
                    "single_product_invoices"
                ),

                pl.col(
                    "invoice_unique_products"
                )
                .filter(
                    pl.col(
                        "invoice_unique_products"
                    ) > 1
                )
                .count()
                .alias(
                    "multi_product_invoices"
                ),

                pl.col("invoice_date")
                .min()
                .alias("first_purchase"),

                pl.col("invoice_date")
                .max()
                .alias("last_purchase"),

                pl.col("invoice_date")
                .dt.month_start()
                .n_unique()
                .alias("purchase_months"),

                pl.col("invoice_date")
                .dt.date()
                .n_unique()
                .alias("purchase_days"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("invoice_count")
                    / (
                        (
                            pl.col("last_purchase")
                            - pl.col("first_purchase")
                        )
                        .dt.total_days()
                        / 30.4375
                    )
                    .clip(lower_bound=1.0)
                )
                .alias(
                    "invoice_intensity_per_month"
                ),

                (
                    pl.col("sales_lines")
                    / pl.col("invoice_count")
                    .clip(lower_bound=1)
                )
                .alias("lines_per_invoice"),

                (
                    pl.col("sales_units")
                    / pl.col("invoice_count")
                    .clip(lower_bound=1)
                )
                .alias("units_per_invoice"),

                (
                    pl.col(
                        "invoice_product_count_sum"
                    )
                    / pl.col("invoice_count")
                    .clip(lower_bound=1)
                )
                .alias(
                    "products_per_invoice"
                ),

                (
                    pl.col(
                        "single_product_invoices"
                    )
                    / pl.col("invoice_count")
                    .clip(lower_bound=1)
                )
                .alias(
                    "single_product_invoice_share"
                ),

                (
                    pl.col(
                        "multi_product_invoices"
                    )
                    / pl.col("invoice_count")
                    .clip(lower_bound=1)
                )
                .alias(
                    "multi_product_invoice_share"
                ),

                (
                    pl.col("last_purchase")
                    - pl.col("first_purchase")
                )
                .dt.total_days()
                .fill_null(0.0)
                .alias("tenure_days"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Product affinity and concentration
    # -------------------------------------------------------------

    customer_product = (
        sales
        .group_by(
            ["Customer ID", "StockCode"]
        )
        .agg(
            [
                pl.col("positive_line_value")
                .sum()
                .alias("product_revenue"),

                pl.col("Invoice")
                .n_unique()
                .alias(
                    "product_invoice_count"
                ),
            ]
        )
    )

    product_profile = (
        customer_product
        .with_columns(
            (
                pl.col("product_revenue")
                / pl.col(
                    "product_revenue"
                )
                .sum()
                .over("Customer ID")
                .clip(lower_bound=1e-9)
            ).alias("product_share")
        )
        .group_by("Customer ID")
        .agg(
            [
                (
                    pl.col("product_share") ** 2
                )
                .sum()
                .alias("product_hhi"),

                pl.col("StockCode")
                .n_unique()
                .alias("unique_products"),

                (
                    (
                        pl.col(
                            "product_invoice_count"
                        ) > 1
                    )
                    .cast(pl.Float64)
                )
                .mean()
                .alias(
                    "repeat_product_ratio"
                ),
            ]
        )
    )

    # -------------------------------------------------------------
    # Price behavior
    # -------------------------------------------------------------

    global_price_q75 = (
        sales
        .select(
            pl.col("Price")
            .quantile(0.75)
        )
        .item()
    )

    price_profile = (
        sales
        .group_by("Customer ID")
        .agg(
            [
                (
                    (
                        pl.col("Price")
                        * pl.col("Quantity")
                    ).sum()
                    / pl.col("Quantity")
                    .sum()
                    .clip(lower_bound=1)
                )
                .alias(
                    "weighted_avg_unit_price"
                ),

                pl.col("Price")
                .median()
                .alias("median_unit_price"),

                (
                    pl.col("Price").quantile(0.75)
                    - pl.col("Price").quantile(0.25)
                )
                .alias("price_iqr"),

                (
                    pl.col("Price")
                    > global_price_q75
                )
                .mean()
                .alias("premium_line_share"),

                (
                    pl.col("Price") <= 0
                )
                .mean()
                .alias("zero_price_line_share"),

                pl.col("Price")
                .mean()
                .alias("mean_unit_price"),

                pl.col("Price")
                .std()
                .fill_null(0.0)
                .alias("std_unit_price"),
            ]
        )
        .with_columns(
            (
                pl.col("std_unit_price")
                / (
                    pl.col(
                        "mean_unit_price"
                    ).abs()
                    + 1e-9
                )
            )
            .fill_null(0.0)
            .alias("line_price_cv")
        )
    )

    # -------------------------------------------------------------
    # Temporal signature
    # -------------------------------------------------------------

    temporal = (
        invoices
        .group_by("Customer ID")
        .agg(
            [
                (
                    pl.col("invoice_weekday")
                    >= 6
                )
                .mean()
                .alias(
                    "weekend_purchase_share"
                ),

                (
                    (
                        pl.col("invoice_hour")
                        >= 9
                    )
                    & (
                        pl.col("invoice_hour")
                        < 18
                    )
                )
                .mean()
                .alias(
                    "business_hour_share"
                ),
            ]
        )
    )

    temporal = temporal.join(
        entropy_feature(
            invoices,
            "hour",
            24.0,
            "hour_entropy",
            date_column="invoice_date",
        ),
        on="Customer ID",
        how="left",
    )

    temporal = temporal.join(
        entropy_feature(
            invoices,
            "weekday",
            7.0,
            "weekday_entropy",
            date_column="invoice_date",
        ),
        on="Customer ID",
        how="left",
    )

    temporal = temporal.join(
        entropy_feature(
            invoices,
            "month",
            12.0,
            "month_entropy",
            date_column="invoice_date",
        ),
        on="Customer ID",
        how="left",
    )

    # -------------------------------------------------------------
    # Returns / cancellations
    # -------------------------------------------------------------

    returns = tx.filter(
        pl.col("is_cancellation_invoice")
        | (pl.col("Quantity") < 0)
    )

    if returns.height:

        return_profile = (
            returns
            .group_by("Customer ID")
            .agg(
                [
                    pl.col("return_units_line")
                    .sum()
                    .alias("return_units"),

                    pl.col("return_value_line")
                    .sum()
                    .alias("return_value"),

                    pl.col("Invoice")
                    .n_unique()
                    .alias("return_invoices"),

                    pl.len()
                    .alias("return_lines"),
                ]
            )
        )

    else:

        return_profile = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "return_units": pl.Float64,
                "return_value": pl.Float64,
                "return_invoices": pl.Float64,
                "return_lines": pl.Float64,
            }
        )

    # -------------------------------------------------------------
    # Combine customer feature views
    # -------------------------------------------------------------

    features = all_customers

    for right in [
        invoice_base,
        invoice_value_shape,
        cadence,
        product_profile,
        price_profile,
        temporal,
        return_profile,
    ]:

        features = features.join(
            right,
            on="Customer ID",
            how="left",
        )

    reference_date = (
        sales
        .select(
            pl.col("InvoiceDate").max()
        )
        .item()
    )

    overall_first_date = (
        sales
        .select(
            pl.col("InvoiceDate").min()
        )
        .item()
    )

    overall_span_days = max(
        1.0,
        float(
            (
                reference_date
                - overall_first_date
            ).total_seconds()
            / 86400.0
        ),
    )

    features = features.with_columns(
        [
            pl.lit(reference_date)
            .alias("reference_date"),

            (
                (
                    pl.lit(reference_date)
                    - pl.col("last_purchase")
                )
                .dt.total_seconds()
                / 86400
            )
            .fill_null(0.0)
            .alias(
                "days_since_last_purchase"
            ),

            (
                (
                    pl.lit(reference_date)
                    .dt.month_start()
                    - pl.col("last_purchase")
                    .dt.month_start()
                )
                .dt.total_days()
                / 30.4375
            )
            .fill_null(0.0)
            .alias(
                "months_since_last_purchase"
            ),

            (
                pl.col("active_purchase_days")
                / (
                    pl.col("tenure_days")
                    + 1.0
                )
            )
            .alias("purchase_day_share"),

            (
                pl.col("tenure_days")
                / pl.lit(overall_span_days)
            )
            .alias(
                "active_span_ratio"
            ),

            (
                pl.col("invoice_count")
                / pl.col(
                    "active_purchase_days"
                )
                .clip(lower_bound=1)
            )
            .alias(
                "invoice_count_per_active_day"
            ),

            (
                pl.col("revenue")
                / pl.col(
                    "active_purchase_days"
                )
                .clip(lower_bound=1)
            )
            .alias(
                "revenue_per_active_day"
            ),

            pl.col("revenue")
            .log1p()
            .alias("log_revenue"),
        ]
    )

    features = features.with_columns(
        [
            pl.col("return_units")
            .fill_null(0.0),

            pl.col("return_value")
            .fill_null(0.0),

            pl.col("return_invoices")
            .fill_null(0.0),

            pl.col("return_lines")
            .fill_null(0.0),
        ]
    )

    features = features.with_columns(
        [
            (
                pl.col("return_units")
                / (
                    pl.col("sales_units")
                    + pl.col("return_units")
                )
                .clip(lower_bound=1.0)
            )
            .alias("unit_return_rate"),

            (
                pl.col("return_value")
                / (
                    pl.col("revenue")
                    + pl.col("return_value")
                )
                .clip(lower_bound=1.0)
            )
            .alias("value_return_rate"),

            (
                pl.col("return_invoices")
                / (
                    pl.col("invoice_count")
                    + pl.col("return_invoices")
                )
                .clip(lower_bound=1.0)
            )
            .alias(
                "return_invoice_rate"
            ),

            (
                pl.col("return_lines")
                / (
                    pl.col("sales_lines")
                    + pl.col("return_lines")
                )
                .clip(lower_bound=1.0)
            )
            .alias("return_line_share"),
        ]
    )

    numeric_features = sorted(
        {
            feature
            for group in FEATURE_GROUPS.values()
            for feature in group
        }
    )

    features = features.with_columns(
        [
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .fill_nan(0.0)
            for column in numeric_features
            if column in features.columns
        ]
    )

    features = features.with_columns(
        pl.col("Customer ID")
        .cast(pl.Int64)
    )

    feature_groups = {
        group: [
            feature
            for feature in features_list
            if feature in features.columns
        ]
        for group, features_list
        in FEATURE_GROUPS.items()
    }

    LOGGER.info(
        "Customers: %s | engineered features: %s",
        f"{features.height:,}",
        len(numeric_features),
    )

    return features, feature_groups


# ---------------------------------------------------------------------
# Robust preprocessing + block balancing
# ---------------------------------------------------------------------

def robust_matrix(
    features: pl.DataFrame,
    feature_groups: Dict[str, List[str]],
) -> Tuple[
    np.ndarray,
    List[str],
    Dict[str, List[str]],
]:

    columns = [
        feature
        for group in feature_groups.values()
        for feature in group
        if feature in features.columns
    ]

    X = (
        features
        .select(columns)
        .to_pandas()
        .astype(float)
        .to_numpy()
    )

    feature_columns = columns

    # Winsorization.
    lower = np.nanpercentile(
        X,
        1.0,
        axis=0,
    )

    upper = np.nanpercentile(
        X,
        99.0,
        axis=0,
    )

    X = np.clip(
        X,
        lower,
        upper,
    )

    # Remove zero-variance columns.
    stds = np.nanstd(
        X,
        axis=0,
    )

    keep = stds > 1e-12

    X = X[:, keep]

    feature_columns = [
        column
        for column, keep_flag
        in zip(feature_columns, keep)
        if keep_flag
    ]

    # Yeo-Johnson handles non-negative, zero-heavy,
    # skewed, and negative return features.
    power_transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    X = power_transformer.fit_transform(X)

    robust_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        unit_variance=True,
    )

    X = robust_scaler.fit_transform(X)

    # Equalize the behavioral views before PCA.
    # Otherwise a large feature block can dominate the latent space.
    used_groups = {}

    for group_name, group_columns in feature_groups.items():

        active = [
            column
            for column in group_columns
            if column in feature_columns
        ]

        if not active:
            continue

        indices = [
            feature_columns.index(column)
            for column in active
        ]

        X[:, indices] /= math.sqrt(
            len(indices)
        )

        used_groups[group_name] = active

    return (
        X,
        feature_columns,
        used_groups,
    )


# ---------------------------------------------------------------------
# HDBSCAN helpers
# ---------------------------------------------------------------------

def build_hdbscan(
    min_cluster_size: int,
    min_samples: int,
):

    if SKHDBSCAN is not None:

        return SKHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            n_jobs=-1,
        )

    if EXTERNAL_HDBSCAN is not None:

        return EXTERNAL_HDBSCAN.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )

    return None


def get_probabilities(
    model,
    labels: np.ndarray,
) -> np.ndarray:

    probabilities = getattr(
        model,
        "probabilities_",
        None,
    )

    if probabilities is None:

        output = np.ones(
            labels.shape[0],
            dtype=float,
        )

        output[labels < 0] = 0.0

        return output

    return np.asarray(
        probabilities,
        dtype=float,
    )


def make_hdbscan_candidates(
    n: int,
) -> List[Tuple[int, int]]:

    if n < 30:

        return [
            (
                max(5, n // 5),
                max(3, n // 10),
            )
        ]

    raw_sizes = [
        max(
            15,
            int(round(n * 0.005)),
        ),
        max(
            20,
            int(round(n * 0.01)),
        ),
        max(
            30,
            int(round(n * 0.02)),
        ),
        max(
            50,
            int(round(n * 0.03)),
        ),
        max(
            75,
            int(round(n * 0.05)),
        ),
    ]

    max_size = max(
        10,
        min(
            250,
            n // 3,
        ),
    )

    cluster_sizes = sorted(
        set(
            min(
                max(10, size),
                max_size,
            )
            for size in raw_sizes
        )
    )

    candidates = []

    for cluster_size in cluster_sizes:

        for fraction in (
            0.5,
            1.0,
        ):

            min_samples = max(
                5,
                min(
                    cluster_size,
                    int(
                        round(
                            cluster_size
                            * fraction
                        )
                    ),
                ),
            )

            candidates.append(
                (
                    cluster_size,
                    min_samples,
                )
            )

    return candidates


def evaluate_hdbscan(
    X: np.ndarray,
    model,
) -> Dict[str, float]:

    labels = np.asarray(
        model.labels_
    )

    clustered = labels >= 0

    number_clusters = (
        len(
            set(
                labels[clustered]
            )
        )
        if clustered.any()
        else 0
    )

    coverage = float(
        clustered.mean()
    )

    probabilities = get_probabilities(
        model,
        labels,
    )

    mean_probability = (
        float(
            probabilities[clustered].mean()
        )
        if clustered.any()
        else 0.0
    )

    if (
        number_clusters >= 2
        and clustered.sum() >= 20
    ):

        sample_size = min(
            3000,
            int(clustered.sum()),
        )

        candidate_indices = np.flatnonzero(
            clustered
        )

        if candidate_indices.size > sample_size:

            rng = np.random.default_rng(
                SEED
            )

            indices = rng.choice(
                candidate_indices,
                size=sample_size,
                replace=False,
            )

        else:

            indices = candidate_indices

        y = labels[indices]

        try:

            silhouette = float(
                silhouette_score(
                    X[indices],
                    y,
                    metric="euclidean",
                )
            )

        except Exception:

            silhouette = -1.0

        try:

            davies_bouldin = float(
                davies_bouldin_score(
                    X[indices],
                    y,
                )
            )

        except Exception:

            davies_bouldin = float("inf")

    else:

        silhouette = -1.0
        davies_bouldin = float("inf")

    normalized_silhouette = (
        silhouette + 1.0
    ) / 2.0

    normalized_db = (
        1.0
        / (
            1.0
            + max(
                davies_bouldin,
                0.0,
            )
        )
        if np.isfinite(davies_bouldin)
        else 0.0
    )

    quality = (
        0.45
        * normalized_silhouette
        + 0.20
        * normalized_db
        + 0.20
        * coverage
        + 0.15
        * mean_probability
    )

    if number_clusters < 2:
        quality *= 0.20

    if number_clusters > 20:
        quality *= 0.85

    return {
        "n_clusters": int(
            number_clusters
        ),
        "coverage": coverage,
        "noise_share": 1.0 - coverage,
        "mean_probability": mean_probability,
        "silhouette": silhouette,
        "davies_bouldin": (
            davies_bouldin
            if np.isfinite(
                davies_bouldin
            )
            else None
        ),
        "quality": float(
            quality
        ),
    }


# ---------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------

def bootstrap_stability(
    X: np.ndarray,
    labels_full: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    repeats: int,
) -> float:

    if (
        build_hdbscan(
            min_cluster_size,
            min_samples,
        )
        is None
        or repeats <= 0
    ):
        return float("nan")

    rng = np.random.default_rng(
        SEED
    )

    n = X.shape[0]

    sample_size = max(
        min(
            n - 1,
            int(0.8 * n),
        ),
        min_cluster_size * 2,
    )

    sample_size = min(
        sample_size,
        n - 1,
    )

    scores = []

    for repeat in range(
        repeats
    ):

        indices = rng.choice(
            n,
            size=sample_size,
            replace=False,
        )

        model = build_hdbscan(
            min_cluster_size,
            min_samples,
        )

        try:

            model.fit(
                X[indices]
            )

            bootstrap_labels = np.asarray(
                model.labels_
            )

            full_labels_subset = (
                labels_full[indices]
            )

            ari = float(
                adjusted_rand_score(
                    full_labels_subset,
                    bootstrap_labels,
                )
            )

            scores.append(
                ari
            )

        except Exception as exc:

            LOGGER.warning(
                "Bootstrap stability repeat %s failed: %s",
                repeat + 1,
                exc,
            )

    if not scores:
        return float("nan")

    return float(
        np.mean(scores)
    )


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def choose_clusters(
    X: np.ndarray,
    repeats: int = 2,
) -> Tuple[
    np.ndarray,
    Dict,
    pd.DataFrame,
    object,
]:

    n = X.shape[0]

    candidates = make_hdbscan_candidates(
        n
    )

    candidate_rows = []
    fitted_models = []

    if build_hdbscan(
        10,
        5,
    ) is not None:

        LOGGER.info(
            "HDBSCAN available; evaluating %s candidate parameterizations.",
            len(candidates),
        )

        for min_cluster_size, min_samples in candidates:

            model = build_hdbscan(
                min_cluster_size,
                min_samples,
            )

            try:

                model.fit(X)

                metrics = evaluate_hdbscan(
                    X,
                    model,
                )

                metrics.update(
                    {
                        "algorithm": "HDBSCAN",
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                    }
                )

                candidate_rows.append(
                    metrics
                )

                fitted_models.append(
                    (
                        metrics,
                        model,
                    )
                )

            except Exception as exc:

                LOGGER.warning(
                    "HDBSCAN candidate failed "
                    "mcs=%s ms=%s: %s",
                    min_cluster_size,
                    min_samples,
                    exc,
                )

        if candidate_rows:

            candidate_df = (
                pd.DataFrame(
                    candidate_rows
                )
                .sort_values(
                    "quality",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            # Stability is evaluated only on the strongest
            # candidates to keep runtime manageable.
            top_candidates = candidate_df.head(
                min(
                    3,
                    len(candidate_df),
                )
            )

            stability_results = []

            for _, row in top_candidates.iterrows():

                min_cluster_size = int(
                    row[
                        "min_cluster_size"
                    ]
                )

                min_samples = int(
                    row[
                        "min_samples"
                    ]
                )

                matching_model = next(
                    model
                    for metrics, model
                    in fitted_models
                    if int(
                        metrics[
                            "min_cluster_size"
                        ]
                    )
                    == min_cluster_size
                    and int(
                        metrics[
                            "min_samples"
                        ]
                    )
                    == min_samples
                )

                stability = bootstrap_stability(
                    X,
                    np.asarray(
                        matching_model.labels_
                    ),
                    min_cluster_size,
                    min_samples,
                    repeats,
                )

                stability_results.append(
                    (
                        min_cluster_size,
                        min_samples,
                        stability,
                    )
                )

            candidate_df["bootstrap_ari"] = np.nan

            for (
                min_cluster_size,
                min_samples,
                stability,
            ) in stability_results:

                mask = (
                    candidate_df.min_cluster_size
                    == min_cluster_size
                ) & (
                    candidate_df.min_samples
                    == min_samples
                )

                candidate_df.loc[
                    mask,
                    "bootstrap_ari",
                ] = stability

            candidate_df[
                "final_selection_score"
            ] = candidate_df[
                "quality"
            ]

            valid_stability = (
                candidate_df[
                    "bootstrap_ari"
                ].notna()
            )

            candidate_df.loc[
                valid_stability,
                "final_selection_score",
            ] = (
                0.70
                * candidate_df.loc[
                    valid_stability,
                    "quality",
                ]
                + 0.30
                * candidate_df.loc[
                    valid_stability,
                    "bootstrap_ari",
                ].clip(
                    lower=0.0,
                    upper=1.0,
                )
            )

            candidate_df = (
                candidate_df
                .sort_values(
                    "final_selection_score",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            best = candidate_df.iloc[0]

            best_min_cluster_size = int(
                best[
                    "min_cluster_size"
                ]
            )

            best_min_samples = int(
                best[
                    "min_samples"
                ]
            )

            best_model = build_hdbscan(
                best_min_cluster_size,
                best_min_samples,
            )

            best_model.fit(X)

            best_labels = np.asarray(
                best_model.labels_
            )

            bootstrap_ari_value = (
                None
                if pd.isna(
                    best.get(
                        "bootstrap_ari"
                    )
                )
                else float(
                    best[
                        "bootstrap_ari"
                    ]
                )
            )

            metadata = {
                "algorithm": "HDBSCAN",
                "min_cluster_size": (
                    best_min_cluster_size
                ),
                "min_samples": (
                    best_min_samples
                ),
                "n_clusters": int(
                    best["n_clusters"]
                ),
                "coverage": float(
                    best["coverage"]
                ),
                "noise_share": float(
                    best["noise_share"]
                ),
                "mean_probability": float(
                    best[
                        "mean_probability"
                    ]
                ),
                "silhouette": float(
                    best["silhouette"]
                ),
                "davies_bouldin": (
                    best["davies_bouldin"]
                ),
                "bootstrap_ari": (
                    bootstrap_ari_value
                ),
                "selection_score": float(
                    best[
                        "final_selection_score"
                    ]
                ),
            }

            return (
                best_labels,
                metadata,
                candidate_df,
                best_model,
            )

    # -----------------------------------------------------------------
    # KMeans fallback
    # -----------------------------------------------------------------

    LOGGER.warning(
        "Neither sklearn HDBSCAN nor external hdbscan is installed. "
        "Using KMeans fallback."
    )

    fallback_rows = []
    best_solution = None

    max_k = min(
        10,
        max(
            2,
            n // 25 + 1,
        ),
    )

    for k in range(
        2,
        max_k + 1,
    ):

        model = KMeans(
            n_clusters=k,
            n_init=25,
            random_state=SEED,
            max_iter=500,
        )

        labels = model.fit_predict(
            X
        )

        sample_size = min(
            3000,
            n,
        )

        if n > sample_size:

            rng = np.random.default_rng(
                SEED
            )

            indices = rng.choice(
                n,
                size=sample_size,
                replace=False,
            )

        else:

            indices = np.arange(n)

        silhouette = float(
            silhouette_score(
                X[indices],
                labels[indices],
            )
        )

        quality = (
            silhouette + 1
        ) / 2

        fallback_rows.append(
            {
                "algorithm": "KMeans_fallback",
                "k": k,
                "silhouette": silhouette,
                "quality": quality,
            }
        )

        if (
            best_solution is None
            or silhouette
            > best_solution[0]
        ):

            best_solution = (
                silhouette,
                model,
                labels,
            )

    if best_solution is None:
        raise RuntimeError(
            "Unable to produce any clustering solution."
        )

    candidate_df = (
        pd.DataFrame(
            fallback_rows
        )
        .sort_values(
            "quality",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best_silhouette, best_model, labels = (
        best_solution
    )

    metadata = {
        "algorithm": "KMeans_fallback",
        "k": int(
            best_model.n_clusters
        ),
        "n_clusters": int(
            best_model.n_clusters
        ),
        "coverage": 1.0,
        "noise_share": 0.0,
        "mean_probability": 1.0,
        "silhouette": float(
            best_silhouette
        ),
        "davies_bouldin": float(
            davies_bouldin_score(
                X,
                labels,
            )
        ),
        "bootstrap_ari": None,
        "selection_score": float(
            (
                best_silhouette
                + 1
            ) / 2
        ),
    }

    return (
        labels,
        metadata,
        candidate_df,
        best_model,
    )


# ---------------------------------------------------------------------
# Profile scoring
# ---------------------------------------------------------------------

def robust_profile_scores(
    features: pl.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:

    X = (
        features
        .select(feature_columns)
        .to_pandas()
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy()
    )

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        unit_variance=True,
    )

    Z = scaler.fit_transform(X)

    return pd.DataFrame(
        Z,
        columns=feature_columns,
    )


def make_profile_tables(
    feature_df: pd.DataFrame,
    profile_z: pd.DataFrame,
    labels: np.ndarray,
    feature_groups: Dict[str, List[str]],
    probabilities: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    group_names = list(
        feature_groups.keys()
    )

    group_scores = pd.DataFrame(
        index=np.arange(
            len(feature_df)
        )
    )

    for group, columns in feature_groups.items():

        active_columns = [
            column
            for column in columns
            if column in profile_z.columns
        ]

        if active_columns:

            group_scores[group] = (
                profile_z[
                    active_columns
                ]
                .median(axis=1)
            )

        else:

            group_scores[group] = 0.0

    group_scores["cluster"] = labels
    group_scores["confidence"] = probabilities

    non_noise = (
        group_scores[
            group_scores["cluster"] >= 0
        ]
    )

    if non_noise.empty:

        group_profile = pd.DataFrame()

    else:

        aggregation = {
            "customers": (
                "cluster",
                "size",
            ),
            "confidence_mean": (
                "confidence",
                "mean",
            ),
        }

        for group in group_names:
            aggregation[group] = (
                group,
                "mean",
            )

        group_profile = (
            non_noise
            .groupby("cluster")
            .agg(**aggregation)
            .reset_index()
        )

        group_profile["share"] = (
            group_profile["customers"]
            / len(feature_df)
        )

        group_profile = (
            group_profile
            .sort_values(
                "customers",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    feature_cluster = profile_z.copy()

    feature_cluster["cluster"] = labels

    feature_cluster = feature_cluster[
        feature_cluster["cluster"] >= 0
    ]

    if feature_cluster.empty:

        feature_medians = pd.DataFrame()

    else:

        feature_medians = (
            feature_cluster
            .groupby("cluster")[
                profile_z.columns
            ]
            .median()
            .reset_index()
        )

    return (
        group_profile,
        feature_medians,
    )


# ---------------------------------------------------------------------
# Visual diagnostics
# ---------------------------------------------------------------------

def save_visuals(
    out_dir: Path,
    features_pd: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    pca_xy: np.ndarray,
    pca_variance: np.ndarray,
    group_profile: pd.DataFrame,
    sample_for_plots: int,
) -> None:

    plots_dir = (
        out_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        SEED
    )

    n = len(
        features_pd
    )

    if n > sample_for_plots:

        indices = rng.choice(
            n,
            size=sample_for_plots,
            replace=False,
        )

    else:

        indices = np.arange(n)

    # -------------------------------------------------------------
    # 01. PCA segment map
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    unique_labels = sorted(
        np.unique(labels)
    )

    for cluster in unique_labels:

        cluster_indices = indices[
            labels[indices] == cluster
        ]

        if cluster == -1:

            ax.scatter(
                pca_xy[
                    cluster_indices,
                    0
                ],
                pca_xy[
                    cluster_indices,
                    1
                ],
                s=9,
                alpha=0.20,
                label="Noise",
            )

        else:

            ax.scatter(
                pca_xy[
                    cluster_indices,
                    0
                ],
                pca_xy[
                    cluster_indices,
                    1
                ],
                s=12,
                alpha=0.55,
                label=f"Cluster {cluster}",
            )

    ax.set_title(
        "Customer Behavioral Segments — PCA Space "
        f"({pca_variance[0] * 100:.1f}% / "
        f"{pca_variance[1] * 100:.1f}% variance)"
    )

    ax.set_xlabel(
        "PC1"
    )

    ax.set_ylabel(
        "PC2"
    )

    ax.legend(
        loc="best",
        ncol=2,
        fontsize=8,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "01_pca_clusters.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # 02. Segment size
    # -------------------------------------------------------------

    sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(
        len(sizes)
    )

    ax.bar(
        x,
        sizes.values,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Noise"
            if label == -1
            else f"Cluster {label}"
            for label in sizes.index
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Customers"
    )

    ax.set_title(
        "Segment Size and Density-Noise Population"
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "02_segment_sizes.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # 03. Behavioral view heatmap
    # -------------------------------------------------------------

    if (
        group_profile is not None
        and not group_profile.empty
    ):

        heatmap_columns = [
            column
            for column in FEATURE_GROUPS.keys()
            if column in group_profile.columns
        ]

        heatmap = (
            group_profile
            .set_index("cluster")[
                heatmap_columns
            ]
        )

        array = heatmap.to_numpy()

        fig, ax = plt.subplots(
            figsize=(
                12,
                max(
                    4.5,
                    0.55
                    * len(heatmap)
                    + 2,
                ),
            )
        )

        image = ax.imshow(
            array,
            aspect="auto",
        )

        ax.set_yticks(
            np.arange(
                len(heatmap)
            )
        )

        ax.set_yticklabels(
            [
                f"Cluster {cluster}"
                for cluster in heatmap.index
            ]
        )

        ax.set_xticks(
            np.arange(
                len(
                    heatmap.columns
                )
            )
        )

        ax.set_xticklabels(
            [
                column
                .replace("_", " ")
                .title()
                for column in heatmap.columns
            ],
            rotation=30,
            ha="right",
        )

        ax.set_title(
            "Behavioral View Profile — "
            "Cluster Median Robust Scores"
        )

        for row_index in range(
            array.shape[0]
        ):

            for column_index in range(
                array.shape[1]
            ):

                ax.text(
                    column_index,
                    row_index,
                    f"{array[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        fig.colorbar(
            image,
            ax=ax,
            shrink=0.8,
            label="Cluster median robust score",
        )

        fig.tight_layout()

        fig.savefig(
            plots_dir
            / "03_behavior_view_heatmap.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

    # -------------------------------------------------------------
    # 04. Economic intensity vs cadence
    # -------------------------------------------------------------

    revenue = (
        features_pd[
            "revenue"
        ]
        .to_numpy()
    )

    invoice_count = (
        features_pd[
            "invoice_count"
        ]
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        np.log1p(
            revenue[indices]
        ),
        np.log1p(
            invoice_count[indices]
        ),
        c=labels[indices],
        s=12,
        alpha=0.35,
    )

    ax.set_xlabel(
        "log(1 + customer revenue)"
    )

    ax.set_ylabel(
        "log(1 + invoice count)"
    )

    ax.set_title(
        "Economic Intensity vs Purchase Cadence"
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "04_value_vs_cadence.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # 05. Return/reversal behavior
    # -------------------------------------------------------------

    return_rate = (
        features_pd[
            "value_return_rate"
        ]
        .to_numpy()
    )

    marker_size = (
        15
        + 80
        * probabilities[indices]
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        np.log1p(
            revenue[indices]
        ),
        return_rate[indices],
        c=labels[indices],
        s=marker_size,
        alpha=0.35,
    )

    ax.set_xlabel(
        "log(1 + customer revenue)"
    )

    ax.set_ylabel(
        "Observed value return / reversal rate"
    )

    ax.set_title(
        "Customer Value vs Reversal Behavior "
        "(marker size = clustering confidence)"
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        plots_dir
        / "05_value_vs_returns.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "plots"
    ).mkdir(
        exist_ok=True
    )

    input_path = (
        Path(
            args.input
        )
        .expanduser()
        .resolve()
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # =============================================================
    # 1. LOAD + CLEAN
    # =============================================================

    raw = load_input(
        input_path,
        args.sheet,
    )

    transactions = clean_transactions(
        raw
    )

    raw = None

    # =============================================================
    # 2. FEATURE ENGINEERING
    # =============================================================

    customer_pl, feature_groups = (
        build_customer_features(
            transactions
        )
    )

    customer_pd = (
        customer_pl
        .to_pandas()
    )

    if (
        len(customer_pd)
        < args.min_customers
    ):

        raise ValueError(
            f"Only {len(customer_pd):,} customers are available; "
            f"need at least {args.min_customers:,}."
        )

    # =============================================================
    # 3. ROBUST FEATURE MATRIX
    # =============================================================

    X, feature_columns, used_groups = (
        robust_matrix(
            customer_pl,
            feature_groups,
        )
    )

    LOGGER.info(
        "Matrix shape after robust preprocessing: %s",
        X.shape,
    )

    # =============================================================
    # 4. PCA
    # =============================================================

    full_pca = PCA(
        n_components=min(
            args.max_pca_components,
            X.shape[1],
        ),
        svd_solver="full",
        random_state=SEED,
    )

    Z = full_pca.fit_transform(
        X
    )

    cumulative_variance = (
        np.cumsum(
            full_pca
            .explained_variance_ratio_
        )
    )

    requested_components = (
        int(
            np.searchsorted(
                cumulative_variance,
                args.pca_variance,
            )
        )
        + 1
    )

    if Z.shape[1] >= 2:

        n_components_for_clustering = min(
            max(
                2,
                requested_components,
            ),
            Z.shape[1],
        )

    else:

        n_components_for_clustering = 1

    Z_cluster = Z[
        :,
        :n_components_for_clustering,
    ]

    LOGGER.info(
        "PCA: %s features -> %s PCs for clustering; "
        "cumulative variance %.2f%%",
        X.shape[1],
        n_components_for_clustering,
        cumulative_variance[
            n_components_for_clustering - 1
        ]
        * 100,
    )

    # =============================================================
    # 5. ROBUST CLUSTERING
    # =============================================================

    (
        labels,
        cluster_metadata,
        candidate_models,
        cluster_model,
    ) = choose_clusters(
        Z_cluster,
        repeats=args.stability_repeats,
    )

    probabilities = get_probabilities(
        cluster_model,
        labels,
    )

    LOGGER.info(
        "Selected %s | clusters=%s | coverage=%.1f%% | "
        "noise=%.1f%% | silhouette=%.3f",
        cluster_metadata.get(
            "algorithm"
        ),
        cluster_metadata.get(
            "n_clusters"
        ),
        cluster_metadata.get(
            "coverage",
            0,
        )
        * 100,
        cluster_metadata.get(
            "noise_share",
            0,
        )
        * 100,
        cluster_metadata.get(
            "silhouette",
            float("nan"),
        ),
    )

    # =============================================================
    # 6. CUSTOMER SEGMENT OUTPUT
    # =============================================================

    customer_pd["segment"] = labels

    customer_pd[
        "segment_confidence"
    ] = probabilities

    segment_names = []

    for label in labels:

        if label == -1:
            segment_names.append(
                "Noise / low-density"
            )
        else:
            segment_names.append(
                f"Segment_{int(label):02d}"
            )

    customer_pd[
        "segment_name"
    ] = segment_names

    customer_pd.to_csv(
        output_dir
        / "customer_segments.csv",
        index=False,
    )

    try:

        segmented_polars = (
            customer_pl
            .with_columns(
                [
                    pl.Series(
                        "segment",
                        labels,
                    ),

                    pl.Series(
                        "segment_confidence",
                        probabilities,
                    ),
                ]
            )
        )

        segmented_polars.write_parquet(
            output_dir
            / "customer_segments.parquet"
        )

    except Exception as exc:

        LOGGER.warning(
            "Parquet export failed: %s",
            exc,
        )

    # =============================================================
    # 7. PCA COORDINATES
    # =============================================================

    pca_data = {
        f"PC{i + 1}": Z[:, i]
        for i in range(
            Z.shape[1]
        )
    }

    pca_pd = pd.DataFrame(
        pca_data
    )

    pca_pd.insert(
        0,
        "Customer ID",
        customer_pd[
            "Customer ID"
        ].to_numpy(),
    )

    pca_pd["segment"] = labels

    pca_pd[
        "segment_confidence"
    ] = probabilities

    pca_pd.to_csv(
        output_dir
        / "pca_coordinates.csv",
        index=False,
    )

    # =============================================================
    # 8. PCA LOADINGS
    # =============================================================

    loadings = pd.DataFrame(
        full_pca.components_.T,
        index=feature_columns,
        columns=[
            f"PC{i + 1}"
            for i in range(
                full_pca.components_.shape[0]
            )
        ],
    )

    loadings.to_csv(
        output_dir
        / "pca_feature_loadings.csv"
    )

    # =============================================================
    # 9. CLUSTER PROFILES
    # =============================================================

    profile_z = robust_profile_scores(
        customer_pl,
        feature_columns,
    )

    (
        group_profile,
        feature_profile,
    ) = make_profile_tables(
        customer_pd,
        profile_z,
        labels,
        used_groups,
        probabilities,
    )

    group_profile.to_csv(
        output_dir
        / "segment_profiles.csv",
        index=False,
    )

    feature_profile.to_csv(
        output_dir
        / "segment_feature_medians_robust.csv",
        index=False,
    )

    raw_medians = (
        customer_pd
        .assign(
            segment=labels
        )
    )

    raw_medians = (
        raw_medians[
            raw_medians["segment"] >= 0
        ]
        .groupby("segment")[
            feature_columns
        ]
        .median()
        .reset_index()
    )

    raw_medians.to_csv(
        output_dir
        / "segment_feature_medians_raw.csv",
        index=False,
    )

    # =============================================================
    # 10. MODEL CANDIDATE DIAGNOSTICS
    # =============================================================

    candidate_models.to_csv(
        output_dir
        / "cluster_model_candidates.csv",
        index=False,
    )

    # =============================================================
    # 11. VISUALS
    # =============================================================

    if Z.shape[1] >= 2:

        pca_visual = Z[
            :,
            :2,
        ]

        pca_visual_variance = np.array(
            [
                full_pca.explained_variance_ratio_[0],
                full_pca.explained_variance_ratio_[1],
            ]
        )

    else:

        pca_visual = np.column_stack(
            [
                Z[:, 0],
                np.zeros(
                    Z.shape[0]
                ),
            ]
        )

        pca_visual_variance = np.array(
            [
                full_pca.explained_variance_ratio_[0],
                0.0,
            ]
        )

    save_visuals(
        out_dir=output_dir,
        features_pd=customer_pd,
        labels=labels,
        probabilities=probabilities,
        pca_xy=pca_visual,
        pca_variance=pca_visual_variance,
        group_profile=group_profile,
        sample_for_plots=args.sample_for_plots,
    )

    # =============================================================
    # 12. FEATURE DICTIONARY
    # =============================================================

    feature_dictionary = []

    for group, columns in used_groups.items():

        for column in columns:

            feature_dictionary.append(
                {
                    "feature": column,
                    "behavioral_view": group,
                }
            )

    pd.DataFrame(
        feature_dictionary
    ).to_csv(
        output_dir
        / "feature_dictionary.csv",
        index=False,
    )

    # =============================================================
    # 13. MODEL METADATA
    # =============================================================

    metadata = {
        "input": str(
            input_path
        ),

        "rows_after_cleaning": int(
            transactions.height
        ),

        "customers": int(
            customer_pd.shape[0]
        ),

        "reference_date_for_recency": str(
            customer_pd[
                "reference_date"
            ].iloc[0]
        ),

        "feature_count_before_variance_filter": int(
            len(
                [
                    feature
                    for group
                    in feature_groups.values()
                    for feature
                    in group
                ]
            )
        ),

        "feature_count_after_variance_filter": int(
            len(feature_columns)
        ),

        "feature_groups": used_groups,

        "pca_components_full": int(
            Z.shape[1]
        ),

        "pca_components_for_clustering": int(
            n_components_for_clustering
        ),

        "pca_cumulative_variance_for_clustering": float(
            cumulative_variance[
                n_components_for_clustering - 1
            ]
        ),

        "pca_explained_variance_ratio": [
            float(value)
            for value
            in full_pca.explained_variance_ratio_
        ],

        "cluster_model": cluster_metadata,

        "seed": SEED,
    }

    with open(
        output_dir
        / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=2,
            default=str,
        )

    # =============================================================
    # 14. RUN MANIFEST
    # =============================================================

    manifest = {
        "files": [
            "customer_segments.csv",
            "customer_segments.parquet",
            "pca_coordinates.csv",
            "pca_feature_loadings.csv",
            "segment_profiles.csv",
            "segment_feature_medians_robust.csv",
            "segment_feature_medians_raw.csv",
            "cluster_model_candidates.csv",
            "feature_dictionary.csv",
            "model_metadata.json",
            "plots/01_pca_clusters.png",
            "plots/02_segment_sizes.png",
            "plots/03_behavior_view_heatmap.png",
            "plots/04_value_vs_cadence.png",
            "plots/05_value_vs_returns.png",
        ]
    }

    with open(
        output_dir
        / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as manifest_file:

        json.dump(
            manifest,
            manifest_file,
            indent=2,
        )

    LOGGER.info(
        "Finished. Outputs written to: %s",
        output_dir,
    )


if __name__ == "__main__":
    main()