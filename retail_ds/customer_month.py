"""
Canonical customer-month panel construction.

This module builds the single reusable customer × calendar_month analytical panel
that serves as the backbone for all downstream analytics (segmentation, cohorts,
CLV, churn, next-purchase, reactivation, decisioning).

Grain: Customer ID × Calendar Month (for months ≥ cohort_month)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

LOGGER = logging.getLogger("retail_ds.customer_month")


@dataclass
class CustomerMonthConfig:
    """Configuration for customer-month panel construction."""
    reactivation_gap_months: int = 1
    min_observation_months: int = 1


def build_customer_month_panel(
    tx: pl.DataFrame,
    config: Optional[CustomerMonthConfig] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict]:
    """
    Build the canonical customer-month dense panel.

    Args:
        tx: Cleaned and classified transactions (output of cleaning.clean_transactions + transactions.classify_transactions)
        config: Optional configuration

    Returns:
        Tuple of (customer_month_sparse, customer_month_dense, metadata)
        - customer_month_sparse: Only months with activity (sales or returns)
        - customer_month_dense: Dense grid Customer ID × Calendar Month with zeros for inactive months
        - metadata: Observation window info
    """
    if config is None:
        config = CustomerMonthConfig()

    sales = tx.filter(pl.col("is_sale") & pl.col("is_positive_price"))
    returns = tx.filter(
        pl.col("is_return") | pl.col("is_cancellation") | pl.col("is_discount") | pl.col("is_postage") | pl.col("is_fee")
    )

    if sales.height == 0:
        raise ValueError("No clean positive sales found.")

    # -------------------------------------------------------------------------
    # Invoice-level aggregation
    # -------------------------------------------------------------------------
    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
                pl.col("Quantity").sum().alias("invoice_units"),
                pl.col("StockCode").n_unique().alias("invoice_unique_products"),
                pl.col("Price").median().alias("invoice_median_price"),
            ]
        )
        .with_columns(
            pl.col("invoice_date").dt.truncate("1mo").alias("calendar_month"),
        )
    )

    # -------------------------------------------------------------------------
    # First purchase / acquisition cohort
    # -------------------------------------------------------------------------
    first_purchase = (
        invoice.group_by("Customer ID")
        .agg(pl.col("invoice_date").min().alias("first_purchase_date"))
        .with_columns(
            pl.col("first_purchase_date").dt.truncate("1mo").alias("cohort_month"),
        )
    )

    # -------------------------------------------------------------------------
    # Sales month aggregation
    # -------------------------------------------------------------------------
    monthly_sales = (
        invoice.group_by(["Customer ID", "calendar_month"])
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("invoice_revenue").sum().alias("gross_revenue"),
                pl.col("invoice_units").sum().alias("units"),
                pl.col("invoice_unique_products").sum().alias("product_breadth"),
                pl.col("invoice_revenue").median().alias("median_order_value"),
            ]
        )
        .with_columns(
            [
                pl.lit(0.0).alias("return_value"),
                pl.lit(0.0).alias("return_units"),
                pl.lit(1).alias("active"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Returns month aggregation
    # -------------------------------------------------------------------------
    if returns.height:
        monthly_returns = (
            returns.with_columns(
                pl.col("InvoiceDate").dt.truncate("1mo").alias("calendar_month"),
            )
            .group_by(["Customer ID", "calendar_month"])
            .agg(
                [
                    pl.col("return_value").sum().alias("return_value"),
                    pl.col("Quantity").filter(pl.col("Quantity") < 0).abs().sum().fill_null(0.0).alias("return_units"),
                ]
            )
            .with_columns(
                [
                    pl.lit(0).alias("orders"),
                    pl.lit(0.0).alias("gross_revenue"),
                    pl.lit(0.0).alias("units"),
                    pl.lit(0.0).alias("product_breadth"),
                    pl.lit(0.0).alias("median_order_value"),
                    pl.lit(0).alias("active"),
                ]
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
                "product_breadth": pl.Float64,
                "median_order_value": pl.Float64,
                "active": pl.Int64,
            }
        )

    # -------------------------------------------------------------------------
    # Combine sales + returns into sparse customer-month
    # -------------------------------------------------------------------------
    customer_month_sparse = (
        pl.concat([monthly_sales, monthly_returns], how="diagonal_relaxed")
        .group_by(["Customer ID", "calendar_month"])
        .agg(
            [
                pl.col("orders").sum().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("product_breadth").sum().alias("product_breadth"),
                pl.col("median_order_value").mean().fill_null(0.0).alias("median_order_value"),
                pl.col("return_value").sum().alias("return_value"),
                pl.col("return_units").sum().alias("return_units"),
                pl.col("active").max().alias("active"),
            ]
        )
        .sort(["Customer ID", "calendar_month"])
    )

    # Net revenue
    customer_month_sparse = customer_month_sparse.with_columns(
        [
            (pl.col("gross_revenue") - pl.col("return_value")).alias("net_revenue"),
            (pl.col("return_value") / (pl.col("gross_revenue") + 1e-9)).alias("return_to_gross_ratio"),
            (pl.col("gross_revenue") / pl.col("orders").clip(lower_bound=1)).alias("aov"),
        ]
    )

    # -------------------------------------------------------------------------
    # Observation window
    # -------------------------------------------------------------------------
    observation_start = customer_month_sparse.select(pl.col("calendar_month").min()).item()
    observation_end = customer_month_sparse.select(pl.col("calendar_month").max()).item()

    # -------------------------------------------------------------------------
    # Dense monthly panel: customer × calendar month after acquisition
    # -------------------------------------------------------------------------
    calendar_count = (
        (observation_end.year - observation_start.year) * 12
        + observation_end.month - observation_start.month + 1
    )

    calendar_rows = [
        {"calendar_month": observation_start + pd.DateOffset(months=i)}
        for i in range(calendar_count)
    ]

    calendar = pl.from_pandas(
        pd.DataFrame(calendar_rows).assign(
            calendar_month=lambda x: pd.to_datetime(x["calendar_month"])
        )
    ).with_columns(pl.col("calendar_month").cast(pl.Datetime("us")))

    customers = first_purchase.select(["Customer ID", "cohort_month"]).unique()

    dense = (
        customers.join(calendar, how="cross")
        .filter(pl.col("calendar_month") >= pl.col("cohort_month"))
        .join(customer_month_sparse, on=["Customer ID", "calendar_month"], how="left")
        .with_columns(
            [
                pl.col("orders").fill_null(0).cast(pl.Float64),
                pl.col("gross_revenue").fill_null(0.0),
                pl.col("net_revenue").fill_null(0.0),
                pl.col("units").fill_null(0.0),
                pl.col("product_breadth").fill_null(0.0),
                pl.col("median_order_value").fill_null(0.0),
                pl.col("return_value").fill_null(0.0),
                pl.col("return_units").fill_null(0.0),
                pl.col("active").fill_null(0).cast(pl.Int8),
            ]
        )
        .with_columns(
            [
                (pl.col("calendar_month").dt.year() * 12 + pl.col("calendar_month").dt.month()).alias("calendar_month_id"),
                (pl.col("cohort_month").dt.year() * 12 + pl.col("cohort_month").dt.month()).alias("cohort_month_id"),
            ]
        )
        .with_columns(
            (pl.col("calendar_month_id") - pl.col("cohort_month_id")).alias("age_month"),
        )
        .sort(["Customer ID", "calendar_month"])
    )

    # -------------------------------------------------------------------------
    # Cumulative features (lifetime-to-date)
    # -------------------------------------------------------------------------
    dense = dense.with_columns(
        [
            pl.col("orders").cum_sum().over("Customer ID").alias("lifetime_orders"),
            pl.col("gross_revenue").cum_sum().over("Customer ID").alias("lifetime_gross_revenue"),
            pl.col("net_revenue").cum_sum().over("Customer ID").alias("lifetime_net_revenue"),
            pl.col("active").cum_sum().over("Customer ID").alias("lifetime_active_months"),
            pl.col("return_value").cum_sum().over("Customer ID").alias("lifetime_return_value"),
        ]
    )

    # -------------------------------------------------------------------------
    # Lagged features
    # -------------------------------------------------------------------------
    for lag in (1, 2, 3):
        dense = dense.with_columns(
            [
                pl.col("orders").shift(lag).over("Customer ID").fill_null(0.0).alias(f"orders_lag_{lag}"),
                pl.col("net_revenue").shift(lag).over("Customer ID").fill_null(0.0).alias(f"net_revenue_lag_{lag}"),
            ]
        )

    dense = dense.with_columns(
        [
            (pl.col("orders") + pl.col("orders_lag_1") + pl.col("orders_lag_2")).alias("orders_last_3m"),
            (pl.col("net_revenue") + pl.col("net_revenue_lag_1") + pl.col("net_revenue_lag_2")).alias("net_revenue_last_3m"),
            (pl.col("orders_lag_1") + pl.col("orders_lag_2") + pl.col("orders_lag_3")).alias("orders_prev_3m"),
            (pl.col("net_revenue_lag_1") + pl.col("net_revenue_lag_2") + pl.col("net_revenue_lag_3")).alias("net_revenue_prev_3m"),
        ]
    )

    dense = dense.with_columns(
        [
            pl.col("net_revenue").rolling_mean(3).over("Customer ID").fill_null(0.0).alias("net_revenue_3m_mean"),
            pl.col("orders").rolling_mean(3).over("Customer ID").fill_null(0.0).alias("orders_3m_mean"),
        ]
    )

    # Last active month ID via cumulative maximum of active month IDs.
    dense = dense.with_columns(
        [
            (pl.col("calendar_month").dt.year() * 12 + pl.col("calendar_month").dt.month()).alias("calendar_month_id"),
        ]
    )

    dense = dense.with_columns(
        pl.when(pl.col("active") == 1).then(pl.col("calendar_month_id")).otherwise(None).alias("active_month_id_candidate")
    )

    dense = dense.with_columns(
        pl.col("active_month_id_candidate").forward_fill().over("Customer ID").alias("last_active_month_id")
    )

    dense = dense.with_columns(
        [
            (pl.col("calendar_month_id") - pl.col("last_active_month_id")).fill_null(pl.col("age_month")).clip(lower_bound=0).alias("recency_months"),
            (pl.col("lifetime_orders") / pl.col("age_month").add(1)).alias("lifetime_order_rate_per_month"),
            (pl.col("lifetime_active_months") / pl.col("age_month").add(1)).alias("lifetime_active_month_share"),
            (pl.col("net_revenue_last_3m") / (pl.col("net_revenue_prev_3m").abs() + 10.0)).alias("recent_value_momentum"),
            (pl.col("orders_last_3m") / pl.col("age_month").clip(lower_bound=1)).alias("recent_orders_intensity"),
            (pl.col("lifetime_net_revenue") / pl.col("lifetime_orders").clip(lower_bound=1)).alias("historical_value_per_order"),
            (pl.col("lifetime_return_value") / (pl.col("lifetime_gross_revenue") + 1e-9)).alias("historical_return_ratio"),
            (pl.col("calendar_month").dt.month().cast(pl.Float64)).alias("calendar_month_number"),
        ]
    )

    # Seasonal harmonics.
    dense = dense.with_columns(
        [
            (2 * math.pi * pl.col("calendar_month_number") / 12.0).sin().alias("month_sin"),
            (2 * math.pi * pl.col("calendar_month_number") / 12.0).cos().alias("month_cos"),
        ]
    )

    # Target is next-month activity / next-month net revenue.
    dense = dense.with_columns(
        [
            pl.col("active").shift(-1).over("Customer ID").alias("next_active"),
            pl.col("net_revenue").shift(-1).over("Customer ID").alias("next_net_revenue"),
            pl.col("calendar_month").shift(-1).over("Customer ID").alias("next_calendar_month"),
        ]
    )

    # Add rolling window features
    dense = add_rolling_features(dense, [1, 3, 6, 12])

    metadata = {
        "observation_start": str(observation_start),
        "observation_end": str(observation_end),
        "calendar_months": calendar_count,
        "unique_customers": customers.height,
    }

    LOGGER.info(
        "Customer-month panel: %s customers × %s months = %s rows (dense)",
        f"{customers.height:,}",
        calendar_count,
        f"{dense.height:,}",
    )

    return customer_month_sparse, dense, metadata


def add_rolling_features(dense: pl.DataFrame, windows: List[int] = [1, 3, 6, 12]) -> pl.DataFrame:
    """Add rolling window features to dense customer-month panel."""
    out = dense

    for w in windows:
        out = out.with_columns(
            [
                pl.col("orders").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"orders_last_{w}m"),
                pl.col("gross_revenue").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"revenue_last_{w}m"),
                pl.col("net_revenue").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"net_revenue_last_{w}m"),
                pl.col("units").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"units_last_{w}m"),
                pl.col("active").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"active_months_last_{w}m"),
                pl.col("return_value").rolling_sum(window_size=w, min_samples=1).over("Customer ID").alias(f"return_value_last_{w}m"),
            ]
        )

        # Rolling averages
        out = out.with_columns(
            [
                (pl.col(f"revenue_last_{w}m") / pl.col(f"orders_last_{w}m").clip(lower_bound=1)).alias(f"aov_last_{w}m"),
                (pl.col(f"return_value_last_{w}m") / (pl.col(f"revenue_last_{w}m") + 1e-9)).alias(f"return_rate_last_{w}m"),
            ]
        )

    # Acceleration features (recent vs previous window)
    for w in [3, 6, 12]:
        if f"revenue_last_{w}m" in out.columns and w * 2 <= max(windows) * 2:
            prev_col = f"revenue_last_{w}m"
            # For simplicity, we'll use shift to get previous window
            out = out.with_columns(
                pl.col(f"revenue_last_{w}m").shift(w).over("Customer ID").fill_null(0.0).alias(f"revenue_prev_{w}m"),
            )
            out = out.with_columns(
                (
                    (pl.col(f"revenue_last_{w}m") - pl.col(f"revenue_prev_{w}m"))
                    / (pl.col(f"revenue_prev_{w}m").abs() + 1e-9)
                ).alias(f"revenue_acceleration_{w}m"),
            )

    return out