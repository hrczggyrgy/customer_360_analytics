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
        --config config/project.yaml \
        --output-dir ./cohort_analysis_output
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

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.config import ProjectConfig, load_config, add_config_args


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
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
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
    add_config_args(parser)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config).",
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
# CUSTOMER-MONTH EVENT MODEL
# =============================================================================

def build_customer_month_events(
    tx: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict]:

    sales = tx.filter(pl.col("is_sale") & pl.col("is_positive_price"))
    returns = tx.filter(pl.col("is_return") | pl.col("is_cancellation"))

    if sales.height == 0:
        raise ValueError("No clean positive sales were found.")

    # -------------------------------------------------------------------------
    # Invoice-level aggregation
    # -------------------------------------------------------------------------

    sale_invoices = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("gross_revenue"),
                pl.col("Quantity").sum().alias("units"),
                pl.col("StockCode").n_unique().alias("unique_products"),
                pl.col("Price").median().alias("median_line_price"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # First purchase / acquisition cohort
    # -------------------------------------------------------------------------

    first_purchase = (
        sale_invoices.group_by("Customer ID")
        .agg(pl.col("invoice_date").min().alias("first_purchase_date"))
        .with_columns(pl.col("first_purchase_date").dt.truncate("1mo").alias("cohort_month"))
    )

    # -------------------------------------------------------------------------
    # Customer-month sales events
    # -------------------------------------------------------------------------

    sales_month = (
        sale_invoices.with_columns(pl.col("invoice_date").dt.truncate("1mo").alias("calendar_month"))
        .group_by(["Customer ID", "calendar_month"])
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("unique_products").sum().alias("product_lines"),
            ]
        )
        .with_columns(
            [
                pl.lit(0.0).alias("return_value"),
                pl.lit(0.0).alias("return_units"),
                pl.lit(1).alias("is_active_purchase_month"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Customer-month return events
    # -------------------------------------------------------------------------

    if returns.height:
        return_month = (
            returns.filter(pl.col("Customer ID").is_not_null())
            .with_columns(pl.col("InvoiceDate").dt.truncate("1mo").alias("calendar_month"))
            .group_by(["Customer ID", "calendar_month"])
            .agg(
                [
                    pl.col("gross_merchandise_revenue").sum().alias("return_value"),
                    pl.col("Quantity").filter(pl.col("Quantity") < 0).abs().sum().fill_null(0.0).alias("return_units"),
                ]
            )
            .with_columns(
                [
                    pl.lit(0).alias("orders"),
                    pl.lit(0.0).alias("gross_revenue"),
                    pl.lit(0.0).alias("units"),
                    pl.lit(0.0).alias("product_lines"),
                    pl.lit(0).alias("is_active_purchase_month"),
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
        pl.concat([sales_month, return_month], how="diagonal_relaxed")
        .group_by(["Customer ID", "calendar_month"])
        .agg(
            [
                pl.col("orders").sum().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("product_lines").sum().alias("product_lines"),
                pl.col("return_value").sum().alias("return_value"),
                pl.col("return_units").sum().alias("return_units"),
                pl.col("is_active_purchase_month").max().alias("is_active_purchase_month"),
            ]
        )
        .join(first_purchase, on="Customer ID", how="left")
        .with_columns(
            [
                (
                    (pl.col("calendar_month").dt.year() - pl.col("cohort_month").dt.year()) * 12
                    + (pl.col("calendar_month").dt.month() - pl.col("cohort_month").dt.month())
                ).alias("age_month"),
                (pl.col("calendar_month").dt.year() * 12 + pl.col("calendar_month").dt.month()).alias("calendar_month_id"),
                (pl.col("cohort_month").dt.year() * 12 + pl.col("cohort_month").dt.month()).alias("cohort_month_id"),
                (pl.col("gross_revenue") - pl.col("return_value")).alias("net_revenue"),
                (pl.col("return_value") / (pl.col("gross_revenue") + 1e-9)).alias("return_to_sales_ratio"),
            ]
        )
        .sort(["Customer ID", "calendar_month"])
    )

    # Exclude anomalous return activity that predates the first observed sale.
    customer_month = customer_month.filter(pl.col("age_month") >= 0)

    # -------------------------------------------------------------------------
    # Reactivation
    # -------------------------------------------------------------------------

    active_months = (
        customer_month.filter(pl.col("is_active_purchase_month") == 1)
        .select(["Customer ID", "calendar_month", "age_month"])
        .sort(["Customer ID", "calendar_month"])
        .with_columns(pl.col("age_month").shift(1).over("Customer ID").alias("previous_active_age_month"))
        .with_columns(
            [
                pl.when(pl.col("previous_active_age_month").is_null())
                .then(pl.lit(0))
                .when((pl.col("age_month") - pl.col("previous_active_age_month")) > 1)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("is_reactivation"),
                (pl.col("age_month") - pl.col("previous_active_age_month")).alias("active_month_gap"),
            ]
        )
    )

    customer_month = (
        customer_month.join(
            active_months.select(["Customer ID", "calendar_month", "is_reactivation", "active_month_gap"]),
            on=["Customer ID", "calendar_month"], how="left"
        )
        .with_columns([pl.col("is_reactivation").fill_null(0), pl.col("active_month_gap").fill_null(0)])
        .sort(["Customer ID", "calendar_month"])
    )

    # -------------------------------------------------------------------------
    # Observation window
    # -------------------------------------------------------------------------

    last_observation_month = sales.select(pl.col("InvoiceDate").max().dt.truncate("1mo")).item()
    first_observation_month = sales.select(pl.col("InvoiceDate").min().dt.truncate("1mo")).item()

    months_observable = (
        (last_observation_month.year - first_observation_month.year) * 12
        + (last_observation_month.month - first_observation_month.month)
        + 1
    )

    data_quality = {
        "raw_rows": int(tx.height),
        "clean_sale_lines": int(sales.height),
        "customer_month_rows": int(customer_month.height),
        "unique_customers": int(first_purchase.height),
        "observation_start": str(first_observation_month),
        "observation_end": str(last_observation_month),
        "observable_calendar_months": int(months_observable),
        "return_rows": int(returns.height),
        "return_only_customer_months": int(
            customer_month.filter(
                (pl.col("is_active_purchase_month") == 0) & (pl.col("return_value") > 0)
            ).height
        ),
    }

    return customer_month, first_purchase, data_quality


# =============================================================================
# COHORT GRID
# =============================================================================

def dense_age_calendar(customer_month: pl.DataFrame, last_observation_month) -> pl.DataFrame:
    cohort_calendar = customer_month.select("cohort_month").unique().sort("cohort_month")
    rows = []
    for cohort in cohort_calendar["cohort_month"].to_list():
        max_age = (
            (last_observation_month.year - cohort.year) * 12
            + (last_observation_month.month - cohort.month)
        )
        for age in range(max_age + 1):
            rows.append({"cohort_month": cohort, "age_month": age})
    return pl.DataFrame(rows)


def build_cohort_tables(customer_month: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    last_observation_month = customer_month.select(pl.col("calendar_month").max()).item()

    cohort_sizes = (
        customer_month.filter(pl.col("age_month") == 0)
        .group_by("cohort_month")
        .agg(pl.col("Customer ID").n_unique().alias("cohort_customers"))
    )

    customer_age = customer_month.select(
        ["Customer ID", "cohort_month", "age_month", "calendar_month",
         "orders", "gross_revenue", "net_revenue", "units", "return_value", "is_reactivation"]
    )

    # Cohort x age metrics
    cohort_age = (
        customer_age.group_by(["cohort_month", "age_month"])
        .agg(
            [
                pl.col("Customer ID").filter(pl.col("orders") > 0).n_unique().alias("active_customers"),
                pl.col("orders").sum().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("net_revenue").sum().alias("net_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("return_value").sum().alias("return_value"),
                pl.col("is_reactivation").sum().alias("reactivated_customer_months"),
            ]
        )
        .join(cohort_sizes, on="cohort_month", how="left")
        .with_columns(
            [
                (pl.col("active_customers") / pl.col("cohort_customers")).alias("logo_retention"),
                (pl.col("orders") / pl.col("cohort_customers")).alias("orders_per_acquired_customer"),
                (pl.col("gross_revenue") / pl.col("cohort_customers")).alias("gross_revenue_per_acquired_customer"),
                (pl.col("net_revenue") / pl.col("cohort_customers")).alias("net_revenue_per_acquired_customer"),
                (pl.col("units") / pl.col("cohort_customers")).alias("units_per_acquired_customer"),
                (pl.col("reactivated_customer_months") / pl.col("active_customers").clip(lower_bound=1))
                .alias("reactivation_share_of_active"),
            ]
        )
        .sort(["cohort_month", "age_month"])
    )

    cohort_age = cohort_age.with_columns(
        pl.when(pl.col("active_customers") > 0)
        .then(pl.col("reactivated_customer_months") / pl.col("active_customers"))
        .otherwise(0.0)
        .alias("reactivation_share_of_active")
    )

    # Cohort baseline
    cohort_baseline = (
        cohort_age.filter(pl.col("age_month") == 0)
        .select(
            [
                "cohort_month",
                pl.col("gross_revenue").alias("baseline_gross_revenue"),
                pl.col("net_revenue").alias("baseline_net_revenue"),
                pl.col("orders").alias("baseline_orders"),
                pl.col("units").alias("baseline_units"),
            ]
        )
    )

    cohort_age = (
        cohort_age.join(cohort_baseline, on="cohort_month", how="left")
        .with_columns(
            [
                (pl.col("gross_revenue") / (pl.col("baseline_gross_revenue") + 1e-9)).alias("gross_revenue_retention"),
                (pl.col("net_revenue") / (pl.col("baseline_net_revenue") + 1e-9)).alias("net_revenue_retention"),
                (pl.col("orders") / (pl.col("baseline_orders") + 1e-9)).alias("order_retention"),
                (pl.col("units") / (pl.col("baseline_units") + 1e-9)).alias("unit_retention"),
            ]
        )
    )

    # Dense maturity-aware grid
    dense = dense_age_calendar(customer_month, last_observation_month)
    dense = (
        dense.join(cohort_age, on=["cohort_month", "age_month"], how="left")
        .join(cohort_sizes, on="cohort_month", how="left")
        .with_columns(pl.lit(True).alias("age_observable"))
        .sort(["cohort_month", "age_month"])
    )

    numeric_defaults = [
        "active_customers", "orders", "gross_revenue", "net_revenue", "units",
        "return_value", "reactivated_customer_months", "logo_retention",
        "orders_per_acquired_customer", "gross_revenue_per_acquired_customer",
        "net_revenue_per_acquired_customer", "units_per_acquired_customer",
        "reactivation_share_of_active", "baseline_gross_revenue", "baseline_net_revenue",
        "baseline_orders", "baseline_units", "gross_revenue_retention",
        "net_revenue_retention", "order_retention", "unit_retention",
    ]

    dense = dense.with_columns(
        [pl.col(c).fill_null(0.0) for c in numeric_defaults if c in dense.columns]
    )

    # Calendar-month performance
    calendar = (
        customer_age.group_by("calendar_month")
        .agg(
            [
                pl.col("Customer ID").filter(pl.col("orders") > 0).n_unique().alias("active_customers"),
                pl.col("orders").sum().alias("orders"),
                pl.col("gross_revenue").sum().alias("gross_revenue"),
                pl.col("net_revenue").sum().alias("net_revenue"),
                pl.col("units").sum().alias("units"),
                pl.col("return_value").sum().alias("return_value"),
                pl.col("is_reactivation").sum().alias("reactivated_customer_months"),
            ]
        )
        .sort("calendar_month")
        .with_columns(
            (pl.col("return_value") / (pl.col("gross_revenue") + 1e-9)).alias("return_to_gross_sales_ratio")
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

def build_cohort_quality(tx: pl.DataFrame, customer_month: pl.DataFrame) -> pl.DataFrame:
    first_month = customer_month.filter(pl.col("age_month") == 0)

    quality = (
        first_month.group_by("cohort_month")
        .agg(
            [
                pl.col("Customer ID").n_unique().alias("new_customers"),
                pl.col("gross_revenue").sum().alias("acquisition_month_gross_revenue"),
                pl.col("net_revenue").sum().alias("acquisition_month_net_revenue"),
                pl.col("orders").sum().alias("acquisition_month_orders"),
                pl.col("units").sum().alias("acquisition_month_units"),
            ]
        )
        .with_columns(
            [
                (pl.col("acquisition_month_gross_revenue") / pl.col("new_customers").clip(lower_bound=1))
                .alias("gross_revenue_per_new_customer"),
                (pl.col("acquisition_month_net_revenue") / pl.col("new_customers").clip(lower_bound=1))
                .alias("net_revenue_per_new_customer"),
                (pl.col("acquisition_month_orders") / pl.col("new_customers").clip(lower_bound=1))
                .alias("orders_per_new_customer"),
                (pl.col("acquisition_month_units") / pl.col("new_customers").clip(lower_bound=1))
                .alias("units_per_new_customer"),
            ]
        )
        .sort("cohort_month")
    )

    acquisition_customer_ids = first_month.select(["Customer ID", "cohort_month"])

    acquisition_lines = (
        tx.filter(pl.col("is_sale") & pl.col("is_positive_price"))
        .with_columns(pl.col("InvoiceDate").dt.truncate("1mo").alias("calendar_month"))
        .join(acquisition_customer_ids, on="Customer ID", how="inner")
        .filter(pl.col("calendar_month") == pl.col("cohort_month"))
    )

    acquisition_behavior = (
        acquisition_lines.group_by("cohort_month")
        .agg(
            [
                pl.col("StockCode").n_unique().alias("unique_products_bought_on_acquisition_month"),
                pl.col("Price").median().alias("median_unit_price_acquisition_month"),
                pl.col("Price").mean().alias("mean_unit_price_acquisition_month"),
                (pl.col("Price") > pl.col("Price").median()).mean().alias("above_cohort_median_price_line_share"),
            ]
        )
    )

    return quality.join(acquisition_behavior, on="cohort_month", how="left")


# =============================================================================
# COHORT SCORECARD
# =============================================================================

def cohort_snapshot_scorecard(
    dense: pl.DataFrame,
    min_cohort_size: int,
    early_age: int,
    mature_age: int,
) -> pl.DataFrame:

    base = dense.filter(pl.col("cohort_customers") >= min_cohort_size)
    observations = []
    cohort_values = base.select("cohort_month").unique().sort("cohort_month")["cohort_month"].to_list()

    for cohort in cohort_values:
        row = {"cohort_month": cohort}
        cohort_data = base.filter(pl.col("cohort_month") == cohort)

        row["cohort_customers"] = cohort_data.select("cohort_customers").item()
        row["max_observable_age"] = cohort_data.select("age_month").max().item()
        row["is_mature_at_early"] = int(row["max_observable_age"] >= early_age)
        row["is_mature_at_mature"] = int(row["max_observable_age"] >= mature_age)

        # Early retention
        if row["is_mature_at_early"]:
            early = cohort_data.filter(pl.col("age_month") == early_age)
            row["early_logo_retention"] = early.select("logo_retention").item() if early.height else None
            row["early_gross_revenue_retention"] = early.select("gross_revenue_retention").item() if early.height else None
            row["early_net_revenue_retention"] = early.select("net_revenue_retention").item() if early.height else None

        # Mature retention
        if row["is_mature_at_mature"]:
            mature = cohort_data.filter(pl.col("age_month") == mature_age)
            row["mature_logo_retention"] = mature.select("logo_retention").item() if mature.height else None
            row["mature_gross_revenue_retention"] = mature.select("gross_revenue_retention").item() if mature.height else None
            row["mature_net_revenue_retention"] = mature.select("net_revenue_retention").item() if mature.height else None

        # Cumulative value at max observable age
        max_age_data = cohort_data.filter(pl.col("age_month") == row["max_observable_age"])
        if max_age_data.height:
            cum_rev = cohort_data.filter(pl.col("age_month") <= row["max_observable_age"]).select(pl.col("net_revenue").sum()).item()
            row["cumulative_net_revenue_per_customer"] = cum_rev / row["cohort_customers"]

        observations.append(row)

    return pl.DataFrame(observations)


# =============================================================================
# MATURITY-AWARE DECAY CURVES
# =============================================================================

def build_maturity_aware_decay(
    dense: pl.DataFrame,
    calendar_month: pl.DataFrame,
    max_age: int,
) -> pl.DataFrame:

    observations = []

    for age in range(max_age + 1):
        age_data = dense.filter(
            (pl.col("age_month") == age) & (pl.col("age_observable"))
        )
        mature = age_data.filter(pl.col("cohort_customers") >= 25)

        if mature.height == 0:
            continue

        obs = {
            "age_month": age,
            "n_cohorts_observable": int(age_data.select(pl.col("cohort_month").n_unique()).item()),
            "n_cohorts_mature": int(mature.select(pl.col("cohort_month").n_unique()).item()),
            "avg_logo_retention_mature": float(mature.select(pl.col("logo_retention").mean()).item()),
            "avg_gross_revenue_retention_mature": float(mature.select(pl.col("gross_revenue_retention").mean()).item()),
            "avg_net_revenue_retention_mature": float(mature.select(pl.col("net_revenue_retention").mean()).item()),
            "avg_order_retention_mature": float(mature.select(pl.col("order_retention").mean()).item()),
            "avg_unit_retention_mature": float(mature.select(pl.col("unit_retention").mean()).item()),
        }
        observations.append(obs)

    return pl.DataFrame(observations)


# =============================================================================
# CUSTOMER LIFECYCLE STATUS
# =============================================================================

def build_customer_lifecycle_status(customer_month: pl.DataFrame) -> pl.DataFrame:
    last_month = customer_month.select(pl.col("calendar_month").max()).item()

    latest = (
        customer_month.filter(pl.col("calendar_month") == last_month)
        .select(
            [
                "Customer ID", "cohort_month", "age_month", "orders", "gross_revenue",
                "net_revenue", "units", "return_value", "is_active_purchase_month",
                "is_reactivation", "consecutive_inactive_months",
            ]
        )
    )

    latest = latest.with_columns(
        [
            (last_month - pl.col("cohort_month").dt.truncate("1mo")).dt.total_days().alias("tenure_days"),
            pl.when(pl.col("is_active_purchase_month") == 1).then(pl.lit("active"))
            .when(pl.col("consecutive_inactive_months") <= 3).then(pl.lit("at_risk"))
            .when(pl.col("consecutive_inactive_months") <= 12).then(pl.lit("stale"))
            .otherwise(pl.lit("dormant")).alias("lifecycle_state"),
        ]
    )

    return latest


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def save_plots(
    out_dir: Path,
    dense: pl.DataFrame,
    cohort_age: pl.DataFrame,
    calendar: pl.DataFrame,
    decay: pl.DataFrame,
    scorecard: pl.DataFrame,
    heatmap_limit: int,
) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Heatmap data preparation
    pivot_cols = [c for c in dense.columns if c.startswith(("logo_", "gross_", "net_", "order_", "unit_")) and c.endswith("_retention")]
    for metric_name in ["logo_retention", "gross_revenue_retention", "net_revenue_retention", "order_retention", "unit_retention"]:
        if metric_name not in dense.columns:
            continue

        pivot = (
            dense.select(["cohort_month", "age_month", metric_name])
            .pivot(values=metric_name, index="cohort_month", on="age_month", aggregate_function="first")
            .sort("cohort_month")
        )

        # Limit columns for heatmap
        age_cols = [c for c in pivot.columns if c != "cohort_month" and isinstance(c, int)]
        age_cols = sorted(age_cols)[:heatmap_limit]
        plot_cols = ["cohort_month"] + age_cols
        pivot = pivot.select(plot_cols)

        fig, ax = plt.subplots(figsize=(max(12, len(age_cols) * 0.5), max(6, pivot.height * 0.35)))
        data = pivot.select(age_cols).to_numpy()
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.2)

        ax.set_yticks(np.arange(pivot.height))
        ax.set_yticklabels([str(c)[:10] for c in pivot["cohort_month"].to_list()], fontsize=7)
        ax.set_xticks(np.arange(len(age_cols)))
        ax.set_xticklabels(age_cols, fontsize=7)
        ax.set_title(f"Cohort {metric_name.replace('_', ' ').title()} Heatmap")
        ax.set_xlabel("Cohort Age (months)")
        ax.set_ylabel("Acquisition Cohort")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Retention Rate")
        fig.tight_layout()
        fig.savefig(plots / f"heatmap_{metric_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Decay curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["logo_retention", "gross_revenue_retention", "net_revenue_retention", "order_retention"]
    for ax, metric in zip(axes.flatten(), metrics):
        if metric not in decay.columns:
            continue
        mature_cols = [c for c in decay.columns if c.startswith(f"avg_{metric}_mature")]
        if not mature_cols:
            continue
        ax.plot(decay["age_month"], decay[mature_cols[0]], marker="o", linewidth=2)
        ax.set_title(f"Mature Cohort {metric.replace('_', ' ').title()} Decay")
        ax.set_xlabel("Cohort Age (months)")
        ax.set_ylabel("Retention Rate")
        ax.grid(alpha=0.15)
        ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(plots / "decay_curves_mature.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Calendar month performance
    fig, ax = plt.subplots(figsize=(14, 6))
    cal_pd = calendar.to_pandas()
    ax.plot(cal_pd["calendar_month"], cal_pd["active_customers"], marker="o", label="Active Customers")
    ax2 = ax.twinx()
    ax2.plot(cal_pd["calendar_month"], cal_pd["net_revenue"], marker="s", color="orange", label="Net Revenue")
    ax.set_xlabel("Calendar Month")
    ax.set_ylabel("Active Customers")
    ax2.set_ylabel("Net Revenue")
    ax.set_title("Calendar Month Performance")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "calendar_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Scorecard table as image
    fig, ax = plt.subplots(figsize=(16, max(4, scorecard.height * 0.3)))
    ax.axis("off")
    sc_pd = scorecard.to_pandas()
    table = ax.table(cellText=sc_pd.values, colLabels=sc_pd.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax.set_title("Cohort Snapshot Scorecard", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(plots / "scorecard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.cohorts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Load canonical transactions
    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"
    if canonical_path.exists():
        LOGGER.info("Loading canonical transactions from %s", canonical_path)
        tx = pl.read_parquet(canonical_path)
    else:
        LOGGER.info("Building canonical transactions from raw data...")
        raw = load_raw_transactions(config.raw_data_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # Build customer-month events
    customer_month, first_purchase, data_quality = build_customer_month_events(tx)

    # Build cohort tables
    tables = build_cohort_tables(customer_month)
    cohort_age = tables["cohort_age"]
    dense = tables["dense_cohort_age"]
    calendar = tables["calendar_month"]
    cohort_sizes = tables["cohort_sizes"]

    # Cohort quality
    quality = build_cohort_quality(tx, customer_month)

    # Scorecard
    scorecard = cohort_snapshot_scorecard(
        dense, args.min_cohort_size, args.early_retention_months, args.mature_retention_months
    )

    # Maturity-aware decay
    max_age = args.max_age_months or dense.select(pl.col("age_month").max()).item()
    decay = build_maturity_aware_decay(dense, calendar, max_age)

    # Customer lifecycle status
    lifecycle = build_customer_lifecycle_status(customer_month)

    # Acquisition cohorts
    acquisition_cohorts = first_purchase.select(["Customer ID", "cohort_month"]).sort("cohort_month")

    # Visualizations
    save_plots(output_dir, dense, cohort_age, calendar, decay, scorecard, args.heatmap_limit)

    # =============================================================================
    # OUTPUTS
    # =============================================================================

    LOGGER.info("Writing outputs...")

    # Wide matrices
    for name in ["logo_retention", "gross_revenue_retention", "net_revenue_retention", "order_retention", "unit_retention"]:
        if name in dense.columns:
            pivot = (
                dense.select(["cohort_month", "age_month", name])
                .pivot(values=name, index="cohort_month", on="age_month", aggregate_function="first")
                .sort("cohort_month")
            )
            pivot.write_csv(output_dir / f"matrix_{name}.csv")

    # Cohort age metrics (long format)
    cohort_age.write_csv(output_dir / "cohort_age_metrics.csv")
    dense.write_csv(output_dir / "cohort_age_metrics_dense.csv")

    # Cohort sizes
    cohort_sizes.write_csv(output_dir / "cohort_sizes.csv")

    # Calendar month performance
    calendar.write_csv(output_dir / "calendar_month_performance.csv")

    # Cohort quality
    quality.write_csv(output_dir / "cohort_acquisition_quality.csv")

    # Scorecard
    scorecard.write_csv(output_dir / "cohort_scorecard.csv")

    # Decay curves
    decay.write_csv(output_dir / "retention_decay_curve.csv")

    # Customer lifecycle status
    lifecycle.write_csv(output_dir / "customer_lifecycle_status.csv")

    # Acquisition cohorts
    acquisition_cohorts.write_csv(output_dir / "customer_acquisition_cohorts.csv")

    # Customer-month panel
    customer_month.write_parquet(output_dir / "customer_month.parquet")

    # Data quality
    with open(output_dir / "data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=2, default=str)

    # Run manifest
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "observation_start": data_quality["observation_start"],
        "observation_end": data_quality["observation_end"],
        "files": [
            "matrix_logo_retention.csv",
            "matrix_net_revenue_retention.csv",
            "matrix_gross_revenue_retention.csv",
            "matrix_order_retention.csv",
            "matrix_unit_retention.csv",
            "cohort_age_metrics.csv",
            "cohort_age_metrics_dense.csv",
            "cohort_sizes.csv",
            "calendar_month_performance.csv",
            "cohort_acquisition_quality.csv",
            "cohort_scorecard.csv",
            "retention_decay_curve.csv",
            "customer_lifecycle_status.csv",
            "customer_acquisition_cohorts.csv",
            "customer_month.parquet",
            "data_quality.json",
            "plots/heatmap_logo_retention.png",
            "plots/heatmap_gross_revenue_retention.png",
            "plots/heatmap_net_revenue_retention.png",
            "plots/heatmap_order_retention.png",
            "plots/heatmap_unit_retention.png",
            "plots/decay_curves_mature.png",
            "plots/calendar_performance.png",
            "plots/scorecard.png",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("Finished. Outputs written to: %s", output_dir)


if __name__ == "__main__":
    main()