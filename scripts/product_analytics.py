#!/usr/bin/env python3
"""
Product Analytics — Product-Level Behavioral Intelligence

This script builds product-level metrics and classifies products into behavioral roles
based on the Online Retail II transaction data.

Metrics per product:
- sales, units, unique customers
- repeat customer rate, customer penetration
- return rate, price distribution
- sales velocity

Product roles (evidence-based):
- acquisition: first purchase for many customers
- repeat: high repeat purchase rate
- basket builder: frequently co-purchased with other products
- retention: purchased by loyal/retained customers
- niche high-value: low volume, high price, specific segments
- volatile: irregular sales pattern

Uses the retail_ds shared package for canonical data processing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("product_analytics")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product analytics for Online Retail II.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--min-sales", type=int, default=10, help="Minimum sales for product inclusion.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_product_metrics(
    tx: pl.DataFrame,
    min_sales: int = 10,
) -> pl.DataFrame:
    """Build comprehensive product-level metrics from canonical transactions.

    One row per StockCode. Description conflicts resolved by most frequent non-null.
    """
    LOGGER.info("Building product-level metrics...")

    # Filter to clean sales
    sales = tx.filter(pl.col("is_clean_sale"))

    # Resolve description conflicts: most frequent non-null per StockCode
    description_map = (
        sales.filter(pl.col("Description").is_not_null() & (pl.col("Description") != ""))
        .group_by(["StockCode", "Description"])
        .len()
        .sort(["StockCode", "len"], descending=[False, True])
        .group_by("StockCode")
        .first()
        .select(["StockCode", "Description"])
    )

    # Product-level aggregations (one row per StockCode)
    product_metrics = (
        sales.group_by("StockCode")
        .agg(
            [
                pl.col("gross_merchandise_revenue").sum().alias("total_revenue"),
                pl.col("Quantity").sum().alias("total_units"),
                pl.col("Invoice").n_unique().alias("total_invoices"),
                pl.col("Customer ID").n_unique().alias("unique_customers"),
                pl.col("Price").mean().alias("avg_price"),
                pl.col("Price").median().alias("median_price"),
                pl.col("Price").std().fill_null(0.0).alias("price_std"),
                pl.col("Price").quantile(0.25).alias("price_q25"),
                pl.col("Price").quantile(0.75).alias("price_q75"),
                pl.col("InvoiceDate").min().alias("first_sale_date"),
                pl.col("InvoiceDate").max().alias("last_sale_date"),
            ]
        )
        .filter(pl.col("total_invoices") >= min_sales)
    )

    # Join resolved description
    product_metrics = product_metrics.join(description_map, on="StockCode", how="left")

    # Calculate repeat customer rate using DISTINCT purchase invoices (not line events)
    # For each product, what % of customers bought it on 2+ distinct invoices?
    customer_product_invoices = sales.group_by(["Customer ID", "StockCode"]).agg(
        pl.col("Invoice").n_unique().alias("distinct_purchase_invoices")
    )
    repeat_customers = (
        customer_product_invoices.filter(pl.col("distinct_purchase_invoices") >= 2)
        .group_by("StockCode")
        .agg(pl.col("Customer ID").n_unique().alias("repeat_customers"))
    )

    product_metrics = product_metrics.join(repeat_customers, on="StockCode", how="left")
    product_metrics = product_metrics.with_columns(
        pl.col("repeat_customers").fill_null(0).cast(pl.Int64)
    )

    # Repeat customer rate: customers with >=2 distinct purchase invoices / unique customers
    product_metrics = product_metrics.with_columns(
        (pl.col("repeat_customers") / pl.col("unique_customers")).alias("repeat_customer_rate"),
    )

    # Customer penetration (customers / total customers)
    total_customers = tx.filter(pl.col("is_clean_sale")).select(pl.col("Customer ID").n_unique()).item()
    product_metrics = product_metrics.with_columns(
        (pl.col("unique_customers") / total_customers).alias("customer_penetration"),
    )

    # Sales velocity: invoices per month of availability
    product_metrics = product_metrics.with_columns(
        (pl.col("last_sale_date") - pl.col("first_sale_date")).dt.total_days().alias("days_active"),
    )
    product_metrics = product_metrics.with_columns(
        (pl.col("total_invoices") / (pl.col("days_active") / 30.4375).clip(lower_bound=1.0)).alias("invoices_per_month"),
    )

    # Return rate for this product
    returns = tx.filter(pl.col("is_return") | pl.col("is_cancellation"))
    if returns.height > 0:
        product_returns = (
            returns.group_by("StockCode")
            .agg(
                pl.col("return_value").sum().alias("return_value"),
                pl.col("Quantity").filter(pl.col("Quantity") < 0).abs().sum().alias("return_units"),
            )
        )
        product_metrics = product_metrics.join(product_returns, on="StockCode", how="left")
        product_metrics = product_metrics.with_columns(
            [
                pl.col("return_value").fill_null(0.0),
                pl.col("return_units").fill_null(0.0),
            ]
        )
    else:
        product_metrics = product_metrics.with_columns(
            pl.lit(0.0).alias("return_value"),
            pl.lit(0.0).alias("return_units"),
        )

    # Return rate (units)
    product_metrics = product_metrics.with_columns(
        (pl.col("return_units") / (pl.col("total_units") + pl.col("return_units"))).alias("unit_return_rate"),
    )

    # Revenue return rate
    product_metrics = product_metrics.with_columns(
        (pl.col("return_value") / (pl.col("total_revenue") + pl.col("return_value"))).alias("revenue_return_rate"),
    )

    # Price variability (CV)
    product_metrics = product_metrics.with_columns(
        (pl.col("price_std") / (pl.col("avg_price") + 1e-9)).alias("price_cv"),
    )

    # Days since last sale (recency)
    max_date = tx.filter(pl.col("is_clean_sale")).select(pl.col("InvoiceDate").max()).item()
    product_metrics = product_metrics.with_columns(
        (pl.lit(max_date) - pl.col("last_sale_date")).dt.total_days().alias("days_since_last_sale"),
    )

    return product_metrics.sort("total_revenue", descending=True)


def classify_product_roles(product_metrics: pl.DataFrame) -> pl.DataFrame:
    """Classify products into evidence-based behavioral roles with explicit mutually exclusive rules.

    Precedence order (first match wins):
    1. Acquisition: High penetration (Q75), low repeat rate (Q25)
    2. Retention: High repeat rate (Q75) AND high revenue (Q75) — loyal high-value
    3. Repeat: High repeat rate (Q75) — frequent repurchase, not high revenue
    4. Basket builder: High velocity (Q75) AND moderate repeat (Q25) — co-purchase signal
    5. Niche high-value: High price (Q75), low penetration (< Q75*0.5)
    6. Volatile: Low repeat (Q25), low velocity (< Q75*0.5)
    7. Other: everything else
    """
    LOGGER.info("Classifying product roles...")

    out = product_metrics.clone()

    # Compute percentiles for thresholding
    revenue_q75 = out.select(pl.col("total_revenue").quantile(0.75)).item()
    repeat_q75 = out.select(pl.col("repeat_customer_rate").quantile(0.75)).item()
    velocity_q75 = out.select(pl.col("invoices_per_month").quantile(0.75)).item()
    price_q75 = out.select(pl.col("avg_price").quantile(0.75)).item()
    penetration_q75 = out.select(pl.col("customer_penetration").quantile(0.75)).item()
    repeat_q25 = out.select(pl.col("repeat_customer_rate").quantile(0.25)).item()

    # Single when-then-otherwise chain for mutual exclusivity
    out = out.with_columns(
        pl.when(
            # 1. Acquisition: High penetration, low repeat rate
            (pl.col("customer_penetration") >= penetration_q75) &
            (pl.col("repeat_customer_rate") < repeat_q25) &
            (pl.col("total_revenue") > 0)
        ).then(pl.lit("acquisition"))
        .when(
            # 2. Retention: High repeat + high revenue (loyal high-value)
            (pl.col("repeat_customer_rate") >= repeat_q75) &
            (pl.col("total_revenue") >= revenue_q75)
        ).then(pl.lit("retention"))
        .when(
            # 3. Repeat: High repeat rate (not already retention)
            (pl.col("repeat_customer_rate") >= repeat_q75)
        ).then(pl.lit("repeat"))
        .when(
            # 4. Basket builder: High velocity + moderate repeat
            (pl.col("invoices_per_month") >= velocity_q75) &
            (pl.col("repeat_customer_rate") >= repeat_q25)
        ).then(pl.lit("basket_builder"))
        .when(
            # 5. Niche high-value: High price, low penetration
            (pl.col("avg_price") >= price_q75) &
            (pl.col("customer_penetration") < penetration_q75 * 0.5) &
            (pl.col("total_revenue") > 0)
        ).then(pl.lit("niche_high_value"))
        .when(
            # 6. Volatile: Low repeat, low velocity
            (pl.col("repeat_customer_rate") < repeat_q25) &
            (pl.col("invoices_per_month") < velocity_q75 * 0.5)
        ).then(pl.lit("volatile"))
        .otherwise(pl.lit("other"))
        .alias("product_role")
    )

    # Log distribution
    role_counts = out.group_by("product_role").len().sort("len", descending=True)
    LOGGER.info(f"Product role distribution:\n{role_counts}")

    return out


def build_co_purchase_matrix(tx: pl.DataFrame, min_cooccurrence: int = 5) -> pl.DataFrame:
    """Build product co-purchase matrix at CUSTOMER level for consistent lift calculation.

    Co-occurrence = number of CUSTOMERS who bought both products (not invoices).
    This ensures consistent observation unit with customer-level product supports.
    """
    LOGGER.info("Building co-purchase matrix (customer-level)...")

    sales = tx.filter(pl.col("is_clean_sale"))

    customer_products = sales.select(["Customer ID", "StockCode"]).unique()

    co_purchase = customer_products.join(
        customer_products.rename({"StockCode": "product_b"}),
        on="Customer ID",
        how="inner"
    ).filter(pl.col("StockCode") < pl.col("product_b"))

    co_purchase = (
        co_purchase.group_by(["StockCode", "product_b"])
        .len()
        .rename({"len": "cooccurrence", "StockCode": "product_a"})
        .filter(pl.col("cooccurrence") >= min_cooccurrence)
        .sort("cooccurrence", descending=True)
    )

    return co_purchase


def build_product_affinity(product_metrics: pl.DataFrame, co_purchase: pl.DataFrame) -> pl.DataFrame:
    """Add product affinity metrics based on co-purchase."""
    out = product_metrics.clone()

    # For each product, find top co-purchased products
    top_affinity = (
        co_purchase.group_by("product_a")
        .agg(
            pl.col("product_b").sort_by("cooccurrence", descending=True).head(5).alias("top_5_co_products"),
            pl.col("cooccurrence").sort_by("cooccurrence", descending=True).head(5).alias("top_5_cooccurrence"),
        )
    ).rename({"product_a": "StockCode"})

    out = out.join(top_affinity, on="StockCode", how="left")

    return out


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.product_analytics_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load canonical transactions
    # -------------------------------------------------------------------------
    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"
    if canonical_path.exists():
        LOGGER.info("Loading canonical transactions...")
        tx = pl.read_parquet(canonical_path)
    else:
        LOGGER.info("Building canonical transactions...")
        raw = load_raw_transactions(input_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # -------------------------------------------------------------------------
    # Run validations
    # -------------------------------------------------------------------------
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # -------------------------------------------------------------------------
    # Build product metrics
    # -------------------------------------------------------------------------
    product_metrics = build_product_metrics(tx, args.min_sales)

    # Classify roles
    product_metrics = classify_product_roles(product_metrics)

    # Co-purchase matrix
    co_purchase = build_co_purchase_matrix(tx)

    # Product affinity
    product_metrics = build_product_affinity(product_metrics, co_purchase)

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    # For CSV, drop nested list columns
    csv_metrics = product_metrics.drop(["top_5_co_products", "top_5_cooccurrence"])
    csv_metrics.write_parquet(output_dir / "product_metrics.parquet")
    csv_metrics.write_csv(output_dir / "product_metrics.csv")

    co_purchase.write_parquet(output_dir / "co_purchase_matrix.parquet")
    co_purchase.write_csv(output_dir / "co_purchase_matrix.csv")

    # Role summary
    role_summary = (
        product_metrics.group_by("product_role")
        .agg(
            pl.len().alias("product_count"),
            pl.col("total_revenue").sum().alias("total_revenue"),
            pl.col("unique_customers").sum().alias("total_customers"),
            pl.col("repeat_customer_rate").mean().alias("avg_repeat_rate"),
        )
        .sort("product_count", descending=True)
    )
    role_summary.write_csv(output_dir / "product_role_summary.csv")

    # Top products by role
    csv_metrics = product_metrics.drop(["top_5_co_products", "top_5_cooccurrence"])
    for role in csv_metrics.select("product_role").unique().to_series().to_list():
        if role == "other":
            continue
        role_products = csv_metrics.filter(pl.col("product_role") == role).sort("total_revenue", descending=True).head(20)
        role_products.write_csv(output_dir / f"top_products_{role}.csv")

    # Model card / documentation
    with open(output_dir / "product_analytics_card.json", "w") as f:
        json.dump({
            "total_products": product_metrics.height,
            "total_revenue": float(product_metrics.select(pl.col("total_revenue").sum()).item()),
            "role_distribution": role_summary.to_dicts(),
            "methodology": "Evidence-based role classification using revenue, repeat rate, velocity, price, penetration",
            "thresholds": "Quartile-based (Q75/Q25) for role assignment",
        }, f, indent=2, default=str)

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "product_metrics.parquet",
            "product_metrics.csv",
            "co_purchase_matrix.parquet",
            "co_purchase_matrix.csv",
            "product_analytics_card.json",
            "product_role_summary.csv",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        plots = output_dir / "plots"
        plots.mkdir(exist_ok=True)

        # Revenue by role
        fig, ax = plt.subplots(figsize=(10, 6))
        role_rev = product_metrics.group_by("product_role").agg(pl.col("total_revenue").sum()).sort("total_revenue", descending=True)
        ax.barh(role_rev["product_role"], role_rev["total_revenue"])
        ax.set_xlabel("Total Revenue")
        ax.set_title("Revenue by Product Role")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "01_revenue_by_role.png", dpi=180)
        plt.close(fig)

        # Product count by role
        fig, ax = plt.subplots(figsize=(10, 6))
        role_count = product_metrics.group_by("product_role").len().sort("len", descending=True)
        ax.barh(role_count["product_role"], role_count["len"])
        ax.set_xlabel("Product Count")
        ax.set_title("Product Count by Role")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "02_count_by_role.png", dpi=180)
        plt.close(fig)

        # Revenue vs Repeat Rate
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter_data = product_metrics.to_pandas()
        for role in scatter_data["product_role"].unique():
            subset = scatter_data[scatter_data["product_role"] == role]
            ax.scatter(subset["repeat_customer_rate"], subset["total_revenue"], label=role, alpha=0.6, s=30)
        ax.set_xlabel("Repeat Customer Rate")
        ax.set_ylabel("Total Revenue (log)")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Product Revenue vs Repeat Rate by Role")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "03_revenue_vs_repeat.png", dpi=180)
        plt.close(fig)

    LOGGER.info("Product analytics complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    main()