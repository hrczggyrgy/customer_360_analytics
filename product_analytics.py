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
    parser.add_argument("--input", default="./data_xslx/online_retail_II.xlsx", help="Input file.")
    parser.add_argument("--output-dir", default="./product_analytics_output", help="Output directory.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--min-sales", type=int, default=10, help="Minimum sales for product inclusion.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_product_metrics(
    tx: pl.DataFrame,
    min_sales: int = 10,
) -> pl.DataFrame:
    """Build comprehensive product-level metrics from canonical transactions."""
    LOGGER.info("Building product-level metrics...")

    # Filter to clean sales
    sales = tx.filter(pl.col("is_clean_sale"))

    # Product-level aggregations
    product_metrics = (
        sales.group_by(["StockCode", "Description"])
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
                pl.col("Customer ID").n_unique().alias("customer_penetration_raw"),
            ]
        )
        .filter(pl.col("total_invoices") >= min_sales)
    )

    # Calculate repeat customer rate
    # For each product, what % of customers bought it more than once?
    customer_product = sales.group_by(["Customer ID", "StockCode"]).agg(
        pl.len().alias("purchase_count")
    )
    repeat_customers = (
        customer_product.filter(pl.col("purchase_count") > 1)
        .group_by("StockCode")
        .agg(pl.col("Customer ID").n_unique().alias("repeat_customers"))
    )

    product_metrics = product_metrics.join(repeat_customers, on="StockCode", how="left")
    product_metrics = product_metrics.with_columns(
        pl.col("repeat_customers").fill_null(0).cast(pl.Int64)
    )

    # Repeat customer rate
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
    """Classify products into evidence-based behavioral roles."""
    LOGGER.info("Classifying product roles...")

    out = product_metrics.clone()

    # Compute percentiles for thresholding
    revenue_q75 = out.select(pl.col("total_revenue").quantile(0.75)).item()
    repeat_q75 = out.select(pl.col("repeat_customer_rate").quantile(0.75)).item()
    velocity_q75 = out.select(pl.col("invoices_per_month").quantile(0.75)).item()
    price_q75 = out.select(pl.col("avg_price").quantile(0.75)).item()
    penetration_q75 = out.select(pl.col("customer_penetration").quantile(0.75)).item()
    repeat_q25 = out.select(pl.col("repeat_customer_rate").quantile(0.25)).item()

    # Role classification logic (prioritized)
    out = out.with_columns(
        pl.lit("other").alias("product_role"),
    )

    # 1. Acquisition: High penetration, low repeat rate (first purchase for many)
    out = out.with_columns(
        pl.when(
            (pl.col("customer_penetration") >= penetration_q75) &
            (pl.col("repeat_customer_rate") < repeat_q25) &
            (pl.col("total_revenue") > 0)
        ).then(pl.lit("acquisition")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # 2. Repeat: High repeat customer rate
    out = out.with_columns(
        pl.when(
            (pl.col("repeat_customer_rate") >= repeat_q75) &
            (pl.col("product_role") == "other")
        ).then(pl.lit("repeat")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # 3. Retention: High repeat rate + high revenue + purchased by loyal customers
    # (simplified: high repeat rate + high revenue)
    out = out.with_columns(
        pl.when(
            (pl.col("repeat_customer_rate") >= repeat_q75) &
            (pl.col("total_revenue") >= revenue_q75) &
            (pl.col("product_role") == "other")
        ).then(pl.lit("retention")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # 4. Basket builder: High velocity + moderate repeat (co-purchase signal)
    out = out.with_columns(
        pl.when(
            (pl.col("invoices_per_month") >= velocity_q75) &
            (pl.col("repeat_customer_rate") >= repeat_q25) &
            (pl.col("product_role") == "other")
        ).then(pl.lit("basket_builder")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # 5. Niche high-value: Low volume, high price, low penetration
    out = out.with_columns(
        pl.when(
            (pl.col("avg_price") >= price_q75) &
            (pl.col("customer_penetration") < penetration_q75 * 0.5) &
            (pl.col("total_revenue") > 0) &
            (pl.col("product_role") == "other")
        ).then(pl.lit("niche_high_value")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # 6. Volatile: Low repeat, low velocity, irregular
    out = out.with_columns(
        pl.when(
            (pl.col("repeat_customer_rate") < repeat_q25) &
            (pl.col("invoices_per_month") < velocity_q75 * 0.5) &
            (pl.col("product_role") == "other")
        ).then(pl.lit("volatile")).otherwise(pl.col("product_role")).alias("product_role"),
    )

    # Log distribution
    role_counts = out.group_by("product_role").len().sort("len", descending=True)
    LOGGER.info(f"Product role distribution:\n{role_counts}")

    return out


def build_co_purchase_matrix(tx: pl.DataFrame, min_cooccurrence: int = 5, max_pairs_per_invoice: int = 50) -> pl.DataFrame:
    """Build product co-purchase matrix for basket analysis (optimized)."""
    LOGGER.info("Building co-purchase matrix...")

    sales = tx.filter(pl.col("is_clean_sale"))

    # Get invoices with multiple products
    invoice_products = (
        sales.group_by("Invoice")
        .agg(pl.col("StockCode").unique().alias("products"))
        .filter(pl.col("products").list.len() > 1)
        .filter(pl.col("products").list.len() <= max_pairs_per_invoice)  # Limit to avoid explosion
    )

    if invoice_products.height == 0:
        return pl.DataFrame(schema={"product_a": pl.Utf8, "product_b": pl.Utf8, "cooccurrence": pl.Int64})

    # Use Polars explode and self-join for co-occurrence
    exploded = invoice_products.explode("products").rename({"products": "product_a"})
    
    # Self-join on Invoice to get all pairs
    co_purchase = exploded.join(
        exploded.select(["Invoice", "product_a"]).rename({"product_a": "product_b"}),
        on="Invoice",
        how="inner"
    ).filter(pl.col("product_a") < pl.col("product_b"))  # Avoid duplicates and self-pairs
    
    co_purchase = (
        co_purchase.group_by(["product_a", "product_b"])
        .len()
        .rename({"len": "cooccurrence"})
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
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load canonical transactions
    # -------------------------------------------------------------------------
    canonical_path = Path("data_quality_output/canonical_transactions.parquet")
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