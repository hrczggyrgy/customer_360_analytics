#!/usr/bin/env python3
"""
Customer 360 — Descriptive current-state customer feature mart + Point-in-time feature engine.

This script produces TWO outputs:
1. customer_360_current.parquet — Descriptive current state (as of max observation date)
   Valid for: dashboard, segmentation, cohort analysis, current customer profiling
   NOT valid for: historical ML prediction (contains future information)

2. Point-in-time feature function: features_at_date(prediction_date)
   Valid for: CLV training, churn training, next-purchase training, reactivation training
   Computes features using ONLY data available up to prediction_date

Architecture:
- Uses retail_ds shared package for canonical ingestion, cleaning, classification
- Builds canonical customer-month panel once
- Derives current-state features from full history
- Provides point-in-time feature computation for ML pipelines
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel, add_rolling_features
from retail_ds.features import build_point_in_time_features, FEATURE_REGISTRY
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.backtesting import rolling_origin_split


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("customer_360")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Build Customer 360 feature mart with point-in-time feature engine."
    )

    parser.add_argument(
        "--input",
        default=str(project_dir / "data_xslx" / "online_retail_II.xlsx"),
        help="Input .xlsx/.xls/.csv/.parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "customer_360_output"),
        help="Output directory.",
    )

    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name.",
    )

    parser.add_argument(
        "--prediction-dates",
        default=None,
        help="Comma-separated prediction dates for point-in-time features (ISO format).",
    )

    parser.add_argument(
        "--windows",
        default="1,3,6,12",
        help="Comma-separated trailing windows in months.",
    )

    parser.add_argument(
        "--stale-after-months",
        type=int,
        default=6,
        help="Inactive months after which active customer is labeled stale.",
    )

    parser.add_argument(
        "--dormant-after-months",
        type=int,
        default=12,
        help="Inactive months after which customer is labeled dormant.",
    )

    parser.add_argument(
        "--reactivation-gap-months",
        type=int,
        default=1,
        help="Inactive months needed before activity qualifies as reactivation.",
    )

    parser.add_argument(
        "--top-products",
        type=int,
        default=10,
        help="Number of top product affinity features.",
    )

    parser.add_argument(
        "--plot-customer-cap",
        type=int,
        default=10000,
        help="Max customers for scatter plots.",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation for faster runs.",
    )

    return parser.parse_args()


def build_current_state_features(
    tx: pl.DataFrame,
    customer_month_dense: pl.DataFrame,
    windows: Sequence[int],
    stale_after_months: int,
    dormant_after_months: int,
    top_products: int,
) -> tuple[pl.DataFrame, Dict[str, List[str]]]:
    """
    Build descriptive CURRENT-STATE Customer 360 features.
    
    Uses FULL history up to max observation date.
    NOT safe for historical prediction - contains future information relative to any earlier date.
    """
    sales = tx.filter(pl.col("is_clean_sale"))
    observation_end = customer_month_dense.select(pl.col("calendar_month").max()).item()

    # Invoice-level for overall behavior
    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
                pl.col("Quantity").sum().alias("invoice_units"),
                pl.col("StockCode").n_unique().alias("invoice_unique_products"),
            ]
        )
        .sort(["Customer ID", "invoice_date"])
        .with_columns(
            [
                pl.col("invoice_date").diff().over("Customer ID").dt.total_seconds().alias("interpurchase_seconds"),
                pl.col("invoice_date").dt.weekday().alias("invoice_weekday"),
                pl.col("invoice_date").dt.hour().alias("invoice_hour"),
            ]
        )
    )

    customer_base = (
        invoice.group_by("Customer ID")
        .agg(
            [
                pl.len().alias("lifetime_orders"),
                pl.col("invoice_date").min().alias("first_purchase_date"),
                pl.col("invoice_date").max().alias("last_purchase_date"),
                pl.col("invoice_revenue").sum().alias("lifetime_gross_revenue"),
                pl.col("invoice_revenue").mean().alias("historical_avg_order_value"),
                pl.col("invoice_revenue").median().alias("historical_median_order_value"),
                pl.col("invoice_revenue").std().fill_null(0.0).alias("historical_order_value_std"),
                pl.col("invoice_units").sum().alias("lifetime_units"),
                pl.col("invoice_unique_products").sum().alias("lifetime_product_line_events"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).median().alias("median_interpurchase_seconds"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).mean().alias("mean_interpurchase_seconds"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).std().fill_null(0.0).alias("std_interpurchase_seconds"),
                pl.col("invoice_unique_products").mean().alias("avg_products_per_order"),
                (pl.col("invoice_weekday") >= 6).mean().alias("weekend_order_share"),
                ((pl.col("invoice_hour") >= 9) & (pl.col("invoice_hour") < 18)).mean().alias("business_hour_order_share"),
            ]
        )
        .with_columns(
            [
                (pl.col("historical_order_value_std") / (pl.col("historical_avg_order_value").abs() + 1e-9)).alias("order_value_cv"),
                (pl.col("std_interpurchase_seconds") / (pl.col("mean_interpurchase_seconds").abs() + 1e-9)).alias("interpurchase_cv"),
                (pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_days().fill_null(0.0).alias("tenure_days"),
                (pl.lit(observation_end) - pl.col("last_purchase_date")).dt.total_days().fill_null(0.0).alias("recency_days"),
                pl.col("first_purchase_date").dt.truncate("1mo").alias("cohort_month"),
            ]
        )
        .with_columns(
            [
                (pl.col("recency_days") / 30.4375).alias("recency_months"),
                (pl.col("lifetime_orders") / (pl.col("tenure_days") / 30.4375).clip(lower_bound=1.0)).alias("orders_per_active_tenure_month"),
                (pl.col("lifetime_gross_revenue") / pl.col("lifetime_orders")).alias("aov_check"),
            ]
        )
    )

    # Returns
    returns = tx.filter(pl.col("is_return") | pl.col("is_cancellation"))
    if returns.height:
        return_features = (
            returns.group_by("Customer ID")
            .agg(
                [
                    pl.col("return_value").sum().alias("lifetime_return_value"),
                    pl.col("Quantity").filter(pl.col("Quantity") < 0).abs().sum().fill_null(0.0).alias("lifetime_return_units"),
                    pl.col("Invoice").n_unique().alias("return_invoice_count"),
                    pl.len().alias("return_line_count"),
                ]
            )
            .with_columns(pl.col("Customer ID").cast(pl.Int64))
        )
    else:
        return_features = pl.DataFrame(schema={
            "Customer ID": pl.Int64, "lifetime_return_value": pl.Float64,
            "lifetime_return_units": pl.Float64, "return_invoice_count": pl.Float64,
            "return_line_count": pl.Float64,
        })

    # Product features
    customer_product = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg(
            [
                pl.col("gross_merchandise_revenue").sum().alias("product_revenue"),
                pl.col("Quantity").sum().alias("product_units"),
                pl.col("Invoice").n_unique().alias("product_order_count"),
            ]
        )
        .with_columns(
            (pl.col("product_revenue") / pl.col("product_revenue").sum().over("Customer ID")).alias("product_revenue_share")
        )
    )

    product_features = (
        customer_product.group_by("Customer ID")
        .agg(
            [
                pl.col("StockCode").n_unique().alias("unique_products"),
                (pl.col("product_revenue_share") ** 2).sum().alias("product_revenue_hhi"),
                (pl.col("product_order_count") > 1).cast(pl.Float64).mean().alias("repeat_product_ratio"),
                pl.col("product_revenue").max().alias("top_product_revenue"),
            ]
        )
        .with_columns(
            (pl.col("repeat_product_ratio") / pl.col("unique_products")).alias("repeat_product_ratio"),
            pl.col("Customer ID").cast(pl.Int64)
        )
    )

    # Price features
    price_features = (
        sales.group_by("Customer ID")
        .agg(
            [
                pl.col("Price").mean().alias("mean_unit_price"),
                pl.col("Price").median().alias("median_unit_price"),
                (pl.col("Price").quantile(0.75) - pl.col("Price").quantile(0.25)).alias("unit_price_iqr"),
                pl.col("Price").std().fill_null(0.0).alias("unit_price_std"),
                pl.col("Price").quantile(0.90).alias("unit_price_p90"),
                (pl.col("Price") <= 0).mean().alias("nonpositive_price_share"),
            ]
        )
        .with_columns(
            (pl.col("unit_price_std") / (pl.col("mean_unit_price").abs() + 1e-9)).alias("unit_price_cv"),
            pl.col("Customer ID").cast(pl.Int64)
        )
    )

    global_price_p75 = sales.select(pl.col("Price").quantile(0.75)).item()
    premium_features = (
        sales.group_by("Customer ID")
        .agg((pl.col("Price") > global_price_p75).mean().alias("premium_price_line_share"))
        .with_columns(pl.col("Customer ID").cast(pl.Int64))
    )

    # Country features
    country_features = (
        sales.group_by("Customer ID")
        .agg(
            [
                pl.col("Country").n_unique().alias("unique_countries"),
                pl.col("Country").mode().first().alias("primary_country"),
                pl.col("Country").first().alias("first_country"),
            ]
        )
        .with_columns(pl.col("Customer ID").cast(pl.Int64))
    )

    # Temporal entropy
    invoice_temporal = invoice.select(["Customer ID", "invoice_date", "invoice_weekday", "invoice_hour"])

    def entropy_from_counts(df, key, bucket, out_name, norm):
        counts = df.group_by([key, bucket]).len("n")
        counts = counts.with_columns((pl.col("n") / pl.col("n").sum().over(key)).alias("p"))
        return counts.group_by(key).agg((-(pl.col("p") * pl.col("p").log()).sum() / np.log(norm)).alias(out_name))

    temporal = invoice_temporal.group_by("Customer ID").agg(
        pl.col("invoice_weekday").mean().alias("avg_order_weekday"),
        pl.col("invoice_hour").mean().alias("avg_order_hour"),
        pl.col("invoice_date").dt.month().n_unique().alias("purchase_month_count"),
    )
    temporal = temporal.join(
        entropy_from_counts(invoice_temporal.with_columns(pl.col("invoice_hour").cast(pl.Int64).alias("hour_bucket")), "Customer ID", "hour_bucket", "hour_entropy", 24.0),
        on="Customer ID", how="left"
    ).join(
        entropy_from_counts(invoice_temporal.with_columns(pl.col("invoice_weekday").cast(pl.Int64).alias("weekday_bucket")), "Customer ID", "weekday_bucket", "weekday_entropy", 7.0),
        on="Customer ID", how="left"
    ).join(
        entropy_from_counts(invoice_temporal.with_columns(pl.col("invoice_date").dt.month().cast(pl.Int64).alias("month_bucket")), "Customer ID", "month_bucket", "month_entropy", 12.0),
        on="Customer ID", how="left"
    ).with_columns(pl.col("Customer ID").cast(pl.Int64))

    # Rolling windows from customer-month
    dynamic = customer_month_dense.filter(pl.col("calendar_month") == observation_end).with_columns(pl.col("Customer ID").cast(pl.Int64))
    for w in windows:
        dynamic = dynamic.with_columns(
            [
                pl.col(f"orders_last_{w}m"),
                pl.col(f"revenue_last_{w}m"),
                pl.col(f"net_revenue_last_{w}m"),
                pl.col(f"active_months_last_{w}m"),
                pl.col(f"return_value_last_{w}m"),
                pl.col(f"aov_last_{w}m"),
                pl.col(f"return_rate_last_{w}m"),
            ]
        )

    # Transitions
    transitions = (
        customer_month_dense.group_by("Customer ID")
        .agg(
            [
                pl.col("reactivation_event").sum().alias("reactivation_count"),
                pl.col("churn_transition").sum().alias("churn_transition_count"),
                pl.col("active").sum().alias("active_month_count"),
            ]
        )
        .with_columns(pl.col("Customer ID").cast(pl.Int64))
    )

    # Top product affinity
    if top_products > 0:
        top_codes = (
            sales.group_by("StockCode")
            .agg(pl.col("gross_merchandise_revenue").sum().alias("product_revenue"))
            .sort("product_revenue", descending=True)
            .head(top_products)
            .get_column("StockCode").to_list()
        )
        top_affinity = (
            sales.filter(pl.col("StockCode").is_in(top_codes))
            .group_by(["Customer ID", "StockCode"])
            .agg(pl.col("gross_merchandise_revenue").sum().alias("product_revenue"))
            .with_columns(
                (pl.col("product_revenue") / pl.col("product_revenue").sum().over("Customer ID")).alias("share")
            )
            .pivot(values="share", index="Customer ID", on="StockCode", aggregate_function="first")
            .fill_null(0.0)
            .with_columns(pl.col("Customer ID").cast(pl.Int64))
        )
        top_affinity = top_affinity.rename({c: f"share_top_product_pool_{c}" for c in top_affinity.columns if c != "Customer ID"})
    else:
        top_affinity = pl.DataFrame(schema={"Customer ID": pl.Int64})

    # Combine all
    customer = customer_base
    for right in [return_features, product_features, price_features, premium_features, country_features, temporal, dynamic, transitions, top_affinity]:
        customer = customer.join(right, on="Customer ID", how="left")

    # Derived
    customer = customer.with_columns(
        [
            (pl.col("lifetime_gross_revenue") - pl.col("lifetime_return_value")).alias("lifetime_net_revenue"),
            (pl.col("lifetime_return_value") / (pl.col("lifetime_gross_revenue") + 1e-9)).alias("lifetime_return_value_rate"),
            (pl.col("lifetime_return_units") / (pl.col("lifetime_units") + pl.col("lifetime_return_units") + 1e-9)).alias("lifetime_return_unit_rate"),
            (pl.col("active_month_count") / (pl.col("tenure_days") / 30.4375).clip(lower_bound=1.0)).alias("active_month_density"),
            (pl.col("lifetime_gross_revenue") / pl.col("unique_products")).alias("revenue_per_unique_product"),
            (pl.col("lifetime_gross_revenue") / pl.col("lifetime_units")).alias("revenue_per_unit"),
            (pl.col("recency_months") <= stale_after_months).alias("within_stale_threshold"),
        ]
    )

    # Lifecycle state (current state)
    customer = customer.with_columns(
        pl.when(pl.col("lifetime_orders") == 1).then(pl.lit("new_single_order"))
        .when((pl.col("recency_months") <= 1) & (pl.col("lifetime_orders") >= 2)).then(pl.lit("active_repeat"))
        .when((pl.col("recency_months") > 1) & (pl.col("recency_months") <= stale_after_months)).then(pl.lit("at_risk"))
        .when((pl.col("recency_months") > stale_after_months) & (pl.col("recency_months") <= dormant_after_months)).then(pl.lit("stale"))
        .otherwise(pl.lit("dormant"))
        .alias("lifecycle_state")
    )

    customer = customer.with_columns(
        pl.when(pl.col("reactivation_count") > 0).then(pl.lit(True)).otherwise(pl.lit(False)).alias("has_reactivated"),
        pl.when(pl.col("lifetime_orders") >= 2 & (pl.col("interpurchase_cv") > 1.0)).then(pl.lit("volatile_cadence")).otherwise(pl.lit("regular_cadence")).alias("cadence_regime"),
    )

    customer = customer.with_columns(
        pl.when(pl.col("has_reactivated") & (pl.col("recency_months") <= 3)).then(pl.lit("reactivated_recently")).otherwise(pl.col("lifecycle_state")).alias("customer_state")
    )

    # Feature groups for documentation
    feature_groups = {
        "lifecycle": ["first_purchase_date", "last_purchase_date", "tenure_days", "recency_days", "recency_months", "active_month_count", "active_month_density", "purchase_month_count", "lifecycle_state", "customer_state"],
        "economic_value": ["lifetime_gross_revenue", "lifetime_net_revenue", "historical_avg_order_value", "historical_median_order_value", "historical_order_value_std", "order_value_cv", "revenue_per_unique_product", "revenue_per_unit"],
        "purchase_cadence": ["lifetime_orders", "median_interpurchase_seconds", "mean_interpurchase_seconds", "std_interpurchase_seconds", "interpurchase_cv", "orders_per_active_tenure_month", "cadence_regime"],
        "assortment": ["unique_products", "product_revenue_hhi", "repeat_product_ratio", "avg_products_per_order", "lifetime_product_line_events"],
        "pricing": ["mean_unit_price", "median_unit_price", "unit_price_iqr", "unit_price_std", "unit_price_p90", "unit_price_cv", "premium_price_line_share"],
        "returns": ["lifetime_return_value", "lifetime_return_units", "return_invoice_count", "return_line_count", "lifetime_return_value_rate", "lifetime_return_unit_rate"],
        "temporal_behavior": ["weekend_order_share", "business_hour_order_share", "avg_order_weekday", "avg_order_hour", "hour_entropy", "weekday_entropy", "month_entropy"],
        "dynamics": [c for c in customer.columns if c.startswith(("orders_last_", "revenue_last_", "net_revenue_last_", "active_months_last_", "return_value_last_", "return_rate_last_", "aov_last_"))],
        "reactivation": ["reactivation_count", "churn_transition_count", "has_reactivated"],
        "geography": ["primary_country", "first_country", "unique_countries"],
        "product_affinity": [c for c in customer.columns if c.startswith("share_top_product_pool_")],
    }

    customer = customer.with_columns(pl.lit(observation_end).alias("snapshot_date"))
    customer = customer.with_columns(pl.col("Customer ID").cast(pl.Int64)).sort("Customer ID")

    return customer, feature_groups


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    windows = [int(w) for w in args.windows.split(",")]

    # -------------------------------------------------------------------------
    # Load canonical transactions (reuse from data_quality if available)
    # -------------------------------------------------------------------------
    canonical_path = Path("data_quality_output/canonical_transactions.parquet")
    if canonical_path.exists():
        LOGGER.info("Loading canonical transactions from data_quality_output...")
        import polars as pl
        tx = pl.read_parquet(canonical_path)
    else:
        LOGGER.info("Building canonical transactions from raw data...")
        raw = load_raw_transactions(input_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # -------------------------------------------------------------------------
    # Build customer-month panel
    # -------------------------------------------------------------------------
    LOGGER.info("Building customer-month panel...")
    _, customer_month_dense, metadata = build_customer_month_panel(tx)
    customer_month_dense = add_rolling_features(customer_month_dense, windows)

    # -------------------------------------------------------------------------
    # Run validations
    # -------------------------------------------------------------------------
    LOGGER.info("Running validations...")
    validation_results = run_all_validations(tx=tx, customer_month_dense=customer_month_dense)
    assert_validations_pass(validation_results)

    # -------------------------------------------------------------------------
    # 1. Descriptive Current-State Customer 360
    # -------------------------------------------------------------------------
    LOGGER.info("Building descriptive current-state Customer 360...")
    customer_current, feature_groups = build_current_state_features(
        tx=tx,
        customer_month_dense=customer_month_dense,
        windows=windows,
        stale_after_months=args.stale_after_months,
        dormant_after_months=args.dormant_after_months,
        top_products=args.top_products,
    )

    # -------------------------------------------------------------------------
    # 2. Point-in-time features for specified prediction dates
    # -------------------------------------------------------------------------
    prediction_dates = []
    if args.prediction_dates:
        prediction_dates = [d.strip() for d in args.prediction_dates.split(",")]
    else:
        # Default: use temporal split origins for CLV/churn training
        prediction_dates = ["2010-09-30", "2010-12-31", "2011-03-31", "2011-06-30", "2011-09-30"]

    point_in_time_features = {}
    for pred_date in prediction_dates:
        LOGGER.info(f"Computing point-in-time features for {pred_date}...")
        features = build_point_in_time_features(tx, pred_date, customer_month_dense, windows)
        point_in_time_features[pred_date] = features
        features.write_parquet(output_dir / f"features_at_{pred_date.replace('-', '')}.parquet")

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    # Current-state Customer 360
    customer_current.write_parquet(output_dir / "customer_360_current.parquet")
    customer_current.write_csv(output_dir / "customer_360_current.csv")

    # Feature metadata
    feature_meta = {name: {"description": meta.description, "dtype": meta.dtype, "source": meta.source,
                          "point_in_time_safe": meta.point_in_time_safe, "requires_as_of_date": meta.requires_as_of_date,
                          "feature_group": meta.feature_group} for name, meta in FEATURE_REGISTRY.items()}
    with open(output_dir / "feature_dictionary.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # Feature groups
    with open(output_dir / "feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    # Customer-month panel
    customer_month_dense.write_parquet(output_dir / "customer_month.parquet")

    # Metadata
    metadata = {
        "observation_start": metadata["observation_start"],
        "observation_end": metadata["observation_end"],
        "unique_customers": customer_current.height,
        "feature_count": len([c for c in customer_current.columns if c != "Customer ID"]),
        "prediction_dates": prediction_dates,
        "windows": windows,
    }
    with open(output_dir / "customer_360_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Validation results
    val_output = [{"check": r.check_name, "passed": r.passed, "message": r.message, "severity": r.severity, "details": r.details} for r in validation_results]
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(val_output, f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        # Recency distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        recency = customer_current.select(pl.col("recency_months")).to_numpy().flatten()
        ax.hist(recency[np.isfinite(recency)], bins=50, edgecolor='white')
        ax.set_xlabel("Recency (months)")
        ax.set_ylabel("Customers")
        ax.set_title("Customer Recency Distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "01_recency_distribution.png", dpi=150)
        plt.close(fig)

        # Value distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        revenue = customer_current.select(pl.col("lifetime_gross_revenue")).to_numpy().flatten()
        ax.hist(np.log1p(revenue[revenue > 0]), bins=50, edgecolor='white')
        ax.set_xlabel("log(1 + Lifetime Gross Revenue)")
        ax.set_ylabel("Customers")
        ax.set_title("Customer Value Distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "02_value_distribution.png", dpi=150)
        plt.close(fig)

        # Lifecycle state
        fig, ax = plt.subplots(figsize=(10, 6))
        state_counts = customer_current.group_by("lifecycle_state").len().sort("len", descending=True)
        ax.barh(state_counts["lifecycle_state"], state_counts["len"])
        ax.set_xlabel("Customers")
        ax.set_title("Customer Lifecycle State")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "03_lifecycle_state.png", dpi=150)
        plt.close(fig)

    LOGGER.info("Customer 360 complete. Outputs in %s", output_dir)
    LOGGER.info("Current-state customers: %s", f"{customer_current.height:,}")
    LOGGER.info("Point-in-time feature sets: %s", len(point_in_time_features))


if __name__ == "__main__":
    main()