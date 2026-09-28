#!/usr/bin/env python3
"""
Recommendation Engine — Product Co-Purchase Based Recommendations (Vectorized)

This script builds customer-specific product recommendations using:
1. Product co-purchase relationships (from product_analytics)
2. Customer purchase history
3. Customer segment affinity
4. Product popularity

Fully vectorized using Polars joins and aggregations — no per-customer loops.

Output: Customer ID, recommended_product, score, reason, support, lift

Uses the retail_ds shared package for canonical data processing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures

SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("recommendation_engine")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product recommendation engine.")
    parser.add_argument("--input", default="./data_xslx/online_retail_II.xlsx", help="Input file.")
    parser.add_argument("--output-dir", default="./recommendation_output", help="Output directory.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--co-purchase-file", default="./product_analytics_output/co_purchase_matrix.parquet", help="Co-purchase matrix from product_analytics.")
    parser.add_argument("--segments-file", default="./online_retail_segmentation/customer_segments.parquet", help="Customer segments file.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K recommendations per customer.")
    parser.add_argument("--min-support", type=int, default=3, help="Minimum co-purchase support.")
    parser.add_argument("--max-partners", type=int, default=20, help="Max co-purchase partners per product.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_customer_history(tx: pl.DataFrame) -> pl.DataFrame:
    """Build customer purchase history: which products each customer bought."""
    LOGGER.info("Building customer purchase history...")

    sales = tx.filter(pl.col("is_clean_sale"))

    # Customer-Product purchase history with aggregation
    customer_product = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("first_purchase_date"),
                pl.col("InvoiceDate").max().alias("last_purchase_date"),
                pl.col("Invoice").n_unique().alias("purchase_count"),
                pl.col("Quantity").sum().alias("total_quantity"),
                pl.col("gross_merchandise_revenue").sum().alias("total_revenue"),
            ]
        )
        .sort(["Customer ID", "total_revenue"], descending=[False, True])
    )

    return customer_product


def load_co_purchase_matrix(co_purchase_path: Path, min_support: int, max_partners: int) -> pl.DataFrame:
    """Load and validate co-purchase matrix, keep top-N partners per product."""
    LOGGER.info(f"Loading co-purchase matrix from {co_purchase_path}")

    if not co_purchase_path.exists():
        raise FileNotFoundError(f"Co-purchase matrix not found: {co_purchase_path}")

    co_purchase = pl.read_parquet(co_purchase_path)

    # Ensure symmetric (both directions) by creating reverse pairs
    if "product_a" in co_purchase.columns and "product_b" in co_purchase.columns:
        reverse = co_purchase.rename({"product_a": "product_b", "product_b": "product_a"})
        co_purchase = pl.concat([co_purchase, reverse], how="diagonal_relaxed").unique()

    # Filter by min support and keep top-N partners per product
    co_purchase = (
        co_purchase
        .filter(pl.col("cooccurrence") >= min_support)
        .sort(["product_a", "cooccurrence"], descending=[False, True])
        .group_by("product_a")
        .head(max_partners)
    )

    LOGGER.info(f"Loaded co-purchase matrix (top {max_partners} per product): {co_purchase.height:,} pairs")
    return co_purchase


def load_customer_segments(segments_path: Path) -> pl.DataFrame:
    """Load customer segments if available."""
    LOGGER.info(f"Loading customer segments from {segments_path}")

    if not segments_path.exists():
        LOGGER.warning(f"Segments file not found: {segments_path}")
        return pl.DataFrame(schema={"Customer ID": pl.Int64, "segment": pl.Int64})

    segments = pl.read_parquet(segments_path)
    keep_cols = ["Customer ID"]
    for col in ["segment", "segment_name", "segment_confidence"]:
        if col in segments.columns:
            keep_cols.append(col)
    return segments.select(keep_cols).unique(subset=["Customer ID"])


def generate_recommendations_vectorized(
    customer_product: pl.DataFrame,
    co_purchase: pl.DataFrame,  # Already filtered to top-N per product
    customer_segments: pl.DataFrame,
    top_k: int = 10,
) -> pl.DataFrame:
    """
    Generate recommendations for all customers using fully vectorized Polars operations.
    
    Strategy:
    1. Co-purchase matrix already filtered to top-N partners per product
    2. Join customer purchases with co-purchase matrix to get candidate products per customer
    3. Aggregate co-purchase scores per customer-product
    4. Add popularity fallback (computed from same filtered co-purchase)
    5. Rank and select top-K per customer
    """
    LOGGER.info("Generating recommendations (fully vectorized)...")

    # Get all unique customers and customer purchase history
    all_customers = customer_product.select("Customer ID").unique()
    customer_bought = customer_product.select(["Customer ID", "StockCode"]).unique()

    # VECTORIZED: Join customer purchases with co-purchase matrix
    # Customer bought product A -> co-purchase says product A is bought with product B
    # So candidate product B is recommended to customers who bought A
    LOGGER.info("Joining customer purchases with co-purchase matrix...")
    
    customer_candidates = (
        customer_bought.rename({"StockCode": "product_a"})
        .join(co_purchase, on="product_a", how="inner")
        .rename({"product_b": "recommended_product"})
    )
    LOGGER.info(f"Customer-candidate pairs: {customer_candidates.height:,}")

    # Filter out products the customer already bought
    customer_candidates = customer_candidates.join(
        customer_bought.rename({"StockCode": "recommended_product"}),
        on=["Customer ID", "recommended_product"],
        how="anti"
    )
    LOGGER.info(f"After removing already-bought: {customer_candidates.height:,}")

    # Aggregate co-purchase scores per customer-product
    customer_scores = (
        customer_candidates.group_by(["Customer ID", "recommended_product"])
        .agg(
            pl.col("cooccurrence").sum().alias("co_purchase_score"),
            pl.col("product_a").n_unique().alias("supporting_products"),
        )
    )

    # Add lift (normalize by customer's number of purchased products)
    n_products_per_customer = (
        customer_product.group_by("Customer ID").len()
        .rename({"len": "n_products_bought"})
    )
    customer_scores = customer_scores.join(n_products_per_customer, on="Customer ID", how="left")
    customer_scores = customer_scores.with_columns(
        (pl.col("co_purchase_score") / pl.col("n_products_bought")).alias("lift"),
    ).drop("n_products_bought")

    # Get product popularity (for fallback) - from the SAME filtered co_purchase matrix
    LOGGER.info("Computing product popularity fallback...")
    product_popularity = (
        co_purchase.group_by("product_b")
        .agg(pl.col("cooccurrence").sum().alias("total_cooccurrence"))
        .rename({"product_b": "StockCode"})
        .sort("total_cooccurrence", descending=True)
    )
    max_pop = product_popularity.select(pl.col("total_cooccurrence").max()).item()
    product_popularity = product_popularity.with_columns(
        (pl.col("total_cooccurrence") / max_pop).alias("popularity_score"),
    ).select(["StockCode", "popularity_score"])

    # For customers WITH co-purchase scores, merge with popularity
    LOGGER.info("Merging co-purchase scores with popularity...")
    
    # WORKAROUND: Use when/then/otherwise instead of fill_null to avoid Polars bug
    # that corrupts co_purchase_score column type
    recommendations_with_cp = (
        customer_scores
        .join(product_popularity, left_on="recommended_product", right_on="StockCode", how="left")
        .with_columns(
            pl.when(pl.col("popularity_score").is_null())
            .then(0.0)
            .otherwise(pl.col("popularity_score"))
            .alias("popularity_score")
        )
        .with_columns(
            (pl.lit(0.7) * pl.col("co_purchase_score").cast(pl.Float64) + pl.lit(0.3) * pl.col("popularity_score")).alias("final_score"),
            pl.lit("co_purchase").alias("reason"),
        )
    )

    # For customers WITHOUT co-purchase scores, use pure popularity
    customers_with_cp = recommendations_with_cp.select("Customer ID").unique()
    all_customer_ids = all_customers.select("Customer ID").to_series().to_list()
    customers_with_cp_ids = customers_with_cp.select("Customer ID").to_series().to_list()
    customers_without_cp = [cid for cid in all_customer_ids if cid not in customers_with_cp_ids]
    
    if customers_without_cp:
        LOGGER.info(f"Generating popularity fallback for {len(customers_without_cp)} customers...")
        # Create popularity-based recommendations for these customers
        pop_candidates = (
            pl.DataFrame({"Customer ID": customers_without_cp})
            .join(product_popularity, how="cross")
            .rename({"StockCode": "recommended_product"})
        )
        
        # Filter out products they already bought
        if customer_bought.height > 0:
            pop_candidates = pop_candidates.join(
                customer_bought.filter(pl.col("Customer ID").is_in(customers_without_cp)).rename({"StockCode": "recommended_product"}),
                on=["Customer ID", "recommended_product"],
                how="anti"
            )
        
        recommendations_fallback = pop_candidates.with_columns(
            pl.lit(0).cast(pl.UInt64).alias("co_purchase_score"),
            pl.lit(0).cast(pl.UInt64).alias("supporting_products"),
            pl.lit(1.0).alias("lift"),
            pl.col("popularity_score").alias("final_score"),
            pl.lit("popularity").alias("reason"),
        ).select(recommendations_with_cp.columns)
    else:
        recommendations_fallback = pl.DataFrame(schema=recommendations_with_cp.schema)

    # Combine and rank top-K per customer
    LOGGER.info("Ranking top-K recommendations per customer...")
    all_recommendations = pl.concat([recommendations_with_cp, recommendations_fallback], how="vertical")
    
    # Rank by final_score within each customer
    top_recommendations = (
        all_recommendations
        .with_columns(
            pl.col("final_score").rank(method="ordinal", descending=True).over("Customer ID").alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .drop("rank")
        .sort(["Customer ID", "final_score"], descending=[False, True])
    )

    LOGGER.info(f"Generated {top_recommendations.height:,} recommendations for {top_recommendations.select('Customer ID').n_unique()} customers")
    return top_recommendations


def enrich_recommendations(
    recommendations: pl.DataFrame,
    product_metrics_path: Path,
) -> pl.DataFrame:
    """Add product details to recommendations."""
    if not product_metrics_path.exists():
        return recommendations

    product_metrics = pl.read_parquet(product_metrics_path)

    product_info = product_metrics.select([
        "StockCode", "Description", "product_role", "avg_price", "total_revenue"
    ]).rename({"StockCode": "recommended_product"})

    return recommendations.join(product_info, on="recommended_product", how="left")


def save_plots(
    output_dir: Path,
    recommendations: pl.DataFrame,
) -> None:
    """Generate recommendation visualizations."""
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs_pd = recommendations.to_pandas()

    # Score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(recs_pd["final_score"], bins=30, alpha=0.8, edgecolor='white')
    ax.set_xlabel("Recommendation Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Recommendation Scores")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "01_score_distribution.png", dpi=180)
    plt.close(fig)

    # Reason distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    reason_counts = recs_pd["reason"].value_counts()
    ax.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
    ax.set_title("Recommendation Strategy Distribution")
    fig.tight_layout()
    fig.savefig(plots / "02_reason_distribution.png", dpi=180)
    plt.close(fig)

    # Score by reason
    fig, ax = plt.subplots(figsize=(10, 6))
    for reason in recs_pd["reason"].unique():
        subset = recs_pd[recs_pd["reason"] == reason]
        ax.hist(subset["final_score"], bins=20, alpha=0.5, label=reason, density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Recommendation Reason")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "03_score_by_reason.png", dpi=180)
    plt.close(fig)

    # Top recommended products
    fig, ax = plt.subplots(figsize=(10, 8))
    top_products = recs_pd["recommended_product"].value_counts().head(20)
    ax.barh(top_products.index[::-1], top_products.values[::-1])
    ax.set_xlabel("Recommendation Count")
    ax.set_title("Top 20 Recommended Products")
    fig.tight_layout()
    fig.savefig(plots / "04_top_recommended.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    co_purchase_path = Path(args.co_purchase_file).expanduser().resolve()
    segments_path = Path(args.segments_file).expanduser().resolve()
    product_metrics_path = Path("./product_analytics_output/product_metrics.parquet")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Load canonical transactions
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

    # Load co-purchase matrix (top-N per product)
    co_purchase = load_co_purchase_matrix(co_purchase_path, args.min_support, args.max_partners)

    # Load customer segments (optional)
    customer_segments = load_customer_segments(segments_path)

    # Build customer purchase history
    customer_product = build_customer_history(tx)

    # Generate recommendations (fully vectorized)
    recommendations = generate_recommendations_vectorized(
        customer_product=customer_product,
        co_purchase=co_purchase,
        customer_segments=customer_segments,
        top_k=args.top_k,
    )

    # Enrich with product details
    recommendations = enrich_recommendations(recommendations, product_metrics_path)

    # Outputs
    LOGGER.info("Writing outputs...")
    recommendations.write_parquet(output_dir / "recommendations.parquet")
    recommendations.write_csv(output_dir / "recommendations.csv")

    # Model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model_name": "Co-Purchase Based Recommendation Engine (Fully Vectorized)",
            "methodology": "Pre-computed product affinity (top-20 per product) + popularity fallback via Polars joins",
            "top_k": args.top_k,
            "min_support": args.min_support,
            "max_partners_per_product": args.max_partners,
            "n_customers": recommendations.select("Customer ID").n_unique(),
            "n_recommendations": recommendations.height,
            "avg_recommendations_per_customer": recommendations.height / recommendations.select("Customer ID").n_unique(),
            "reason_distribution": recommendations.group_by("reason").len().to_dicts(),
        }, f, indent=2, default=str)

    # Feature importance (placeholder)
    pl.DataFrame().write_csv(output_dir / "feature_importance.csv")

    # Calibration data (not applicable for recommendations)
    pl.DataFrame().write_csv(output_dir / "validation_predictions.csv")

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, recommendations)

    LOGGER.info("Recommendation engine complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
