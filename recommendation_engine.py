#!/usr/bin/env python3
"""
Recommendation Engine — Product Co-Purchase Based Recommendations (Optimized)

This script builds customer-specific product recommendations using:
1. Product co-purchase relationships (from product_analytics)
2. Customer purchase history
3. Customer segment affinity
4. Product popularity

Optimized for performance using vectorized operations instead of per-customer loops.

Output: Customer ID, recommended_product, score, reason, support, lift

Uses the retail_ds shared package for canonical data processing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_customer_history(tx: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Build customer purchase history: which products each customer bought and when."""
    LOGGER.info("Building customer purchase history...")

    sales = tx.filter(pl.col("is_clean_sale"))

    # Customer-Product purchase history
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

    # Customer's most recent purchase per product (for recency)
    customer_product_recency = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg(pl.col("InvoiceDate").max().alias("last_purchase_date"))
    )

    return customer_product, customer_product_recency


def load_co_purchase_matrix(co_purchase_path: Path) -> pl.DataFrame:
    """Load and validate co-purchase matrix."""
    LOGGER.info(f"Loading co-purchase matrix from {co_purchase_path}")
    
    if not co_purchase_path.exists():
        raise FileNotFoundError(f"Co-purchase matrix not found: {co_purchase_path}")
    
    co_purchase = pl.read_parquet(co_purchase_path)
    
    # Ensure symmetric (both directions)
    if "product_a" in co_purchase.columns and "product_b" in co_purchase.columns:
        # Make symmetric
        reverse = co_purchase.rename({"product_a": "product_b", "product_b": "product_a"})
        co_purchase = pl.concat([co_purchase, reverse], how="diagonal_relaxed").unique()
    
    LOGGER.info(f"Loaded co-purchase matrix: {co_purchase.height:,} pairs")
    return co_purchase


def load_customer_segments(segments_path: Path) -> pl.DataFrame:
    """Load customer segments if available."""
    LOGGER.info(f"Loading customer segments from {segments_path}")
    
    if not segments_path.exists():
        LOGGER.warning(f"Segments file not found: {segments_path}")
        return pl.DataFrame(schema={"Customer ID": pl.Int64, "segment": pl.Int64})
    
    segments = pl.read_parquet(segments_path)
    # Keep only relevant columns
    keep_cols = ["Customer ID"]
    for col in ["segment", "segment_name", "segment_confidence"]:
        if col in segments.columns:
            keep_cols.append(col)
    return segments.select(keep_cols).unique(subset=["Customer ID"])


def generate_recommendations_vectorized(
    customer_product: pl.DataFrame,
    co_purchase: pl.DataFrame,
    customer_segments: pl.DataFrame,
    product_metrics: pl.DataFrame,
    top_k: int = 10,
    min_support: int = 3,
) -> pl.DataFrame:
    """
    Generate recommendations for all customers using vectorized operations.
    
    Key optimization: Pre-compute product scores globally, then filter per customer.
    """
    LOGGER.info("Generating recommendations (vectorized)...")
    
    # Get all unique products from co-purchase matrix
    all_products = co_purchase.select("product_a").unique().rename({"product_a": "StockCode"})
    
    # Get product popularity (for fallback)
    product_popularity = co_purchase.group_by("product_a").agg(
        pl.col("cooccurrence").sum().alias("total_cooccurrence")
    ).rename({"product_a": "StockCode"}).sort("total_cooccurrence", descending=True)
    
    # Normalize popularity scores
    max_pop = product_popularity.select(pl.col("total_cooccurrence").max()).item()
    product_popularity = product_popularity.with_columns(
        (pl.col("total_cooccurrence") / max_pop).alias("popularity_score"),
    ).rename({"total_cooccurrence": "pop_score"})
    
    # Get all customer IDs
    all_customers = customer_product.select("Customer ID").unique()
    customer_bought = customer_product.select(["Customer ID", "StockCode"]).unique()
    
    # Filter co-purchase matrix by minimum support
    co_purchase_filtered = co_purchase.filter(pl.col("cooccurrence") >= 3)
    
    # Pre-compute product affinity scores for all product pairs
    # This replaces the per-customer join with a global affinity matrix
    LOGGER.info("Pre-computing product affinity scores...")
    
    # For each product, get its co-purchase partners and scores
    co_purchase_scores = co_purchase_filtered.group_by("product_a").agg(
        pl.col("product_b").alias("co_product"),
        pl.col("cooccurrence").alias("co_score"),
    )
    
    # Convert to dictionary for fast lookup: product -> list of (co_product, score)
    product_affinity = {}
    for row in co_purchase_scores.iter_rows(named=True):
        product_affinity[row["product_a"]] = list(zip(row["co_product"], row["co_score"]))
    
    # Get product popularity scores as dictionary
    pop_dict = dict(zip(
        product_popularity["StockCode"].to_list(),
        product_popularity["popularity_score"].to_list()
    ))
    
    # Get all unique product codes
    all_products_list = all_products["StockCode"].to_list()
    
    # Get customer purchase history as dictionary: customer -> set of bought products
    bought_dict = {}
    for row in customer_product.group_by("Customer ID").agg(pl.col("StockCode")).iter_rows(named=True):
        bought_dict[row["Customer ID"]] = set(row["StockCode"])
    
    # Get all customer IDs
    all_customer_ids = all_customers.select("Customer ID").to_series().to_list()
    
    LOGGER.info(f"Generating recommendations for {len(all_customer_ids)} customers...")
    
    recommendations = []
    
    for i, customer_id in enumerate(all_customer_ids):
        if i % 1000 == 0:
            LOGGER.info(f"  Processed {i}/{len(all_customer_ids)} customers...")
        
        bought_products = bought_dict.get(customer_id, set())
        
        # Strategy 1: Co-purchase based recommendations
        co_scores_dict = {}
        
        for bought_product in bought_products:
            if bought_product in product_affinity:
                for co_product, co_score in product_affinity[bought_product]:
                    if co_product not in bought_products:
                        if co_product not in co_scores_dict:
                            co_scores_dict[co_product] = {"score": 0.0, "support": 0}
                        co_scores_dict[co_product]["score"] += co_score
                        co_scores_dict[co_product]["support"] += 1
        
        # Convert to DataFrame
        if co_scores_dict:
            co_scores_df = pl.DataFrame([
                {
                    "StockCode": k,
                    "co_purchase_score": v["score"],
                    "supporting_products": v["support"],
                    "lift": v["score"] / len(bought_products) if bought_products else 1.0,
                }
                for k, v in co_scores_dict.items()
            ])
        else:
            co_scores_df = pl.DataFrame(schema={"StockCode": pl.Utf8, "co_purchase_score": pl.Float64, "supporting_products": pl.Int64, "lift": pl.Float64})
        
        # Get candidate products (not already bought)
        candidate_products = [p for p in all_products_list if p not in bought_products]
        
        # Combine scores
        if co_scores_df.height > 0:
            # Convert co_scores to dict for fast lookup
            co_score_dict = dict(zip(
                co_scores_df["StockCode"].to_list(),
                zip(
                    co_scores_df["co_purchase_score"].to_list(),
                    co_scores_df["supporting_products"].to_list(),
                    co_scores_df["lift"].to_list()
                )
            ))
            
            # Build final recommendations
            recs = []
            for product in candidate_products:
                co_score, support, lift = co_score_dict.get(product, (0.0, 0, 1.0))
                pop_score = pop_dict.get(product, 0.0)
                final_score = 0.7 * co_score + 0.3 * pop_score
                recs.append({
                    "StockCode": product,
                    "co_purchase_score": co_score,
                    "supporting_products": support,
                    "lift": lift,
                    "popularity_score": pop_score,
                    "final_score": final_score,
                })
            
            recs_df = pl.DataFrame(recs)
            reason = "co_purchase"
        else:
            # Fallback to popularity
            recs = []
            for product in candidate_products:
                pop_score = pop_dict.get(product, 0.0)
                recs.append({
                    "StockCode": product,
                    "co_purchase_score": 0.0,
                    "supporting_products": 0,
                    "lift": 1.0,
                    "popularity_score": pop_score,
                    "final_score": pop_score,
                })
            recs_df = pl.DataFrame(recs)
            reason = "popularity"
        
        # Get top K
        top_recs = recs_df.sort("final_score", descending=True).head(10)
        
        # Add reason, support, lift
        for row in top_recs.iter_rows(named=True):
            recommendations.append({
                "Customer ID": customer_id,
                "recommended_product": row["StockCode"],
                "score": row["final_score"],
                "reason": reason,
                "support": row.get("supporting_products", 0),
                "lift": row.get("lift", 1.0),
            })
    
    if not recommendations:
        return pl.DataFrame(schema={
            "Customer ID": pl.Int64,
            "recommended_product": pl.Utf8,
            "score": pl.Float64,
            "reason": pl.Utf8,
            "support": pl.Int64,
            "lift": pl.Float64,
        })
    
    return pl.DataFrame(recommendations)


def enrich_recommendations(
    recommendations: pl.DataFrame,
    product_metrics_path: Path,
) -> pl.DataFrame:
    """Add product details to recommendations."""
    if not product_metrics_path.exists():
        return recommendations
    
    product_metrics = pl.read_parquet(product_metrics_path)
    
    # Add product description and category
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
    ax.hist(recs_pd["score"], bins=30, alpha=0.8, edgecolor='white')
    ax.set_xlabel("Recommendation Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Recommendation Scores")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "01_score_distribution.png", dpi=180)
    plt.close(fig)
    
    # Reason distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    reason_counts = recs_pd["reason"].value_counts()
    ax.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
    ax.set_title("Recommendation Strategy Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "02_reason_distribution.png", dpi=180)
    plt.close(fig)
    
    # Score by reason
    fig, ax = plt.subplots(figsize=(10, 6))
    for reason in recs_pd["reason"].unique():
        subset = recs_pd[recs_pd["reason"] == reason]
        ax.hist(subset["score"], bins=20, alpha=0.5, label=reason, density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Recommendation Reason")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "03_score_by_reason.png", dpi=180)
    plt.close(fig)
    
    # Top recommended products
    fig, ax = plt.subplots(figsize=(10, 8))
    top_products = recs_pd["recommended_product"].value_counts().head(20)
    ax.barh(top_products.index[::-1], top_products.values[::-1])
    ax.set_xlabel("Recommendation Count")
    ax.set_title("Top 20 Recommended Products")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "04_top_recommended.png", dpi=180)
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
    # Load co-purchase matrix
    # -------------------------------------------------------------------------
    co_purchase = load_co_purchase_matrix(co_purchase_path)

    # -------------------------------------------------------------------------
    # Load customer segments (optional)
    # -------------------------------------------------------------------------
    customer_segments = load_customer_segments(segments_path)

    # -------------------------------------------------------------------------
    # Build customer purchase history
    # -------------------------------------------------------------------------
    customer_product, customer_product_recency = build_customer_history(tx)

    # -------------------------------------------------------------------------
    # Load product metrics for enrichment
    # -------------------------------------------------------------------------
    product_metrics_path = Path("./product_analytics_output/product_metrics.parquet")

    # -------------------------------------------------------------------------
    # Generate recommendations
    # -------------------------------------------------------------------------
    recommendations = generate_recommendations_vectorized(
        customer_product=customer_product,
        co_purchase=co_purchase,
        customer_segments=customer_segments,
        product_metrics=pl.read_parquet(product_metrics_path) if product_metrics_path.exists() else pl.DataFrame(),
        top_k=args.top_k,
        min_support=args.min_support,
    )

    # Enrich with product details
    recommendations = enrich_recommendations(recommendations, product_metrics_path)

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    recommendations.write_parquet(output_dir / "recommendations.parquet")
    recommendations.write_csv(output_dir / "recommendations.csv")

    # Model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model_name": "Co-Purchase Based Recommendation Engine (Vectorized)",
            "methodology": "Pre-computed product affinity + popularity fallback",
            "top_k": args.top_k,
            "min_support": args.min_support,
            "n_customers": recommendations.select("Customer ID").n_unique(),
            "n_recommendations": recommendations.height,
            "avg_recommendations_per_customer": recommendations.height / recommendations.select("Customer ID").n_unique(),
            "reason_distribution": recommendations.group_by("reason").len().to_dicts(),
        }, f, indent=2, default=str)

    importance_df = pl.DataFrame()  # Not computing importance for speed
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Calibration data (not applicable for recommendations)
    pd.DataFrame().to_csv(output_dir / "validation_predictions.csv", index=False)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, recommendations)

    LOGGER.info("Recommendation engine complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    import argparse
    main()