#!/usr/bin/env python3
"""
Recommendation Engine — Product Co-Purchase Based Recommendations (Vectorized)

This script builds customer-specific product recommendations using:
1. Product co-purchase relationships (from product_analytics)
2. Customer purchase history
3. Product popularity (from transactional history)
4. Normalized co-purchase affinity as lift proxy

Fully vectorized using Polars joins and aggregations — no per-customer loops.

Output contract (matching Streamlit dashboard expectations):
    Customer ID, recommended_product, score, reason, support, lift

Additional diagnostic columns preserved:
    co_purchase_score, popularity_score, Description, product_role, avg_price, total_revenue
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
    parser.add_argument("--top-k", type=int, default=10, help="Top K recommendations per customer.")
    parser.add_argument("--min-support", type=int, default=3, help="Minimum co-purchase support.")
    parser.add_argument("--max-partners", type=int, default=20, help="Max co-purchase partners per product.")
    parser.add_argument("--fallback-pool", type=int, default=200, help="Popularity fallback candidate pool size.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_partners < 1:
        parser.error("--max-partners must be >= 1")
    if args.min_support < 1:
        parser.error("--min-support must be >= 1")
    if args.fallback_pool < 1:
        parser.error("--fallback-pool must be >= 1")
    
    return args


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

    LOGGER.info(f"Loaded co-purchase matrix (top {max_partners} per product, min_support={min_support}): {co_purchase.height:,} pairs")
    return co_purchase


def normalize_recommendation_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure both co-purchase and fallback branches have identical schema for vertical concat."""
    return df.with_columns(
        pl.col("co_purchase_score").cast(pl.Float64),
        pl.col("supporting_products").cast(pl.Int64),
        pl.col("lift").cast(pl.Float64),
        pl.col("popularity_score").cast(pl.Float64),
        pl.col("final_score").cast(pl.Float64),
        pl.col("reason").cast(pl.Utf8),
    )


def generate_recommendations_vectorized(
    customer_product: pl.DataFrame,
    co_purchase: pl.DataFrame,  # Already filtered to top-N per product
    top_k: int = 10,
    fallback_pool_size: int = 200,
) -> pl.DataFrame:
    """
    Generate recommendations for all customers using fully vectorized Polars operations.
    
    Strategy:
    1. Co-purchase matrix already filtered to top-N partners per product
    2. Join customer purchases with co-purchase matrix to get candidate products per customer
    3. Aggregate co-purchase features per customer-product
    4. Compute product popularity from transactional history
    5. Normalize and blend scores
    6. Rank and select top-K per customer (deterministic)
    """
    LOGGER.info("Generating recommendations (fully vectorized)...")

    # Handle empty co-purchase matrix
    if co_purchase.is_empty():
        LOGGER.warning("No co-purchase pairs remain after filtering. Generating popularity-only recommendations.")
        return generate_popularity_only_recommendations(customer_product, top_k)

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
    # customer_bought has columns: Customer ID, StockCode
    # Rename StockCode to recommended_product for the anti join
    customer_candidates = customer_candidates.join(
        customer_bought.rename({"StockCode": "recommended_product"}),
        on=["Customer ID", "recommended_product"],
        how="anti"
    )
    LOGGER.info(f"After removing already-bought: {customer_candidates.height:,}")

    # Aggregate co-purchase features per customer-product
    customer_scores = (
        customer_candidates.group_by(["Customer ID", "recommended_product"])
        .agg(
            pl.col("cooccurrence").sum().alias("co_purchase_score"),
            pl.col("product_a").n_unique().alias("supporting_products"),
        )
    )

    # Compute lift proxy: normalized affinity
    # lift = co_purchase_score / (supporting_products * avg_support_per_product)
    # This is a proxy for association-rule lift, not true lift
    total_coocc = co_purchase.select(pl.col("cooccurrence").sum()).item()
    n_unique_products = co_purchase.select("product_a").n_unique()
    avg_support = total_coocc / n_unique_products
    customer_scores = customer_scores.with_columns(
        (pl.col("co_purchase_score") / (pl.col("supporting_products") * avg_support)).alias("lift"),
    )

    # Product popularity from TRANSACTIONAL HISTORY (not filtered co-purchase)
    LOGGER.info("Computing product popularity from transactional history...")
    product_popularity = (
        customer_product
        .group_by("StockCode")
        .agg(
            pl.col("purchase_count").sum().alias("product_invoice_count"),
            pl.col("Customer ID").n_unique().alias("unique_customers"),
            pl.col("total_revenue").sum().alias("product_total_revenue"),
        )
        .sort("unique_customers", descending=True)
    )
    
    # Normalize popularity
    max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
    product_popularity = product_popularity.with_columns(
        (pl.col("unique_customers") / max_pop).alias("popularity_score"),
    )
    # Keep only columns needed for the join and downstream
    product_popularity_for_join = product_popularity.select(["StockCode", "popularity_score"])

    # For customers WITH co-purchase scores, merge with popularity
    LOGGER.info("Merging co-purchase scores with popularity...")
    
    # WORKAROUND: Use when/then/otherwise instead of fill_null to avoid Polars bug
    # that corrupts co_purchase_score column type
    recommendations_with_cp = (
        customer_scores
        .join(product_popularity_for_join, left_on="recommended_product", right_on="StockCode", how="left")
        .with_columns(
            pl.when(pl.col("popularity_score").is_null())
            .then(0.0)
            .otherwise(pl.col("popularity_score"))
            .alias("popularity_score")
        )
    )
    
    # Normalize co_purchase_score per customer to [0, 1] before blending
    recommendations_with_cp = recommendations_with_cp.with_columns(
        (pl.col("co_purchase_score") / pl.col("co_purchase_score").max().over("Customer ID"))
        .fill_nan(0.0)
        .alias("co_purchase_score_norm")
    )
    
    # Blend normalized scores: 70% co-purchase affinity, 30% popularity
    recommendations_with_cp = recommendations_with_cp.with_columns(
        (pl.lit(0.7) * pl.col("co_purchase_score_norm") + pl.lit(0.3) * pl.col("popularity_score"))
        .alias("final_score"),
        pl.lit("co_purchase").alias("reason"),
    )

    # For customers WITHOUT co-purchase scores, use pure popularity
    customers_with_cp = recommendations_with_cp.select("Customer ID").unique()
    all_customer_ids = all_customers.select("Customer ID").to_series().to_list()
    customers_with_cp_ids = customers_with_cp.select("Customer ID").to_series().to_list()
    customers_without_cp = [cid for cid in all_customer_ids if cid not in customers_with_cp_ids]
    
    if customers_without_cp:
        LOGGER.info(f"Generating popularity fallback for {len(customers_without_cp)} customers...")
        # Use bounded fallback pool for efficiency
        popular_products = product_popularity.select(["StockCode", "popularity_score"]).head(fallback_pool_size)
        
        pop_candidates = (
            pl.DataFrame({"Customer ID": customers_without_cp})
            .join(popular_products, how="cross")
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
            pl.lit(0.0).alias("co_purchase_score"),
            pl.lit(0).cast(pl.UInt64).alias("supporting_products"),
            pl.lit(1.0).alias("lift"),
            pl.lit(0.0).alias("co_purchase_score_norm"),
            pl.col("popularity_score").alias("final_score"),
            pl.lit("popularity").alias("reason"),
        ).select([
            "Customer ID", "recommended_product", 
            "co_purchase_score", "supporting_products", "lift", 
            "co_purchase_score_norm", "popularity_score", "final_score", "reason"
        ])
    else:
        recommendations_fallback = pl.DataFrame(schema=recommendations_with_cp.schema)

    # Normalize both branches to identical schema before concat
    # First, drop extra columns from recommendations_with_cp (co_purchase_score_norm will be dropped in final select anyway)
    # We only need columns that both branches have for concat
    common_columns = [
        "Customer ID", "recommended_product",
        "co_purchase_score", "supporting_products", "lift",
        "co_purchase_score_norm", "popularity_score", "final_score", "reason"
    ]
    recommendations_with_cp = recommendations_with_cp.select(common_columns)
    recommendations_fallback = recommendations_fallback.select(common_columns)

    # Normalize schema
    recommendations_with_cp = normalize_recommendation_schema(recommendations_with_cp)
    recommendations_fallback = normalize_recommendation_schema(recommendations_fallback)

    # Combine and rank top-K per customer (DETERMINISTIC: sort by score then product code)
    LOGGER.info("Ranking top-K recommendations per customer...")
    all_recommendations = pl.concat([recommendations_with_cp, recommendations_fallback], how="vertical")
    
    # Sort first for deterministic tie-breaking
    all_recommendations = all_recommendations.sort(
        ["Customer ID", "final_score", "recommended_product"],
        descending=[False, True, False]
    )
    
    top_recommendations = (
        all_recommendations
        .with_columns(
            pl.col("final_score")
            .rank(method="ordinal", descending=True)
            .over("Customer ID")
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .drop("rank")
        # Output contract matching Streamlit dashboard
        .select(
            "Customer ID",
            "recommended_product",
            pl.col("final_score").alias("score"),
            "reason",
            pl.col("supporting_products").alias("support"),
            "lift",
            "co_purchase_score",
            "popularity_score",
        )
    )

    LOGGER.info(f"Generated {top_recommendations.height:,} recommendations for {top_recommendations.select('Customer ID').n_unique()} customers")
    return top_recommendations


def generate_popularity_only_recommendations(
    customer_product: pl.DataFrame,
    top_k: int,
) -> pl.DataFrame:
    """Generate recommendations using only product popularity when no co-purchase data exists."""
    LOGGER.info("Generating popularity-only recommendations...")
    
    # Product popularity from transactional history
    product_popularity = (
        customer_product
        .group_by("StockCode")
        .agg(
            pl.col("purchase_count").sum().alias("product_invoice_count"),
            pl.col("Customer ID").n_unique().alias("unique_customers"),
        )
        .sort("unique_customers", descending=True)
    )
    
    max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
    product_popularity = product_popularity.with_columns(
        (pl.col("unique_customers") / max_pop).alias("popularity_score"),
    ).select(["StockCode", "popularity_score"])
    
    all_customers = customer_product.select("Customer ID").unique()
    customer_bought = customer_product.select(["Customer ID", "StockCode"]).unique()
    
    # Cross join with all popular products
    pop_candidates = all_customers.join(product_popularity, how="cross").rename({"StockCode": "recommended_product"})
    
    # Filter out already-bought
    pop_candidates = pop_candidates.join(
        customer_bought.rename({"StockCode": "recommended_product"}),
        on=["Customer ID", "recommended_product"],
        how="anti"
    )
    
    recommendations = pop_candidates.with_columns(
        pl.lit(0.0).alias("co_purchase_score"),
        pl.lit(0).cast(pl.UInt64).alias("supporting_products"),
        pl.lit(1.0).alias("lift"),
        pl.col("popularity_score").alias("final_score"),
        pl.lit("popularity").alias("reason"),
    )
    
    # Deterministic ranking
    recommendations = recommendations.sort(
        ["Customer ID", "final_score", "recommended_product"],
        descending=[False, True, False]
    )
    
    top_recommendations = (
        recommendations
        .with_columns(
            pl.col("final_score")
            .rank(method="ordinal", descending=True)
            .over("Customer ID")
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .drop("rank")
        .select(
            "Customer ID",
            "recommended_product",
            pl.col("final_score").alias("score"),
            "reason",
            pl.col("supporting_products").alias("support"),
            "lift",
            "co_purchase_score",
            "popularity_score",
        )
    )
    
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
    ax.hist(recs_pd["score"], bins=30, alpha=0.8, edgecolor='white')
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
        ax.hist(subset["score"], bins=20, alpha=0.5, label=reason, density=True)
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

    # Build customer purchase history
    customer_product = build_customer_history(tx)

    # Generate recommendations (fully vectorized)
    recommendations = generate_recommendations_vectorized(
        customer_product=customer_product,
        co_purchase=co_purchase,
        top_k=args.top_k,
        fallback_pool_size=args.fallback_pool,
    )

    # Enrich with product details
    recommendations = enrich_recommendations(recommendations, product_metrics_path)

    # Outputs
    LOGGER.info("Writing outputs...")
    recommendations.write_parquet(output_dir / "recommendations.parquet")
    recommendations.write_csv(output_dir / "recommendations.csv")

    # Model card - honest about what this is
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model_name": "Co-Purchase Based Recommendation Engine (Vectorized)",
            "methodology": "Pre-computed product affinity (top-20 per product) + popularity fallback via Polars joins",
            "top_k": args.top_k,
            "min_support": args.min_support,
            "max_partners_per_product": args.max_partners,
            "fallback_pool_size": args.fallback_pool,
            "n_customers": recommendations.select("Customer ID").n_unique(),
            "n_recommendations": recommendations.height,
            "avg_recommendations_per_customer": recommendations.height / recommendations.select("Customer ID").n_unique(),
            "reason_distribution": recommendations.group_by("reason").len().to_dicts(),
            "limitations": [
                "Observational co-purchase only - no causal recommendation effect",
                "No explicit segment affinity (segment feature loaded but not used)",
                "Lift is normalized affinity proxy, not true association-rule lift",
                "Popularity fallback based on bounded candidate pool",
                "No temporal holdout validation - offline metrics not computed",
            ],
        }, f, indent=2, default=str)

    # Feature importance (placeholder - not applicable for heuristic engine)
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
