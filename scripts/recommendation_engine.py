#!/usr/bin/env python3
"""
Recommendation Engine — Product Co-Purchase Based Recommendations (Fully Vectorized)

This script builds customer-specific product recommendations using:
1. Product co-purchase relationships (from product_analytics)
2. Customer purchase history
3. Product popularity
4. True association-rule lift

Optimized for performance using fully vectorized Polars operations.

Output: Customer ID, recommended_product, rank, score, reason, support, association_lift,
        co_purchase_score, popularity_score, Description, product_role, avg_price, total_revenue

Uses the retail_ds shared package for canonical data processing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.validation import run_all_validations, assert_validations_pass
from scripts.product_analytics import build_co_purchase_matrix


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
    parser = argparse.ArgumentParser(description="Product recommendation engine (fully vectorized).")
    parser.add_argument("--input", default="./data_xslx/online_retail_II.xlsx", help="Input file.")
    parser.add_argument("--output-dir", default="./recommendation_output", help="Output directory.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--co-purchase-file", default="./product_analytics_output/co_purchase_matrix.parquet", help="Co-purchase matrix from product_analytics.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K recommendations per customer.")
    parser.add_argument("--min-support", type=int, default=3, help="Minimum co-purchase support.")
    parser.add_argument("--max-partners", type=int, default=20, help="Max co-purchase partners per product.")
    parser.add_argument("--fallback-pool-size", type=int, default=200, help="Size of popularity fallback pool.")
    parser.add_argument("--temporal-eval", action="store_true", help="Run temporal holdout evaluation.")
    parser.add_argument("--eval-cutoff", default="2011-06-30", help="Temporal cutoff for evaluation (YYYY-MM-DD).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_customer_history(tx: pl.DataFrame) -> pl.DataFrame:
    """Build customer purchase history: which products each customer bought."""
    LOGGER.info("Building customer purchase history...")

    sales = tx.filter(pl.col("is_clean_sale"))

    # Customer-Product purchase history
    customer_product = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg([
            pl.col("InvoiceDate").min().alias("first_purchase_date"),
            pl.col("InvoiceDate").max().alias("last_purchase_date"),
            pl.col("Invoice").n_unique().alias("purchase_count"),
            pl.col("Quantity").sum().alias("total_quantity"),
            pl.col("gross_merchandise_revenue").sum().alias("total_revenue"),
        ])
        .sort(["Customer ID", "total_revenue"], descending=[False, True])
    )

    return customer_product


def load_co_purchase_matrix(
    co_purchase_path: Path,
    max_partners: int,
    min_support: int,
    product_popularity_df: pl.DataFrame = None,
) -> tuple:
    """Load and process co-purchase matrix with true association lift.

    Args:
        co_purchase_path: Path to co-purchase matrix parquet
        max_partners: Max co-purchase partners per product
        min_support: Minimum co-occurrence support
        product_popularity_df: Optional DataFrame with StockCode, unique_customers for true support

    Returns:
        Tuple of (co_purchase_top, product_supports)
        product_supports has columns: StockCode, product_support (unique customers)
    """
    LOGGER.info(f"Loading co-purchase matrix from {co_purchase_path}")

    if not co_purchase_path.exists():
        raise FileNotFoundError(f"Co-purchase matrix not found: {co_purchase_path}")

    co_purchase = pl.read_parquet(co_purchase_path)

    required_cols = ["product_a", "product_b", "cooccurrence"]
    for col in required_cols:
        if col not in co_purchase.columns:
            raise ValueError(f"Co-purchase matrix missing required column: {col}")

    # Filter by minimum support
    co_purchase = co_purchase.filter(pl.col("cooccurrence") >= min_support)

    # Compute TRUE product supports: unique customers per product
    if product_popularity_df is not None and "unique_customers" in product_popularity_df.columns:
        product_supports = product_popularity_df.select(["StockCode", "unique_customers"]).rename(
            {"unique_customers": "product_support"}
        )
    else:
        # Fallback: sum of co-occurrences (less accurate)
        LOGGER.warning("No product popularity data provided; using co-occurrence sum for support (approximate)")
        product_supports = co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("product_support")
        ).rename({"product_a": "StockCode"})

    # Make symmetric: for each pair (a,b), add (b,a) with same cooccurrence
    reverse = co_purchase.rename({"product_a": "product_b", "product_b": "product_a"})
    co_purchase_sym = pl.concat([co_purchase, reverse], how="diagonal_relaxed").unique()

    # Keep top max_partners per product by cooccurrence
    co_purchase_top = (
        co_purchase_sym
        .sort(["product_a", "cooccurrence"], descending=[False, True])
        .group_by("product_a")
        .head(max_partners)
    )

    LOGGER.info(f"Loaded co-purchase matrix: {co_purchase_top.height:,} pairs (top {max_partners} per product)")
    return co_purchase_top, product_supports


def compute_association_lift(
    co_purchase: pl.DataFrame,
    product_supports: pl.DataFrame,
    total_customers: int,
) -> pl.DataFrame:
    """Compute true association lift for co-purchase pairs.

    Lift = P(A,B) / (P(A) * P(B)) = cooccurrence / (support_a * support_b / N)
    where N = total customers
    """
    # Join product supports
    cp = co_purchase.join(
        product_supports.rename({"StockCode": "product_a", "product_support": "support_a"}),
        on="product_a",
        how="left",
    ).join(
        product_supports.rename({"StockCode": "product_b", "product_support": "support_b"}),
        on="product_b",
        how="left",
    )

    # Compute lift
    cp = cp.with_columns([
        (pl.col("cooccurrence") * total_customers / (pl.col("support_a") * pl.col("support_b"))).alias("association_lift"),
    ])

    # Replace infinite/NaN with null
    cp = cp.with_columns([
        pl.when(pl.col("association_lift").is_infinite() | pl.col("association_lift").is_nan())
        .then(None)
        .otherwise(pl.col("association_lift"))
        .alias("association_lift"),
    ])

    return cp


def generate_recommendations_vectorized(
    customer_product: pl.DataFrame,
    co_purchase: pl.DataFrame,
    product_popularity: pl.DataFrame,
    top_k: int,
    fallback_pool_size: int,
) -> pl.DataFrame:
    """
    Generate recommendations for all customers using fully vectorized Polars operations.

    Strategy:
    1. For each customer, get their purchased products
    2. Join with co-purchase matrix to get candidate products
    3. Remove already-purchased products
    4. Score by co-purchase score and popularity
    4. Rank and take top-K per customer
    """
    LOGGER.info("Generating recommendations (fully vectorized)...")

    # Get all customers and their purchased products
    customer_bought = customer_product.select(["Customer ID", "StockCode"]).unique()

    # Get candidate products from co-purchase matrix
    # For each customer-product, get co-purchase partners
    candidates = customer_bought.join(
        co_purchase.rename({"product_a": "StockCode", "product_b": "candidate_product", "cooccurrence": "co_score", "association_lift": "lift"}),
        on="StockCode",
        how="inner",
    )

    # Filter out already purchased products
    # Anti-join with customer_bought on (Customer ID, candidate_product)
    candidates = candidates.join(
        customer_bought.rename({"StockCode": "candidate_product"}),
        on=["Customer ID", "candidate_product"],
        how="anti",
    )

    # Aggregate scores per customer-candidate
    # Sum co-purchase scores, count supporting products, max lift
    scored = (
        candidates.group_by(["Customer ID", "candidate_product"])
        .agg([
            pl.col("co_score").sum().alias("co_purchase_score"),
            pl.col("lift").max().alias("association_lift"),
            pl.len().alias("supporting_products"),
        ])
    )

    # Join with product popularity
    scored = scored.join(
        product_popularity.rename({"StockCode": "candidate_product", "popularity_score": "popularity_score"}),
        on="candidate_product",
        how="left",
    ).with_columns([
        pl.col("popularity_score").fill_null(0.0),
    ])

    # Normalize co-purchase score per customer (0-1)
    scored = scored.with_columns([
        (pl.col("co_purchase_score") / pl.col("co_purchase_score").max().over("Customer ID")).alias("co_purchase_score_norm"),
    ])

    # Final score: weighted combination
    scored = scored.with_columns([
        (0.7 * pl.col("co_purchase_score_norm") + 0.3 * pl.col("popularity_score")).alias("final_score"),
    ])

    # Rank per customer
    scored = scored.with_columns([
        pl.col("final_score").rank(descending=True).over("Customer ID").alias("rank"),
    ])

    # Take top-K
    top_k_recs = scored.filter(pl.col("rank") <= top_k).sort(["Customer ID", "rank"])

    # Add reason
    top_k_recs = top_k_recs.with_columns([
        pl.when(pl.col("co_purchase_score") > 0).then(pl.lit("co_purchase")).otherwise(pl.lit("popularity")).alias("reason"),
    ])

    # Rename columns
    top_k_recs = top_k_recs.rename({
        "candidate_product": "recommended_product",
        "co_purchase_score": "co_purchase_score",
        "supporting_products": "support",
        "association_lift": "association_lift",
        "popularity_score": "popularity_score",
        "final_score": "score",
    })

    return top_k_recs.select([
        "Customer ID",
        "recommended_product",
        "rank",
        "score",
        "reason",
        "support",
        "association_lift",
        "co_purchase_score",
        "popularity_score",
    ])


def generate_popularity_fallback(
    customer_product: pl.DataFrame,
    product_popularity: pl.DataFrame,
    top_k: int,
    fallback_pool_size: int,
    customers_with_copurchase: set,
) -> pl.DataFrame:
    """Generate popularity-only recommendations for customers with NO co-purchase candidates."""
    LOGGER.info("Generating popularity fallback recommendations...")

    # Get all customers
    all_customers = customer_product.select("Customer ID").unique()

    # Filter to only customers WITHOUT co-purchase candidates
    customers_without_copurchase = all_customers.filter(
        ~pl.col("Customer ID").is_in(customers_with_copurchase)
    )

    if customers_without_copurchase.height == 0:
        LOGGER.info("All customers have co-purchase candidates, no fallback needed")
        return pl.DataFrame(schema={
            "Customer ID": pl.Int64,
            "recommended_product": pl.Utf8,
            "rank": pl.Int64,
            "score": pl.Float64,
            "reason": pl.Utf8,
            "support": pl.Int64,
            "association_lift": pl.Float64,
            "co_purchase_score": pl.Float64,
            "popularity_score": pl.Float64,
        })

    # Get top fallback products
    fallback_products = product_popularity.head(fallback_pool_size)["StockCode"].to_list()

    # Get customer purchase history for filtering
    customer_bought = customer_product.select(["Customer ID", "StockCode"]).unique()

    # Cross join customers without co-purchase with fallback products
    fallback_candidates = (
        customers_without_copurchase.join(
            pl.DataFrame({"recommended_product": fallback_products}),
            how="cross",
        )
    )

    # Remove already purchased
    fallback_candidates = fallback_candidates.join(
        customer_bought.rename({"StockCode": "recommended_product"}),
        on=["Customer ID", "recommended_product"],
        how="anti",
    )

    # Join popularity
    fallback_candidates = fallback_candidates.join(
        product_popularity.rename({"StockCode": "recommended_product", "popularity_score": "popularity_score"}),
        on="recommended_product",
        how="left",
    ).with_columns([
        pl.col("popularity_score").fill_null(0.0),
    ])

    # Rank and take top-K
    fallback_candidates = fallback_candidates.with_columns([
        pl.col("popularity_score").rank(descending=True).over("Customer ID").alias("rank"),
    ])

    top_fallback = fallback_candidates.filter(pl.col("rank") <= top_k).sort(["Customer ID", "rank"])

    top_fallback = top_fallback.with_columns([
        pl.lit("popularity").alias("reason"),
        pl.lit(0.0).alias("co_purchase_score"),
        pl.lit(0).alias("support"),
        pl.lit(None).alias("association_lift"),
        pl.col("popularity_score").alias("score"),
    ])

    return top_fallback.select([
        "Customer ID",
        "recommended_product",
        "rank",
        "score",
        "reason",
        "support",
        "association_lift",
        "co_purchase_score",
        "popularity_score",
    ])


def enrich_recommendations(
    recommendations: pl.DataFrame,
    product_metrics_path: Path,
) -> pl.DataFrame:
    """Add product details to recommendations."""
    if not product_metrics_path.exists():
        LOGGER.warning(f"Product metrics not found: {product_metrics_path}")
        return recommendations

    product_metrics = pl.read_parquet(product_metrics_path)

    product_info = (
        product_metrics.select([
            "StockCode", "Description", "product_role", "avg_price", "total_revenue"
        ])
        .unique(subset=["StockCode"], maintain_order=True)
        .rename({"StockCode": "recommended_product"})
    )

    return recommendations.join(product_info, on="recommended_product", how="left")


def temporal_holdout_evaluation(
    tx: pl.DataFrame,
    customer_product_train: pl.DataFrame,
    co_purchase: pl.DataFrame,
    cutoff_date: str,
    top_k: int,
    fallback_pool_size: int,
) -> dict:
    """Run temporal holdout evaluation of recommendations.

    Args:
        tx: Full canonical transactions
        customer_product_train: Customer-product history up to cutoff
        co_purchase: Co-purchase matrix from training period
        cutoff_date: Evaluation cutoff date
        top_k: Top-K for evaluation
        fallback_pool_size: Fallback pool size

    Returns:
        Dictionary with evaluation metrics
    """
    LOGGER.info(f"Running temporal holdout evaluation with cutoff {cutoff_date}...")

    # Compute TRUE product popularity from TRAINING data only
    train_sales = tx.filter(pl.col("is_clean_sale") & (pl.col("InvoiceDate") <= pl.lit(cutoff_date).str.strptime(pl.Datetime)))
    product_popularity_train = (
        train_sales.group_by("StockCode")
        .agg(pl.col("Customer ID").n_unique().alias("unique_customers"))
        .sort("unique_customers", descending=True)
    )
    max_pop = product_popularity_train.select(pl.col("unique_customers").max()).item()
    product_popularity_train = product_popularity_train.with_columns([
        (pl.col("unique_customers") / max_pop).alias("popularity_score"),
    ])

    # Generate recommendations for training period using TRAINING popularity
    recs_train = generate_recommendations_vectorized(
        customer_product_train, co_purchase, product_popularity_train, top_k, fallback_pool_size
    )

    # Get test period transactions (after cutoff)
    cutoff = pl.lit(cutoff_date).str.strptime(pl.Datetime)
    test_sales = tx.filter(
        (pl.col("is_clean_sale")) & (pl.col("InvoiceDate") > cutoff)
    )

    if test_sales.height == 0:
        LOGGER.warning("No test period sales found")
        return {"hit_rate_at_5": 0.0, "hit_rate_at_10": 0.0, "mrr_at_10": 0.0, "coverage": 0.0, "catalog_coverage": 0.0}

    # Get test period purchases per customer
    test_purchases = test_sales.group_by(["Customer ID", "StockCode"]).agg(
        pl.len().alias("test_count")
    )

    # Catalog size: eligible products from training period
    catalog_size = train_sales.select(pl.col("StockCode").n_unique()).item()

    # Evaluate
    metrics = evaluate_recommendations(
        recs_train, test_purchases, top_k, catalog_size
    )

    # Add baseline comparison using TRAINING popularity
    baseline_metrics = evaluate_popularity_baseline_from_training(
        test_purchases, product_popularity_train, top_k
    )

    metrics["baseline"] = baseline_metrics

    return metrics


def evaluate_recommendations(
    recs: pl.DataFrame,
    test_purchases: pl.DataFrame,
    top_k: int,
    catalog_size: int = None,
) -> dict:
    """Evaluate recommendations against test purchases."""

    # Join recommendations with test purchases
    eval_df = recs.join(
        test_purchases.rename({"StockCode": "recommended_product"}),
        on=["Customer ID", "recommended_product"],
        how="left",
    )

    # Mark hits
    eval_df = eval_df.with_columns([
        pl.col("test_count").is_not_null().alias("hit"),
    ])

    # Metrics at different K
    metrics = {}
    for k in [5, 10]:
        if k > top_k:
            continue
        k_eval = eval_df.filter(pl.col("rank") <= k)

        # Hit Rate @ K: fraction of customers with at least one hit in top-K
        hit_rate = k_eval.group_by("Customer ID").agg(
            pl.col("hit").any().alias("has_hit")
        )["has_hit"].mean()

        # Recall @ K: fraction of test purchases captured
        total_test = test_purchases.group_by("Customer ID").agg(
            pl.col("test_count").sum().alias("total_test")
        )
        hits = k_eval.group_by("Customer ID").agg(
            pl.col("test_count").sum().alias("hits")
        )
        recall_df = total_test.join(hits, on="Customer ID", how="left").fill_null(0)
        recall = (recall_df["hits"] / recall_df["total_test"]).mean()

        metrics[f"hit_rate_at_{k}"] = float(hit_rate)
        metrics[f"recall_at_{k}"] = float(recall)

    # MRR @ 10
    if top_k >= 10:
        mrr_df = eval_df.filter(pl.col("rank") <= 10).filter(pl.col("hit"))
        mrr = mrr_df.group_by("Customer ID").agg(
            (1.0 / pl.col("rank").min()).alias("rr")
        )["rr"].mean()
        metrics["mrr_at_10"] = float(mrr) if mrr is not None else 0.0

    # Coverage
    n_customers_eval = eval_df["Customer ID"].n_unique()
    n_customers_with_recs = eval_df.filter(pl.col("rank").is_not_null())["Customer ID"].n_unique()
    metrics["recommendation_coverage"] = float(n_customers_with_recs / n_customers_eval) if n_customers_eval > 0 else 0.0

    # Catalog coverage: unique recommended products / eligible catalog size
    n_recommended_products = recs["recommended_product"].n_unique()
    if catalog_size and catalog_size > 0:
        metrics["catalog_coverage"] = float(n_recommended_products / catalog_size)
    else:
        metrics["catalog_coverage"] = float(n_recommended_products)

    metrics["n_customers_evaluated"] = int(n_customers_eval)
    metrics["n_customers_with_recommendations"] = int(n_customers_with_recs)
    metrics["n_recommendations"] = int(recs.height)

    return metrics


def evaluate_popularity_baseline_from_training(
    test_purchases: pl.DataFrame,
    product_popularity_train: pl.DataFrame,
    top_k: int,
) -> dict:
    """Evaluate global popularity baseline using TRAINING period popularity."""
    top_pop = product_popularity_train.head(top_k)["StockCode"].to_list()

    test_cust = test_purchases.group_by("Customer ID").agg(
        pl.col("StockCode").alias("test_products"),
    )

    hits = 0
    total = 0
    for row in test_cust.iter_rows(named=True):
        cust_products = set(row["test_products"])
        if cust_products & set(top_pop):
            hits += 1
        total += 1

    return {
        "hit_rate": hits / total if total > 0 else 0.0,
    }


def evaluate_popularity_baseline(
    test_purchases: pl.DataFrame,
    top_k: int,
) -> dict:
    """Evaluate global popularity baseline."""
    # Global popularity: products by total test purchases
    global_pop = test_purchases.group_by("StockCode").agg(
        pl.col("test_count").sum().alias("total_test")
    ).sort("total_test", descending=True)

    top_pop = global_pop.head(top_k)["StockCode"].to_list()

    # For each customer, check if their test purchases are in top_pop
    test_cust = test_purchases.group_by("Customer ID").agg(
        pl.col("StockCode").alias("test_products"),
    )

    hits = 0
    total = 0
    for row in test_cust.iter_rows(named=True):
        cust_products = set(row["test_products"])
        if cust_products & set(top_pop):
            hits += 1
        total += 1

    return {
        "hit_rate": hits / total if total > 0 else 0.0,
    }


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

    # Association lift distribution
    if "association_lift" in recs_pd.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        lift_data = recs_pd["association_lift"].dropna()
        if len(lift_data) > 0:
            ax.hist(lift_data, bins=30, alpha=0.8, edgecolor='white')
            ax.set_xlabel("Association Lift")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Association Lift")
            ax.grid(alpha=0.15)
            fig.tight_layout()
            fig.savefig(output_dir / "plots" / "05_lift_distribution.png", dpi=180)
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
    # Load product metrics for true popularity
    # -------------------------------------------------------------------------
    if product_metrics_path.exists():
        product_metrics_for_popularity = pl.read_parquet(product_metrics_path)
    else:
        product_metrics_for_popularity = None

    # -------------------------------------------------------------------------
    # Load co-purchase matrix and compute association lift
    # -------------------------------------------------------------------------
    co_purchase, product_supports = load_co_purchase_matrix(
        co_purchase_path, args.max_partners, args.min_support, product_metrics_for_popularity
    )

    total_customers = tx.filter(pl.col("is_clean_sale")).select(pl.col("Customer ID").n_unique()).item()
    co_purchase = compute_association_lift(co_purchase, product_supports, total_customers)

    # -------------------------------------------------------------------------
    # Build customer purchase history
    # -------------------------------------------------------------------------
    customer_product = build_customer_history(tx)

    # -------------------------------------------------------------------------
    # Product popularity (for fallback) — TRUE popularity from transaction data
    # -------------------------------------------------------------------------
    # Load product metrics for true popularity (unique customers per product)
    product_metrics_popularity = pl.read_parquet(product_metrics_path)
    if "unique_customers" in product_metrics_popularity.columns:
        product_popularity = product_metrics_popularity.select(["StockCode", "unique_customers"]).sort(
            "unique_customers", descending=True
        )
        max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
        product_popularity = product_popularity.with_columns([
            (pl.col("unique_customers") / max_pop).alias("popularity_score"),
        ])
    else:
        # Fallback if unique_customers not available
        LOGGER.warning("unique_customers not in product_metrics; using co-occurrence as popularity proxy")
        product_popularity = co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("total_cooccurrence")
        ).rename({"product_a": "StockCode"}).sort("total_cooccurrence", descending=True)
        max_pop = product_popularity.select(pl.col("total_cooccurrence").max()).item()
        product_popularity = product_popularity.with_columns([
            (pl.col("total_cooccurrence") / max_pop).alias("popularity_score"),
        ]).rename({"total_cooccurrence": "pop_score"})

    # -------------------------------------------------------------------------
    # Generate recommendations
    # -------------------------------------------------------------------------
    recommendations = generate_recommendations_vectorized(
        customer_product=customer_product,
        co_purchase=co_purchase.select(["product_a", "product_b", "cooccurrence", "association_lift"]),
        product_popularity=product_popularity,
        top_k=args.top_k,
        fallback_pool_size=args.fallback_pool_size,
    )

    # Get customers who have co-purchase candidates
    customers_with_copurchase = set(recommendations.select("Customer ID").unique().to_series().to_list())

    # Generate fallback for customers with no co-purchase candidates
    fallback_recs = generate_popularity_fallback(
        customer_product=customer_product,
        product_popularity=product_popularity,
        top_k=args.top_k,
        fallback_pool_size=args.fallback_pool_size,
        customers_with_copurchase=customers_with_copurchase,
    )

    # Combine: co-purchase first, then fallback
    all_recs = pl.concat([recommendations, fallback_recs], how="diagonal_relaxed")

    # Re-rank combined (co-purchase ranked first, then fallback)
    all_recs = all_recs.with_columns([
        pl.when(pl.col("reason") == "co_purchase").then(pl.lit(0)).otherwise(pl.lit(1)).alias("reason_priority"),
    ]).sort(["Customer ID", "reason_priority", "rank"]).drop("reason_priority")

    # Sort by score descending so highest score comes first
    all_recs = all_recs.sort(["Customer ID", "score"], descending=[False, True])

    all_recs = all_recs.unique(subset=["Customer ID", "recommended_product"], maintain_order=True)

    # Re-rank final
    all_recs = all_recs.with_columns([
        pl.col("score").rank(descending=True).over("Customer ID").alias("rank"),
    ]).filter(pl.col("rank") <= args.top_k).sort(["Customer ID", "rank"])

    # -------------------------------------------------------------------------
    # Temporal holdout evaluation (optional)
    # -------------------------------------------------------------------------
    evaluation_results = None
    if args.temporal_eval:
        # Build training customer-product up to cutoff
        cutoff = pl.lit(args.eval_cutoff).str.strptime(pl.Datetime)
        tx_train = tx.filter(pl.col("InvoiceDate") <= cutoff)
        customer_product_train = build_customer_history(tx_train)

        # Build co-purchase matrix from training data only
        co_purchase_train = build_co_purchase_matrix(tx_train, args.min_support, args.max_partners)
        total_customers_train = tx_train.filter(pl.col("is_clean_sale")).select(pl.col("Customer ID").n_unique()).item()

        # Compute TRUE product supports from training data
        product_supports_train = (
            tx_train.filter(pl.col("is_clean_sale"))
            .group_by("StockCode")
            .agg(pl.col("Customer ID").n_unique().alias("product_support"))
        )

        # Compute lift for training co-purchase matrix
        co_purchase_train = compute_association_lift(co_purchase_train, product_supports_train, total_customers_train)

        evaluation_results = temporal_holdout_evaluation(
            tx=tx,
            customer_product_train=customer_product_train,
            co_purchase=co_purchase_train.select(["product_a", "product_b", "cooccurrence", "association_lift"]),
            cutoff_date=args.eval_cutoff,
            top_k=args.top_k,
            fallback_pool_size=args.fallback_pool_size,
        )
        LOGGER.info(f"Temporal evaluation results: {evaluation_results}")

    # -------------------------------------------------------------------------
    # Enrich with product details (use deduplicated all_recs)
    # -------------------------------------------------------------------------
    all_recs = enrich_recommendations(all_recs, product_metrics_path)

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    all_recs.write_parquet(output_dir / "recommendations.parquet")
    all_recs.write_csv(output_dir / "recommendations.csv")

    # Model card
    model_card = {
        "model_name": "Co-Purchase Based Recommendation Engine (Fully Vectorized + True Lift)",
        "methodology": "Pre-computed product affinity with true association lift + popularity fallback",
        "top_k": args.top_k,
        "min_support": args.min_support,
        "max_partners_per_product": args.max_partners,
        "fallback_pool_size": args.fallback_pool_size,
        "score_weights": {"co_purchase": 0.7, "popularity": 0.3},
        "n_customers": all_recs.select("Customer ID").n_unique(),
        "n_recommendations": all_recs.height,
        "avg_recommendations_per_customer": all_recs.height / all_recs.select("Customer ID").n_unique(),
        "reason_distribution": all_recs.group_by("reason").len().to_dicts(),
        "association_lift_stats": {
            "mean": float(all_recs.select(pl.col("association_lift").mean()).item()) if "association_lift" in all_recs.columns else None,
            "median": float(all_recs.select(pl.col("association_lift").median()).item()) if "association_lift" in all_recs.columns else None,
        },
    }

    if evaluation_results:
        model_card["temporal_evaluation"] = evaluation_results

    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    # Placeholder files (not applicable for this engine)
    pl.DataFrame().write_csv(output_dir / "feature_importance.csv")
    pl.DataFrame().write_csv(output_dir / "validation_predictions.csv")

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, all_recs)

    LOGGER.info("Recommendation engine complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()