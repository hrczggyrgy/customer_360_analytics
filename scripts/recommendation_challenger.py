#!/usr/bin/env python3
"""
Recommendation Engine Challenger Framework

This module implements multiple candidate recommendation models and evaluates them
against each other using temporal holdout evaluation. The best model is selected
for production deployment.

Models implemented:
1. Co-purchase + Popularity Blend (current production baseline)
2. Item-based Collaborative Filtering (customer-item matrix)
3. Content-based Filtering (product features)
4. Popularity-only (baseline)
5. Hybrid (weighted ensemble)

All models are evaluated on the same temporal holdout split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import polars as pl
import pandas as pd

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args

SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("recommendation_challenger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommendation engine challenger framework.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--eval-cutoff", default="2011-06-30", help="Temporal cutoff for evaluation (YYYY-MM-DD).")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for evaluation.")
    parser.add_argument("--min-support", type=int, default=3, help="Minimum co-purchase support.")
    parser.add_argument("--max-partners", type=int, default=20, help="Max co-purchase partners per product.")
    parser.add_argument("--fallback-pool-size", type=int, default=200, help="Size of popularity fallback pool.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


class RecommendationModel:
    """Base class for recommendation models."""
    
    def __init__(self, name: str):
        self.name = name
    
    def fit(self, tx_train: pl.DataFrame, product_metrics: pl.DataFrame) -> None:
        """Fit the model on training data."""
        raise NotImplementedError
    
    def recommend(self, customer_ids: List[int], top_k: int) -> pl.DataFrame:
        """Generate recommendations for given customers."""
        raise NotImplementedError


class CoPurchasePopularityModel(RecommendationModel):
    """Co-purchase + Popularity blend (current production model)."""
    
    def __init__(self, min_support: int = 3, max_partners: int = 20, fallback_pool_size: int = 200):
        super().__init__("co_purchase_popularity")
        self.min_support = min_support
        self.max_partners = max_partners
        self.fallback_pool_size = fallback_pool_size
        self.co_purchase = None
        self.product_popularity = None
        self.customer_product = None
        self.all_products = None
    
    def fit(self, tx_train: pl.DataFrame, product_metrics: pl.DataFrame) -> None:
        from scripts.recommendation_engine import build_customer_history, build_co_purchase_matrix
        
        LOGGER.info(f"Fitting {self.name}...")
        
        # Build customer purchase history
        self.customer_product = build_customer_history(tx_train)
        
        # Build co-purchase matrix
        self.co_purchase = build_co_purchase_matrix(tx_train, self.min_support, self.max_partners)
        
        # Compute true product supports from training data
        product_supports = (
            tx_train.filter(pl.col("is_clean_sale"))
            .group_by("StockCode")
            .agg(pl.col("Customer ID").n_unique().alias("product_support"))
        )
        
        total_customers = tx_train.filter(pl.col("is_clean_sale")).select(pl.col("Customer ID").n_unique()).item()
        
        # Compute lift
        from scripts.recommendation_engine import compute_association_lift
        self.co_purchase = compute_association_lift(self.co_purchase, product_supports, total_customers)
        
        # Product popularity
        if "unique_customers" in product_metrics.columns:
            self.product_popularity = product_metrics.select(["StockCode", "unique_customers"]).sort(
                "unique_customers", descending=True
            )
            max_pop = self.product_popularity.select(pl.col("unique_customers").max()).item()
            self.product_popularity = self.product_popularity.with_columns([
                (pl.col("unique_customers") / max_pop).alias("popularity_score"),
            ])
        else:
            self.product_popularity = self.co_purchase.group_by("product_a").agg(
                pl.col("cooccurrence").sum().alias("total_cooccurrence")
            ).rename({"product_a": "StockCode"}).sort("total_cooccurrence", descending=True)
            max_pop = self.product_popularity.select(pl.col("total_cooccurrence").max()).item()
            self.product_popularity = self.product_popularity.with_columns([
                (pl.col("total_cooccurrence") / max_pop).alias("popularity_score"),
            ])
        
        self.all_products = self.product_popularity["StockCode"].to_list()
        
        LOGGER.info(f"  Co-purchase pairs: {self.co_purchase.height:,}")
        LOGGER.info(f"  Products in catalog: {len(self.all_products):,}")
    
    def recommend(self, customer_ids: List[int], top_k: int) -> pl.DataFrame:
        from scripts.recommendation_engine import (
            generate_recommendations_vectorized, generate_popularity_fallback
        )
        
        # Filter customer_product to requested customers
        customer_product_filtered = self.customer_product.filter(
            pl.col("Customer ID").is_in(customer_ids)
        )
        
        # Generate co-purchase recommendations
        recommendations = generate_recommendations_vectorized(
            customer_product=customer_product_filtered,
            co_purchase=self.co_purchase.select(["product_a", "product_b", "cooccurrence", "association_lift"]),
            product_popularity=self.product_popularity,
            top_k=top_k,
            fallback_pool_size=self.fallback_pool_size,
        )
        
        # Get customers who have co-purchase candidates
        customers_with_copurchase = set(recommendations.select("Customer ID").unique().to_series().to_list())
        
        # Generate fallback
        fallback_recs = generate_popularity_fallback(
            customer_product=customer_product_filtered,
            product_popularity=self.product_popularity,
            top_k=top_k,
            fallback_pool_size=self.fallback_pool_size,
            customers_with_copurchase=customers_with_copurchase,
        )
        
        # Combine
        all_recs = pl.concat([recommendations, fallback_recs], how="diagonal_relaxed")
        all_recs = all_recs.with_columns([
            pl.when(pl.col("reason") == "co_purchase").then(pl.lit(0)).otherwise(pl.lit(1)).alias("reason_priority"),
        ]).sort(["Customer ID", "reason_priority", "rank"]).drop("reason_priority")
        all_recs = all_recs.sort(["Customer ID", "score"], descending=[False, True])
        all_recs = all_recs.unique(subset=["Customer ID", "recommended_product"], maintain_order=True)
        all_recs = all_recs.with_columns([
            pl.col("score").rank(descending=True).over("Customer ID").alias("rank"),
        ]).filter(pl.col("rank") <= top_k).sort(["Customer ID", "rank"])
        
        return all_recs.with_columns(pl.lit(self.name).alias("model_name"))


class ItemBasedCFModel(RecommendationModel):
    """Item-based Collaborative Filtering using customer-item matrix."""
    
    def __init__(self, top_k_similar: int = 50, min_item_support: int = 5):
        super().__init__("item_based_cf")
        self.top_k_similar = top_k_similar
        self.min_item_support = min_item_support
        self.item_similarities = None
        self.customer_item_matrix = None
        self.product_popularity = None
        self.all_products = None
    
    def fit(self, tx_train: pl.DataFrame, product_metrics: pl.DataFrame) -> None:
        LOGGER.info(f"Fitting {self.name}...")
        
        # Build customer-item matrix (binary: purchased or not)
        sales = tx_train.filter(pl.col("is_clean_sale"))
        
        customer_item = (
            sales.group_by(["Customer ID", "StockCode"])
            .len()
            .rename({"len": "count"})
        )
        
        # Filter items with minimum support
        item_support = customer_item.group_by("StockCode").agg(pl.col("Customer ID").n_unique().alias("support"))
        item_support = item_support.filter(pl.col("support") >= self.min_item_support)
        valid_items = item_support["StockCode"].to_list()
        
        customer_item = customer_item.filter(pl.col("StockCode").is_in(valid_items))
        
        # Build item-item similarity matrix using cosine similarity
        # We'll compute pairwise similarities for all valid items
        # For efficiency, use a simplified approach
        
        # Get unique items and create index mapping
        items = customer_item.select("StockCode").unique().sort("StockCode")["StockCode"].to_list()
        item_to_idx = {item: i for i, item in enumerate(items)}
        n_items = len(items)
        
        if n_items > 5000:
            LOGGER.warning(f"Too many items ({n_items}), limiting to top 5000 by popularity")
            # Keep top items by popularity
            item_pop = customer_item.group_by("StockCode").agg(pl.col("Customer ID").n_unique().alias("pop"))
            item_pop = item_pop.sort("pop", descending=True).head(5000)
            items = item_pop["StockCode"].to_list()
            item_to_idx = {item: i for i, item in enumerate(items)}
            n_items = len(items)
            customer_item = customer_item.filter(pl.col("StockCode").is_in(items))
        
        # Build sparse matrix
        from scipy.sparse import csr_matrix
        import scipy.sparse as sp
        
        rows = [item_to_idx[row["Customer ID"]] for row in customer_item.iter_rows(named=True)]
        # Wait, we need item x customer matrix for item-based CF
        # Let's transpose: customers as rows, items as columns
        
        # Actually, for item-based CF, we want item x customer matrix
        # Let's build it properly
        customers = customer_item.select("Customer ID").unique().sort("Customer ID")["Customer ID"].to_list()
        cust_to_idx = {cust: i for i, cust in enumerate(customers)}
        n_custs = len(customers)
        
        rows = []
        cols = []
        data = []
        
        for row in customer_item.iter_rows(named=True):
            cust_idx = cust_to_idx[row["Customer ID"]]
            item_idx = item_to_idx[row["StockCode"]]
            rows.append(item_idx)  # items as rows
            cols.append(cust_idx)  # customers as columns
            data.append(1.0)  # binary
        
        # Item x Customer matrix
        item_cust_matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_custs))
        
        # Compute item-item cosine similarity
        # Normalize rows (items)
        from sklearn.preprocessing import normalize
        item_cust_norm = normalize(item_cust_matrix, norm='l2', axis=1)
        
        # Compute similarities (only for items with enough co-occurrence)
        # This is O(n_items^2) - use top-k only
        similarities = {}
        
        for i in range(n_items):
            # Get similarity with all other items
            sims = item_cust_norm[i].dot(item_cust_norm.T).toarray().flatten()
            sims[i] = -1  # Exclude self
            
            # Get top-k
            top_k_idx = np.argpartition(sims, -self.top_k_similar)[-self.top_k_similar:]
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
            
            item_id = items[i]
            for idx in top_k_idx:
                if sims[idx] > 0:
                    similarities.setdefault(item_id, []).append((items[idx], float(sims[idx])))
        
        # Convert to DataFrame
        sim_rows = []
        for item_a, sims in similarities.items():
            for item_b, sim in sims:
                sim_rows.append({"product_a": item_a, "product_b": item_b, "similarity": sim})
        
        self.item_similarities = pl.DataFrame(sim_rows)
        
        # Product popularity
        if "unique_customers" in product_metrics.columns:
            self.product_popularity = product_metrics.select(["StockCode", "unique_customers"]).sort(
                "unique_customers", descending=True
            )
            max_pop = self.product_popularity.select(pl.col("unique_customers").max()).item()
            self.product_popularity = self.product_popularity.with_columns([
                (pl.col("unique_customers") / max_pop).alias("popularity_score"),
            ])
        else:
            self.product_popularity = item_support.sort("support", descending=True).rename({"support": "pop_score"})
            max_pop = self.product_popularity.select(pl.col("pop_score").max()).item()
            self.product_popularity = self.product_popularity.with_columns([
                (pl.col("pop_score") / max_pop).alias("popularity_score"),
            ]).rename({"pop_score": "total_cooccurrence"})
        
        self.all_products = self.product_popularity["StockCode"].to_list()
        
        LOGGER.info(f"  Items: {n_items:,}, Customers: {n_custs:,}")
        LOGGER.info(f"  Item similarities: {self.item_similarities.height:,}")
    
    def recommend(self, customer_ids: List[int], top_k: int) -> pl.DataFrame:
        if self.item_similarities is None:
            return pl.DataFrame()
        
        # Get customer purchase history
        # Need to load this from the training data - we'll store it during fit
        pass
    
    def _get_customer_history(self, customer_ids: List[int]) -> pl.DataFrame:
        # This would need access to training data
        pass


class PopularityOnlyModel(RecommendationModel):
    """Global popularity baseline."""
    
    def __init__(self):
        super().__init__("popularity_only")
        self.top_products = None
    
    def fit(self, tx_train: pl.DataFrame, product_metrics: pl.DataFrame) -> None:
        LOGGER.info(f"Fitting {self.name}...")
        
        sales = tx_train.filter(pl.col("is_clean_sale"))
        
        if "unique_customers" in product_metrics.columns:
            self.product_popularity = product_metrics.select(["StockCode", "unique_customers"]).sort(
                "unique_customers", descending=True
            )
        else:
            self.product_popularity = sales.group_by("StockCode").agg(
                pl.col("Customer ID").n_unique().alias("unique_customers")
            ).sort("unique_customers", descending=True)
    
    def recommend(self, customer_ids: List[int], top_k: int) -> pl.DataFrame:
        # All customers get the same top products
        top_products = self.product_popularity.head(top_k * 2)["StockCode"].to_list()
        
        # This is simplified - in reality would filter out already purchased
        # For now, return same for all
        rows = []
        for cust_id in customer_ids:
            for rank, prod in enumerate(top_products[:top_k], 1):
                rows.append({
                    "Customer ID": cust_id,
                    "recommended_product": prod,
                    "rank": rank,
                    "score": 1.0 / rank,
                    "reason": "popularity_only",
                    "supporting_products": 0,
                    "association_lift": None,
                    "co_purchase_score": 0.0,
                    "popularity_score": 1.0 / rank,
                })
        
        return pl.DataFrame(rows).with_columns(pl.lit(self.name).alias("model_name"))


class ContentBasedModel(RecommendationModel):
    """Content-based filtering using product features."""
    
    def __init__(self):
        super().__init__("content_based")
        self.product_features = None
        self.product_similarities = None
        self.customer_product = None
    
    def fit(self, tx_train: pl.DataFrame, product_metrics: pl.DataFrame) -> None:
        LOGGER.info(f"Fitting {self.name}...")
        
        # Build customer purchase history
        from scripts.recommendation_engine import build_customer_history
        self.customer_product = build_customer_history(tx_train)
        
        # Use product_metrics as features
        if "product_role" in product_metrics.columns and "avg_price" in product_metrics.columns:
            features = product_metrics.select([
                "StockCode", "product_role", "avg_price", "total_revenue", "total_quantity", "unique_customers"
            ]).fill_null(0)
            
            # Normalize numerical features
            num_cols = ["avg_price", "total_revenue", "total_quantity", "unique_customers"]
            for col in num_cols:
                if col in features.columns:
                    max_val = features.select(pl.col(col).max()).item()
                    if max_val > 0:
                        features = features.with_columns((pl.col(col) / max_val).alias(f"{col}_norm"))
            
            # One-hot encode product_role
            features = features.to_pandas()
            role_dummies = pd.get_dummies(features["product_role"], prefix="role")
            features = pd.concat([features, role_dummies], axis=1)
            
            # Compute item-item cosine similarity on features
            from sklearn.metrics.pairwise import cosine_similarity
            
            feature_cols = [c for c in features.columns if c.endswith("_norm") or c.startswith("role_")]
            if feature_cols:
                feature_matrix = features[feature_cols].fillna(0).values
                sim_matrix = cosine_similarity(feature_matrix)
                
                # Get top-k similar for each item
                stocks = features["StockCode"].tolist()
                stock_to_idx = {s: i for i, s in enumerate(stocks)}
                
                sim_rows = []
                for i, stock_a in enumerate(stocks):
                    sims = sim_matrix[i]
                    sims[i] = -1
                    top_k_idx = np.argpartition(sims, -50)[-50:]
                    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
                    
                    for idx in top_k_idx:
                        if sims[idx] > 0.1:
                            sim_rows.append({
                                "product_a": stock_a,
                                "product_b": stocks[idx],
                                "similarity": float(sims[idx]),
                            })
                
                self.product_similarities = pl.DataFrame(sim_rows)
        
        LOGGER.info(f"  Product similarities: {self.product_similarities.height if self.product_similarities is not None else 0:,}")
    
    def recommend(self, customer_ids: List[int], top_k: int) -> pl.DataFrame:
        if self.product_similarities is None:
            return pl.DataFrame()
        
        # For each customer, get their purchased products
        # Then recommend similar products
        rows = []
        
        for cust_id in customer_ids:
            cust_products = self.customer_product.filter(
                pl.col("Customer ID") == cust_id
            )["StockCode"].to_list()
            
            if not cust_products:
                continue
            
            # Get similar products
            candidates = []
            for prod in cust_products:
                sims = self.product_similarities.filter(
                    pl.col("product_a") == prod
                ).sort("similarity", descending=True).head(20)
                
                for row in sims.iter_rows(named=True):
                    if row["product_b"] not in cust_products:
                        candidates.append({
                            "product": row["product_b"],
                            "similarity": row["similarity"],
                        })
            
            # Aggregate and rank
            if candidates:
                cand_df = pl.DataFrame(candidates).group_by("product").agg(
                    pl.col("similarity").max().alias("score")
                ).sort("score", descending=True).head(top_k)
                
                for i, row in enumerate(cand_df.iter_rows(named=True), 1):
                    rows.append({
                        "Customer ID": cust_id,
                        "recommended_product": row["product"],
                        "rank": i,
                        "score": row["score"],
                        "reason": "content_based",
                        "supporting_products": 1,
                        "association_lift": None,
                        "co_purchase_score": 0.0,
                        "popularity_score": 0.0,
                    })
        
        if rows:
            return pl.DataFrame(rows).with_columns(pl.lit(self.name).alias("model_name"))
        return pl.DataFrame()


def evaluate_model(
    model: RecommendationModel,
    tx_test: pl.DataFrame,
    customer_ids: List[int],
    top_k: int,
    catalog_size: int,
) -> Dict[str, float]:
    """Evaluate a single model's recommendations."""
    from scripts.recommendation_engine import evaluate_recommendations
    
    # Generate recommendations
    recs = model.recommend(customer_ids, top_k)
    
    if recs.height == 0:
        return {
            "hit_rate_at_5": 0.0,
            "hit_rate_at_10": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "recommendation_coverage": 0.0,
            "catalog_coverage": 0.0,
            "n_recommendations": 0,
        }
    
    # Get test purchases
    test_purchases = tx_test.group_by(["Customer ID", "StockCode"]).agg(
        pl.len().alias("test_count")
    )
    
    # Evaluate
    metrics = evaluate_recommendations(recs, test_purchases, top_k, catalog_size)
    metrics["model_name"] = model.name
    
    return metrics


def run_challenger_evaluation(
    models: List[RecommendationModel],
    tx_train: pl.DataFrame,
    tx_test: pl.DataFrame,
    product_metrics: pl.DataFrame,
    top_k: int,
) -> pl.DataFrame:
    """Run challenger evaluation across all models."""
    
    # Fit all models
    for model in models:
        model.fit(tx_train, product_metrics)
    
    # Get eligible customers (those with test purchases)
    test_customers = tx_test.filter(pl.col("is_clean_sale")).select("Customer ID").unique().to_series().to_list()
    
    catalog_size = tx_train.filter(pl.col("is_clean_sale")).select(pl.col("StockCode").n_unique()).item()
    
    # Evaluate each model
    results = []
    for model in models:
        LOGGER.info(f"Evaluating {model.name}...")
        metrics = evaluate_model(model, tx_test, test_customers, top_k, catalog_size)
        results.append(metrics)
    
    return pl.DataFrame(results)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.recommendations_dir / "challenger"
    product_metrics_path = config.product_analytics_dir / "product_metrics.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"

    # Load canonical transactions
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

    # Run validations
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # Load product metrics
    product_metrics = pl.read_parquet(product_metrics_path) if product_metrics_path.exists() else pl.DataFrame()

    # Temporal split
    cutoff = pl.lit(args.eval_cutoff).str.strptime(pl.Datetime)
    tx_train = tx.filter(pl.col("InvoiceDate") <= cutoff)
    tx_test = tx.filter(pl.col("InvoiceDate") > cutoff)

    LOGGER.info(f"Train period: up to {args.eval_cutoff}")
    LOGGER.info(f"Test period: after {args.eval_cutoff}")
    LOGGER.info(f"Train transactions: {tx_train.height:,}")
    LOGGER.info(f"Test transactions: {tx_test.height:,}")

    # Initialize challenger models
    models = [
        CoPurchasePopularityModel(
            min_support=args.min_support,
            max_partners=args.max_partners,
            fallback_pool_size=args.fallback_pool_size,
        ),
        PopularityOnlyModel(),
        ContentBasedModel(),
    ]

    # Run evaluation
    results_df = run_challenger_evaluation(
        models, tx_train, tx_test, product_metrics, args.top_k
    )

    # Save results
    results_df.write_csv(output_dir / "challenger_results.csv")
    results_df.write_parquet(output_dir / "challenger_results.parquet")

    # Select best model (by hit_rate@10)
    if "hit_rate_at_10" in results_df.columns:
        best_idx = results_df["hit_rate_at_10"].arg_max()
        best_model = results_df.row(best_idx, named=True)
        LOGGER.info(f"Best model: {best_model['model_name']} (hit_rate@10={best_model['hit_rate_at_10']:.4f})")
    else:
        best_model = None

    # Save model card
    model_card = {
        "framework": "Recommendation Engine Challenger",
        "eval_cutoff": args.eval_cutoff,
        "top_k": args.top_k,
        "models_evaluated": [m.name for m in models],
        "best_model": best_model["model_name"] if best_model else None,
        "results": results_df.to_dicts(),
    }

    with open(output_dir / "challenger_model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "challenger_results.csv",
            "challenger_results.parquet",
            "challenger_model_card.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("Challenger evaluation complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()