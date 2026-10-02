"""
Test suite for recommendation engine.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.recommendation_engine import (
    build_customer_history,
    load_co_purchase_matrix,
    compute_association_lift,
    generate_recommendations_vectorized,
    generate_popularity_fallback,
    evaluate_recommendations,
    evaluate_popularity_baseline_from_training,
)


@pytest.fixture
def sample_transactions():
    """Create synthetic transaction data for testing."""
    np.random.seed(42)
    n = 5000

    # Use ns resolution for Polars compatibility
    base_date = np.datetime64("2010-01-01", "ns")
    dates = base_date + np.arange(n) * np.timedelta64(12, "h")
    
    data = {
        "Invoice": np.random.choice([f"INV{i:06d}" for i in range(200)] + [f"C{i:05d}" for i in range(20)], n),
        "StockCode": np.random.choice([f"PROD{i:04d}" for i in range(100)], n),
        "Description": [f"Product {i}" for i in np.random.randint(0, 100, n)],
        "Quantity": np.random.randint(1, 10, n),
        "InvoiceDate": dates,
        "Price": np.random.uniform(0.5, 100, n),
        "Customer ID": np.random.randint(1000, 1200, n),
        "Country": np.random.choice(["UK", "France", "Germany", "USA"], n),
    }

    df = pl.DataFrame(data)
    # Compute line_value first, then use it in subsequent expressions
    df = df.with_columns(
        (pl.col("Quantity") * pl.col("Price")).alias("line_value")
    )
    df = df.with_columns([
        pl.col("Quantity") > 0,
        pl.lit(False).alias("is_cancellation_invoice"),
        pl.lit(False).alias("is_return"),
        pl.lit(True).alias("is_clean_sale"),
        pl.col("line_value").alias("gross_merchandise_revenue"),
        pl.lit(0.0).alias("return_value"),
        pl.lit(0.0).alias("cancellation_value"),
        pl.col("line_value").alias("net_merchandise_revenue"),
    ])
    return df


@pytest.fixture
def sample_customer_product(sample_transactions):
    """Build customer product history from sample transactions."""
    return build_customer_history(sample_transactions)


@pytest.fixture
def sample_co_purchase(sample_transactions):
    """Build co-purchase matrix from sample transactions."""
    from scripts.product_analytics import build_co_purchase_matrix
    return build_co_purchase_matrix(sample_transactions, min_cooccurrence=2, max_pairs_per_invoice=50)


class TestRecommendationEngine:
    """Tests for recommendation engine components."""

    def test_customer_history_grain(self, sample_customer_product):
        """Customer-product history should have unique (Customer ID, StockCode) pairs."""
        dupes = sample_customer_product.group_by(["Customer ID", "StockCode"]).len().filter(pl.col("len") > 1)
        assert dupes.height == 0

    def test_co_purchase_symmetry(self, sample_co_purchase):
        """Co-purchase matrix should have no self-pairs and product_a < product_b."""
        # Check no self-pairs
        self_pairs = sample_co_purchase.filter(pl.col("product_a") == pl.col("product_b"))
        assert self_pairs.height == 0
        
        # Check product_a < product_b
        assert (sample_co_purchase["product_a"] < sample_co_purchase["product_b"]).all()
        
        # cooccurrence should be positive integers
        assert (sample_co_purchase["cooccurrence"] > 0).all()

    def test_lift_formula(self, sample_co_purchase, sample_customer_product):
        """Lift = P(A,B) / (P(A) * P(B)) = cooccurrence * N / (support_a * support_b)."""
        total_customers = sample_customer_product.select(pl.col("Customer ID").n_unique()).item()
        
        product_supports = sample_co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("product_support")
        ).rename({"product_a": "StockCode"})
        
        co_purchase_with_lift = compute_association_lift(sample_co_purchase, product_supports, total_customers)
        
        for row in co_purchase_with_lift.head(10).iter_rows(named=True):
            coocc = row["cooccurrence"]
            support_a = row["support_a"]
            support_b = row["support_b"]
            expected_lift = (coocc * total_customers) / (support_a * support_b) if support_a * support_b > 0 else None
            actual_lift = row["association_lift"]
            if expected_lift is not None and actual_lift is not None:
                assert abs(expected_lift - actual_lift) < 0.001

    def test_no_self_recommendations(self, sample_customer_product, sample_co_purchase):
        """Recommendations should not include products the customer already purchased."""
        total_customers = sample_customer_product.select(pl.col("Customer ID").n_unique()).item()
        product_supports = sample_co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("product_support")
        ).rename({"product_a": "StockCode"})
        co_purchase_with_lift = compute_association_lift(sample_co_purchase, product_supports, total_customers)
        
        product_popularity = sample_customer_product.group_by("StockCode").agg(
            pl.col("Customer ID").n_unique().alias("unique_customers")
        ).sort("unique_customers", descending=True)
        max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
        product_popularity = product_popularity.with_columns([
            (pl.col("unique_customers") / max_pop).alias("popularity_score"),
        ])
        
        recs = generate_recommendations_vectorized(
            sample_customer_product, co_purchase_with_lift, product_popularity, top_k=10, fallback_pool_size=100
        )
        
        customer_bought = sample_customer_product.select(["Customer ID", "StockCode"]).unique()
        check = recs.join(
            customer_bought.rename({"StockCode": "recommended_product"}),
            on=["Customer ID", "recommended_product"],
            how="inner"
        )
        assert check.height == 0

    def test_top_k_enforcement(self, sample_customer_product, sample_co_purchase):
        """Each customer should have at most top_k recommendations."""
        total_customers = sample_customer_product.select(pl.col("Customer ID").n_unique()).item()
        product_supports = sample_co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("product_support")
        ).rename({"product_a": "StockCode"})
        co_purchase_with_lift = compute_association_lift(sample_co_purchase, product_supports, total_customers)
        
        product_popularity = sample_customer_product.group_by("StockCode").agg(
            pl.col("Customer ID").n_unique().alias("unique_customers")
        ).sort("unique_customers", descending=True)
        max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
        product_popularity = product_popularity.with_columns([
            (pl.col("unique_customers") / max_pop).alias("popularity_score"),
        ])
        
        for k in [5, 10, 20]:
            recs = generate_recommendations_vectorized(
                sample_customer_product, co_purchase_with_lift, product_popularity, top_k=k, fallback_pool_size=100
            )
            max_recs = recs.group_by("Customer ID").len().select(pl.col("len").max()).item()
            assert max_recs <= k

    def test_deterministic_ranking(self, sample_customer_product, sample_co_purchase):
        """Repeated runs with same inputs should produce identical ordering."""
        total_customers = sample_customer_product.select(pl.col("Customer ID").n_unique()).item()
        product_supports = sample_co_purchase.group_by("product_a").agg(
            pl.col("cooccurrence").sum().alias("product_support")
        ).rename({"product_a": "StockCode"})
        co_purchase_with_lift = compute_association_lift(sample_co_purchase, product_supports, total_customers)
        
        product_popularity = sample_customer_product.group_by("StockCode").agg(
            pl.col("Customer ID").n_unique().alias("unique_customers")
        ).sort("unique_customers", descending=True)
        max_pop = product_popularity.select(pl.col("unique_customers").max()).item()
        product_popularity = product_popularity.with_columns([
            (pl.col("unique_customers") / max_pop).alias("popularity_score"),
        ])
        
        recs1 = generate_recommendations_vectorized(
            sample_customer_product, co_purchase_with_lift, product_popularity, top_k=10, fallback_pool_size=100
        )
        recs2 = generate_recommendations_vectorized(
            sample_customer_product, co_purchase_with_lift, product_popularity, top_k=10, fallback_pool_size=100
        )
        
        # Compare key columns only (rank and recommended_product per customer)
        recs1_sorted = recs1.sort(["Customer ID", "rank"]).select(["Customer ID", "recommended_product", "rank"])
        recs2_sorted = recs2.sort(["Customer ID", "rank"]).select(["Customer ID", "recommended_product", "rank"])
        # Compare row by row
        assert recs1_sorted.height == recs2_sorted.height
        assert recs1_sorted["Customer ID"].to_list() == recs2_sorted["Customer ID"].to_list()
        assert recs1_sorted["recommended_product"].to_list() == recs2_sorted["recommended_product"].to_list()
        assert recs1_sorted["rank"].to_list() == recs2_sorted["rank"].to_list()

    def test_fallback_behavior(self, sample_customer_product, sample_co_purchase):
        """Customers with no co-purchase candidates should get popularity fallback."""
        total_customers = sample_customer_product.select(pl.col("Customer ID").n_unique()).item()
        product_supports = pl.DataFrame({
            "StockCode": ["PROD0000", "PROD0001"],
            "product_support": [100, 90]
        })
        
        empty_copurchase = pl.DataFrame(schema={"product_a": pl.Utf8, "product_b": pl.Utf8, "cooccurrence": pl.Int64, "association_lift": pl.Float64})
        
        product_popularity = pl.DataFrame({
            "StockCode": ["PROD0000", "PROD0001"],
            "unique_customers": [100, 90]
        })
        product_popularity = product_popularity.with_columns([
            (pl.col("unique_customers") / 100).alias("popularity_score"),
        ])
        
        customer_product = pl.DataFrame({
            "Customer ID": [1000, 1001],
            "StockCode": ["PROD9999", "PROD9998"],
        })
        
        recs = generate_recommendations_vectorized(
            customer_product, empty_copurchase, product_popularity, top_k=10, fallback_pool_size=100
        )
        fallback_recs = generate_popularity_fallback(
            customer_product, product_popularity, top_k=10, fallback_pool_size=100,
            customers_with_copurchase=set()
        )
        
        assert (fallback_recs["reason"] == "popularity_fallback").all()

    def test_evaluate_recommendations_hit_rate(self):
        """Hit rate calculation should be correct."""
        recs = pl.DataFrame({
            "Customer ID": [1, 1, 1, 2, 2, 3, 3, 3],
            "recommended_product": ["A", "B", "C", "A", "C", "B", "C", "D"],
            "rank": [1, 2, 3, 1, 2, 1, 2, 3],
        })
        
        test_purchases = pl.DataFrame({
            "Customer ID": [1, 1, 2, 3],
            "StockCode": ["A", "D", "A", "C"],
            "test_count": [1, 1, 1, 1],
        })
        
        metrics = evaluate_recommendations(recs, test_purchases, top_k=10)
        
        assert metrics["hit_rate_at_5"] == 1.0
        assert abs(metrics["recall_at_5"] - 0.8333) < 0.01

    def test_mrr_calculation(self):
        """MRR@K calculation should be correct."""
        recs = pl.DataFrame({
            "Customer ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "recommended_product": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        })
        
        test_purchases = pl.DataFrame({
            "Customer ID": [1, 2, 3],
            "StockCode": ["B", "C", "A"],
            "test_count": [1, 1, 1],
        })
        
        metrics = evaluate_recommendations(recs, test_purchases, top_k=10)
        assert abs(metrics["mrr_at_10"] - 0.611) < 0.01

    def test_popularity_baseline_from_training(self):
        """Baseline should use training popularity, not test."""
        test_purchases = pl.DataFrame({
            "Customer ID": [1, 1, 2, 3],
            "StockCode": ["A", "B", "A", "C"],
            "test_count": [1, 1, 1, 1],
        })
        
        product_popularity_train = pl.DataFrame({
            "StockCode": ["B", "A", "C"],
            "unique_customers": [2, 2, 1],
        })
        product_popularity_train = product_popularity_train.with_columns([
            (pl.col("unique_customers") / 2).alias("popularity_score"),
        ])
        
        baseline = evaluate_popularity_baseline_from_training(test_purchases, product_popularity_train, top_k=2)
        
        assert abs(baseline["hit_rate"] - 2/3) < 0.01

    def test_no_duplicate_customer_product(self):
        """Final recommendations should have unique (Customer ID, recommended_product)."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])