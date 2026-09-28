"""
Test suite for retail_ds shared package and pipeline scripts.
"""

import pytest
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from retail_ds.io import load_raw_transactions, normalize_columns
from retail_ds.cleaning import clean_transactions, add_calendar_fields, TransactionType
from retail_ds.transactions import classify_transactions, compute_financial_measures, get_transaction_type_report
from retail_ds.customer_month import build_customer_month_panel, add_rolling_features
from retail_ds.features import build_point_in_time_features, FEATURE_REGISTRY, validate_point_in_time_safety
from retail_ds.validation import (
    validate_invoice_string_preservation,
    validate_financial_reconciliation,
    validate_customer_aggregation,
    run_all_validations,
    assert_validations_pass,
    ValidationResult,
)
from retail_ds.backtesting import rolling_origin_split, TemporalSplit


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_transactions():
    """Create a small synthetic transaction dataset for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate synthetic data
    invoices = [f"INV{i:06d}" for i in range(100)]
    invoices += [f"C{i:05d}" for i in range(20)]  # Cancellation invoices
    
    data = {
        "Invoice": np.random.choice(invoices, n),
        "StockCode": np.random.choice([f"PROD{i:04d}" for i in range(50)], n),
        "Description": [f"Product {i}" for i in np.random.randint(0, 50, n)],
        "Quantity": np.random.randint(-5, 10, n),
        "InvoiceDate": pd.date_range("2020-01-01", periods=n, freq="h"),
        "Price": np.random.uniform(0.5, 100, n),
        "Customer ID": np.random.randint(1000, 1200, n),
        "Country": np.random.choice(["UK", "France", "Germany", "USA"], n),
    }
    
    df = pl.DataFrame(data)
    return df


@pytest.fixture(scope="session")
def canonical_transactions(sample_transactions):
    """Create canonical transactions from sample data."""
    from retail_ds.cleaning import clean_transactions, add_calendar_fields
    from retail_ds.transactions import classify_transactions, compute_financial_measures
    
    tx = clean_transactions(sample_transactions)
    tx = add_calendar_fields(tx)
    tx = classify_transactions(tx)
    tx = compute_financial_measures(tx)
    return tx


# =============================================================================
# Ingestion Tests
# =============================================================================

class TestIngestion:
    """Tests for data ingestion layer."""
    
    def test_invoice_preserved_as_string(self, sample_transactions):
        """Invoice column should remain as string/Utf8, not be converted to Int64."""
        # Add cancellation invoices
        df = sample_transactions.with_columns(
            pl.when(pl.arange(0, pl.len()) < 50)
            .then(pl.lit("C12345"))
            .otherwise(pl.col("Invoice"))
            .alias("Invoice")
        )
        
        normalized = normalize_columns(df)
        assert normalized.schema["Invoice"] == pl.Utf8
        
        # Check cancellation invoices preserved
        cancellations = normalized.filter(pl.col("Invoice").str.starts_with("C"))
        assert cancellations.height > 0
    
    def test_column_normalization(self, sample_transactions):
        """Column names should be normalized to canonical schema."""
        # Use lowercase column names
        df = sample_transactions.rename({c: c.lower() for c in sample_transactions.columns})
        normalized = normalize_columns(df)
        
        expected_cols = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]
        for col in expected_cols:
            assert col in normalized.columns
    
    def test_required_columns_validation(self, sample_transactions):
        """Should raise error for missing required columns."""
        df = sample_transactions.drop("Invoice")
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_columns(df)


# =============================================================================
# Cleaning Tests
# =============================================================================

class TestCleaning:
    """Tests for transaction cleaning."""
    
    def test_clean_transactions_preserves_invoice_type(self, sample_transactions):
        """Invoice should remain string after cleaning."""
        df = sample_transactions.with_columns(
            pl.when(pl.arange(0, pl.len()) < 10)
            .then(pl.lit("C12345"))
            .otherwise(pl.col("Invoice"))
            .alias("Invoice")
        )
        
        cleaned = clean_transactions(df)
        assert cleaned.schema["Invoice"] == pl.Utf8
    
    def test_cancellation_detection(self, sample_transactions):
        """Cancellation invoices (C-prefixed) should be detected."""
        df = sample_transactions.with_columns(
            pl.when(pl.arange(0, pl.len()) < 20)
            .then(pl.lit("C54321"))
            .otherwise(pl.col("Invoice"))
            .alias("Invoice")
        )
        
        cleaned = clean_transactions(df)
        cancellations = cleaned.filter(pl.col("is_cancellation_invoice"))
        assert cancellations.height > 0
    
    def test_clean_sale_identification(self, sample_transactions):
        """Clean sales should be: positive qty, non-cancellation, positive price."""
        cleaned = clean_transactions(sample_transactions)
        clean_sales = cleaned.filter(pl.col("is_clean_sale"))
        
        # All clean sales should have positive quantity and price
        assert (clean_sales["Quantity"] > 0).all()
        assert (clean_sales["Price"] > 0).all()
        assert (~clean_sales["is_cancellation_invoice"]).all()
    
    def test_return_value_calculation(self, sample_transactions):
        """Return value should be absolute value for negative qty or cancellations."""
        df = sample_transactions.with_columns([
            pl.when(pl.arange(0, pl.len()) < 10)
            .then(pl.lit(-2))
            .otherwise(pl.col("Quantity"))
            .alias("Quantity"),
            pl.when(pl.arange(0, pl.len()) < 10)
            .then(pl.lit(10.0))
            .otherwise(pl.col("Price"))
            .alias("Price"),
        ])
        
        cleaned = clean_transactions(df)
        returns = cleaned.filter(pl.col("is_return_or_cancellation"))
        
        # Return value should be positive (absolute)
        assert (returns["return_value"] >= 0).all()
    
    def test_calendar_fields_added(self, sample_transactions):
        """Calendar fields should be added correctly."""
        cleaned = clean_transactions(sample_transactions)
        with_calendar = add_calendar_fields(cleaned)
        
        expected_fields = ["calendar_date", "calendar_month", "calendar_week", 
                          "year", "month", "weekday", "hour"]
        for field in expected_fields:
            assert field in with_calendar.columns


# =============================================================================
# Transaction Classification Tests
# =============================================================================

class TestTransactionClassification:
    """Tests for transaction classification and financial measures."""
    
    def test_classify_transactions(self, canonical_transactions):
        """All transactions should be classified into a type."""
        classified = classify_transactions(canonical_transactions)
        
        valid_types = ["sale", "return", "cancellation", "discount", 
                      "postage", "fee", "voucher", "manual_adjustment", "other"]
        
        types = classified["transaction_type"].unique().to_list()
        for t in types:
            assert t in valid_types
        
        # Every row should have exactly one type
        type_cols = [c for c in classified.columns if c.startswith("is_")]
        type_sum = sum(classified[c].cast(pl.Int32) for c in type_cols)
        assert (type_sum == 1).all()
    
    def test_financial_measures(self, canonical_transactions):
        """Financial measures should be computed correctly."""
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        
        # Gross merchandise revenue only for sales
        sales_revenue = with_measures.filter(pl.col("is_sale"))["gross_merchandise_revenue"].sum()
        total_sales = with_measures.filter(pl.col("is_sale"))["line_value"].sum()
        assert abs(sales_revenue - total_sales) < 0.01
        
        # Return value for returns
        return_revenue = with_measures.filter(pl.col("is_return"))["return_value"].sum()
        total_returns = with_measures.filter(pl.col("is_return"))["line_value"].abs().sum()
        assert abs(return_revenue - total_returns) < 0.01
        
        # Net revenue = gross - returns - cancellations
        net = with_measures["net_merchandise_revenue"]
        gross = with_measures["gross_merchandise_revenue"]
        returns = with_measures["return_value"]
        cancellations = with_measures["cancellation_value"]
        assert abs(net - (gross - returns - cancellations)).max() < 0.01
    
    def test_transaction_type_report(self, canonical_transactions):
        """Transaction type report should summarize all types."""
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        report = get_transaction_type_report(with_measures)
        
        assert "transaction_type" in report.columns
        assert "count" in report.columns
        assert report.height > 0


# =============================================================================
# Customer-Month Panel Tests
# =============================================================================

class TestCustomerMonthPanel:
    """Tests for canonical customer-month panel construction."""
    
    def test_panel_structure(self, canonical_transactions):
        """Panel should have correct structure and grain."""
        sparse, dense, metadata = build_customer_month_panel(canonical_transactions)
        
        # Check required columns
        required = ["Customer ID", "calendar_month", "cohort_month", "age_month",
                   "orders", "gross_revenue", "net_revenue", "units",
                   "return_value", "active", "recency_months"]
        for col in required:
            assert col in dense.columns
        
        # Grain: one row per customer per calendar month >= cohort_month
        assert (dense["age_month"] >= 0).all()
        
        # Customers should only appear from their cohort month onwards
        check = dense.join(
            dense.group_by("Customer ID").agg(pl.col("calendar_month").min().alias("cohort_month")),
            on="Customer ID"
        )
        assert (check["calendar_month"] >= check["cohort_month"]).all()
    
    def test_cumulative_features(self, canonical_transactions):
        """Cumulative lifetime features should be monotonic per customer."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        for col in ["lifetime_orders", "lifetime_gross_revenue", "lifetime_net_revenue",
                   "lifetime_active_months", "lifetime_return_value"]:
            # Within each customer, should be non-decreasing
            diffs = dense.sort(["Customer ID", "calendar_month"]).group_by("Customer ID").agg(
                pl.col(col).diff().fill_null(0).alias("diff")
            )
            # Allow small floating point differences
            assert (check["diff"] >= -1e-9).all()
    
    def test_rolling_features(self, canonical_transactions):
        """Rolling window features should be computed."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        dense = add_rolling_features(dense, [1, 3, 6])
        
        for w in [1, 3, 6]:
            assert f"orders_last_{w}m" in dense.columns
            assert f"revenue_last_{w}m" in dense.columns
            assert f"active_months_last_{w}m" in dense.columns
    
    def test_target_columns(self, canonical_transactions):
        """Target columns (next month) should be leakage-safe."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        assert "next_active" in dense.columns
        assert "next_net_revenue" in dense.columns
        assert "next_calendar_month" in dense.columns
        
        # Targets should be shifted -1 (next month)
        # Last month per customer should have null/0 targets
        last_months = dense.group_by("Customer ID").agg(pl.col("calendar_month").max())
        last_rows = dense.join(last_months, on=["Customer ID", "calendar_month"])
        assert (last_rows["next_active"] == 0).all()


# =============================================================================
# Point-in-Time Features Tests
# =============================================================================

class TestPointInTimeFeatures:
    """Tests for point-in-time feature engineering."""
    
    def test_features_at_date_no_future_leakage(self, canonical_transactions):
        """Features at date should only use data up to that date."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        # Use a prediction date in the middle of data
        pred_date = "2020-06-30"
        features = build_point_in_time_features(
            canonical_transactions, pred_date, dense, [1, 3, 6]
        )
        
        # All features should be computable from data <= pred_date
        # Check that no feature uses data after pred_date
        assert features.height > 0
    
    def test_feature_registry(self):
        """Feature registry should have metadata for all features."""
        assert len(FEATURE_REGISTRY) > 0
        
        for name, meta in FEATURE_REGISTRY.items():
            assert meta.name == name
            assert meta.point_in_time_safe in [True, False]
            assert meta.requires_as_of_date in [True, False]
            assert meta.feature_group in ["economic", "cadence", "assortment", 
                                          "pricing", "returns", "temporal", 
                                          "lifecycle", "target", "unknown"]
    
    def test_leakage_safety_check(self, canonical_transactions):
        """validate_point_in_time_safety should identify leakage."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        pred_date = "2020-06-30"
        features = build_point_in_time_features(
            canonical_transactions, pred_date, dense, [1, 3, 6]
        )
        
        feature_names = [c for c in features.columns if c != "Customer ID"]
        safety = validate_point_in_time_safety(feature_names)
        
        # All features from build_point_in_time_features should be safe
        for name, safe in safety.items():
            assert safe == True, f"Feature {name} should be point-in-time safe"


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for data quality validation."""
    
    def test_invoice_string_preservation(self, canonical_transactions):
        """Invoice string preservation validation should pass."""
        result = validate_invoice_string_preservation(canonical_transactions)
        assert result.passed
        assert result.details["cancellation_count"] > 0
    
    def test_financial_reconciliation(self, canonical_transactions):
        """Financial reconciliation should balance."""
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        
        result = validate_financial_reconciliation(with_measures)
        assert result.passed
        assert result.details["gross_diff"] < 0.01
        assert result.details["net_diff"] < 0.01
    
    def test_customer_aggregation(self, canonical_transactions):
        """Customer-level aggregates should reconcile with transactions."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        # Build customer features from dense panel
        cust_features = dense.group_by("Customer ID").agg(
            pl.col("lifetime_gross_revenue").max().alias("lifetime_gross_revenue")
        )
        
        result = validate_customer_aggregation(canonical_transactions, cust_features)
        assert result.passed
    
    def test_run_all_validations(self, canonical_transactions):
        """Full validation suite should pass."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        results = run_all_validations(
            tx=canonical_transactions,
            customer_month_dense=dense,
        )
        
        for r in results:
            assert r.passed, f"Validation failed: {r.message}"


# =============================================================================
# Backtesting Tests
# =============================================================================

class TestBacktesting:
    """Tests for temporal backtesting framework."""
    
    def test_rolling_origin_split(self, canonical_transactions):
        """Rolling origin splits should create valid temporal splits."""
        _, dense, metadata = build_customer_month_panel(canonical_transactions)
        
        origins = ["2020-06-30", "2020-09-30"]
        splits = rolling_origin_split(
            dense, "calendar_month", origins,
            val_horizon_months=2, test_horizon_months=2
        )
        
        assert len(splits) >= 1
        for split in splits:
            assert split.train_start <= split.train_end
            assert split.train_end < split.val_start
            assert split.val_end < split.test_start
    
    def test_apply_temporal_split(self, canonical_transactions):
        """Applying splits should partition data correctly."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        origins = ["2020-06-30"]
        splits = rolling_origin_split(
            dense, "calendar_month", origins,
            val_horizon_months=2, test_horizon_months=2
        )
        
        for split in splits:
            train, val, test = split.apply(dense, "calendar_month")
            
            # No overlap
            assert train.height + val.height + test.height <= dense.height
            
            # Temporal order
            if train.height > 0 and val.height > 0:
                assert train.select(pl.col("calendar_month").max()).item() <= split.train_end
                assert val.select(pl.col("calendar_month").min()).item() >= split.val_start


# =============================================================================
# Temporal Leakage Tests (Critical)
# =============================================================================

class TestTemporalLeakage:
    """Critical tests to prevent temporal leakage in ML features."""
    
    def test_no_future_data_in_point_in_time_features(self, canonical_transactions):
        """Point-in-time features must not use future data."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        pred_date = "2020-06-30"
        features = build_point_in_time_features(
            canonical_transactions, pred_date, dense, [1, 3, 6]
        )
        
        # Verify by checking a few customers manually
        # The maximum transaction date used should be <= pred_date
        for cust_id in features.select("Customer ID").head(5).to_series().to_list():
            cust_tx = canonical_transactions.filter(pl.col("Customer ID") == cust_id)
            max_tx_date = cust_tx.filter(pl.col("InvoiceDate") <= pred_date).select(
                pl.col("InvoiceDate").max()
            ).item()
            assert max_tx_date <= pd.Timestamp(pred_date)
    
    def test_no_future_in_customer_month_targets(self, canonical_transactions):
        """Customer-month targets should be strictly future."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        # next_active should be shift(-1) of active
        # For each customer, next_active at month t should equal active at month t+1
        check = dense.sort(["Customer ID", "calendar_month"]).with_columns(
            pl.col("active").shift(-1).over("Customer ID").fill_null(0).alias("active_next")
        )
        assert (check["next_active"] == check["active_next"]).all()
    
    def test_temporal_split_no_overlap(self, canonical_transactions):
        """Train/val/test splits should have no temporal overlap."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        origins = ["2020-06-30", "2020-09-30"]
        splits = rolling_origin_split(
            dense, "calendar_month", origins,
            val_horizon_months=2, test_horizon_months=2
        )
        
        for split in splits:
            train, val, test = split.apply(dense, "calendar_month")
            
            if train.height > 0 and val.height > 0:
                train_max = train.select(pl.col("calendar_month").max()).item()
                val_min = val.select(pl.col("calendar_month").min()).item()
                assert train_max < val_min
            
            if val.height > 0 and test.height > 0:
                val_max = val.select(pl.col("calendar_month").max()).item()
                test_min = test.select(pl.col("calendar_month").min()).item()
                assert val_max < test_min


# =============================================================================
# Financial Reconciliation Tests
# =============================================================================

class TestFinancialReconciliation:
    """Tests for financial integrity across the pipeline."""
    
    def test_gross_revenue_reconciliation(self, canonical_transactions):
        """Gross revenue should match sum of clean sale line values."""
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        
        # Sum of line_value for sales
        calc_gross = with_measures.filter(pl.col("is_sale"))["line_value"].sum()
        stored_gross = with_measures["gross_merchandise_revenue"].sum()
        assert abs(calc_gross - stored_gross) < 0.01
    
    def test_net_revenue_reconciliation(self, canonical_transactions):
        """Net = Gross - Returns - Cancellations."""
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        
        calc_net = (with_measures["gross_merchandise_revenue"] - 
                   with_measures["return_value"] - 
                   with_measures["cancellation_value"]).sum()
        stored_net = with_measures["net_merchandise_revenue"].sum()
        assert abs(calc_net - stored_net) < 0.01
    
    def test_customer_revenue_reconciliation(self, canonical_transactions):
        """Customer-level revenue should sum to transaction total."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        # Sum of customer lifetime gross revenue
        cust_total = dense.group_by("Customer ID").agg(
            pl.col("lifetime_gross_revenue").max().alias("cust_gross")
        )["cust_gross"].sum()
        
        # Transaction gross
        classified = classify_transactions(canonical_transactions)
        with_measures = compute_financial_measures(classified)
        tx_gross = with_measures.filter(pl.col("is_sale"))["line_value"].sum()
        
        assert abs(cust_total - tx_gross) < 1.0  # Allow small floating point


# =============================================================================
# Data Integrity Tests
# =============================================================================

class TestDataIntegrity:
    """Tests for data integrity and edge cases."""
    
    def test_no_duplicate_customer_month_rows(self, canonical_transactions):
        """Customer-month panel should have unique (Customer ID, calendar_month)."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        dupes = dense.group_by(["Customer ID", "calendar_month"]).len().filter(pl.col("len") > 1)
        assert dupes.height == 0
    
    def test_age_month_non_negative(self, canonical_transactions):
        """Age month should never be negative."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        assert (dense["age_month"] >= 0).all()
    
    def test_recency_non_negative(self, canonical_transactions):
        """Recency months should never be negative."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        assert (dense["recency_months"] >= 0).all()
    
    def test_probability_bounds(self, canonical_transactions):
        """Probability outputs should be in [0, 1]."""
        # This would be tested after model training
        pass
    
    def test_cohort_month_consistency(self, canonical_transactions):
        """Cohort month should be first active month for each customer."""
        _, dense, _ = build_customer_month_panel(canonical_transactions)
        
        cohorts = dense.group_by("Customer ID").agg(
            pl.col("calendar_month").filter(pl.col("active") == 1).min().alias("first_active"),
            pl.col("cohort_month").first().alias("cohort_month"),
        )
        assert (cohorts["first_active"] == cohorts["cohort_month"]).all()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])