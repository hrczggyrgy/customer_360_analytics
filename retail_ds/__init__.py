"""
retail_ds — Shared retail data science library for Online Retail II.

This package provides canonical implementations for:
- Data ingestion with type safety (Invoice preserved as string)
- Transaction cleaning and classification
- Customer-month panel construction
- Point-in-time feature engineering
- Data quality validation and reconciliation
- Temporal backtesting framework
"""

from retail_ds.io import load_raw_transactions, normalize_columns
from retail_ds.cleaning import clean_transactions, TransactionType
from retail_ds.transactions import (
    classify_transactions,
    compute_financial_measures,
    TransactionClassification,
)
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.features import (
    build_point_in_time_features,
    FeatureMetadata,
    FEATURE_REGISTRY,
)
from retail_ds.validation import (
    validate_schema,
    run_all_validations,
    assert_validations_pass,
    ValidationResult,
)
from retail_ds.backtesting import rolling_origin_split, TemporalSplit

__version__ = "0.1.0"

__all__ = [
    # IO
    "load_raw_transactions",
    "normalize_columns",
    # Cleaning
    "clean_transactions",
    "TransactionType",
    # Transactions
    "classify_transactions",
    "compute_financial_measures",
    "TransactionClassification",
    # Customer-month
    "build_customer_month_panel",
    # Features
    "build_point_in_time_features",
    "FeatureMetadata",
    "FEATURE_REGISTRY",
    # Validation
    "validate_schema",
    "validate_reconciliation",
    "ValidationResult",
    # Backtesting
    "rolling_origin_split",
    "TemporalSplit",
]