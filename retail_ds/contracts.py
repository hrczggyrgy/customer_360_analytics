"""
Data contracts and schema validation for the Retail Intelligence pipeline.

This module defines explicit schemas for all pipeline artifacts and provides
validation functions to ensure data quality and consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl

LOGGER = logging.getLogger("retail_ds.contracts")


@dataclass
class ContractViolation:
    """A single contract violation."""
    field: str
    expected: str
    actual: str
    severity: str = "error"  # "error", "warning", "info"
    message: str = ""


@dataclass
class ValidationReport:
    """Result of contract validation."""
    passed: bool
    violations: List[ContractViolation] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def add_violation(self, field: str, expected: str, actual: str, severity: str = "error", message: str = "") -> None:
        self.violations.append(ContractViolation(field, expected, actual, severity, message))
        if severity == "error":
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "violations": [
                {
                    "field": v.field,
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                    "message": v.message,
                }
                for v in self.violations
            ],
        }


# =============================================================================
# CANONICAL SCHEMAS
# =============================================================================

CANONICAL_TRANSACTION_SCHEMA = {
    "Invoice": pl.Utf8,
    "StockCode": pl.Utf8,
    "Description": pl.Utf8,
    "Quantity": pl.Float64,
    "InvoiceDate": pl.Datetime,
    "Price": pl.Float64,
    "Customer ID": pl.Int64,
    "Country": pl.Utf8,
    "calendar_date": pl.Date,
    "calendar_month": pl.Datetime,
    "calendar_week": pl.Datetime,
    "year": pl.Int32,
    "month": pl.Int8,
    "weekday": pl.Int8,
    "hour": pl.Int8,
    "is_positive_quantity": pl.Boolean,
    "is_negative_quantity": pl.Boolean,
    "is_positive_price": pl.Boolean,
    "is_zero_or_negative_price": pl.Boolean,
    "line_value": pl.Float64,
    "is_clean_sale": pl.Boolean,
    "return_value": pl.Float64,
    "return_units": pl.Float64,
    "transaction_type": pl.Utf8,
    "is_sale": pl.Boolean,
    "is_return": pl.Boolean,
    "is_cancellation": pl.Boolean,
    "is_discount": pl.Boolean,
    "is_postage": pl.Boolean,
    "is_fee": pl.Boolean,
    "is_voucher": pl.Boolean,
    "is_manual_adjustment": pl.Boolean,
    "is_other": pl.Boolean,
    "gross_merchandise_revenue": pl.Float64,
    "cancellation_value": pl.Float64,
    "net_merchandise_revenue": pl.Float64,
}

CUSTOMER_MONTH_SCHEMA = {
    "Customer ID": pl.Int64,
    "cohort_month": pl.Datetime,
    "calendar_month": pl.Datetime,
    "age_month": pl.Int32,
    "orders": pl.Float64,
    "gross_revenue": pl.Float64,
    "net_revenue": pl.Float64,
    "units": pl.Float64,
    "product_breadth": pl.Float64,
    "median_order_value": pl.Float64,
    "return_value": pl.Float64,
    "return_units": pl.Float64,
    "active": pl.Int8,
    "calendar_month_id": pl.Int32,
    "cohort_month_id": pl.Int32,
    "return_to_gross_ratio": pl.Float64,
    "aov": pl.Float64,
    "lifetime_orders": pl.Float64,
    "lifetime_gross_revenue": pl.Float64,
    "lifetime_net_revenue": pl.Float64,
    "lifetime_active_months": pl.Float64,
    "lifetime_return_value": pl.Float64,
    "consecutive_inactive_months": pl.Float64,
    "orders_lag_1": pl.Float64,
    "orders_lag_2": pl.Float64,
    "orders_lag_3": pl.Float64,
    "net_revenue_lag_1": pl.Float64,
    "net_revenue_lag_2": pl.Float64,
    "net_revenue_lag_3": pl.Float64,
    "orders_last_3m": pl.Float64,
    "net_revenue_last_3m": pl.Float64,
    "orders_prev_3m": pl.Float64,
    "net_revenue_prev_3m": pl.Float64,
    "net_revenue_3m_mean": pl.Float64,
    "orders_3m_mean": pl.Float64,
    "recency_months": pl.Float64,
    "lifetime_order_rate_per_month": pl.Float64,
    "lifetime_active_month_share": pl.Float64,
    "recent_value_momentum": pl.Float64,
    "recent_orders_intensity": pl.Float64,
    "historical_value_per_order": pl.Float64,
    "historical_return_ratio": pl.Float64,
    "calendar_month_number": pl.Float64,
    "month_sin": pl.Float64,
    "month_cos": pl.Float64,
    "next_active": pl.Float64,
    "next_net_revenue": pl.Float64,
    "next_calendar_month": pl.Datetime,
    "prev_month_active": pl.Float64,
    "two_months_ago_active": pl.Float64,
    "reactivation_event": pl.Int64,
    "churn_transition": pl.Int64,
}

CUSTOMER_360_CURRENT_SCHEMA = {
    "Customer ID": pl.Int64,
    "lifetime_orders": pl.Float64,
    "lifetime_gross_revenue": pl.Float64,
    "lifetime_net_revenue": pl.Float64,
    "first_purchase_date": pl.Datetime,
    "last_purchase_date": pl.Datetime,
    "tenure_days": pl.Float64,
    "recency_days": pl.Float64,
    "recency_months": pl.Float64,
    "cohort_month": pl.Datetime,
    "lifecycle_state": pl.Utf8,
    "customer_state": pl.Utf8,
    "has_reactivated": pl.Boolean,
    "active_month_count": pl.Float64,
    "active_month_density": pl.Float64,
    "reactivation_count": pl.Float64,
    "churn_transition_count": pl.Float64,
    "unique_products": pl.Float64,
    "product_revenue_hhi": pl.Float64,
    "repeat_product_ratio": pl.Float64,
    "avg_products_per_order": pl.Float64,
    "lifetime_product_line_events": pl.Float64,
    "historical_avg_order_value": pl.Float64,
    "historical_median_order_value": pl.Float64,
    "historical_order_value_std": pl.Float64,
    "order_value_cv": pl.Float64,
    "median_interpurchase_seconds": pl.Float64,
    "mean_interpurchase_seconds": pl.Float64,
    "std_interpurchase_seconds": pl.Float64,
    "interpurchase_cv": pl.Float64,
    "mean_unit_price": pl.Float64,
    "median_unit_price": pl.Float64,
    "unit_price_iqr": pl.Float64,
    "unit_price_std": pl.Float64,
    "unit_price_p90": pl.Float64,
    "unit_price_cv": pl.Float64,
    "nonpositive_price_share": pl.Float64,
    "premium_price_line_share": pl.Float64,
    "lifetime_return_value": pl.Float64,
    "lifetime_return_units": pl.Float64,
    "lifetime_return_value_rate": pl.Float64,
    "lifetime_return_unit_rate": pl.Float64,
    "hour_entropy": pl.Float64,
    "weekday_entropy": pl.Float64,
    "month_entropy": pl.Float64,
    "weekend_order_share": pl.Float64,
    "business_hour_order_share": pl.Float64,
    "avg_order_weekday": pl.Float64,
    "avg_order_hour": pl.Float64,
    "purchase_month_count": pl.Float64,
    "orders_last_1m": pl.Float64,
    "revenue_last_1m": pl.Float64,
    "net_revenue_last_1m": pl.Float64,
    "active_months_last_1m": pl.Float64,
    "return_value_last_1m": pl.Float64,
    "aov_last_1m": pl.Float64,
    "return_rate_last_1m": pl.Float64,
    "orders_last_3m": pl.Float64,
    "revenue_last_3m": pl.Float64,
    "net_revenue_last_3m": pl.Float64,
    "active_months_last_3m": pl.Float64,
    "return_value_last_3m": pl.Float64,
    "aov_last_3m": pl.Float64,
    "return_rate_last_3m": pl.Float64,
    "orders_last_6m": pl.Float64,
    "revenue_last_6m": pl.Float64,
    "net_revenue_last_6m": pl.Float64,
    "active_months_last_6m": pl.Float64,
    "return_value_last_6m": pl.Float64,
    "aov_last_6m": pl.Float64,
    "return_rate_last_6m": pl.Float64,
    "orders_last_12m": pl.Float64,
    "revenue_last_12m": pl.Float64,
    "net_revenue_last_12m": pl.Float64,
    "active_months_last_12m": pl.Float64,
    "return_value_last_12m": pl.Float64,
    "aov_last_12m": pl.Float64,
    "return_rate_last_12m": pl.Float64,
    "unique_countries": pl.Float64,
    "primary_country": pl.Utf8,
    "first_country": pl.Utf8,
    "snapshot_date": pl.Datetime,
}

# Contract registry
CONTRACTS = {
    "canonical_transactions": CANONICAL_TRANSACTION_SCHEMA,
    "customer_month": CUSTOMER_MONTH_SCHEMA,
    "customer_360_current": CUSTOMER_360_CURRENT_SCHEMA,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_schema(df: pl.DataFrame, schema_name: str) -> ValidationReport:
    """
    Validate a DataFrame against a registered schema.
    
    Args:
        df: DataFrame to validate
        schema_name: Name of the registered schema to validate against
        
    Returns:
        ValidationReport with results
    """
    if schema_name not in CONTRACTS:
        report = ValidationReport(passed=False)
        report.add_violation(
            "_schema", "registered", schema_name,
            "error", f"Unknown schema: {schema_name}"
        )
        return report

    expected_schema = CONTRACTS[schema_name]
    report = ValidationReport(passed=True, row_count=df.height, column_count=len(df.columns))

    # Check for missing columns
    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            report.add_violation(
                col, str(expected_dtype), "MISSING",
                "error", f"Required column '{col}' is missing"
            )
        else:
            actual_dtype = df.schema[col]
            if actual_dtype != expected_dtype:
                report.add_violation(
                    col, str(expected_dtype), str(actual_dtype),
                    "warning", f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
                )

    # Check for extra columns (warning only)
    extra_cols = set(df.columns) - set(expected_schema.keys())
    for col in extra_cols:
        report.add_violation(
            col, "NOT_IN_SCHEMA", str(df.schema[col]),
            "info", f"Extra column '{col}' not in schema"
        )

    return report


def validate_canonical_transactions(df: pl.DataFrame) -> ValidationReport:
    """Validate canonical transactions with business logic checks."""
    report = validate_schema(df, "canonical_transactions")
    
    if not report.passed and len(report.violations) > 0:
        # Only run business logic checks if basic schema passes
        return report

    # Invoice string preservation
    if "Invoice" in df.columns:
        cancellations = df.filter(pl.col("Invoice").str.to_uppercase().str.starts_with("C"))
        if cancellations.height == 0:
            report.add_violation(
                "Invoice", "C-prefixed cancellations present", "0 found",
                "warning", "No cancellation invoices detected - possible ingestion issue"
            )

    # Financial reconciliation
    if all(c in df.columns for c in ["is_sale", "is_return", "is_cancellation", "line_value"]):
        gross = df.filter(pl.col("is_sale")).select(pl.col("line_value").sum()).item()
        returns = df.filter(pl.col("is_return")).select(pl.col("line_value").abs().sum()).item()
        cancels = df.filter(pl.col("is_cancellation")).select(pl.col("line_value").abs().sum()).item()
        net = gross - returns - cancels

        if "gross_merchandise_revenue" in df.columns:
            stored_gross = df.select(pl.col("gross_merchandise_revenue").sum()).item()
            if abs(gross - stored_gross) > 0.01:
                report.add_violation(
                    "gross_merchandise_revenue", f"{gross:.2f}", f"{stored_gross:.2f}",
                    "error", "Gross revenue reconciliation failed"
                )

        if "net_merchandise_revenue" in df.columns:
            stored_net = df.select(pl.col("net_merchandise_revenue").sum()).item()
            if abs(net - stored_net) > 0.01:
                report.add_violation(
                    "net_merchandise_revenue", f"{net:.2f}", f"{stored_net:.2f}",
                    "error", "Net revenue reconciliation failed"
                )

    # Probability bounds
    prob_cols = [c for c in df.columns if c.endswith("_probability") or c.endswith("_prob")]
    for col in prob_cols:
        below = df.filter(pl.col(col) < 0).height
        above = df.filter(pl.col(col) > 1).height
        nulls = df.filter(pl.col(col).is_null()).height
        if below > 0 or above > 0 or nulls > 0:
            report.add_violation(
                col, "values in [0, 1]", f"below={below}, above={above}, nulls={nulls}",
                "error", f"Probability column '{col}' has out-of-bounds values"
            )

    return report


def validate_customer_month(df: pl.DataFrame) -> ValidationReport:
    """Validate customer-month panel."""
    report = validate_schema(df, "customer_month")
    
    # Additional business logic checks
    if "age_month" in df.columns:
        neg_age = df.filter(pl.col("age_month") < 0).height
        if neg_age > 0:
            report.add_violation(
                "age_month", ">= 0", f"{neg_age} negative values",
                "error", "Negative age_month values found"
            )

    if "active" in df.columns:
        invalid = df.filter(~pl.col("active").is_in([0, 1])).height
        if invalid > 0:
            report.add_violation(
                "active", "0 or 1", f"{invalid} invalid values",
                "error", "Active flag must be 0 or 1"
            )

    # Check for duplicate customer-month pairs
    dup_count = df.group_by(["Customer ID", "calendar_month"]).len().filter(pl.col("len") > 1).height
    if dup_count > 0:
        report.add_violation(
            "customer_month_grain", "unique (Customer ID, calendar_month)", f"{dup_count} duplicates",
            "error", "Duplicate customer-month pairs found"
        )

    return report


def validate_customer_360(df: pl.DataFrame) -> ValidationReport:
    """Validate customer 360 current-state features."""
    report = validate_schema(df, "customer_360_current")

    # Check for duplicate customers
    if "Customer ID" in df.columns:
        dup_count = df.group_by("Customer ID").len().filter(pl.col("len") > 1).height
        if dup_count > 0:
            report.add_violation(
                "Customer ID", "unique", f"{dup_count} duplicates",
                "error", "Duplicate Customer ID in customer_360_current"
            )

    # Probability bounds for lifecycle features
    prob_cols = [c for c in df.columns if "probability" in c.lower() or "prob" in c.lower()]
    for col in prob_cols:
        below = df.filter(pl.col(col) < 0).height
        above = df.filter(pl.col(col) > 1).height
        if below > 0 or above > 0:
            report.add_violation(
                col, "values in [0, 1]", f"below={below}, above={above}",
                "error", f"Probability column '{col}' has out-of-bounds values"
            )

    return report


def validate_artifact(df: pl.DataFrame, artifact_type: str) -> ValidationReport:
    """
    Validate a DataFrame based on its artifact type.
    
    Args:
        df: DataFrame to validate
        artifact_type: Type of artifact (canonical_transactions, customer_month, customer_360_current)
        
    Returns:
        ValidationReport
    """
    validators = {
        "canonical_transactions": validate_canonical_transactions,
        "customer_month": validate_customer_month,
        "customer_360_current": validate_customer_360,
    }
    
    if artifact_type not in validators:
        report = ValidationReport(passed=False)
        report.add_violation("_type", "known", artifact_type, "error", f"Unknown artifact type: {artifact_type}")
        return report
    
    return validators[artifact_type](df)


# =============================================================================
# DUPLICATE DIAGNOSTICS
# =============================================================================

def analyze_duplicates(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive duplicate analysis for a DataFrame.
    
    Returns a dictionary with:
    - exact_duplicates: fully identical rows
    - key_duplicates: duplicates on primary key columns
    - cross_source_duplicates: potential cross-source contamination
    """
    results = {
        "total_rows": df.height,
        "unique_rows": df.unique().height,
        "exact_duplicate_count": df.height - df.unique().height,
        "exact_duplicate_pct": (df.height - df.unique().height) / df.height * 100,
    }

    # Key-based duplicates (if Customer ID exists)
    if "Customer ID" in df.columns:
        cust_dups = df.group_by("Customer ID").len().filter(pl.col("len") > 1)
        results["customer_id_duplicates"] = cust_dups.height
        results["customer_id_duplicate_rows"] = cust_dups.select(pl.col("len").sum()).item()

    # Invoice-based duplicates
    if "Invoice" in df.columns:
        inv_dups = df.group_by("Invoice").len().filter(pl.col("len") > 1)
        results["invoice_duplicates"] = inv_dups.height
        results["invoice_duplicate_rows"] = inv_dups.select(pl.col("len").sum()).item()

    # Invoice + StockCode duplicates (line-level)
    if "Invoice" in df.columns and "StockCode" in df.columns:
        line_dups = df.group_by(["Invoice", "StockCode"]).len().filter(pl.col("len") > 1)
        results["invoice_stockcode_duplicates"] = line_dups.height
        results["invoice_stockcode_duplicate_rows"] = line_dups.select(pl.col("len").sum()).item()

    return results


def log_validation_report(report: ValidationReport, context: str = "") -> None:
    """Log validation report with context."""
    prefix = f"[{context}] " if context else ""
    if report.passed:
        LOGGER.info(f"{prefix}Validation PASSED: {report.row_count:,} rows, {report.column_count} columns")
    else:
        error_count = sum(1 for v in report.violations if v.severity == "error")
        warning_count = sum(1 for v in report.violations if v.severity == "warning")
        LOGGER.error(f"{prefix}Validation FAILED: {error_count} errors, {warning_count} warnings")
        for v in report.violations:
            if v.severity == "error":
                LOGGER.error(f"  ✗ {v.field}: {v.message} (expected: {v.expected}, actual: {v.actual})")
            elif v.severity == "warning":
                LOGGER.warning(f"  ⚠ {v.field}: {v.message} (expected: {v.expected}, actual: {v.actual})")