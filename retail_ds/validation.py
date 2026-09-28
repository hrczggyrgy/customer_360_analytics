"""
Data quality validation and reconciliation gates.

This module provides reusable validation functions that can be run at any
stage of the pipeline to ensure data integrity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import polars as pl

LOGGER = logging.getLogger("retail_ds.validation")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error", "warning", "info"

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} [{self.severity.upper()}] {self.check_name}: {self.message}"


def validate_schema(df: pl.DataFrame, expected_schema: Dict[str, pl.DataType], check_name: str = "schema") -> ValidationResult:
    """Validate DataFrame schema matches expected types."""
    missing = [c for c in expected_schema if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_schema]
    mismatched = []

    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = df.schema[col]
            if actual_dtype != expected_dtype:
                mismatched.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")

    passed = len(missing) == 0 and len(mismatched) == 0
    details = {"missing_columns": missing, "extra_columns": extra, "mismatched": mismatched}

    if not passed:
        msg = f"Schema validation failed. Missing: {missing}, Mismatched: {mismatched}"
    else:
        msg = f"Schema valid. Columns: {len(df.columns)}, Rows: {df.height:,}"

    return ValidationResult(check_name, passed, msg, details, "error" if not passed else "info")


def validate_invoice_string_preservation(df: pl.DataFrame, check_name: str = "invoice_string") -> ValidationResult:
    """Validate that Invoice column is string and cancellations are preserved."""
    if "Invoice" not in df.columns:
        return ValidationResult(check_name, False, "Invoice column missing", {}, "error")

    dtype = df.schema["Invoice"]
    is_string = dtype == pl.Utf8

    # Check for cancellation invoices
    cancellations = df.filter(pl.col("Invoice").str.to_uppercase().str.starts_with("C"))
    n_cancellations = cancellations.height

    details = {
        "invoice_dtype": str(dtype),
        "is_string": is_string,
        "cancellation_count": n_cancellations,
    }

    if not is_string:
        return ValidationResult(check_name, False, f"Invoice is {dtype}, not string. Cancellations will be lost.", details, "error")

    if n_cancellations == 0:
        return ValidationResult(check_name, False, "No cancellation invoices found (C-prefixed). Ingestion may have corrupted data.", details, "warning")

    return ValidationResult(check_name, True, f"Invoice preserved as string. {n_cancellations:,} cancellations detected.", details, "info")


def validate_date_validity(df: pl.DataFrame, date_col: str = "InvoiceDate", check_name: str = "date_validity") -> ValidationResult:
    """Validate dates are reasonable (not null, not future, not too old)."""
    if date_col not in df.columns:
        return ValidationResult(check_name, False, f"{date_col} column missing", {}, "error")

    null_count = df.select(pl.col(date_col).is_null().sum()).item()
    future_count = df.select(pl.col(date_col) > pl.datetime(2025, 1, 1)).sum().item()
    too_old = df.select(pl.col(date_col) < pl.datetime(2000, 1, 1)).sum().item()

    details = {"null_dates": null_count, "future_dates": future_count, "pre_2000_dates": too_old}
    passed = null_count == 0 and future_count == 0 and too_old == 0

    msg = f"Date validity: nulls={null_count}, future={future_count}, pre-2000={too_old}"
    return ValidationResult(check_name, passed, msg, details, "error" if not passed else "info")


def validate_quantity_price(df: pl.DataFrame, check_name: str = "quantity_price") -> ValidationResult:
    """Validate quantity and price fields."""
    issues = []

    if "Quantity" in df.columns:
        null_qty = df.select(pl.col("Quantity").is_null().sum()).item()
        if null_qty > 0:
            issues.append(f"Null Quantity: {null_qty}")

    if "Price" in df.columns:
        null_price = df.select(pl.col("Price").is_null().sum()).item()
        neg_price = df.select((pl.col("Price") < 0).sum()).item()
        if null_price > 0:
            issues.append(f"Null Price: {null_price}")
        if neg_price > 0:
            issues.append(f"Negative Price: {neg_price}")

    passed = len(issues) == 0
    msg = "Quantity/Price valid" if passed else "; ".join(issues)
    return ValidationResult(check_name, passed, msg, {"issues": issues}, "error" if not passed else "info")


def validate_probability_bounds(df: pl.DataFrame, prob_cols: List[str], check_name: str = "probability_bounds") -> ValidationResult:
    """Validate probability columns are in [0, 1]."""
    issues = []
    for col in prob_cols:
        if col in df.columns:
            below = df.select((pl.col(col) < 0).sum()).item()
            above = df.select((pl.col(col) > 1).sum()).item()
            nulls = df.select(pl.col(col).is_null().sum()).item()
            if below > 0 or above > 0 or nulls > 0:
                issues.append(f"{col}: below_0={below}, above_1={above}, nulls={nulls}")

    passed = len(issues) == 0
    msg = "All probabilities in [0,1]" if passed else "; ".join(issues)
    return ValidationResult(check_name, passed, msg, {"issues": issues}, "error" if not passed else "info")


def validate_financial_reconciliation(
    tx: pl.DataFrame,
    check_name: str = "financial_reconciliation"
) -> ValidationResult:
    """
    Validate financial reconciliation invariants.

    Invariants (using transaction_type classification):
    1. Gross merchandise revenue = sum of line_value for transaction_type == "sale"
    2. Return value = sum of |line_value| for transaction_type == "return"
    3. Cancellation value = sum of |line_value| for transaction_type == "cancellation"
    4. Net revenue = Gross - Returns - Cancellations
    """
    # Need classified transactions with line_value
    required_cols = ["transaction_type", "line_value", "gross_merchandise_revenue", "return_value", "cancellation_value", "net_merchandise_revenue"]
    missing = [c for c in required_cols if c not in tx.columns]
    if missing:
        return ValidationResult(check_name, False, f"Missing columns for reconciliation: {missing}", {"missing": missing}, "error")

    # Gross revenue reconciliation
    calc_gross = tx.filter(pl.col("transaction_type") == "sale").select(pl.col("line_value").sum()).item()
    stored_gross = tx.select(pl.col("gross_merchandise_revenue").sum()).item()
    gross_diff = abs(calc_gross - stored_gross)

    # Return value reconciliation
    calc_return = tx.filter(pl.col("transaction_type") == "return").select(pl.col("line_value").abs().sum()).item()
    stored_return = tx.select(pl.col("return_value").sum()).item()
    return_diff = abs(calc_return - stored_return)

    # Cancellation value reconciliation
    calc_cancel = tx.filter(pl.col("transaction_type") == "cancellation").select(pl.col("line_value").abs().sum()).item()
    stored_cancel = tx.select(pl.col("cancellation_value").sum()).item()
    cancel_diff = abs(calc_cancel - stored_cancel)

    # Net revenue reconciliation
    calc_net = calc_gross - calc_return - calc_cancel
    stored_net = tx.select(pl.col("net_merchandise_revenue").sum()).item()
    net_diff = abs(calc_net - stored_net)

    tolerance = 0.01  # 1 cent tolerance for floating point
    passed = gross_diff < tolerance and return_diff < tolerance and cancel_diff < tolerance and net_diff < tolerance

    details = {
        "calc_gross": calc_gross,
        "stored_gross": stored_gross,
        "gross_diff": gross_diff,
        "calc_return": calc_return,
        "stored_return": stored_return,
        "return_diff": return_diff,
        "calc_cancellation": calc_cancel,
        "stored_cancellation": stored_cancel,
        "cancel_diff": cancel_diff,
        "calc_net": calc_net,
        "stored_net": stored_net,
        "net_diff": net_diff,
    }

    msg = f"Gross diff: {gross_diff:.4f}, Return diff: {return_diff:.4f}, Cancel diff: {cancel_diff:.4f}, Net diff: {net_diff:.4f}"
    return ValidationResult(check_name, passed, msg, details, "error" if not passed else "info")


def validate_customer_aggregation(
    tx: pl.DataFrame,
    customer_features: pl.DataFrame,
    check_name: str = "customer_aggregation"
) -> ValidationResult:
    """Validate customer-level aggregates reconcile with transactions."""
    # Sum of customer lifetime_gross_revenue should match transaction gross
    if "lifetime_gross_revenue" not in customer_features.columns:
        return ValidationResult(check_name, False, "lifetime_gross_revenue missing from customer features", {}, "error")

    tx_gross = tx.filter(pl.col("is_sale")).select(pl.col("line_value").sum()).item()
    cust_gross = customer_features.select(pl.col("lifetime_gross_revenue").sum()).item()
    diff = abs(tx_gross - cust_gross)

    passed = diff < 0.01
    details = {"tx_gross": tx_gross, "cust_gross": cust_gross, "diff": diff}
    msg = f"Customer aggregation: tx={tx_gross:,.2f}, cust={cust_gross:,.2f}, diff={diff:.4f}"
    return ValidationResult(check_name, passed, msg, details, "error" if not passed else "info")


def validate_cohort_aggregation(
    customer_month_dense: pl.DataFrame,
    cohort_summary: pl.DataFrame,
    check_name: str = "cohort_aggregation"
) -> ValidationResult:
    """Validate cohort-level aggregates reconcile with customer-month panel."""
    # Cohort sizes should match
    if "cohort_month" not in customer_month_dense.columns or "cohort_month" not in cohort_summary.columns:
        return ValidationResult(check_name, False, "cohort_month missing from inputs", {}, "error")

    cm_cohorts = customer_month_dense.filter(pl.col("age_month") == 0).group_by("cohort_month").agg(
        pl.col("Customer ID").n_unique().alias("cm_cohort_size")
    )
    merged = cm_cohorts.join(cohort_summary.select(["cohort_month", "cohort_customers"]), on="cohort_month", how="inner")

    if merged.height == 0:
        return ValidationResult(check_name, False, "No matching cohorts found", {}, "error")

    mismatched = merged.filter(pl.col("cm_cohort_size") != pl.col("cohort_customers"))
    passed = mismatched.height == 0

    details = {
        "cohorts_compared": merged.height,
        "mismatched": mismatched.height,
        "mismatch_details": mismatched.to_dicts() if mismatched.height > 0 else [],
    }

    msg = f"Cohort reconciliation: {merged.height} cohorts compared, {mismatched.height} mismatched"
    return ValidationResult(check_name, passed, msg, details, "error" if not passed else "info")


def run_all_validations(
    tx: pl.DataFrame,
    customer_features: Optional[pl.DataFrame] = None,
    customer_month_dense: Optional[pl.DataFrame] = None,
    cohort_summary: Optional[pl.DataFrame] = None,
) -> List[ValidationResult]:
    """Run all validation checks and return results."""
    results = []

    # Schema and basic checks
    results.append(validate_invoice_string_preservation(tx))
    results.append(validate_date_validity(tx))
    results.append(validate_quantity_price(tx))

    # Financial reconciliation
    if all(c in tx.columns for c in ["is_clean_sale", "is_return", "is_cancellation", "gross_merchandise_revenue", "return_value", "cancellation_value", "net_merchandise_revenue"]):
        results.append(validate_financial_reconciliation(tx))

    # Customer aggregation
    if customer_features is not None:
        results.append(validate_customer_aggregation(tx, customer_features))

    # Cohort aggregation
    if customer_month_dense is not None and cohort_summary is not None:
        results.append(validate_cohort_aggregation(customer_month_dense, cohort_summary))

    # Log results
    for r in results:
        LOGGER.info(str(r))

    return results


def assert_validations_pass(results: List[ValidationResult], fail_on_warning: bool = False) -> None:
    """Raise exception if any validation failed (or warned if fail_on_warning)."""
    failed = [r for r in results if not r.passed]
    warnings = [r for r in results if r.passed and r.severity == "warning"]

    if failed:
        msgs = "\n".join(str(r) for r in failed)
        raise ValueError(f"Validation failures:\n{msgs}")

    if fail_on_warning and warnings:
        msgs = "\n".join(str(r) for r in warnings)
        raise ValueError(f"Validation warnings (treated as failures):\n{msgs}")