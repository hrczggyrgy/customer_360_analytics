"""
Semantic formatters for Retail Customer Intelligence Streamlit App.

This module provides explicit semantic formatting functions that replace
the heuristic `pct()` function. Each formatter has a clear semantic contract.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# SEMANTIC TYPE DEFINITIONS
# =============================================================================

class SemanticType:
    """Enumeration of semantic data types for explicit formatting."""
    CURRENCY = "currency"
    PROBABILITY = "probability"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    COUNT = "count"
    DATE = "date"
    MONTH = "month"
    DURATION_MONTHS = "duration_months"
    DURATION_DAYS = "duration_days"
    SCORE = "score"
    UNKNOWN = "unknown"


# =============================================================================
# EXPLICIT FORMAT MAPS FOR HIGH-VALUE UI TABLES
# =============================================================================

# Column name -> SemanticType mapping for known fields
COLUMN_FORMAT_MAP: dict[str, str] = {
    # Currency fields
    "clv": SemanticType.CURRENCY,
    "clv_mean": SemanticType.CURRENCY,
    "clv_lower": SemanticType.CURRENCY,
    "clv_upper": SemanticType.CURRENCY,
    "clv_p10": SemanticType.CURRENCY,
    "clv_p50": SemanticType.CURRENCY,
    "clv_p90": SemanticType.CURRENCY,
    "revenue": SemanticType.CURRENCY,
    "net_revenue": SemanticType.CURRENCY,
    "gross_revenue": SemanticType.CURRENCY,
    "total_revenue": SemanticType.CURRENCY,
    "lifetime_gross_revenue": SemanticType.CURRENCY,
    "lifetime_net_revenue": SemanticType.CURRENCY,
    "historical_avg_order_value": SemanticType.CURRENCY,
    "historical_median_order_value": SemanticType.CURRENCY,
    "mean_unit_price": SemanticType.CURRENCY,
    "median_unit_price": SemanticType.CURRENCY,
    "avg_price": SemanticType.CURRENCY,
    "total_clv": SemanticType.CURRENCY,
    "expected_value_proxy": SemanticType.CURRENCY,
    "avg_expected_value": SemanticType.CURRENCY,
    "total_expected_value": SemanticType.CURRENCY,
    "return_value": SemanticType.CURRENCY,
    "cancellation_value": SemanticType.CURRENCY,
    
    # Probability fields (0-1 scale)
    "churn_probability": SemanticType.PROBABILITY,
    "churn_prob": SemanticType.PROBABILITY,
    "prob_churn": SemanticType.PROBABILITY,
    "next_purchase_probability": SemanticType.PROBABILITY,
    "next_purchase_probability_7d": SemanticType.PROBABILITY,
    "next_purchase_probability_30d": SemanticType.PROBABILITY,
    "next_purchase_probability_60d": SemanticType.PROBABILITY,
    "purchase_probability_7d": SemanticType.PROBABILITY,
    "purchase_probability_30d": SemanticType.PROBABILITY,
    "purchase_probability_60d": SemanticType.PROBABILITY,
    "reactivation_probability": SemanticType.PROBABILITY,
    "reactivation_prob": SemanticType.PROBABILITY,
    "survival_3m": SemanticType.PROBABILITY,
    "survival_6m": SemanticType.PROBABILITY,
    "survival_12m": SemanticType.PROBABILITY,
    "decision_confidence": SemanticType.PROBABILITY,
    "segment_confidence": SemanticType.PROBABILITY,
    "confidence": SemanticType.PROBABILITY,
    
    # Percentage fields (already in % scale or basis points)
    "retention": SemanticType.PERCENTAGE,
    "logo_retention": SemanticType.PERCENTAGE,
    "net_revenue_retention": SemanticType.RATIO,  # Can exceed 100%
    "gross_revenue_retention": SemanticType.RATIO,
    "repeat_customer_rate": SemanticType.PERCENTAGE,
    "customer_penetration": SemanticType.PERCENTAGE,
    "unit_return_rate": SemanticType.PERCENTAGE,
    "revenue_return_rate": SemanticType.PERCENTAGE,
    "lifetime_return_value_rate": SemanticType.PERCENTAGE,
    "lifetime_return_unit_rate": SemanticType.PERCENTAGE,
    "return_to_gross_ratio": SemanticType.PERCENTAGE,
    "weekend_order_share": SemanticType.PERCENTAGE,
    "business_hour_order_share": SemanticType.PERCENTAGE,
    "nonpositive_price_share": SemanticType.PERCENTAGE,
    
    # Ratio/index fields
    "nrr": SemanticType.RATIO,
    "net_revenue_retention_index": SemanticType.RATIO,
    "product_revenue_hhi": SemanticType.RATIO,
    "price_cv": SemanticType.RATIO,
    "interpurchase_cv": SemanticType.RATIO,
    "order_value_cv": SemanticType.RATIO,
    "unit_price_cv": SemanticType.RATIO,
    "repeat_product_ratio": SemanticType.RATIO,
    "active_month_density": SemanticType.RATIO,
    "orders_per_active_tenure_month": SemanticType.RATIO,
    "aov_check": SemanticType.RATIO,
    "aov_last_1m": SemanticType.RATIO,
    "aov_last_3m": SemanticType.RATIO,
    "aov_last_6m": SemanticType.RATIO,
    "aov_last_12m": SemanticType.RATIO,
    "return_rate_last_1m": SemanticType.PERCENTAGE,
    "return_rate_last_3m": SemanticType.PERCENTAGE,
    "return_rate_last_6m": SemanticType.PERCENTAGE,
    "return_rate_last_12m": SemanticType.PERCENTAGE,
    
    # Count fields
    "orders": SemanticType.COUNT,
    "lifetime_orders": SemanticType.COUNT,
    "invoice_count": SemanticType.COUNT,
    "unique_products": SemanticType.COUNT,
    "unique_customers": SemanticType.COUNT,
    "total_invoices": SemanticType.COUNT,
    "unique_countries": SemanticType.COUNT,
    "reactivation_count": SemanticType.COUNT,
    "churn_transition_count": SemanticType.COUNT,
    "return_invoice_count": SemanticType.COUNT,
    "return_line_count": SemanticType.COUNT,
    "active_month_count": SemanticType.COUNT,
    "customer_count": SemanticType.COUNT,
    "customers": SemanticType.COUNT,
    "cohort_customers": SemanticType.COUNT,
    "product_count": SemanticType.COUNT,
    "repeat_customers": SemanticType.COUNT,
    
    # Duration fields (months)
    "recency_months": SemanticType.DURATION_MONTHS,
    "consecutive_inactive_months": SemanticType.DURATION_MONTHS,
    "age_month": SemanticType.DURATION_MONTHS,
    "calendar_month_id": SemanticType.DURATION_MONTHS,
    
    # Duration fields (days)
    "tenure_days": SemanticType.DURATION_DAYS,
    "recency_days": SemanticType.DURATION_DAYS,
    "interpurchase_days": SemanticType.DURATION_DAYS,
    "median_interpurchase_days": SemanticType.DURATION_DAYS,
    "expected_days_to_next_purchase": SemanticType.DURATION_DAYS,
    "days_since_last_purchase": SemanticType.DURATION_DAYS,
    "days_since_last_sale": SemanticType.DURATION_DAYS,
    
    # Decision confidence is a score, not a probability
    "decision_confidence": SemanticType.SCORE,
    
    # Duration fields (seconds - treated as days for display)
    "interpurchase_seconds": SemanticType.DURATION_DAYS,
    "mean_interpurchase_seconds": SemanticType.DURATION_DAYS,
    "median_interpurchase_seconds": SemanticType.DURATION_DAYS,
    "std_interpurchase_seconds": SemanticType.DURATION_DAYS,
    
    # Score fields
    "score": SemanticType.SCORE,
    "priority_score": SemanticType.SCORE,
    "top_score": SemanticType.SCORE,
    "co_purchase_score": SemanticType.SCORE,
    "popularity_score": SemanticType.SCORE,
    "final_score": SemanticType.SCORE,
    "lift": SemanticType.SCORE,
    "support": SemanticType.COUNT,
    "pop_score": SemanticType.SCORE,
    "protect_value_score": SemanticType.SCORE,
    "accelerate_purchase_score": SemanticType.SCORE,
    "reactivate_score": SemanticType.SCORE,
    "cross_sell_score": SemanticType.SCORE,
    "nurture_score": SemanticType.SCORE,
    
    # Date fields
    "first_purchase_date": SemanticType.DATE,
    "last_purchase_date": SemanticType.DATE,
    "cohort_month": SemanticType.MONTH,
    "calendar_month": SemanticType.MONTH,
    "prediction_date": SemanticType.DATE,
    "invoice_date": SemanticType.DATE,
    
    # Month fields
    "calendar_month": SemanticType.MONTH,
    "cohort_month": SemanticType.MONTH,
}


# =============================================================================
# FORMATTER FUNCTIONS
# =============================================================================

def format_currency(value: Any, decimals: int = 0, prefix: str = "£") -> str:
    """Format a monetary value in GBP.
    
    Args:
        value: Numeric value (can be None, NaN, or numeric)
        decimals: Decimal places for values < 1000
        prefix: Currency prefix
    
    Returns:
        Formatted string like "£1,234", "£1.23M", "£12.3K", or "—" for missing
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    if value == 0:
        return f"{prefix}0"
    
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1_000_000:
        return f"{sign}{prefix}{abs_val / 1_000_000:.{max(1, decimals)}f}M"
    elif abs_val >= 1_000:
        return f"{sign}{prefix}{abs_val / 1_000:.{max(1, decimals)}f}K"
    else:
        return f"{sign}{prefix}{abs_val:,.{decimals}f}"


def format_probability(value: Any, decimals: int = 1) -> str:
    """Format a probability (0-1 scale) as percentage with % sign.
    
    Args:
        value: Probability in [0, 1] range
        decimals: Decimal places
    
    Returns:
        Formatted string like "23.5%", "—", or "Invalid" for out-of-range
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    # Validate probability range
    if value < 0 or value > 1:
        return "Invalid"
    
    return f"{value * 100:.{decimals}f}%"


def format_percent(value: Any, decimals: int = 1) -> str:
    """Format a percentage value (already in 0-100 or 0-1 scale).
    
    Unlike format_probability, this assumes the value may already be in
    percentage scale (e.g., 23.5 = 23.5%). Values > 1 are treated as
    already in percentage scale; values <= 1 are multiplied by 100.
    
    Args:
        value: Percentage value
        decimals: Decimal places
    
    Returns:
        Formatted string like "23.5%"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    # Use explicit semantic contract: values > 1 are treated as already in percentage scale
    # This matches the explicit COLUMN_FORMAT_MAP assignments
    if abs(value) > 1:
        return f"{value:.{decimals}f}%"
    else:
        return f"{value * 100:.{decimals}f}%"


def format_days(value: Any, decimals: int = 0) -> str:
    """Format a duration in days.
    
    Args:
        value: Duration in days
        decimals: Decimal places
    
    Returns:
        Formatted string like "45 days" or "—"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    return f"{value:.{decimals}f} days"


def format_ratio(value: Any, decimals: int = 2) -> str:
    """Format a ratio/index value (e.g., 1.23 = 1.23x or 123%).
    
    Args:
        value: Ratio value (1.0 = baseline)
        decimals: Decimal places
    
    Returns:
        Formatted string like "1.23x" or "123%"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    return f"{value:.{decimals}f}x"


def format_count(value: Any) -> str:
    """Format an integer count with thousand separators.
    
    Args:
        value: Integer count
    
    Returns:
        Formatted string like "1,234" or "—"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "—"
    
    return f"{value:,}"


def format_date(value: Any, fmt: str = "%Y-%m-%d") -> str:
    """Format a date value.
    
    Args:
        value: Date/datetime value
        fmt: strftime format string
    
    Returns:
        Formatted date string or "—"
    """
    if value is None:
        return "—"
    
    try:
        if isinstance(value, (datetime, date)):
            dt = value
        elif isinstance(value, str):
            dt = pd.to_datetime(value)
        elif isinstance(value, (int, float)):
            dt = pd.Timestamp(value, unit="ms")
        else:
            dt = pd.to_datetime(value)
        return dt.strftime(fmt)
    except Exception:
        return "—"


def format_month(value: Any, fmt: str = "%Y-%m") -> str:
    """Format a month value (YYYY-MM).
    
    Args:
        value: Month value (datetime, date, string, or Timestamp)
        fmt: strftime format string
    
    Returns:
        Formatted month string or "—"
    """
    if value is None:
        return "—"
    
    try:
        if isinstance(value, (datetime, date)):
            dt = value
        elif isinstance(value, str):
            dt = pd.to_datetime(value)
        elif isinstance(value, (int, float)):
            dt = pd.Timestamp(value, unit="ms")
        else:
            dt = pd.to_datetime(value)
        return dt.strftime(fmt)
    except Exception:
        return "—"


def format_duration_months(value: Any, decimals: int = 1) -> str:
    """Format a duration in months.
    
    Args:
        value: Duration in months (can be fractional)
        decimals: Decimal places
    
    Returns:
        Formatted string like "12.5 mo" or "—"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    if value < 1:
        return f"{value * 30:.0f} days"
    else:
        return f"{value:.{decimals}f} mo"


def format_score(value: Any, decimals: int = 2) -> str:
    """Format a generic score value.
    
    Args:
        value: Score value
        decimals: Decimal places
    
    Returns:
        Formatted string like "0.87" or "—"
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    
    return f"{value:.{decimals}f}"


# =============================================================================
# SEMANTIC TYPE INFERENCE
# =============================================================================

def infer_semantic_type(column_name: str, sample_values: Optional[list] = None) -> str:
    """Infer semantic type from column name and optional sample values.
    
    Args:
        column_name: Name of the column
        sample_values: Optional sample of values for heuristic inference
    
    Returns:
        SemanticType string
    """
    # Normalize column name
    normalized = column_name.lower().strip()
    
    # Check explicit mapping first
    for key, sem_type in COLUMN_FORMAT_MAP.items():
        key_lower = key.lower()
        if key_lower == normalized or f"_{key_lower}_" in f"_{normalized}_" or normalized.startswith(key_lower + "_") or normalized.endswith("_" + key_lower):
            return sem_type
    
    # Pattern-based inference
    if any(kw in normalized for kw in ["revenue", "price", "value", "amount", "clv", "cost", "spend", "spent", "gross", "net", "return", "cancellation", "avg_order", "avg_price", "unit_price", "mean_price", "median_price"]):
        return SemanticType.CURRENCY
    
    if any(kw in normalized for kw in ["probability", "prob_", "churn", "survival", "reactivation", "confidence", "penetration"]):
        return SemanticType.PROBABILITY
    
    if any(kw in normalized for kw in ["ratio", "index", "hhi", "cv_", "c_v", "lift"]):
        return SemanticType.RATIO
    
    if any(kw in normalized for kw in ["retention", "rate", "share", "percentage", "pct_"]):
        return SemanticType.PERCENTAGE
    
    if any(kw in normalized for kw in ["count", "customers", "orders", "invoices", "products", "countries", "reactivation", "churn_transition", "return_invoice", "return_line", "active_month"]):
        return SemanticType.COUNT
    
    if any(kw in normalized for kw in ["months", "duration", "tenure", "recency", "interpurchase", "age_", "consecutive_inactive", "calendar_month_id", "cohort_month_id"]):
        return SemanticType.DURATION_MONTHS
    
    if any(kw in normalized for kw in ["days", "expected_days", "days_since"]):
        return SemanticType.DURATION_DAYS
    
    if any(kw in normalized for kw in ["score", "lift", "priority", "support", "pop_score", "aov_", "co_purchase", "popularity", "final_score"]):
        return SemanticType.SCORE
    
    if any(kw in normalized for kw in ["date", "first_purchase", "last_purchase", "prediction_date", "invoice_date"]):
        return SemanticType.DATE
    
    if any(kw in normalized for kw in ["cohort_month", "calendar_month", "month"]):
        return SemanticType.MONTH
    
    return SemanticType.UNKNOWN


# =============================================================================
# AUTO-FORMATTER
# =============================================================================

def auto_format(value: Any, column_name: str = "") -> str:
    """Automatically format a value based on its column name.
    
    Args:
        value: Value to format
        column_name: Column name for semantic inference
    
    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    
    sem_type = infer_semantic_type(column_name)
    
    if sem_type == SemanticType.CURRENCY:
        return format_currency(value)
    elif sem_type == SemanticType.PROBABILITY:
        return format_probability(value)
    elif sem_type == SemanticType.PERCENTAGE:
        return format_percent(value)
    elif sem_type == SemanticType.RATIO:
        return format_ratio(value)
    elif sem_type == SemanticType.COUNT:
        return format_count(value)
    elif sem_type == SemanticType.DATE:
        return format_date(value)
    elif sem_type == SemanticType.MONTH:
        return format_month(value)
    elif sem_type == SemanticType.DURATION_MONTHS:
        return format_duration_months(value)
    elif sem_type == SemanticType.DURATION_DAYS:
        return format_days(value)
    elif sem_type == SemanticType.SCORE:
        return format_score(value)
    else:
        # No heuristic fallback - return raw value as string
        if isinstance(value, (int, np.integer)):
            return format_count(value)
        elif isinstance(value, (float, np.floating)):
            return f"{value:.2f}"
        elif isinstance(value, (datetime, date, pd.Timestamp)):
            return format_date(value)
        elif isinstance(value, str):
            return value
        else:
            return str(value)


# =============================================================================
# DATAFRAME FORMATTING UTILITIES
# =============================================================================

def format_dataframe_columns(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """Format specified columns in a DataFrame using semantic formatters.
    
    Args:
        df: Input DataFrame
        columns: List of column names to format (default: all)
    
    Returns:
        DataFrame with formatted columns (as strings)
    """
    if df.empty:
        return df
    
    result = df.copy()
    target_cols = columns if columns is not None else df.columns
    
    for col in target_cols:
        if col not in df.columns:
            continue
        
        sem_type = infer_semantic_type(col)
        
        if sem_type == SemanticType.CURRENCY:
            result[col] = result[col].apply(format_currency)
        elif sem_type == SemanticType.PROBABILITY:
            result[col] = result[col].apply(format_probability)
        elif sem_type == SemanticType.PERCENTAGE:
            result[col] = result[col].apply(format_percent)
        elif sem_type == SemanticType.RATIO:
            result[col] = result[col].apply(format_ratio)
        elif sem_type == SemanticType.COUNT:
            result[col] = result[col].apply(format_count)
        elif sem_type == SemanticType.DATE:
            result[col] = result[col].apply(format_date)
        elif sem_type == SemanticType.MONTH:
            result[col] = result[col].apply(format_month)
        elif sem_type == SemanticType.DURATION_MONTHS:
            result[col] = result[col].apply(format_duration_months)
        elif sem_type == SemanticType.SCORE:
            result[col] = result[col].apply(format_score)
    
    return result