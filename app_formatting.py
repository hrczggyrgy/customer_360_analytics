"""
Centralized formatting utilities for the Retail Customer Intelligence application.

Provides semantic formatters that explicitly handle different data types
rather than inferring semantics from magnitude.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# SEMANTIC FORMATTERS
# =============================================================================

def format_currency(
    value: Union[float, int, None],
    decimals: int = 0,
    currency_symbol: str = "£",
    abbreviate: bool = True,
) -> str:
    """Format a monetary value.

    Args:
        value: The numeric value to format.
        decimals: Number of decimal places.
        currency_symbol: Currency symbol to prefix.
        abbreviate: Whether to use K/M/B abbreviations for large values.

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    if abbreviate:
        if abs(value) >= 1_000_000_000:
            return f"{currency_symbol}{value / 1_000_000_000:.{max(1, decimals)}f}B"
        if abs(value) >= 1_000_000:
            return f"{currency_symbol}{value / 1_000_000:.{max(1, decimals)}f}M"
        if abs(value) >= 1_000:
            return f"{currency_symbol}{value / 1_000:.{max(1, decimals)}f}K"

    return f"{currency_symbol}{value:,.{decimals}f}"


def format_probability(
    value: Union[float, int, None],
    decimals: int = 1,
    as_percent: bool = True,
) -> str:
    """Format a probability value (0-1 range).

    Args:
        value: Probability in [0, 1] range.
        decimals: Decimal places for percentage.
        as_percent: Whether to display as percentage (87%) or decimal (0.87).

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    # Clamp to valid probability range
    value = max(0.0, min(1.0, value))

    if as_percent:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals + 2}f}"


def format_percent(
    value: Union[float, int, None],
    decimals: int = 1,
    already_percentage: bool = False,
) -> str:
    """Format a percentage value.

    Args:
        value: The value to format.
        decimals: Decimal places.
        already_percentage: If True, value is already in percentage units (e.g., 87.5).
                           If False, value is a ratio (e.g., 0.875) and will be multiplied by 100.

    Returns:
        Formatted string with % sign, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    if not already_percentage:
        value *= 100

    return f"{value:.{decimals}f}%"


def format_ratio(
    value: Union[float, int, None],
    decimals: int = 2,
    as_percent: bool = False,
) -> str:
    """Format a ratio/index value (e.g., NRR index of 3.28 = 328%).

    Args:
        value: The ratio value.
        decimals: Decimal places.
        as_percent: Whether to display as percentage (328%) or decimal (3.28).

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    if as_percent:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def format_count(
    value: Union[float, int, None],
    abbreviate: bool = True,
) -> str:
    """Format an integer count.

    Args:
        value: The count value.
        abbreviate: Whether to use K/M abbreviations.

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = int(value) if isinstance(value, float) and value.is_integer() else float(value)

    if abbreviate and isinstance(value, (int, float)):
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_date(
    value: Union[str, pd.Timestamp, datetime.date, None],
    format_str: str = "%Y-%m-%d",
) -> str:
    """Format a date value.

    Args:
        value: Date value in various formats.
        format_str: strftime format string.

    Returns:
        Formatted date string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"

    try:
        if isinstance(value, str):
            # Try parsing common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    value = pd.Timestamp(value)
                    break
                except Exception:
                    continue
            else:
                return str(value)

        if isinstance(value, pd.Timestamp):
            return value.strftime(format_str)
        if hasattr(value, "strftime"):
            return value.strftime(format_str)

        return str(value)
    except Exception:
        return str(value)


def format_month(
    value: Union[str, pd.Timestamp, datetime.date, int, None],
) -> str:
    """Format a calendar month (YYYY-MM).

    Args:
        value: Month value.

    Returns:
        Formatted month string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"

    try:
        if isinstance(value, int):
            # Assume YYYYMM format
            year = value // 100
            month = value % 100
            if 1 <= month <= 12 and 2000 <= year <= 2100:
                return f"{year:04d}-{month:02d}"

        if isinstance(value, str):
            # Try to parse YYYY-MM
            if len(value) >= 7 and value[4] == "-":
                return value[:7]
            # Try YYYYMM
            if len(value) == 6 and value.isdigit():
                year = int(value[:4])
                month = int(value[4:])
                if 1 <= month <= 12:
                    return f"{year:04d}-{month:02d}"

        ts = pd.Timestamp(value)
        return ts.strftime("%Y-%m")
    except Exception:
        return str(value)


def format_duration_months(
    value: Union[float, int, None],
    decimals: int = 1,
) -> str:
    """Format a duration in months.

    Args:
        value: Number of months.
        decimals: Decimal places.

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    if value < 1:
        return f"{value * 30:.0f} days"
    if value < 12:
        return f"{value:.{decimals}f} months"
    years = value / 12
    if years < 2:
        return f"{years:.{decimals}f} year"
    return f"{years:.{decimals}f} years"


def format_score(
    value: Union[float, int, None],
    decimals: int = 2,
) -> str:
    """Format a generic score/priority value.

    Args:
        value: The score value.
        decimals: Decimal places.

    Returns:
        Formatted string, or "—" for missing values.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    return f"{float(value):.{decimals}f}"


def format_basis_points(
    value: Union[float, int, None],
    decimals: int = 0,
) -> str:
    """Format a value in basis points (1 bp = 0.01%).

    Args:
        value: Value as a ratio (0.01 = 100bp) or already in basis points.
        decimals: Decimal places.

    Returns:
        Formatted string with "bp" suffix.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    value = float(value)

    # Heuristic: if value < 1, assume it's a ratio and convert
    if abs(value) < 1:
        value *= 10000

    return f"{value:.{decimals}f} bp"


# =============================================================================
# CONVENIENCE ALIASES FOR COMMON USE CASES
# =============================================================================

# CLV-specific formatting
format_clv = lambda v, d=0: format_currency(v, d, abbreviate=True)
format_clv_full = lambda v, d=2: format_currency(v, d, abbreviate=False)

# Retention formatting (already in percentage units from cohort analysis)
format_retention = lambda v, d=1: format_percent(v, d, already_percentage=True)

# Churn risk (probability)
format_churn_risk = lambda v, d=1: format_probability(v, d, as_percent=True)

# Next purchase probability
format_next_purchase = lambda v, d=1: format_probability(v, d, as_percent=True)

# NRR index (can exceed 100%)
format_nrr_index = lambda v, d=1: format_ratio(v, d, as_percent=True)

# Revenue retention (already percentage)
format_revenue_retention = lambda v, d=1: format_percent(v, d, already_percentage=True)


# =============================================================================
# DATA TYPE DETECTION HELPERS
# =============================================================================

def infer_semantic_type(column_name: str, sample_values: Optional[list] = None) -> str:
    """Infer the semantic type of a column for appropriate formatting.

    Args:
        column_name: The column name.
        sample_values: Optional sample of values for inference.

    Returns:
        Semantic type: "currency", "probability", "percent", "ratio", "count", "date", "month", "duration", "score", "text"
    """
    col_lower = column_name.lower()

    # Currency/revenue
    if any(term in col_lower for term in ["revenue", "value", "clv", "price", "amount", "spend", "monetary"]):
        return "currency"

    # Probability (0-1)
    if any(term in col_lower for term in ["probability", "prob_", "propensity", "likelihood", "risk", "churn", "survival"]):
        if "rate" not in col_lower or "return" in col_lower:  # return_rate is a percent
            return "probability"

    # Percentage (already in % units or rate)
    if any(term in col_lower for term in ["rate", "pct", "percent", "share", "retention"]) and "probability" not in col_lower:
        return "percent"

    # Ratio/index (can exceed 1)
    if any(term in col_lower for term in ["nrr", "index", "ratio", "multiplier", "lift"]):
        return "ratio"

    # Count
    if any(term in col_lower for term in ["count", "orders", "units", "customers", "frequency", "n_", "num_"]):
        return "count"

    # Date
    if any(term in col_lower for term in ["date", "cohort_month", "calendar_month"]):
        if "month" in col_lower:
            return "month"
        return "date"

    # Duration
    if any(term in col_lower for term in ["tenure", "recency", "days_since", "months_since", "age_month", "horizon"]):
        return "duration"

    # Score
    if any(term in col_lower for term in ["score", "priority", "confidence", "rank"]):
        return "score"

    return "text"


def auto_format(value: Union[float, int, str, None], column_name: str) -> str:
    """Automatically format a value based on column name semantics.

    Args:
        value: The value to format.
        column_name: The column name for semantic inference.

    Returns:
        Formatted string.
    """
    semantic = infer_semantic_type(column_name)

    if semantic == "currency":
        return format_currency(value)
    elif semantic == "probability":
        return format_probability(value)
    elif semantic == "percent":
        return format_percent(value, already_percentage=True)
    elif semantic == "ratio":
        return format_ratio(value, as_percent=True)
    elif semantic == "count":
        return format_count(value)
    elif semantic == "date":
        return format_date(value)
    elif semantic == "month":
        return format_month(value)
    elif semantic == "duration":
        return format_duration_months(value)
    elif semantic == "score":
        return format_score(value)
    else:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return str(value)


# =============================================================================
# TABLE FORMATTING HELPERS
# =============================================================================

def format_dataframe_for_display(
    df: pd.DataFrame,
    max_rows: int = 1000,
    column_format_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Format a DataFrame for display in Streamlit.

    Args:
        df: Input DataFrame.
        max_rows: Maximum rows to return (samples if exceeded).
        column_format_map: Optional explicit format overrides {column: semantic_type}.

    Returns:
        Formatted DataFrame with string values ready for display.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Sample if too large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    result = df.copy()

    for col in result.columns:
        semantic = column_format_map.get(col) if column_format_map else None
        if not semantic:
            semantic = infer_semantic_type(col)

        if semantic == "currency":
            result[col] = result[col].apply(lambda v: format_currency(v))
        elif semantic == "probability":
            result[col] = result[col].apply(lambda v: format_probability(v))
        elif semantic == "percent":
            result[col] = result[col].apply(lambda v: format_percent(v, already_percentage=True))
        elif semantic == "ratio":
            result[col] = result[col].apply(lambda v: format_ratio(v, as_percent=True))
        elif semantic == "count":
            result[col] = result[col].apply(lambda v: format_count(v))
        elif semantic == "date":
            result[col] = result[col].apply(lambda v: format_date(v))
        elif semantic == "month":
            result[col] = result[col].apply(lambda v: format_month(v))
        elif semantic == "duration":
            result[col] = result[col].apply(lambda v: format_duration_months(v))
        elif semantic == "score":
            result[col] = result[col].apply(lambda v: format_score(v))
        else:
            # Convert to string, handling NaN
            result[col] = result[col].apply(lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))

    return result


# Import at module level for format_dataframe_for_display
import math
from datetime import datetime