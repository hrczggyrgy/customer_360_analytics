"""
Transaction cleaning and type classification.

This module provides the SINGLE canonical implementation of transaction cleaning.
All scripts must import from here rather than duplicating cleaning logic.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import polars as pl

LOGGER = logging.getLogger("retail_ds.cleaning")


class TransactionType(str, Enum):
    """Explicit transaction type taxonomy for Online Retail II."""

    SALE = "sale"                    # Clean positive sale: Qty>0, non-cancellation, Price>0
    RETURN = "return"                # Negative quantity (customer return)
    CANCELLATION = "cancellation"    # Invoice starts with 'C' (cancellation)
    DISCOUNT = "discount"            # StockCode patterns indicating discounts
    POSTAGE = "postage"              # StockCode POST or postage-related
    FEE = "fee"                      # Bank fees, handling fees
    VOUCHER = "voucher"              # Gift vouchers, coupons
    MANUAL_ADJUSTMENT = "manual_adjustment"  # Manual adjustments, write-offs
    OTHER = "other"                  # Unclassified


def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Canonical transaction cleaning with Invoice preserved as string.

    This is the ONLY cleaning function that should be used across the project.
    """
    raw_rows = df.height

    # Ensure Invoice is string from the start (should already be from io.load_raw_transactions)
    out = df.with_columns(
        [
            pl.col("Invoice").cast(pl.Utf8).str.strip_chars().alias("Invoice"),
            pl.col("StockCode").cast(pl.Utf8).str.strip_chars().alias("StockCode"),
            pl.col("Description").cast(pl.Utf8).fill_null("").str.strip_chars().alias("Description"),
            pl.col("Country").cast(pl.Utf8).fill_null("Unknown").str.strip_chars().alias("Country"),
            pl.col("Quantity").cast(pl.Float64, strict=False).alias("Quantity"),
            pl.col("Price").cast(pl.Float64, strict=False).alias("Price"),
            pl.col("Customer ID").cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False).alias("Customer ID"),
        ]
    )

    # Parse InvoiceDate
    if out.schema["InvoiceDate"] not in {pl.Date, pl.Datetime}:
        out = out.with_columns(
            pl.col("InvoiceDate").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, exact=False).alias("InvoiceDate")
        )
    else:
        out = out.with_columns(pl.col("InvoiceDate").cast(pl.Datetime, strict=False).alias("InvoiceDate"))

    # Filter nulls in critical fields
    out = out.filter(
        pl.col("Invoice").is_not_null()
        & pl.col("Customer ID").is_not_null()
        & pl.col("InvoiceDate").is_not_null()
        & pl.col("Quantity").is_not_null()
        & pl.col("Price").is_not_null()
    )

    # Core classification flags
    out = out.with_columns(
        [
            pl.col("Invoice").str.to_uppercase().str.starts_with("C").alias("is_cancellation_invoice"),
            (pl.col("Quantity") > 0).alias("is_positive_quantity"),
            (pl.col("Quantity") < 0).alias("is_negative_quantity"),
            (pl.col("Price") > 0).alias("is_positive_price"),
            (pl.col("Price") <= 0).alias("is_zero_or_negative_price"),
            (pl.col("Quantity") * pl.col("Price")).alias("line_value"),
        ]
    )

    # Clean sale: positive quantity, non-cancellation, positive price
    out = out.with_columns(
        (
            pl.col("is_positive_quantity")
            & ~pl.col("is_cancellation_invoice")
            & pl.col("is_positive_price")
        ).alias("is_clean_sale"),
    )

    # Return or cancellation
    out = out.with_columns(
        (
            pl.col("is_negative_quantity")
            | pl.col("is_cancellation_invoice")
        ).alias("is_return_or_cancellation"),
    )

    # Financial measures
    out = out.with_columns(
        [
            pl.when(pl.col("is_clean_sale"))
            .then(pl.col("line_value"))
            .otherwise(0.0)
            .alias("gross_sale_value"),

            pl.when(pl.col("is_return_or_cancellation"))
            .then(pl.col("line_value").abs())
            .otherwise(0.0)
            .alias("return_value"),

            pl.when(pl.col("is_return_or_cancellation"))
            .then(pl.col("Quantity").abs())
            .otherwise(0.0)
            .alias("return_units"),
        ]
    )

    LOGGER.info(
        "Raw rows=%s | cleaned rows=%s | clean sales=%s",
        f"{raw_rows:,}",
        f"{out.height:,}",
        f"{out.filter(pl.col('is_clean_sale')).height:,}",
    )

    return out


def add_calendar_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Add calendar-derived fields to transaction dataframe."""
    return df.with_columns(
        [
            pl.col("InvoiceDate").dt.date().alias("calendar_date"),
            pl.col("InvoiceDate").dt.truncate("1mo").alias("calendar_month"),
            pl.col("InvoiceDate").dt.truncate("1w").alias("calendar_week"),
            pl.col("InvoiceDate").dt.year().alias("year"),
            pl.col("InvoiceDate").dt.month().alias("month"),
            pl.col("InvoiceDate").dt.weekday().alias("weekday"),
            pl.col("InvoiceDate").dt.hour().alias("hour"),
        ]
    )