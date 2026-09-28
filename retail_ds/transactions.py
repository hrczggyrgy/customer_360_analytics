"""
Explicit transaction classification and financial measures.

This module implements a transparent transaction taxonomy appropriate to the
actual Online Retail II data, based on StockCode/Description patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import polars as pl

LOGGER = logging.getLogger("retail_ds.transactions")


@dataclass
class TransactionClassification:
    """Result of transaction classification."""
    transaction_type: str
    is_sale: bool
    is_return: bool
    is_cancellation: bool
    is_discount: bool
    is_postage: bool
    is_fee: bool
    is_voucher: bool
    is_manual_adjustment: bool
    gross_merchandise_revenue: float
    return_value: float
    cancellation_value: float
    net_merchandise_revenue: float


def classify_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classify each transaction row into explicit type based on StockCode/Description.

    Classification logic (based on UCI Online Retail II data patterns):
    - Cancellation: Invoice starts with 'C'
    - Return: Quantity < 0 (and not a cancellation)
    - Postage: StockCode == 'POST' or Description contains 'POSTAGE'
    - Discount: StockCode == 'D' or Description contains 'DISCOUNT'
    - Fee: Description contains 'FEE', 'BANK', 'CHARGE'
    - Voucher: Description contains 'VOUCHER', 'GIFT VOUCHER', 'COUPON'
    - Manual adjustment: Description contains 'ADJUST', 'WRITE OFF', 'MANUAL'
    - Sale: Everything else with Quantity > 0, Price > 0, non-cancellation
    """
    # StockCode-based classification
    stock_patterns = {
        "POST": "postage",
        "D": "discount",
        "M": "manual_adjustment",  # Manual entries often start with M
        "B": "fee",  # Bank charges
    }

    # Description-based classification (case-insensitive)
    desc_patterns = {
        "postage": "postage",
        "post": "postage",
        "discount": "discount",
        "discounted": "discount",
        "fee": "fee",
        "bank charge": "fee",
        "bank fee": "fee",
        "handling": "fee",
        "voucher": "voucher",
        "gift voucher": "voucher",
        "coupon": "voucher",
        "adjust": "manual_adjustment",
        "write off": "manual_adjustment",
        "write-off": "manual_adjustment",
        "manual": "manual_adjustment",
        "amazon": "fee",  # Amazon fees
        "dotcom": "postage",  # Dotcomgiftshop postage
    }

    out = df.with_columns(
        [
            pl.col("Invoice").str.to_uppercase().str.starts_with("C").alias("is_cancellation_invoice"),
            (pl.col("Quantity") < 0).alias("is_negative_qty"),
            (pl.col("Quantity") > 0).alias("is_positive_qty"),
            (pl.col("Price") > 0).alias("is_positive_price"),
            (pl.col("Price") <= 0).alias("is_zero_neg_price"),
            pl.col("StockCode").str.to_uppercase().alias("stock_upper"),
            pl.col("Description").str.to_lowercase().alias("desc_lower"),
        ]
    )

    # Initialize as sale
    out = out.with_columns(pl.lit("sale").alias("transaction_type"))

    # Apply classification rules in priority order
    # 1. Cancellations (highest priority - Invoice level)
    out = out.with_columns(
        pl.when(pl.col("is_cancellation_invoice"))
        .then(pl.lit("cancellation"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 2. Returns (negative quantity, not cancellation)
    out = out.with_columns(
        pl.when(pl.col("is_negative_qty") & ~pl.col("is_cancellation_invoice"))
        .then(pl.lit("return"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 3. Postage (StockCode POST or description)
    out = out.with_columns(
        pl.when(
            (pl.col("stock_upper") == "POST")
            | pl.col("desc_lower").str.contains("postage|post")
        )
        .then(pl.lit("postage"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 4. Discounts
    out = out.with_columns(
        pl.when(
            (pl.col("stock_upper") == "D")
            | pl.col("desc_lower").str.contains("discount")
        )
        .then(pl.lit("discount"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 5. Fees
    out = out.with_columns(
        pl.when(
            pl.col("desc_lower").str.contains("fee|bank charge|bank fee|handling|amazon fee")
        )
        .then(pl.lit("fee"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 6. Vouchers
    out = out.with_columns(
        pl.when(
            pl.col("desc_lower").str.contains("voucher|coupon|gift voucher")
        )
        .then(pl.lit("voucher"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 7. Manual adjustments
    out = out.with_columns(
        pl.when(
            pl.col("desc_lower").str.contains("adjust|write.?off|manual")
        )
        .then(pl.lit("manual_adjustment"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # 8. Zero/negative price non-sales -> other
    out = out.with_columns(
        pl.when(
            (pl.col("transaction_type") == "sale")
            & ~pl.col("is_positive_price")
        )
        .then(pl.lit("other"))
        .otherwise(pl.col("transaction_type"))
        .alias("transaction_type")
    )

    # Add boolean flags for each type
    for t in ["sale", "return", "cancellation", "discount", "postage", "fee", "voucher", "manual_adjustment"]:
        out = out.with_columns(
            (pl.col("transaction_type") == t).alias(f"is_{t}")
        )

    # Other catch-all
    out = out.with_columns(
        ~pl.col("transaction_type").is_in([
            "sale", "return", "cancellation", "discount", "postage", "fee", "voucher", "manual_adjustment"
        ]).alias("is_other")
    )

    # Log distribution
    type_counts = out.group_by("transaction_type").len().sort("len", descending=True)
    LOGGER.info("Transaction type distribution:\n%s", type_counts)

    return out


def compute_financial_measures(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute canonical financial measures from classified transactions.

    Definitions (based on final transaction_type):
    - gross_merchandise_revenue: Sum of line_value for type == "sale"
    - return_value: Sum of |line_value| for type == "return"
    - cancellation_value: Sum of |line_value| for type == "cancellation"
    - net_merchandise_revenue: gross_merchandise_revenue - return_value - cancellation_value
    """
    out = df.with_columns(
        (pl.col("Quantity") * pl.col("Price")).alias("line_value")
    )

    # Gross merchandise revenue (clean sales only)
    out = out.with_columns(
        pl.when(pl.col("transaction_type") == "sale")
        .then(pl.col("line_value"))
        .otherwise(0.0)
        .alias("gross_merchandise_revenue")
    )

    # Return value (absolute) - only for explicit returns
    out = out.with_columns(
        pl.when(pl.col("transaction_type") == "return")
        .then(pl.col("line_value").abs())
        .otherwise(0.0)
        .alias("return_value")
    )

    # Cancellation value (absolute)
    out = out.with_columns(
        pl.when(pl.col("transaction_type") == "cancellation")
        .then(pl.col("line_value").abs())
        .otherwise(0.0)
        .alias("cancellation_value")
    )

    # Net merchandise revenue
    out = out.with_columns(
        (
            pl.col("gross_merchandise_revenue")
            - pl.col("return_value")
            - pl.col("cancellation_value")
        ).alias("net_merchandise_revenue")
    )

    return out


def get_transaction_type_report(df: pl.DataFrame) -> pl.DataFrame:
    """Generate a summary report of transaction types."""
    return df.group_by("transaction_type").agg(
        pl.len().alias("count"),
        pl.col("line_value").sum().alias("total_line_value"),
        pl.col("Quantity").sum().alias("total_quantity"),
        pl.col("Customer ID").n_unique().alias("unique_customers"),
        pl.col("Invoice").n_unique().alias("unique_invoices"),
    ).sort("count", descending=True)