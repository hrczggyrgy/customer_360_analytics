"""
Point-in-time feature engineering for customer-level prediction.

This module provides features_at_date() function that computes features using
ONLY information available up to the prediction date. This is critical for
preventing temporal leakage in ML models.

Also provides a feature registry with metadata for model governance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

LOGGER = logging.getLogger("retail_ds.features")


@dataclass
class FeatureMetadata:
    """Metadata for a point-in-time feature."""
    name: str
    description: str
    dtype: str
    source: str  # e.g., "transactions", "customer_month", "derived"
    point_in_time_safe: bool  # Can be computed at prediction date without future info
    requires_as_of_date: bool  # Requires explicit as_of_date parameter
    feature_group: str  # e.g., "economic", "cadence", "assortment", "returns", etc.
    transformation: Optional[str] = None  # e.g., "log", "sqrt", "none"
    notes: str = ""


# Feature registry for model governance
FEATURE_REGISTRY: Dict[str, FeatureMetadata] = {
    # Economic features
    "lifetime_gross_revenue": FeatureMetadata(
        name="lifetime_gross_revenue",
        description="Total gross revenue from clean sales up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "lifetime_net_revenue": FeatureMetadata(
        name="lifetime_net_revenue",
        description="Total net revenue (gross - returns - cancellations) up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "lifetime_units": FeatureMetadata(
        name="lifetime_units",
        description="Total units purchased up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "lifetime_product_line_events": FeatureMetadata(
        name="lifetime_product_line_events",
        description="Total product line events (unique product per invoice) up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "historical_avg_order_value": FeatureMetadata(
        name="historical_avg_order_value",
        description="Mean order value up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "historical_median_order_value": FeatureMetadata(
        name="historical_median_order_value",
        description="Median order value up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "historical_order_value_std": FeatureMetadata(
        name="historical_order_value_std",
        description="Standard deviation of order values up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "aov_check": FeatureMetadata(
        name="aov_check",
        description="Average order value check (gross revenue / orders)",
        dtype="float64",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),
    "lifetime_return_units": FeatureMetadata(
        name="lifetime_return_units",
        description="Total return units up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "orders_per_active_tenure_month": FeatureMetadata(
        name="orders_per_active_tenure_month",
        description="Order frequency normalized by active tenure",
        dtype="float64",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="economic",
    ),

    # Cadence features
    "lifetime_orders": FeatureMetadata(
        name="lifetime_orders",
        description="Total number of orders up to as_of_date",
        dtype="int64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "median_interpurchase_days": FeatureMetadata(
        name="median_interpurchase_days",
        description="Median days between purchases up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "median_interpurchase_seconds": FeatureMetadata(
        name="median_interpurchase_seconds",
        description="Median seconds between purchases up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "mean_interpurchase_seconds": FeatureMetadata(
        name="mean_interpurchase_seconds",
        description="Mean seconds between purchases up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "std_interpurchase_seconds": FeatureMetadata(
        name="std_interpurchase_seconds",
        description="Standard deviation of interpurchase seconds up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "interpurchase_cv": FeatureMetadata(
        name="interpurchase_cv",
        description="Coefficient of variation of interpurchase days",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "avg_products_per_order": FeatureMetadata(
        name="avg_products_per_order",
        description="Average products per order up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "weekend_order_share": FeatureMetadata(
        name="weekend_order_share",
        description="Share of orders on weekends up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "business_hour_order_share": FeatureMetadata(
        name="business_hour_order_share",
        description="Share of orders during business hours up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),
    "order_value_cv": FeatureMetadata(
        name="order_value_cv",
        description="Coefficient of variation of order values up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="cadence",
    ),

    # Assortment features
    "unique_products": FeatureMetadata(
        name="unique_products",
        description="Number of unique products purchased up to as_of_date",
        dtype="int64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="assortment",
    ),
    "product_revenue_hhi": FeatureMetadata(
        name="product_revenue_hhi",
        description="Herfindahl-Hirschman Index of product revenue concentration",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="assortment",
    ),
    "repeat_product_ratio": FeatureMetadata(
        name="repeat_product_ratio",
        description="Ratio of repeat product purchases up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="assortment",
    ),

    # Pricing features
    "mean_unit_price": FeatureMetadata(
        name="mean_unit_price",
        description="Mean unit price paid up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "median_unit_price": FeatureMetadata(
        name="median_unit_price",
        description="Median unit price paid up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "unit_price_cv": FeatureMetadata(
        name="unit_price_cv",
        description="Coefficient of variation of unit prices",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "unit_price_iqr": FeatureMetadata(
        name="unit_price_iqr",
        description="Interquartile range of unit prices up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "unit_price_p90": FeatureMetadata(
        name="unit_price_p90",
        description="90th percentile of unit prices up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "unit_price_std": FeatureMetadata(
        name="unit_price_std",
        description="Standard deviation of unit prices up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),
    "nonpositive_price_share": FeatureMetadata(
        name="nonpositive_price_share",
        description="Share of non-positive prices up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="pricing",
    ),

    # Returns features
    "lifetime_return_value": FeatureMetadata(
        name="lifetime_return_value",
        description="Total return/cancellation value up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "lifetime_return_units": FeatureMetadata(
        name="lifetime_return_units",
        description="Total return units up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "return_invoice_count": FeatureMetadata(
        name="return_invoice_count",
        description="Number of return invoices up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "return_line_count": FeatureMetadata(
        name="return_line_count",
        description="Number of return line items up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "lifetime_return_value_rate": FeatureMetadata(
        name="lifetime_return_value_rate",
        description="Return value / gross revenue up to as_of_date",
        dtype="float64",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "lifetime_return_unit_rate": FeatureMetadata(
        name="lifetime_return_unit_rate",
        description="Return units / total units up to as_of_date",
        dtype="float64",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="returns",
    ),
    "active_month_density": FeatureMetadata(
        name="active_month_density",
        description="Active months / tenure months up to as_of_date",
        dtype="float64",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "lifecycle_state": FeatureMetadata(
        name="lifecycle_state",
        description="Customer lifecycle state as of as_of_date",
        dtype="string",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "has_reactivated": FeatureMetadata(
        name="has_reactivated",
        description="Whether customer has ever reactivated up to as_of_date",
        dtype="bool",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "customer_state": FeatureMetadata(
        name="customer_state",
        description="Customer state including reactivation status as of as_of_date",
        dtype="string",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "prediction_date": FeatureMetadata(
        name="prediction_date",
        description="The prediction date used for feature computation",
        dtype="datetime",
        source="derived",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),

    # Temporal features
    "hour_entropy": FeatureMetadata(
        name="hour_entropy",
        description="Entropy of purchase hours (0-23) up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "weekend_order_share": FeatureMetadata(
        name="weekend_order_share",
        description="Share of orders on weekends (Sat/Sun) up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "avg_order_weekday": FeatureMetadata(
        name="avg_order_weekday",
        description="Average weekday of orders up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "avg_order_hour": FeatureMetadata(
        name="avg_order_hour",
        description="Average hour of orders up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "purchase_month_count": FeatureMetadata(
        name="purchase_month_count",
        description="Number of unique purchase months up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "hour_entropy": FeatureMetadata(
        name="hour_entropy",
        description="Entropy of purchase hours up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "weekday_entropy": FeatureMetadata(
        name="weekday_entropy",
        description="Entropy of purchase weekdays up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),
    "month_entropy": FeatureMetadata(
        name="month_entropy",
        description="Entropy of purchase months up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="temporal",
    ),

    # Lifecycle features
    "tenure_days": FeatureMetadata(
        name="tenure_days",
        description="Days since first purchase up to as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "first_purchase_date": FeatureMetadata(
        name="first_purchase_date",
        description="Date of first purchase (historical fact available at any as_of_date after first purchase)",
        dtype="datetime",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "last_purchase_date": FeatureMetadata(
        name="last_purchase_date",
        description="Date of last purchase up to as_of_date",
        dtype="datetime",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "cohort_month": FeatureMetadata(
        name="cohort_month",
        description="Acquisition cohort month (first purchase month)",
        dtype="datetime",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "recency_days": FeatureMetadata(
        name="recency_days",
        description="Days since last purchase as of as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "recency_months": FeatureMetadata(
        name="recency_months",
        description="Months since last purchase as of as_of_date",
        dtype="float64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "active_month_count": FeatureMetadata(
        name="active_month_count",
        description="Number of active months up to as_of_date",
        dtype="int64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "reactivation_count": FeatureMetadata(
        name="reactivation_count",
        description="Number of reactivation events up to as_of_date",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),
    "churn_transition_count": FeatureMetadata(
        name="churn_transition_count",
        description="Number of churn transitions up to as_of_date",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="lifecycle",
    ),

    # Descriptive (NOT point-in-time safe for historical prediction)
    "current_recency": FeatureMetadata(
        name="current_recency",
        description="Recency at final observation date (DESCRIPTIVE ONLY)",
        dtype="float64",
        source="transactions",
        point_in_time_safe=False,
        requires_as_of_date=False,
        feature_group="lifecycle",
        notes="Uses final snapshot date - NOT safe for historical prediction",
    ),
    "future_12m_revenue": FeatureMetadata(
        name="future_12m_revenue",
        description="Actual revenue in next 12 months (TARGET - never a feature)",
        dtype="float64",
        source="transactions",
        point_in_time_safe=False,
        requires_as_of_date=False,
        feature_group="target",
        notes="OUTCOME VARIABLE - cannot be used as feature",
    ),

    # Rolling window features (generated dynamically for windows 1, 3, 6, 12)
    "orders_last_1m": FeatureMetadata(
        name="orders_last_1m",
        description="Orders in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "revenue_last_1m": FeatureMetadata(
        name="revenue_last_1m",
        description="Gross revenue in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "net_revenue_last_1m": FeatureMetadata(
        name="net_revenue_last_1m",
        description="Net revenue in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "active_months_last_1m": FeatureMetadata(
        name="active_months_last_1m",
        description="Active months in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_value_last_1m": FeatureMetadata(
        name="return_value_last_1m",
        description="Return value in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "aov_last_1m": FeatureMetadata(
        name="aov_last_1m",
        description="Average order value in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_rate_last_1m": FeatureMetadata(
        name="return_rate_last_1m",
        description="Return rate in last 1 month",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),

    "orders_last_3m": FeatureMetadata(
        name="orders_last_3m",
        description="Orders in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "revenue_last_3m": FeatureMetadata(
        name="revenue_last_3m",
        description="Gross revenue in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "net_revenue_last_3m": FeatureMetadata(
        name="net_revenue_last_3m",
        description="Net revenue in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "active_months_last_3m": FeatureMetadata(
        name="active_months_last_3m",
        description="Active months in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_value_last_3m": FeatureMetadata(
        name="return_value_last_3m",
        description="Return value in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "aov_last_3m": FeatureMetadata(
        name="aov_last_3m",
        description="Average order value in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_rate_last_3m": FeatureMetadata(
        name="return_rate_last_3m",
        description="Return rate in last 3 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),

    "orders_last_6m": FeatureMetadata(
        name="orders_last_6m",
        description="Orders in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "revenue_last_6m": FeatureMetadata(
        name="revenue_last_6m",
        description="Gross revenue in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "net_revenue_last_6m": FeatureMetadata(
        name="net_revenue_last_6m",
        description="Net revenue in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "active_months_last_6m": FeatureMetadata(
        name="active_months_last_6m",
        description="Active months in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_value_last_6m": FeatureMetadata(
        name="return_value_last_6m",
        description="Return value in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "aov_last_6m": FeatureMetadata(
        name="aov_last_6m",
        description="Average order value in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_rate_last_6m": FeatureMetadata(
        name="return_rate_last_6m",
        description="Return rate in last 6 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),

    "orders_last_12m": FeatureMetadata(
        name="orders_last_12m",
        description="Orders in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "revenue_last_12m": FeatureMetadata(
        name="revenue_last_12m",
        description="Gross revenue in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "net_revenue_last_12m": FeatureMetadata(
        name="net_revenue_last_12m",
        description="Net revenue in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "active_months_last_12m": FeatureMetadata(
        name="active_months_last_12m",
        description="Active months in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_value_last_12m": FeatureMetadata(
        name="return_value_last_12m",
        description="Return value in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "aov_last_12m": FeatureMetadata(
        name="aov_last_12m",
        description="Average order value in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),
    "return_rate_last_12m": FeatureMetadata(
        name="return_rate_last_12m",
        description="Return rate in last 12 months",
        dtype="float64",
        source="customer_month",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="dynamic",
    ),

    # Geography features
    "unique_countries": FeatureMetadata(
        name="unique_countries",
        description="Number of unique countries up to as_of_date",
        dtype="int64",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="geography",
    ),
    "primary_country": FeatureMetadata(
        name="primary_country",
        description="Primary country (most frequent) up to as_of_date",
        dtype="string",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="geography",
    ),
    "first_country": FeatureMetadata(
        name="first_country",
        description="First purchase country",
        dtype="string",
        source="transactions",
        point_in_time_safe=True,
        requires_as_of_date=True,
        feature_group="geography",
    ),
}


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, floor: float = 1e-9) -> pl.Expr:
    """Safe division with floor."""
    return pl.when(denominator.abs() > floor).then(numerator / denominator).otherwise(0.0)


def month_id_expr(column: str) -> pl.Expr:
    """Convert datetime to month ID (year * 12 + month)."""
    return pl.col(column).dt.year() * 12 + pl.col(column).dt.month()


def build_point_in_time_features(
    tx: pl.DataFrame,
    prediction_date: str,
    customer_month_dense: Optional[pl.DataFrame] = None,
    windows: Sequence[int] = (1, 3, 6, 12),
) -> pl.DataFrame:
    """
    Build point-in-time customer features as of a specific prediction date.

    CRITICAL: All features are computed using ONLY transactions dated <= prediction_date.
    This prevents temporal leakage.

    Args:
        tx: Canonical transactions (cleaned + classified)
        prediction_date: ISO format date string (e.g., "2011-06-30")
        customer_month_dense: Pre-computed dense customer-month panel (optional)
        windows: Trailing windows in months

    Returns:
        DataFrame with one row per customer, features as of prediction_date
    """
    pred_dt = pl.lit(prediction_date).str.strptime(pl.Datetime)
    sales = tx.filter(pl.col("is_sale") & pl.col("is_positive_price") & (pl.col("InvoiceDate") <= pred_dt))

    if sales.height == 0:
        raise ValueError(f"No clean sales found up to {prediction_date}")

    # Filter customer-month panel if provided
    if customer_month_dense is not None:
        cm = customer_month_dense.filter(pl.col("calendar_month") <= pred_dt)
    else:
        cm = None

    # -------------------------------------------------------------------------
    # Invoice-level for cadence/order value
    # -------------------------------------------------------------------------
    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
                pl.col("Quantity").sum().alias("invoice_units"),
                pl.col("StockCode").n_unique().alias("invoice_unique_products"),
            ]
        )
        .sort(["Customer ID", "invoice_date"])
    )

    invoice = invoice.with_columns(
        [
            pl.col("invoice_date").diff().over("Customer ID").dt.total_seconds().fill_null(0.0).alias("interpurchase_seconds"),
            pl.col("invoice_date").dt.weekday().alias("invoice_weekday"),
            pl.col("invoice_date").dt.hour().alias("invoice_hour"),
        ]
    )

    # -------------------------------------------------------------------------
    # Customer base features
    # -------------------------------------------------------------------------
    customer_base = (
        invoice.group_by("Customer ID")
        .agg(
            [
                pl.len().alias("lifetime_orders"),
                pl.col("invoice_date").min().alias("first_purchase_date"),
                pl.col("invoice_date").max().alias("last_purchase_date"),
                pl.col("invoice_revenue").sum().alias("lifetime_gross_revenue"),
                pl.col("invoice_revenue").mean().alias("historical_avg_order_value"),
                pl.col("invoice_revenue").median().alias("historical_median_order_value"),
                pl.col("invoice_revenue").std().fill_null(0.0).alias("historical_order_value_std"),
                pl.col("invoice_units").sum().alias("lifetime_units"),
                pl.col("invoice_unique_products").sum().alias("lifetime_product_line_events"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).median().alias("median_interpurchase_seconds"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).mean().alias("mean_interpurchase_seconds"),
                pl.col("interpurchase_seconds").filter(pl.col("interpurchase_seconds") > 0).std().fill_null(0.0).alias("std_interpurchase_seconds"),
                pl.col("invoice_unique_products").mean().alias("avg_products_per_order"),
                (pl.col("invoice_weekday") >= 6).mean().alias("weekend_order_share"),
                ((pl.col("invoice_hour") >= 9) & (pl.col("invoice_hour") < 18)).mean().alias("business_hour_order_share"),
            ]
        )
        .with_columns(
            [
                safe_divide(pl.col("historical_order_value_std"), pl.col("historical_avg_order_value")).alias("order_value_cv"),
                safe_divide(pl.col("std_interpurchase_seconds"), pl.col("mean_interpurchase_seconds")).alias("interpurchase_cv"),
                (pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_days().fill_null(0.0).alias("tenure_days"),
                (pred_dt - pl.col("last_purchase_date")).dt.total_days().fill_null(0.0).alias("recency_days"),
                pl.col("first_purchase_date").dt.truncate("1mo").alias("cohort_month"),
            ]
        )
        .with_columns(
            [
                (pl.col("recency_days") / 30.4375).alias("recency_months"),
                safe_divide(pl.col("lifetime_orders"), (pl.col("tenure_days") / 30.4375).clip(lower_bound=1.0)).alias("orders_per_active_tenure_month"),
                safe_divide(pl.col("lifetime_gross_revenue"), pl.col("lifetime_orders")).alias("aov_check"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Returns features
    # -------------------------------------------------------------------------
    returns = tx.filter(
        (pl.col("is_return") | pl.col("is_cancellation")) & (pl.col("InvoiceDate") <= pred_dt)
    )

    if returns.height:
        return_features = (
            returns.group_by("Customer ID")
            .agg(
                [
                    pl.col("return_value").sum().alias("lifetime_return_value"),
                    pl.col("return_units").sum().alias("lifetime_return_units"),
                    pl.col("Invoice").n_unique().alias("return_invoice_count"),
                    pl.len().alias("return_line_count"),
                ]
            )
        )
    else:
        return_features = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "lifetime_return_value": pl.Float64,
                "lifetime_return_units": pl.Float64,
                "return_invoice_count": pl.Float64,
                "return_line_count": pl.Float64,
            }
        )

    # -------------------------------------------------------------------------
    # Product features
    # -------------------------------------------------------------------------
    customer_product = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg(
            [
                pl.col("gross_merchandise_revenue").sum().alias("product_revenue"),
                pl.col("Quantity").sum().alias("product_units"),
                pl.col("Invoice").n_unique().alias("product_order_count"),
            ]
        )
        .with_columns(
            safe_divide(pl.col("product_revenue"), pl.col("product_revenue").sum().over("Customer ID")).alias("product_revenue_share")
        )
    )

    product_features = (
        customer_product.group_by("Customer ID")
        .agg(
            [
                pl.col("StockCode").n_unique().alias("unique_products"),
                (pl.col("product_revenue_share") ** 2).sum().alias("product_revenue_hhi"),
                (pl.col("product_order_count") > 1).cast(pl.Float64).mean().alias("repeat_product_ratio"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Price features
    # -------------------------------------------------------------------------
    price_features = (
        sales.group_by("Customer ID")
        .agg(
            [
                pl.col("Price").mean().alias("mean_unit_price"),
                pl.col("Price").median().alias("median_unit_price"),
                (pl.col("Price").quantile(0.75) - pl.col("Price").quantile(0.25)).alias("unit_price_iqr"),
                pl.col("Price").std().fill_null(0.0).alias("unit_price_std"),
                pl.col("Price").quantile(0.90).alias("unit_price_p90"),
                (pl.col("Price") <= 0).mean().alias("nonpositive_price_share"),
            ]
        )
        .with_columns(
            safe_divide(pl.col("unit_price_std"), pl.col("mean_unit_price")).alias("unit_price_cv")
        )
    )

    # -------------------------------------------------------------------------
    # Temporal entropy
    # -------------------------------------------------------------------------
    invoice_temporal = invoice.select(["Customer ID", "invoice_date", "invoice_weekday", "invoice_hour"])

    def entropy_from_counts(df: pl.DataFrame, key: str, bucket_col: str, out_name: str, denom: float) -> pl.DataFrame:
        counts = df.group_by([key, bucket_col]).len("n")
        counts = counts.with_columns((pl.col("n") / pl.col("n").sum().over(key)).alias("p"))
        return counts.group_by(key).agg(
            (-(pl.col("p") * pl.col("p").log()).sum() / math.log(denom)).alias(out_name)
        )

    temporal = invoice_temporal.group_by("Customer ID").agg(
        pl.col("invoice_weekday").mean().alias("avg_order_weekday"),
        pl.col("invoice_hour").mean().alias("avg_order_hour"),
        pl.col("invoice_date").dt.month().n_unique().alias("purchase_month_count"),
    )

    temporal = temporal.join(
        entropy_from_counts(
            invoice_temporal.with_columns(pl.col("invoice_hour").cast(pl.Int64).alias("hour_bucket")),
            "Customer ID", "hour_bucket", "hour_entropy", 24.0
        ),
        on="Customer ID", how="left"
    ).join(
        entropy_from_counts(
            invoice_temporal.with_columns(pl.col("invoice_weekday").cast(pl.Int64).alias("weekday_bucket")),
            "Customer ID", "weekday_bucket", "weekday_entropy", 7.0
        ),
        on="Customer ID", how="left"
    ).join(
        entropy_from_counts(
            invoice_temporal.with_columns(pl.col("invoice_date").dt.month().cast(pl.Int64).alias("month_bucket")),
            "Customer ID", "month_bucket", "month_entropy", 12.0
        ),
        on="Customer ID", how="left"
    )

    # -------------------------------------------------------------------------
    # Rolling window features from customer-month panel
    # -------------------------------------------------------------------------
    dynamic_features = pl.DataFrame(schema={"Customer ID": pl.Int64})
    if cm is not None and cm.height > 0:
        window_datas = []
        for w in windows:
            window_start = pred_dt - pl.duration(days=int(w * 30.4375))

            window_data = cm.filter(
                (pl.col("calendar_month") >= window_start) & (pl.col("calendar_month") <= pred_dt)
            ).group_by("Customer ID").agg(
                [
                    pl.col("orders").sum().alias(f"orders_last_{w}m"),
                    pl.col("gross_revenue").sum().alias(f"revenue_last_{w}m"),
                    pl.col("net_revenue").sum().alias(f"net_revenue_last_{w}m"),
                    pl.col("active").sum().alias(f"active_months_last_{w}m"),
                    pl.col("return_value").sum().alias(f"return_value_last_{w}m"),
                ]
            )

            window_data = window_data.with_columns(
                [
                    safe_divide(pl.col(f"revenue_last_{w}m"), pl.col(f"orders_last_{w}m").clip(lower_bound=1)).alias(f"aov_last_{w}m"),
                    safe_divide(pl.col(f"return_value_last_{w}m"), pl.col(f"revenue_last_{w}m") + 1e-9).alias(f"return_rate_last_{w}m"),
                ]
            )

            window_datas.append(window_data)

        # Join all window data at once
        for wd in window_datas:
            dynamic_features = dynamic_features.join(wd, on="Customer ID", how="full", suffix="_right")
            if "Customer ID_right" in dynamic_features.columns:
                dynamic_features = dynamic_features.drop("Customer ID_right")
    country_features = (
        sales.group_by("Customer ID")
        .agg(
            [
                pl.col("Country").n_unique().alias("unique_countries"),
                pl.col("Country").mode().first().alias("primary_country"),
                pl.col("Country").first().alias("first_country"),
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Reactivation / lifecycle from customer-month
    # -------------------------------------------------------------------------
    transitions = pl.DataFrame(schema={"Customer ID": pl.Int64})
    if cm is not None and cm.height > 0:
        transitions = (
            cm.group_by("Customer ID")
            .agg(
                [
                    pl.col("reactivation_event").sum().alias("reactivation_count"),
                    pl.col("churn_transition").sum().alias("churn_transition_count"),
                    pl.col("active").sum().alias("active_month_count"),
                ]
            )
        )

    # -------------------------------------------------------------------------
    # Combine all features
    # -------------------------------------------------------------------------
    customer = customer_base

    for right in [return_features, product_features, price_features, temporal, dynamic_features, transitions, country_features]:
        customer = customer.join(right, on="Customer ID", how="left")

    # Derived features
    customer = customer.with_columns(
        [
            (pl.col("lifetime_gross_revenue") - pl.col("lifetime_return_value")).alias("lifetime_net_revenue"),
            safe_divide(pl.col("lifetime_return_value"), pl.col("lifetime_gross_revenue")).alias("lifetime_return_value_rate"),
            safe_divide(pl.col("lifetime_return_units"), pl.col("lifetime_units") + pl.col("lifetime_return_units")).alias("lifetime_return_unit_rate"),
            safe_divide(pl.col("active_month_count"), (pl.col("tenure_days") / 30.4375).clip(lower_bound=1.0)).alias("active_month_density"),
        ]
    )

    # Lifecycle state (as of prediction_date)
    customer = customer.with_columns(
        pl.when(pl.col("lifetime_orders") == 1).then(pl.lit("new_single_order"))
        .when((pl.col("recency_months") <= 1) & (pl.col("lifetime_orders") >= 2)).then(pl.lit("active_repeat"))
        .when((pl.col("recency_months") > 1) & (pl.col("recency_months") <= 6)).then(pl.lit("at_risk"))
        .when((pl.col("recency_months") > 6) & (pl.col("recency_months") <= 12)).then(pl.lit("stale"))
        .otherwise(pl.lit("dormant"))
        .alias("lifecycle_state")
    )

    customer = customer.with_columns(
        pl.when(pl.col("reactivation_count") > 0).then(pl.lit(True)).otherwise(pl.lit(False)).alias("has_reactivated"),
    )

    customer = customer.with_columns(
        pl.when(pl.col("has_reactivated") & (pl.col("recency_months") <= 3)).then(pl.lit("reactivated_recently"))
        .otherwise(pl.col("lifecycle_state"))
        .alias("customer_state")
    )

    # Add prediction date for traceability
    customer = customer.with_columns(pl.lit(prediction_date).alias("prediction_date"))

    # Ensure Customer ID is Int64
    customer = customer.with_columns(pl.col("Customer ID").cast(pl.Int64))

    return customer.sort("Customer ID")


def get_feature_metadata(feature_names: List[str]) -> List[FeatureMetadata]:
    """Get metadata for a list of feature names."""
    return [FEATURE_REGISTRY.get(name, FeatureMetadata(
        name=name,
        description="Unknown feature",
        dtype="unknown",
        source="unknown",
        point_in_time_safe=False,
        requires_as_of_date=False,
        feature_group="unknown",
    )) for name in feature_names]


def validate_point_in_time_safety(feature_names: List[str]) -> Dict[str, bool]:
    """Check which features are point-in-time safe."""
    return {name: FEATURE_REGISTRY.get(name, FeatureMetadata(
        name=name, description="", dtype="", source="",
        point_in_time_safe=False, requires_as_of_date=False, feature_group=""
    )).point_in_time_safe for name in feature_names}