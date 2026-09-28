"""
Temporal backtesting framework for rolling-origin validation.

This module provides reusable utilities for time-aware model validation,
ensuring all predictive models use consistent temporal splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

LOGGER = logging.getLogger("retail_ds.backtesting")


@dataclass
class TemporalSplit:
    """Single temporal train/validation/test split."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    origin_label: str  # e.g., "2011-06"

    def to_dict(self) -> Dict:
        return {
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "val_start": str(self.val_start),
            "val_end": str(self.val_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "origin_label": self.origin_label,
        }

    def apply(self, df: pl.DataFrame, date_column: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split a DataFrame into train/val/test based on this TemporalSplit."""
        return apply_temporal_split(df, date_column, self)


def rolling_origin_split(
    df: pl.DataFrame,
    date_column: str,
    prediction_origins: List[str],
    train_lookback_months: Optional[int] = None,
    val_horizon_months: int = 3,
    test_horizon_months: int = 3,
    min_train_months: int = 6,
) -> List[TemporalSplit]:
    """
    Generate rolling-origin temporal splits for backtesting.

    Args:
        df: DataFrame with a date column (customer-month panel or similar)
        date_column: Name of the date column (e.g., "calendar_month")
        prediction_origins: List of origin dates as strings (e.g., ["2010-09", "2010-12", "2011-03"])
        train_lookback_months: Optional fixed lookback window for training. If None, uses all history.
        val_horizon_months: Validation horizon in months
        test_horizon_months: Test horizon in months
        min_train_months: Minimum training months required

    Returns:
        List of TemporalSplit objects
    """
    # Get unique months from data
    months = sorted(df.select(pl.col(date_column).dt.truncate("1mo").unique()).to_series().to_list())
    months = [pd.Timestamp(m) for m in months]

    if len(months) < min_train_months + val_horizon_months + test_horizon_months:
        raise ValueError(f"Insufficient data: {len(months)} months, need at least {min_train_months + val_horizon_months + test_horizon_months}")

    splits = []

    for origin_str in prediction_origins:
        origin = pd.Timestamp(origin_str).to_period("M").to_timestamp()

        if origin not in months:
            LOGGER.warning(f"Origin {origin_str} not in data months, skipping")
            continue

        origin_idx = months.index(origin)

        # Test period
        test_start_idx = origin_idx + 1
        test_end_idx = min(test_start_idx + test_horizon_months - 1, len(months) - 1)
        if test_start_idx > test_end_idx:
            LOGGER.warning(f"Insufficient future data for test at origin {origin_str}, skipping")
            continue

        # Validation period (before test)
        val_end_idx = test_start_idx - 1
        val_start_idx = max(val_end_idx - val_horizon_months + 1, 0)

        # Training period (all history before validation)
        if train_lookback_months:
            train_start_idx = max(val_start_idx - train_lookback_months, 0)
        else:
            train_start_idx = 0
        train_end_idx = val_start_idx - 1

        if train_end_idx < train_start_idx + min_train_months - 1:
            LOGGER.warning(f"Insufficient training data at origin {origin_str}, skipping")
            continue

        split = TemporalSplit(
            train_start=months[train_start_idx],
            train_end=months[train_end_idx],
            val_start=months[val_start_idx],
            val_end=months[val_end_idx],
            test_start=months[test_start_idx],
            test_end=months[test_end_idx],
            origin_label=origin_str,
        )
        splits.append(split)

    LOGGER.info(f"Created {len(splits)} temporal splits from {len(prediction_origins)} origins")
    return splits


def apply_temporal_split(
    df: pl.DataFrame,
    date_column: str,
    split: TemporalSplit,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into train/val/test based on a TemporalSplit.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train = df.filter(
        (pl.col(date_column) >= split.train_start) & (pl.col(date_column) <= split.train_end)
    )
    val = df.filter(
        (pl.col(date_column) >= split.val_start) & (pl.col(date_column) <= split.val_end)
    )
    test = df.filter(
        (pl.col(date_column) >= split.test_start) & (pl.col(date_column) <= split.test_end)
    )

    return train, val, test


def expanding_window_splits(
    df: pl.DataFrame,
    date_column: str,
    n_splits: int = 5,
    initial_train_months: int = 12,
    step_months: int = 3,
    val_horizon_months: int = 3,
    test_horizon_months: int = 3,
) -> List[TemporalSplit]:
    """
    Generate expanding-window temporal splits (like sklearn's TimeSeriesSplit but with explicit periods).

    Train window expands with each split; val and test windows are fixed.
    """
    months = sorted(df.select(pl.col(date_column).dt.truncate("1mo").unique()).to_series().to_list())
    months = [pd.Timestamp(m) for m in months]

    if len(months) < initial_train_months + val_horizon_months + test_horizon_months:
        raise ValueError(f"Insufficient data for expanding window splits")

    splits = []

    for i in range(n_splits):
        train_end_idx = initial_train_months - 1 + i * step_months
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_horizon_months - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_horizon_months - 1

        if test_end_idx >= len(months):
            break

        split = TemporalSplit(
            train_start=months[0],
            train_end=months[train_end_idx],
            val_start=months[val_start_idx],
            val_end=months[val_end_idx],
            test_start=months[test_start_idx],
            test_end=months[test_end_idx],
            origin_label=str(months[val_start_idx].to_period("M")),
        )
        splits.append(split)

    return splits


def sliding_window_splits(
    df: pl.DataFrame,
    date_column: str,
    window_months: int = 12,
    step_months: int = 3,
    val_horizon_months: int = 3,
    test_horizon_months: int = 3,
) -> List[TemporalSplit]:
    """
    Generate sliding-window temporal splits (fixed-size training window).
    """
    months = sorted(df.select(pl.col(date_column).dt.truncate("1mo").unique()).to_series().to_list())
    months = [pd.Timestamp(m) for m in months]

    splits = []
    train_start_idx = 0

    while True:
        train_end_idx = train_start_idx + window_months - 1
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_horizon_months - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_horizon_months - 1

        if test_end_idx >= len(months):
            break

        split = TemporalSplit(
            train_start=months[train_start_idx],
            train_end=months[train_end_idx],
            val_start=months[val_start_idx],
            val_end=months[val_end_idx],
            test_start=months[test_start_idx],
            test_end=months[test_end_idx],
            origin_label=str(months[val_start_idx].to_period("M")),
        )
        splits.append(split)

        train_start_idx += step_months

    return splits


def get_cv_splits_for_modeling(
    customer_month_dense: pl.DataFrame,
    strategy: str = "rolling_origin",
    **kwargs,
) -> List[TemporalSplit]:
    """
    Convenience function to get CV splits for customer-month panel.

    Strategies:
    - "rolling_origin": Fixed prediction origins (default)
    - "expanding": Expanding window
    - "sliding": Sliding window
    """
    if strategy == "rolling_origin":
        default_origins = ["2010-09", "2010-12", "2011-03", "2011-06", "2011-09"]
        origins = kwargs.get("prediction_origins", default_origins)
        return rolling_origin_split(customer_month_dense, "calendar_month", origins, **kwargs)
    elif strategy == "expanding":
        return expanding_window_splits(customer_month_dense, "calendar_month", **kwargs)
    elif strategy == "sliding":
        return sliding_window_splits(customer_month_dense, "calendar_month", **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")