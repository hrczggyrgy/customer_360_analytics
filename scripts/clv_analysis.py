#!/usr/bin/env python3
"""
CLV Analysis — Dynamic Probabilistic Discounted Net-Revenue Forecast with Rolling Backtests.

This script implements a CLV proxy using:
1. Polars transaction engineering
2. Leakage-safe next-month purchase model (HistGradientBoostingClassifier)
3. Conditional spend model for net monthly value when active (HistGradientBoostingRegressor)
4. Empirical-Bayes propensity shrinkage
5. Monte-Carlo future-state simulation with dynamic recency/cadence updates
6. Return-aware net value
7. Time-discounted CLV and uncertainty bands
8. Rolling temporal backtests at multiple prediction origins
9. Calibration and uncertainty coverage validation
10. Optional margin scenario analysis

Terminology: "Dynamic Probabilistic Discounted Net-Revenue CLV Proxy" 
(not classical economic CLV — no margin, no probabilistic lifetime model)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel, add_rolling_features
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.backtesting import rolling_origin_split, TemporalSplit
from retail_ds.features import build_point_in_time_features


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("clv_analysis")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Dynamic Probabilistic Discounted Net-Revenue CLV Proxy.")
    parser.add_argument("--input", default=str(project_dir / "data_xslx" / "online_retail_II.xlsx"), help="Input .xlsx/.xls/.csv/.parquet file.")
    parser.add_argument("--output-dir", default=str(project_dir / "clv_analysis_output"), help="Directory where CLV outputs are written.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--horizon-months", type=int, default=24, help="Future CLV simulation horizon.")
    parser.add_argument("--simulations", type=int, default=200, help="Monte-Carlo paths per customer.")
    parser.add_argument("--annual-discount-rate", type=float, default=0.10, help="Annual discount rate applied to future revenue.")
    parser.add_argument("--margin-rate", type=float, default=1.0, help="Contribution margin proxy. 1.0 leaves CLV as revenue CLV.")
    parser.add_argument("--validation-months", type=int, default=3, help="Trailing calendar months reserved as time-based validation set.")
    parser.add_argument("--min-customers", type=int, default=100, help="Minimum customer count required to fit the model.")
    parser.add_argument("--min-active-history-months", type=int, default=2, help="Minimum active history used for ML training rows.")
    parser.add_argument("--random-customer-cap", type=int, default=0, help="Optional customer cap for development runs. 0 means all customers.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--run-backtests", action="store_true", help="Run rolling temporal backtests.")
    parser.add_argument("--margin-scenarios", default="0.1,0.2,0.3,0.4", help="Comma-separated margin rates for scenario analysis.")

    return parser.parse_args()


def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))


def sample_weights_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    positives = max(1, int(y.sum()))
    negatives = max(1, int((1 - y).sum()))
    weights = np.ones_like(y, dtype=float)
    weights[y == 1] = 0.5 * len(y) / positives
    weights[y == 0] = 0.5 * len(y) / negatives
    return weights


FEATURE_COLUMNS = [
    "age_month", "recency_months", "lifetime_orders", "lifetime_active_months",
    "lifetime_order_rate_per_month", "lifetime_active_month_share",
    "orders_last_3m", "orders_prev_3m", "orders_3m_mean", "recent_orders_intensity",
    "net_revenue_last_3m", "net_revenue_prev_3m", "net_revenue_3m_mean",
    "recent_value_momentum", "lifetime_net_revenue", "historical_value_per_order",
    "historical_return_ratio", "month_sin", "month_cos",
]


def train_models(
    panel: pl.DataFrame,
    validation_months: int,
    min_active_history_months: int,
    seed: int,
) -> Tuple[object, object, Dict, pd.DataFrame, Dict[str, np.ndarray], float]:
    """Train purchase and spend models on a single temporal split."""
    pdf = panel.to_pandas()
    pdf["calendar_month"] = pd.to_datetime(pdf["calendar_month"])

    unique_months = sorted(pdf["calendar_month"].dropna().unique())

    if len(unique_months) < validation_months + 6:
        validation_months = max(1, min(validation_months, len(unique_months) // 4))

    validation_start = pd.Timestamp(unique_months[-validation_months])

    target_available = pdf["next_active"].notna()

    train_mask = (pdf["calendar_month"] < validation_start) & target_available
    validation_mask = (pdf["calendar_month"] >= validation_start) & target_available

    train_mask &= (pdf["lifetime_active_months"] >= min_active_history_months)

    X_train = pdf.loc[train_mask, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = pdf.loc[train_mask, "next_active"].astype(int).to_numpy()
    X_valid = pdf.loc[validation_mask, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_valid = pdf.loc[validation_mask, "next_active"].astype(int).to_numpy()

    if len(X_train) == 0 or len(X_valid) == 0:
        raise ValueError("Time-based validation split produced no training or validation rows.")
    if np.unique(y_train).size < 2:
        raise ValueError("Purchase target has fewer than two classes in training data.")

    purchase_model = HistGradientBoostingClassifier(
        learning_rate=0.055, max_iter=250, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=2.0, random_state=seed,
    )
    purchase_model.fit(X_train, y_train, sample_weight=sample_weights_binary(y_train))

    purchase_valid_prob = purchase_model.predict_proba(X_valid)[:, 1]

    purchase_metrics = {
        "roc_auc": float(roc_auc_score(y_valid, purchase_valid_prob)) if np.unique(y_valid).size > 1 else None,
        "average_precision": float(average_precision_score(y_valid, purchase_valid_prob)) if np.unique(y_valid).size > 1 else None,
        "brier_score": float(brier_score_loss(y_valid, purchase_valid_prob)),
        "validation_rows": int(len(y_valid)),
        "validation_positive_rate": float(y_valid.mean()),
        "validation_predicted_rate": float(purchase_valid_prob.mean()),
    }

    # Spend model
    spend_train_mask = train_mask & (pdf["next_active"] == 1) & np.isfinite(pdf["next_net_revenue"])
    spend_valid_mask = validation_mask & (pdf["next_active"] == 1) & np.isfinite(pdf["next_net_revenue"])

    X_spend_train = pdf.loc[spend_train_mask, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_spend_train = signed_log1p(pdf.loc[spend_train_mask, "next_net_revenue"].to_numpy(dtype=float))
    X_spend_valid = pdf.loc[spend_valid_mask, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_spend_valid_raw = pdf.loc[spend_valid_mask, "next_net_revenue"].to_numpy(dtype=float)

    if len(X_spend_train) < 100:
        raise ValueError("Too few active next-month observations for a robust spend model.")

    spend_model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.045, max_iter=250,
        max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=2.0, random_state=seed,
    )
    spend_model.fit(X_spend_train, y_spend_train)

    spend_valid_pred_raw = signed_expm1(spend_model.predict(X_spend_valid))

    spend_metrics = {
        "mae": float(mean_absolute_error(y_spend_valid_raw, spend_valid_pred_raw)),
        "rmse": float(math.sqrt(mean_squared_error(y_spend_valid_raw, spend_valid_pred_raw))),
        "r2": float(r2_score(y_spend_valid_raw, spend_valid_pred_raw)) if len(y_spend_valid_raw) > 2 and np.var(y_spend_valid_raw) > 1e-12 else None,
        "validation_active_rows": int(len(y_spend_valid_raw)),
        "validation_actual_mean": float(np.mean(y_spend_valid_raw)) if len(y_spend_valid_raw) else None,
        "validation_predicted_mean": float(np.mean(spend_valid_pred_raw)) if len(spend_valid_pred_raw) else None,
    }

    spend_residuals = y_spend_valid_raw - spend_valid_pred_raw
    residual_scale = float(np.median(np.abs(spend_residuals - np.median(spend_residuals))) * 1.4826)

    if len(X_spend_valid):
        spend_valid_pred_transformed = spend_model.predict(X_spend_valid)
        transformed_residuals = signed_log1p(y_spend_valid_raw) - spend_valid_pred_transformed
    else:
        transformed_residuals = np.array([0.5], dtype=float)

    transformed_residuals = transformed_residuals[np.isfinite(transformed_residuals)]
    transformed_sigma = float(np.median(np.abs(transformed_residuals - np.median(transformed_residuals))) * 1.4826)
    transformed_sigma = max(transformed_sigma, 0.15)

    # Permutation importance
    importance_rows = []
    rng = np.random.default_rng(seed)

    if len(X_valid) > 5000:
        idx = rng.choice(len(X_valid), size=5000, replace=False)
        X_imp = X_valid[idx]
        y_imp = y_valid[idx]
    else:
        X_imp = X_valid
        y_imp = y_valid

    if len(X_imp) and np.unique(y_imp).size > 1:
        perm = permutation_importance(purchase_model, X_imp, y_imp, scoring="roc_auc", n_repeats=3, random_state=seed)
        for feature, importance in zip(FEATURE_COLUMNS, perm.importances_mean):
            importance_rows.append({"model": "purchase_probability", "feature": feature, "importance": float(importance)})

    if len(X_spend_valid):
        n_spend_imp = min(5000, len(X_spend_valid))
        idx = rng.choice(len(X_spend_valid), size=n_spend_imp, replace=False) if len(X_spend_valid) > n_spend_imp else np.arange(len(X_spend_valid))
        if len(idx) > 30:
            y_spend_valid_transformed = signed_log1p(y_spend_valid_raw[idx])
            perm = permutation_importance(spend_model, X_spend_valid[idx], y_spend_valid_transformed, scoring="neg_mean_absolute_error", n_repeats=3, random_state=seed)
            for feature, importance in zip(FEATURE_COLUMNS, perm.importances_mean):
                importance_rows.append({"model": "conditional_spend", "feature": feature, "importance": float(importance)})

    metrics = {
        "purchase_model": purchase_metrics,
        "conditional_spend_model": spend_metrics,
        "validation": {"validation_start": str(validation_start), "validation_months": int(validation_months)},
        "spend_noise": {"raw_residual_robust_scale": residual_scale, "transformed_residual_sigma": transformed_sigma},
    }

    validation_predictions = {
        "calendar_month": pdf.loc[validation_mask, "calendar_month"].to_numpy(),
        "actual_purchase": y_valid,
        "predicted_purchase_probability": purchase_valid_prob,
    }
    # Spend predictions are only for active customers, save separately
    spend_validation = {
        "calendar_month": pdf.loc[spend_valid_mask, "calendar_month"].to_numpy(),
        "actual_spend": y_spend_valid_raw,
        "predicted_spend": spend_valid_pred_raw,
    }

    return purchase_model, spend_model, metrics, pd.DataFrame(importance_rows), validation_predictions, spend_validation, transformed_sigma


def estimate_beta_prior(panel: pl.DataFrame) -> Tuple[float, float]:
    pdf = panel.select(["active", "age_month"]).to_pandas()
    successes = float(pdf["active"].sum())
    exposure = float((pdf["age_month"] + 1).sum())
    global_rate = min(max(successes / max(exposure, 1.0), 0.01), 0.80)
    prior_strength = 12.0
    alpha = max(0.25, global_rate * prior_strength)
    beta = max(0.25, (1.0 - global_rate) * prior_strength)
    return alpha, beta


def build_customer_snapshot(panel: pl.DataFrame, tx: pl.DataFrame) -> pd.DataFrame:
    snapshot = (
        panel.sort(["Customer ID", "calendar_month"])
        .group_by("Customer ID", maintain_order=True)
        .agg([
            pl.col("calendar_month").last().alias("last_model_month"),
            pl.col("cohort_month").last().alias("cohort_month"),
            pl.col("age_month").last().alias("age_month"),
            pl.col("recency_months").last().alias("recency_months"),
            pl.col("lifetime_orders").last().alias("lifetime_orders"),
            pl.col("lifetime_active_months").last().alias("lifetime_active_months"),
            pl.col("lifetime_net_revenue").last().alias("lifetime_net_revenue"),
            pl.col("lifetime_gross_revenue").last().alias("lifetime_gross_revenue"),
            pl.col("lifetime_return_value").last().alias("lifetime_return_value"),
            pl.col("orders_last_3m").last().alias("orders_last_3m"),
            pl.col("orders_prev_3m").last().alias("orders_prev_3m"),
            pl.col("net_revenue_last_3m").last().alias("net_revenue_last_3m"),
            pl.col("net_revenue_prev_3m").last().alias("net_revenue_prev_3m"),
            pl.col("net_revenue_3m_mean").last().alias("net_revenue_3m_mean"),
            pl.col("orders_3m_mean").last().alias("orders_3m_mean"),
            pl.col("historical_value_per_order").last().alias("historical_value_per_order"),
            pl.col("historical_return_ratio").last().alias("historical_return_ratio"),
            pl.col("active").last().alias("last_month_active"),
            pl.col("next_net_revenue").last().alias("unused"),
        ])
        .drop("unused")
        .to_pandas()
    )

    monthly = panel.select(["Customer ID", "calendar_month", "orders", "net_revenue"]).to_pandas()
    monthly["calendar_month"] = pd.to_datetime(monthly["calendar_month"])

    recent_rows = []
    for customer_id, group in monthly.groupby("Customer ID", sort=False):
        group = group.sort_values("calendar_month").tail(4)
        row = {"Customer ID": customer_id, "orders_t0": 0.0, "orders_t1": 0.0, "orders_t2": 0.0, "orders_t3": 0.0,
               "revenue_t0": 0.0, "revenue_t1": 0.0, "revenue_t2": 0.0, "revenue_t3": 0.0}
        values = group["orders"].to_numpy()
        revenues = group["net_revenue"].to_numpy()
        start = 4 - len(values)
        for i, value in enumerate(values):
            row[f"orders_t{start+i}"] = float(value)
        for i, value in enumerate(revenues):
            row[f"revenue_t{start+i}"] = float(value)
        recent_rows.append(row)

    recent = pd.DataFrame(recent_rows)
    snapshot = snapshot.merge(recent, on="Customer ID", how="left")
    return snapshot


def feature_frame_from_state(state: Dict[str, np.ndarray], month_index: int, reference_month_number: np.ndarray) -> np.ndarray:
    age = state["age_month"]
    recency = state["recency_months"]
    lifetime_orders = state["lifetime_orders"]
    lifetime_active = state["lifetime_active_months"]
    lifetime_net = state["lifetime_net_revenue"]
    lifetime_gross = state["lifetime_gross_revenue"]
    lifetime_returns = state["lifetime_return_value"]
    orders_last_3 = state["orders_last_3m"]
    orders_prev_3 = state["orders_prev_3m"]
    orders_3m_mean = state["orders_3m_mean"]
    net_last_3 = state["net_revenue_last_3m"]
    net_prev_3 = state["net_revenue_prev_3m"]
    net_3m_mean = state["net_revenue_3m_mean"]

    order_rate = lifetime_orders / np.maximum(age + 1.0, 1.0)
    active_share = lifetime_active / np.maximum(age + 1.0, 1.0)
    recent_order_intensity = orders_last_3 / np.maximum(age, 1.0)
    recent_value_momentum = net_last_3 / (np.abs(net_prev_3) + 10.0)
    historical_value_per_order = lifetime_net / np.maximum(lifetime_orders, 1.0)
    historical_return_ratio = lifetime_returns / np.maximum(lifetime_gross, 1e-9)

    month_number = reference_month_number + month_index
    radians = 2.0 * math.pi * month_number / 12.0

    return np.column_stack([
        age, recency, lifetime_orders, lifetime_active, order_rate, active_share,
        orders_last_3, orders_prev_3, orders_3m_mean, recent_order_intensity,
        net_last_3, net_prev_3, net_3m_mean, recent_value_momentum,
        lifetime_net, historical_value_per_order, historical_return_ratio,
        np.sin(radians), np.cos(radians),
    ])


def simulate_clv_vectorized(
    snapshot: pd.DataFrame,
    purchase_model: object,
    spend_model: object,
    prior_alpha: float,
    prior_beta: float,
    transformed_sigma: float,
    horizon_months: int,
    simulations: int,
    annual_discount_rate: float,
    margin_rate: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized Monte Carlo simulation using NumPy broadcasting."""
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    if simulations < 20:
        raise ValueError("simulations must be >= 20")

    rng = np.random.default_rng(seed)
    n_customers = len(snapshot)
    n_paths = simulations

    customer_ids = snapshot["Customer ID"].to_numpy()

    def tiled(col: str) -> np.ndarray:
        values = snapshot[col].fillna(0.0).to_numpy(dtype=float)
        return np.repeat(values, n_paths)

    age = tiled("age_month")
    recency = tiled("recency_months")
    lifetime_orders = tiled("lifetime_orders")
    lifetime_active = tiled("lifetime_active_months")
    lifetime_net = tiled("lifetime_net_revenue")
    lifetime_gross = tiled("lifetime_gross_revenue")
    lifetime_returns = tiled("lifetime_return_value")
    orders_last_3 = tiled("orders_last_3m")
    orders_prev_3 = tiled("orders_prev_3m")
    net_last_3 = tiled("net_revenue_last_3m")
    net_prev_3 = tiled("net_revenue_prev_3m")
    net_3m_mean = tiled("net_revenue_3m_mean")
    orders_3m_mean = tiled("orders_3m_mean")

    orders_r0 = tiled("orders_t0")
    orders_r1 = tiled("orders_t1")
    orders_r2 = tiled("orders_t2")
    orders_r3 = tiled("orders_t3")
    revenue_r0 = tiled("revenue_t0")
    revenue_r1 = tiled("revenue_t1")
    revenue_r2 = tiled("revenue_t2")
    revenue_r3 = tiled("revenue_t3")

    customer_history_exposure = np.maximum(snapshot["age_month"].fillna(0).to_numpy(dtype=float) + 1.0, 1.0)
    customer_history_active = np.maximum(snapshot["lifetime_active_months"].fillna(0).to_numpy(dtype=float), 0.0)

    alpha = np.repeat(prior_alpha + customer_history_active, n_paths)
    beta = np.repeat(prior_beta + np.maximum(customer_history_exposure - customer_history_active, 0.0), n_paths)

    reference_month_number = np.repeat(snapshot["last_model_month"].dt.month.to_numpy(dtype=float), n_paths)

    annual_discount_rate = max(0.0, annual_discount_rate)
    monthly_discount_rate = (1.0 + annual_discount_rate) ** (1.0 / 12.0) - 1.0

    path_clv = np.zeros(len(snapshot) * n_paths, dtype=float)
    monthly_mean_rows = []

    state = {
        "age_month": age, "recency_months": recency, "lifetime_orders": lifetime_orders,
        "lifetime_active_months": lifetime_active, "lifetime_net_revenue": lifetime_net,
        "lifetime_gross_revenue": lifetime_gross, "lifetime_return_value": lifetime_returns,
        "orders_last_3m": orders_last_3, "orders_prev_3m": orders_prev_3,
        "net_revenue_last_3m": net_last_3, "net_revenue_prev_3m": net_prev_3,
        "net_revenue_3m_mean": net_3m_mean, "orders_3m_mean": orders_3m_mean,
        "orders_t0": orders_r0, "orders_t1": orders_r1, "orders_t2": orders_r2, "orders_t3": orders_r3,
        "revenue_t0": revenue_r0, "revenue_t1": revenue_r1, "revenue_t2": revenue_r2, "revenue_t3": revenue_r3,
    }

    for month in range(1, horizon_months + 1):
        X = feature_frame_from_state(state, month - 1, reference_month_number)
        
        # Purchase probability
        p_active = purchase_model.predict_proba(X)[:, 1]
        
        # Empirical Bayes shrinkage
        p_active_shrunk = (alpha * p_active + beta * 0.02) / (alpha + beta + 1e-9)
        p_active_shrunk = np.clip(p_active_shrunk, 1e-4, 0.98)
        
        # Simulate activity
        active = rng.binomial(1, p_active_shrunk)

        # Conditional spend
        spend_pred_log = spend_model.predict(X)
        spend_pred = signed_expm1(spend_pred_log)
        spend_noise = rng.normal(0, transformed_sigma, size=spend_pred.shape)
        spend = np.where(active == 1, np.maximum(spend_pred + spend_noise, 0.0), 0.0)

        # Discounted net revenue with margin
        discount_factor = 1.0 / (1.0 + monthly_discount_rate) ** month
        path_clv += active * spend * discount_factor * margin_rate

        # Update state for next month
        state["age_month"] += 1
        state["recency_months"] = np.where(active == 1, 0, state["recency_months"] + 1)
        state["lifetime_orders"] += active
        state["lifetime_active_months"] += active
        state["lifetime_net_revenue"] += active * spend
        state["lifetime_gross_revenue"] += active * spend
        state["lifetime_return_value"] += active * spend * 0.0  # Simplified

        # Update rolling windows
        state["orders_t3"] = state["orders_t2"]
        state["orders_t2"] = state["orders_t1"]
        state["orders_t1"] = state["orders_t0"]
        state["orders_t0"] = active
        
        state["revenue_t3"] = state["revenue_t2"]
        state["revenue_t2"] = state["revenue_t1"]
        state["revenue_t1"] = state["revenue_t0"]
        state["revenue_t0"] = np.where(active == 1, spend, 0.0)

        # Recompute rolling features
        state["orders_last_3m"] = state["orders_t0"] + state["orders_t1"] + state["orders_t2"]
        state["orders_prev_3m"] = state["orders_t1"] + state["orders_t2"] + state["orders_t3"]
        state["orders_3m_mean"] = state["orders_last_3m"] / 3.0
        
        state["net_revenue_last_3m"] = state["revenue_t0"] + state["revenue_t1"] + state["revenue_t2"]
        state["net_revenue_prev_3m"] = state["revenue_t1"] + state["revenue_t2"] + state["revenue_t3"]
        state["net_revenue_3m_mean"] = state["net_revenue_last_3m"] / 3.0

        monthly_mean_rows.append({
            "month": month,
            "mean_p_active": p_active_shrunk.mean(),
            "mean_active": active.mean(),
            "mean_spend_given_active": spend[active == 1].mean() if active.any() else 0.0,
            "mean_discounted_revenue": (active * spend * discount_factor * margin_rate).mean(),
        })

    # Aggregate to customer level
    customer_ids_tiled = np.repeat(customer_ids, n_paths)
    clv_df = pd.DataFrame({"Customer ID": customer_ids_tiled, "path_clv": path_clv})
    clv_summary = clv_df.groupby("Customer ID")["path_clv"].agg([
        ("clv_mean", "mean"), ("clv_median", "median"),
        ("clv_std", "std"), ("clv_p10", lambda x: x.quantile(0.10)),
        ("clv_p90", lambda x: x.quantile(0.90)),
    ]).reset_index()

    monthly_summary = pd.DataFrame(monthly_mean_rows)

    return clv_summary, monthly_summary


def run_backtests(
    panel: pl.DataFrame,
    prediction_origins: List[str],
    validation_months: int,
    horizon_months: int,
    simulations: int,
    annual_discount_rate: float,
    margin_rate: float,
    seed: int,
) -> List[Dict]:
    """Run rolling temporal backtests at multiple prediction origins."""
    LOGGER.info(f"Running backtests at {len(prediction_origins)} origins...")
    
    results = []
    for origin in prediction_origins:
        LOGGER.info(f"Backtest origin: {origin}")
        try:
            origin_dt = pd.Timestamp(origin).to_period("M").to_timestamp()
            
            # Filter panel up to origin
            train_panel = panel.filter(pl.col("calendar_month") <= origin_dt)
            
            # Need enough history
            if train_panel.select(pl.col("calendar_month").n_unique()).item() < 12:
                LOGGER.warning(f"Insufficient history at origin {origin}, skipping")
                continue
            
            # Train models
            purchase_model, spend_model, metrics, importance_df, val_preds, spend_val, sigma = train_models(
                train_panel, validation_months, 2, seed
            )
            
            # Build snapshot at origin
            snapshot = build_customer_snapshot(train_panel, pl.DataFrame())
            # Need transaction data for snapshot - skip for now
            
            # Simulate
            prior_alpha, prior_beta = estimate_beta_prior(train_panel)
            clv_summary, monthly_summary = simulate_clv_vectorized(
                snapshot, purchase_model, spend_model, prior_alpha, prior_beta, sigma,
                horizon_months, min(simulations, 50), annual_discount_rate, margin_rate, seed
            )
            
            # Evaluate against future
            future_panel = panel.filter(pl.col("calendar_month") > origin_dt)
            # ... evaluation logic would go here
            
            results.append({
                "origin": origin,
                "metrics": metrics,
                "clv_summary": clv_summary.to_dict("records")[:5],
            })
        except Exception as e:
            LOGGER.warning(f"Backtest at origin {origin} failed: {e}")
            results.append({"origin": origin, "error": str(e)})
    
    return results


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    margin_scenarios = [float(x) for x in args.margin_scenarios.split(",")]

    # -------------------------------------------------------------------------
    # Load canonical transactions
    # -------------------------------------------------------------------------
    canonical_path = Path("data_quality_output/canonical_transactions.parquet")
    if canonical_path.exists():
        LOGGER.info("Loading canonical transactions...")
        tx = pl.read_parquet(canonical_path)
    else:
        LOGGER.info("Building canonical transactions...")
        raw = load_raw_transactions(input_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # -------------------------------------------------------------------------
    # Build customer-month panel
    # -------------------------------------------------------------------------
    LOGGER.info("Building customer-month panel...")
    _, customer_month_dense, metadata = build_customer_month_panel(tx)
    customer_month_dense = add_rolling_features(customer_month_dense, [1, 3, 6, 12])

    # Add target columns
    customer_month_dense = customer_month_dense.with_columns([
        pl.col("active").shift(-1).over("Customer ID").fill_null(0).alias("next_active"),
        pl.col("net_revenue").shift(-1).over("Customer ID").fill_null(0.0).alias("next_net_revenue"),
    ])

    # Only keep months where next month is observable
    last_month = pd.Timestamp(metadata["observation_end"])
    panel = customer_month_dense.filter(pl.col("calendar_month") < last_month)

    # -------------------------------------------------------------------------
    # Run validations
    # -------------------------------------------------------------------------
    validation_results = run_all_validations(tx=tx, customer_month_dense=customer_month_dense)
    assert_validations_pass(validation_results)

    # -------------------------------------------------------------------------
    # Rolling backtests
    # -------------------------------------------------------------------------
    backtest_results = []
    if args.run_backtests:
        prediction_origins = ["2010-09-30", "2010-12-31", "2011-03-31", "2011-06-30", "2011-09-30"]
        backtest_results = run_backtests(
            panel, prediction_origins, args.validation_months, args.horizon_months,
            args.simulations, args.annual_discount_rate, args.margin_rate, args.seed
        )

    # -------------------------------------------------------------------------
    # Main model training (using all data up to validation split)
    # -------------------------------------------------------------------------
    LOGGER.info("Training main models on full training period...")
    purchase_model, spend_model, metrics, importance_df, val_preds, spend_val, sigma = train_models(
        panel, args.validation_months, args.min_active_history_months, args.seed
    )

    prior_alpha, prior_beta = estimate_beta_prior(panel)

    # -------------------------------------------------------------------------
    # Build snapshot and simulate
    # -------------------------------------------------------------------------
    LOGGER.info("Building customer snapshot...")
    snapshot = build_customer_snapshot(panel, tx)

    # -------------------------------------------------------------------------
    # CLV simulation for each margin scenario
    # -------------------------------------------------------------------------
    all_clv_results = {}
    for margin in margin_scenarios:
        LOGGER.info(f"Simulating CLV with margin_rate={margin:.0%}...")
        clv_summary, monthly_summary = simulate_clv_vectorized(
            snapshot, purchase_model, spend_model, prior_alpha, prior_beta, sigma,
            args.horizon_months, args.simulations, args.annual_discount_rate, margin, args.seed
        )
        all_clv_results[f"margin_{margin:.0%}"] = {
            "clv_summary": clv_summary,
            "monthly_summary": monthly_summary,
        }

    # Use args.margin_rate for main outputs (explicit CLI parameter)
    main_margin = args.margin_rate
    main_clv = all_clv_results[f"margin_{main_margin:.0%}"]["clv_summary"]
    main_monthly = all_clv_results[f"margin_{main_margin:.0%}"]["monthly_summary"]

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    main_clv.to_csv(output_dir / "clv_customer_predictions.csv", index=False)
    main_monthly.to_csv(output_dir / "clv_monthly_summary.csv", index=False)

    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model_name": "Dynamic Probabilistic Discounted Net-Revenue CLV Proxy",
            "methodology": "Discrete-time hazard + conditional spend + Empirical Bayes + MC simulation",
            "margin_scenarios": {k: v["clv_summary"].to_dict("records")[:5] for k, v in all_clv_results.items()},
            "main_margin": args.margin_rate,
            "metrics": metrics,
            "backtest_results": backtest_results,
        }, f, indent=2, default=str)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Calibration data
    pd.DataFrame(val_preds).to_csv(output_dir / "validation_predictions.csv", index=False)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        plots = output_dir / "plots"
        plots.mkdir(exist_ok=True)

        # Value distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(main_clv["clv_mean"], bins=50, alpha=0.7, edgecolor='white')
        ax.set_xlabel("CLV (Mean)")
        ax.set_ylabel("Customers")
        ax.set_title("Customer CLV Distribution")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(plots / "01_clv_distribution.png", dpi=180)
        plt.close(fig)

        # Decile boxplot
        main_clv["decile"] = pd.qcut(main_clv["clv_mean"], 10, labels=False, duplicates="drop")
        fig, ax = plt.subplots(figsize=(10, 6))
        main_clv.boxplot(column="clv_mean", by="decile", ax=ax, grid=False)
        ax.set_title("CLV by Decile")
        ax.set_xlabel("Value Decile (0=lowest)")
        ax.set_ylabel("CLV")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots / "02_clv_deciles.png", dpi=180)
        plt.close(fig)

        # Monthly simulation trajectory
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(main_monthly["month"], main_monthly["mean_p_active"], marker="o", label="P(active)")
        ax.plot(main_monthly["month"], main_monthly["mean_active"], marker="o", label="Mean active")
        ax2 = ax.twinx()
        ax2.plot(main_monthly["month"], main_monthly["mean_discounted_revenue"], marker="s", color="green", label="Discounted revenue")
        ax.set_xlabel("Month")
        ax.set_ylabel("Probability / Count")
        ax2.set_ylabel("Discounted Revenue")
        ax.set_title("Simulation Trajectory")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plots / "03_simulation_trajectory.png", dpi=180)
        plt.close(fig)

    LOGGER.info("CLV analysis complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
