#!/usr/bin/env python3
"""
Reactivation Model — Inactive Customer Return Prediction

This script predicts whether an inactive customer will return within a specified
horizon using point-in-time features and temporal validation.

Target: customer inactive at t → purchases again within N months

Features:
- Recency, historical cadence, cadence volatility
- Prior reactivation count, past revenue, CLV proxy
- Segment, assortment, returns, seasonality
- Point-in-time safe (only data available at prediction date)

Uses the retail_ds shared package for canonical data processing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import QuantileTransformer

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel, add_rolling_features
from retail_ds.features import build_point_in_time_features
from retail_ds.validation import run_all_validations, assert_validations_pass


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("reactivation_model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reactivation model for inactive customers.")
    parser.add_argument("--input", default="./data_xslx/online_retail_II.xlsx", help="Input file.")
    parser.add_argument("--output-dir", default="./reactivation_output", help="Output directory.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--horizon-months", type=int, default=3, help="Reactivation horizon in months.")
    parser.add_argument("--inactive-threshold-months", type=int, default=3, help="Months of inactivity to define inactive.")
    parser.add_argument("--validation-months", type=int, default=3, help="Trailing months for validation.")
    parser.add_argument("--min-customers", type=int, default=50, help="Minimum customers for training.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, floor: float = 1e-9) -> pl.Expr:
    return pl.when(denominator.abs() > floor).then(numerator / denominator).otherwise(0.0)


def build_reactivation_panel(
    tx: pl.DataFrame,
    horizon_months: int,
    inactive_threshold: int,
) -> Tuple[pl.DataFrame, Dict, pd.Timestamp]:
    """
    Build customer-month panel with reactivation targets.

    Target: inactive customer at t → active within horizon_months
    """
    LOGGER.info("Building reactivation modeling panel...")

    _, customer_month_dense, metadata = build_customer_month_panel(tx)
    customer_month_dense = add_rolling_features(customer_month_dense, [1, 3, 6, 12])

    observation_end = pd.Timestamp(metadata["observation_end"])

    # Add month_end_date
    customer_month_dense = customer_month_dense.with_columns(
        (pl.col("calendar_month") + pl.duration(days=32)).dt.truncate("1mo").alias("month_end_date"),
    )

    # Define inactive at current month: active=0 AND was active previously
    customer_month_dense = customer_month_dense.with_columns(
        [
            pl.col("active").shift(1).over("Customer ID").fill_null(0).cast(pl.Int8).alias("prev_active"),
            pl.col("active").shift(-1).over("Customer ID").fill_null(0).cast(pl.Int8).alias("next_active"),
        ]
    )

    # Inactive definition: currently inactive, was active at some point, and has been inactive for >= threshold months
    customer_month_dense = customer_month_dense.with_columns(
        pl.when(
            (pl.col("active") == 0) & 
            (pl.col("lifetime_active_months") > 0) &  # Had at least one active month in past
            (pl.col("consecutive_inactive_months") >= inactive_threshold)
        ).then(1).otherwise(0).alias("is_inactive_candidate"),
    )

    # Target: becomes active within horizon_months
    # We need to check if any of the next horizon_months have active=1
    reactivation_target = pl.lit(0)
    for h in range(1, horizon_months + 1):
        reactivation_target = reactivation_target | (
            pl.col("active").shift(-h).over("Customer ID").fill_null(0).cast(pl.Int8)
        )

    customer_month_dense = customer_month_dense.with_columns(
        reactivation_target.alias("reactivates_within_horizon"),
    )

    # Only keep rows where customer is an inactive candidate
    panel_model = customer_month_dense.filter(
        (pl.col("is_inactive_candidate") == 1) & 
        (pl.col("calendar_month") < customer_month_dense.select(pl.col("calendar_month").max()).item())
    )

    LOGGER.info(f"Reactivation panel: {panel_model.height:,} inactive-candidate months, {panel_model.select('Customer ID').n_unique():,} customers")
    return panel_model, metadata, observation_end


def prepare_reactivation_features(panel: pl.DataFrame) -> List[str]:
    """Select feature columns for reactivation model."""
    exclude_cols = {
        "Customer ID", "calendar_month", "cohort_month", "calendar_month_id",
        "cohort_month_id", "age_month", "is_inactive_candidate", "reactivates_within_horizon",
        "next_active", "prev_active", "month_end_date", "consecutive_inactive_months",
        "last_invoice_date_month", "last_purchase_date_asof_month", "month_end_proxy",
        "inactive_indicator", "inactive_run_counter", "last_active_counter",
        "reactivation_event", "churn_transition",
    }
    feature_cols = [
        c for c in panel.columns 
        if c not in exclude_cols 
        and panel.schema[c] in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
    ]

    # Remove features with too much null or zero variance
    final_features = []
    for c in feature_cols:
        null_frac = panel.select(pl.col(c).is_null().mean()).item()
        if null_frac > 0.5:
            continue
        var = panel.select(pl.col(c).cast(pl.Float64).fill_null(0).var()).item()
        if var is None or not np.isfinite(var) or var <= 1e-12:
            continue
        final_features.append(c)

    LOGGER.info(f"Reactivation features: {len(final_features)}")
    return final_features


def temporal_split(
    panel: pl.DataFrame,
    test_months: int,
    validation_months: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split panel by calendar month."""
    months = sorted(panel.select(pl.col("calendar_month").unique()).to_series().to_list())
    months = [pd.Timestamp(m) for m in months]

    if len(months) < validation_months + test_months + 6:
        raise ValueError("Not enough monthly observations for requested temporal split.")

    test_month_values = months[-test_months:]
    validation_month_values = months[-(test_months + validation_months):-test_months]
    train_month_values = months[:-(test_months + validation_months)]

    train_df = panel.filter(pl.col("calendar_month").is_in(train_month_values))
    val_df = panel.filter(pl.col("calendar_month").is_in(validation_month_values))
    test_df = panel.filter(pl.col("calendar_month").is_in(test_month_values))

    LOGGER.info(f"Train: {train_df.height:,} | Val: {val_df.height:,} | Test: {test_df.height:,}")
    return train_df, val_df, test_df


def train_calibrated_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Tuple[HistGradientBoostingClassifier, QuantileTransformer, LogisticRegression, Dict]:
    """Train and calibrate model."""
    qt = QuantileTransformer(n_quantiles=min(1000, max(50, X_train.shape[0] // 20)), output_distribution="normal", subsample=10000, random_state=SEED)
    X_train_t = qt.fit_transform(X_train)
    X_val_t = qt.transform(X_val)

    # Class weights for imbalance
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
        class_weight={0: 1.0, 1: float(pos_weight)},
    )
    model.fit(X_train_t, y_train)

    p_val_raw = model.predict_proba(X_val_t)[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    calibrator.fit(p_val_raw.reshape(-1, 1), y_val)
    p_val = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
        "pr_auc": float(average_precision_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
        "brier": float(brier_score_loss(y_val, p_val)),
        "log_loss": float(log_loss(y_val, np.clip(p_val, 1e-7, 1-1e-7))) if len(np.unique(y_val)) > 1 else None,
        "val_positive_rate": float(y_val.mean()),
        "val_predicted_rate": float(p_val.mean()),
    }

    return model, qt, calibrator, metrics


def save_plots(
    output_dir: Path,
    y_val: np.ndarray, p_val: np.ndarray,
    y_test: np.ndarray, p_test: np.ndarray,
    feature_names: List[str],
    importance: pd.DataFrame,
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Calibration
    frac_pos_val, mean_pred_val = calibration_curve(y_val, p_val, n_bins=10, strategy="quantile")
    frac_pos_test, mean_pred_test = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    ax.plot(mean_pred_val, frac_pos_val, marker="o", linewidth=2, label="Validation")
    ax.plot(mean_pred_test, frac_pos_test, marker="o", linewidth=2, label="Test")
    ax.set_title("Reactivation Probability Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "01_reactivation_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    top_imp = importance.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"])
    ax.set_title("Reactivation Model — Permutation Importance")
    ax.set_xlabel("Average precision decrease after permutation")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "02_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Score distribution
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(p_test, bins=30, alpha=0.80)
    ax.set_title("Distribution of Reactivation Probability (Test)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Inactive candidate months")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "03_reactivation_probability_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

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
    # Build reactivation panel
    # -------------------------------------------------------------------------
    panel, metadata, observation_end = build_reactivation_panel(
        tx, args.horizon_months, args.inactive_threshold_months
    )

    # -------------------------------------------------------------------------
    # Run validations
    # -------------------------------------------------------------------------
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # -------------------------------------------------------------------------
    # Feature selection and temporal split
    # -------------------------------------------------------------------------
    feature_cols = prepare_reactivation_features(panel)
    train_df, val_df, test_df = temporal_split(panel, args.validation_months, args.validation_months)

    # -------------------------------------------------------------------------
    # Train reactivation model
    # -------------------------------------------------------------------------
    LOGGER.info("Training reactivation model...")
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("reactivates_within_horizon").to_numpy().flatten()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("reactivates_within_horizon").to_numpy().flatten()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select("reactivates_within_horizon").to_numpy().flatten()

    if y_train.sum() < 10 or (len(y_train) - y_train.sum()) < 10:
        raise ValueError("Insufficient class balance for reactivation target")

    model, qt, calibrator, metrics = train_calibrated_model(X_train, y_train, X_val, y_val)

    X_test_t = qt.transform(X_test)
    p_test_raw = model.predict_proba(X_test_t)[:, 1]
    p_test = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, p_test)),
        "log_loss": float(log_loss(y_test, np.clip(p_test, 1e-7, 1-1e-7))) if len(np.unique(y_test)) > 1 else None,
    }

    # Validation predictions for calibration plot
    X_val_t = qt.transform(X_val)
    p_val_raw = model.predict_proba(X_val_t)[:, 1]
    p_val = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------
    importance = permutation_importance(
        model, X_val_t, y_val,
        n_repeats=3, scoring="average_precision", random_state=SEED,
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # -------------------------------------------------------------------------
    # Customer-level predictions at latest inactive state
    # -------------------------------------------------------------------------
    # For each customer, get their latest inactive candidate month
    latest_inactive = panel.filter(
        pl.col("is_inactive_candidate") == 1
    ).sort(["Customer ID", "calendar_month"]).group_by("Customer ID").tail(1)

    X_latest = latest_inactive.select(feature_cols).to_numpy()
    X_latest_t = qt.transform(X_latest)
    reactivation_prob = calibrator.predict_proba(model.predict_proba(X_latest_t)[:, 1].reshape(-1, 1))[:, 1]

    predictions = pd.DataFrame({
        "Customer ID": latest_inactive.select("Customer ID").to_numpy().flatten(),
        "calendar_month": latest_inactive.select("calendar_month").to_numpy().flatten(),
        "reactivation_probability": reactivation_prob,
    })

    # Add customer-level features for context
    customer_context = latest_inactive.select(["Customer ID", "lifetime_orders", "lifetime_gross_revenue", "recency_months", "consecutive_inactive_months"]).to_pandas()
    predictions = predictions.merge(customer_context, on="Customer ID", how="left")

    predictions.to_csv(output_dir / "reactivation_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # Model card
    # -------------------------------------------------------------------------
    model_card = {
        "model_name": "Inactive Customer Reactivation Model",
        "target": f"Reactivates within {args.horizon_months} months",
        "inactive_definition": f"Inactive for >= {args.inactive_threshold_months} months with prior activity",
        "validation": metrics,
        "test": test_metrics,
        "features": feature_cols,
        "horizon_months": args.horizon_months,
        "inactive_threshold_months": args.inactive_threshold_months,
    }
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Panel
    panel.write_parquet(output_dir / "reactivation_panel.parquet")

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, y_val, p_val, y_test, p_test, feature_cols, importance_df)

    LOGGER.info("Reactivation model complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    main()