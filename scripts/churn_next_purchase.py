#!/usr/bin/env python3
"""
Customer Churn / Next-Purchase Intelligence — Fixed target alignment.

This script fixes the critical IndexError by building targets and features
from the SAME customer-month panel before temporal splitting.

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
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("churn_next_purchase")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer churn/next-purchase modeling.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--test-months", type=int, default=3, help="Final calendar months reserved for test.")
    parser.add_argument("--validation-months", type=int, default=2, help="Months before test reserved for validation.")
    parser.add_argument("--horizon-months", type=int, default=12, help="Maximum monthly survival horizon.")
    parser.add_argument("--windows", default="1,3,6,12", help="Rolling windows in months.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, floor: float = 1e-9) -> pl.Expr:
    return pl.when(denominator.abs() > floor).then(numerator / denominator).otherwise(0.0)


def build_modeling_panel(
    tx: pl.DataFrame,
    windows: List[int],
) -> Tuple[pl.DataFrame, Dict, pd.Timestamp]:
    """
    Build a single customer-month panel with features AND targets.
    This ensures feature rows and target rows are perfectly aligned.
    """
    LOGGER.info("Building modeling panel with aligned features and targets...")

    _, customer_month_dense, metadata = build_customer_month_panel(tx)
    customer_month_dense = add_rolling_features(customer_month_dense, windows)

    observation_end = pd.Timestamp(metadata["observation_end"])

    # Add month_end_date for next-purchase target computation
    customer_month_dense = customer_month_dense.with_columns(
        (pl.col("calendar_month") + pl.duration(days=32)).dt.truncate("1mo").alias("month_end_date"),
    )

    # -------------------------------------------------------------------------
    # FORWARD TARGET CONSTRUCTION (in the SAME panel)
    # -------------------------------------------------------------------------
    # Next-month purchase (churn target)
    customer_month_dense = customer_month_dense.with_columns(
        pl.col("active").shift(-1).over("Customer ID").fill_null(0).cast(pl.Int8).alias("purchase_next_month"),
    )

    LOGGER.info(f"Modeling panel: {customer_month_dense.height:,} rows, {customer_month_dense.select('Customer ID').n_unique():,} customers")
    return customer_month_dense, metadata, observation_end


def add_next_purchase_targets(
    panel: pl.DataFrame,
    tx: pl.DataFrame,
    horizons: List[int] = [7, 30, 60],
) -> pl.DataFrame:
    """Add exact next-purchase targets using invoice dates — all in Polars."""
    sales = tx.filter(pl.col("is_clean_sale"))
    
    # Get invoice dates per customer
    invoices = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(pl.col("InvoiceDate").min().alias("invoice_date"))
        .sort(["Customer ID", "invoice_date"])
    )

    # For each invoice, find the next invoice date for the same customer
    invoices = invoices.with_columns(
        pl.col("invoice_date").shift(-1).over("Customer ID").alias("next_invoice_date"),
    )

    # Use join_asof to find the last invoice ON OR BEFORE month_end_date
    # and get its next_invoice_date
    panel = panel.join_asof(
        invoices.select(["Customer ID", "invoice_date", "next_invoice_date"]),
        left_on="month_end_date",
        right_on="invoice_date",
        by="Customer ID",
        strategy="backward",
    )
    
    # The asof join gives us the last invoice ON OR BEFORE month_end_date
    # and its next_invoice_date. We can then compute days to next purchase.
    panel = panel.with_columns(
        (pl.col("next_invoice_date") - pl.col("month_end_date")).dt.total_days().alias("days_to_next_purchase"),
    )

    for h in horizons:
        panel = panel.with_columns(
            pl.when(pl.col("days_to_next_purchase").is_not_null() & (pl.col("days_to_next_purchase") <= h))
            .then(1).otherwise(0)
            .cast(pl.Int8).alias(f"next_purchase_{h}d"),
        )

    return panel


def prepare_model_data(
    panel: pl.DataFrame,
    test_months: int,
    validation_months: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    """
    Split panel into train/validation/test with perfectly aligned features and targets.
    """
    # Get unique months
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

    # Feature columns (exclude targets and identifiers)
    exclude_cols = {
        "Customer ID", "calendar_month", "cohort_month", "calendar_month_id",
        "cohort_month_id", "age_month", "purchase_next_month", "next_calendar_month",
        "month_end_date", "days_to_next_purchase",
        "last_invoice_date_month", "last_purchase_date_asof_month", "month_end_proxy",
        "inactive_indicator", "inactive_run_counter", "last_active_counter",
    }
    feature_cols = [
        c for c in panel.columns 
        if c not in exclude_cols 
        and not c.startswith("next_purchase_")
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

    LOGGER.info(f"Features: {len(final_features)} | Train: {train_df.height:,} | Val: {val_df.height:,} | Test: {test_df.height:,}")
    return train_df, val_df, test_df, final_features


def train_calibrated_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Tuple[HistGradientBoostingClassifier, QuantileTransformer, LogisticRegression, Dict]:
    """Train and calibrate model."""
    qt = QuantileTransformer(n_quantiles=min(1000, max(50, X_train.shape[0] // 20)), output_distribution="normal", subsample=10000, random_state=SEED)
    X_train_t = qt.fit_transform(X_train)
    X_val_t = qt.transform(X_val)

    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
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
    }

    return model, qt, calibrator, metrics


def build_survival_curves(
    model: HistGradientBoostingClassifier,
    qt: QuantileTransformer,
    panel: pl.DataFrame,
    calibrator: LogisticRegression,
    horizon: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Build survival curves from discrete-time hazard."""
    # Get latest month for each customer
    latest_month = panel.select(pl.col("calendar_month").max()).item()
    latest = panel.filter(pl.col("calendar_month") == latest_month)
    
    latest_pd = latest.select(feature_cols).to_pandas()
    latest_pd = latest_pd.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    X_latest_t = qt.transform(latest_pd)

    base_hazard = model.predict_proba(X_latest_t)[:, 1]
    base_hazard = calibrator.predict_proba(base_hazard.reshape(-1, 1))[:, 1]

    survival = np.ones_like(base_hazard)
    output = {"Customer ID": latest.select("Customer ID").to_numpy().flatten()}

    for month in range(1, horizon + 1):
        shrink = 1.0 - np.exp(-month / 8.0)
        pop_hazard = float(np.mean(base_hazard))
        hazard_t = (1.0 - 0.25 * shrink) * base_hazard + (0.25 * shrink) * pop_hazard
        hazard_t = np.clip(0.02 + (hazard_t - 0.02) / np.sqrt(1.0 + np.maximum(0, month) / 24.0), 1e-4, 0.98)
        survival *= 1.0 - hazard_t
        output[f"survival_month_{month}"] = survival.copy()

    return pd.DataFrame(output)


def save_plots(
    output_dir: Path,
    y_val: np.ndarray, p_val: np.ndarray,
    y_test: np.ndarray, p_test: np.ndarray,
    feature_names: List[str],
    importance: pd.DataFrame,
    survival_df: pd.DataFrame,
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
    ax.set_title("Next-Purchase Probability Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "01_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    top_imp = importance.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"])
    ax.set_title("Churn Model — Permutation Importance")
    ax.set_xlabel("Average precision decrease after permutation")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots / "02_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Survival curve
    survival_cols = [c for c in survival_df.columns if c.startswith("survival_month_")]
    survival_values = survival_df[survival_cols].mean(axis=0).to_numpy()
    months = np.arange(1, len(survival_values) + 1)
    q10 = survival_df[survival_cols].quantile(0.10, axis=0).to_numpy()
    q90 = survival_df[survival_cols].quantile(0.90, axis=0).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(months, survival_values, marker="o", linewidth=2, label="Mean predicted survival")
    ax.fill_between(months, q10, q90, alpha=0.20, label="Customer prediction band (10th-90th pct.)")
    ax.set_title("Predicted Customer Survival Curve")
    ax.set_xlabel("Months after forecast origin")
    ax.set_ylabel("Probability customer remains commercially active")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "03_survival_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import pandas as pd
    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.churn_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    windows = [int(w) for w in args.windows.split(",")]

    # -------------------------------------------------------------------------
    # Load canonical transactions
    # -------------------------------------------------------------------------
    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"
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
    # Build modeling panel with aligned features and targets
    # -------------------------------------------------------------------------
    panel, metadata, observation_end = build_modeling_panel(tx, windows)

    # Add next-purchase targets
    panel = add_next_purchase_targets(panel, tx, [7, 30, 60])

    # Only keep months where next month is observable (for churn target)
    last_month = pd.Timestamp(metadata["observation_end"])
    panel_model = panel.filter(pl.col("calendar_month") < last_month)

    # -------------------------------------------------------------------------
    # Temporal split with aligned features/targets
    # -------------------------------------------------------------------------
    train_df, val_df, test_df, feature_cols = prepare_model_data(
        panel_model, args.test_months, args.validation_months
    )

    # -------------------------------------------------------------------------
    # CHURN MODEL (next-month purchase)
    # -------------------------------------------------------------------------
    LOGGER.info("Training churn model...")
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("purchase_next_month").to_numpy().flatten()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("purchase_next_month").to_numpy().flatten()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select("purchase_next_month").to_numpy().flatten()

    churn_model, qt, calibrator, churn_metrics = train_calibrated_model(X_train, y_train, X_val, y_val)

    X_test_t = qt.transform(X_test)
    p_test_raw = churn_model.predict_proba(X_test_t)[:, 1]
    p_test = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    churn_test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, p_test)),
        "log_loss": float(log_loss(y_test, np.clip(p_test, 1e-7, 1-1e-7))) if len(np.unique(y_test)) > 1 else None,
    }

    # Validation predictions for calibration plot
    X_val_t = qt.transform(val_df.select(feature_cols).to_numpy())
    p_val_raw = churn_model.predict_proba(X_val_t)[:, 1]
    p_val = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]

    # -------------------------------------------------------------------------
    # SURVIVAL CURVES
    # -------------------------------------------------------------------------
    LOGGER.info("Building survival curves...")
    survival_df = build_survival_curves(churn_model, qt, panel_model, calibrator, args.horizon_months, feature_cols)

    # -------------------------------------------------------------------------
    # NEXT-PURCHASE MODELS
    # -------------------------------------------------------------------------
    next_purchase_results = {}
    for h in [7, 30, 60]:
        LOGGER.info(f"Training next-purchase {h}d model...")
        y_train_np = train_df.select(f"next_purchase_{h}d").to_numpy().flatten()
        y_val_np = val_df.select(f"next_purchase_{h}d").to_numpy().flatten()
        y_test_np = test_df.select(f"next_purchase_{h}d").to_numpy().flatten()

        if y_train_np.sum() < 10 or (len(y_train_np) - y_train_np.sum()) < 10:
            LOGGER.warning(f"Insufficient class balance for {h}d target, skipping")
            continue

        np_model, np_qt, np_calibrator, np_metrics = train_calibrated_model(X_train, y_train_np, X_val, y_val_np)
        X_test_t = np_qt.transform(X_test)
        np_p_test_raw = np_model.predict_proba(X_test_t)[:, 1]
        np_p_test = np_calibrator.predict_proba(np_p_test_raw.reshape(-1, 1))[:, 1]

        next_purchase_results[h] = {
            "model": np_model,
            "qt": np_qt,
            "calibrator": np_calibrator,
            "metrics": {
                "roc_auc": float(roc_auc_score(y_test_np, np_p_test)) if len(np.unique(y_test_np)) > 1 else None,
                "pr_auc": float(average_precision_score(y_test_np, np_p_test)) if len(np.unique(y_test_np)) > 1 else None,
                "brier": float(brier_score_loss(y_test_np, np_p_test)),
                "log_loss": float(log_loss(y_test_np, np.clip(np_p_test, 1e-7, 1-1e-7))) if len(np.unique(y_test_np)) > 1 else None,
            },
        }

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    importance = permutation_importance(
        churn_model,
        X_val_t,
        y_val,
        n_repeats=3,
        scoring="average_precision",
        random_state=SEED,
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # -------------------------------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")

    # Customer-level predictions at latest month
    latest_month = panel_model.select(pl.col("calendar_month").max()).item()
    latest = panel_model.filter(pl.col("calendar_month") == latest_month)

    X_latest = latest.select(feature_cols).to_numpy()
    X_latest_t = qt.transform(X_latest)
    churn_prob = calibrator.predict_proba(churn_model.predict_proba(X_latest_t)[:, 1].reshape(-1, 1))[:, 1]

    predictions = pd.DataFrame({
        "Customer ID": latest.select("Customer ID").to_numpy().flatten(),
        "churn_probability": churn_prob,
    })

    for h in [3, 6, 12]:
        if f"survival_month_{h}" in survival_df.columns:
            survival_vals = survival_df[f"survival_month_{h}"].values
            predictions[f"survival_{h}m"] = survival_vals

    for h in [7, 30, 60]:
        if h in next_purchase_results:
            res = next_purchase_results[h]
            X_latest_t_np = res["qt"].transform(X_latest)
            pred = res["calibrator"].predict_proba(res["model"].predict_proba(X_latest_t_np)[:, 1].reshape(-1, 1))[:, 1]
            predictions[f"next_purchase_{h}d_probability"] = pred

    predictions.to_csv(output_dir / "customer_churn_next_purchase.csv", index=False)

    # Model card
    model_card = {
        "churn": {
            "validation": churn_metrics,
            "test": churn_test_metrics,
            "features": feature_cols,
        },
        "next_purchase": {str(h): v["metrics"] for h, v in next_purchase_results.items()},
        "survival": {"horizon_months": args.horizon_months},
    }
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    # Panel
    panel.write_parquet(output_dir / "customer_month_panel.parquet")

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "observation_start": metadata["observation_start"],
        "observation_end": metadata["observation_end"],
        "files": [
            "customer_churn_next_purchase.csv",
            "customer_month_panel.parquet",
            "model_card.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, y_val, p_val, y_test, p_test, feature_cols, importance_df, survival_df)

    LOGGER.info("Churn/Next-Purchase complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    import pandas as pd
    main()
