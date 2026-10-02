#!/usr/bin/env python3
"""
BG/NBD Benchmark for CLV — Classical Probabilistic Lifetime Model

This script implements the Beta-Geometric/Negative Binomial Distribution (BG/NBD) model
as a benchmark against the current Monte Carlo CLV approach.

BG/NBD is a "Buy 'Til You Die" model that models:
- Transaction process: NBD (Negative Binomial) - number of transactions while alive
- Dropout process: BG (Beta-Geometric) - probability of customer being alive

This provides a classical probabilistic CLV baseline for comparison.

Key differences from current approach:
- BG/NBD: Explicit lifetime model with dropout probability
- Current: Discrete-time hazard + conditional spend + MC simulation
- BG/NBD: Analytical expected transactions (no simulation needed)
- Current: Simulation-based with dynamic feature updates
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args

try:
    from lifetimes import BetaGeoFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    BetaGeoFitter = None

SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("bgnbd_benchmark")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BG/NBD benchmark for CLV.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--eval-cutoff", default="2011-06-30", help="Temporal cutoff for evaluation (YYYY-MM-DD).")
    parser.add_argument("--horizon-days", type=int, default=730, help="Prediction horizon in days (default 2 years).")
    parser.add_argument("--annual-discount-rate", type=float, default=0.10, help="Annual discount rate.")
    parser.add_argument("--margin-rate", type=float, default=0.3, help="Margin rate for CLV calculation.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def prepare_rfm_data(
    tx: pl.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Prepare RFM (Recency, Frequency, Monetary) data for BG/NBD.
    
    BG/NBD requires:
    - frequency: number of repeat purchases (total - 1)
    - recency: days since first purchase to last purchase
    - T: days since first purchase to observation date
    - monetary_value: average monetary value per transaction
    """
    sales = tx.filter(
        (pl.col("is_sale") & pl.col("is_positive_price") & (pl.col("InvoiceDate") <= pl.lit(cutoff_date).str.strptime(pl.Datetime)))
    )
    
    if sales.height == 0:
        raise ValueError("No clean sales found for BG/NBD")
    
    # Invoice-level
    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
            ]
        )
    )
    
    # Customer-level
    customer = (
        invoice.group_by("Customer ID")
        .agg(
            [
                pl.col("invoice_date").min().alias("first_purchase"),
                pl.col("invoice_date").max().alias("last_purchase"),
                pl.len().alias("total_orders"),
                pl.col("invoice_revenue").sum().alias("total_revenue"),
                pl.col("invoice_revenue").mean().alias("avg_order_value"),
            ]
        )
    )
    
    cutoff_dt = pd.Timestamp(cutoff_date)
    
    # BG/NBD format
    rfm = customer.to_pandas()
    rfm["frequency"] = rfm["total_orders"] - 1  # repeat purchases
    rfm["recency"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["T"] = (cutoff_dt - rfm["first_purchase"]).dt.days
    rfm["monetary_value"] = rfm["avg_order_value"]
    
    # Filter: need at least 1 purchase (frequency >= 0)
    rfm = rfm[rfm["frequency"] >= 0]
    rfm = rfm[rfm["T"] > 0]
    
    return rfm


def fit_bgnbd(
    rfm_train: pd.DataFrame,
    penalty_coef: float = 0.0,
) -> "BetaGeoFitter":
    """Fit BG/NBD model on training data."""
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes package not installed. Install with: pip install lifetimes")
    
    bgf = BetaGeoFitter(penalty_coef=penalty_coef)
    bgf.fit(rfm_train["frequency"], rfm_train["recency"], rfm_train["T"])
    
    LOGGER.info(f"BG/NBD fitted: params = {bgf._unload_params()}")
    return bgf


def predict_bgnbd(
    bgf: "BetaGeoFitter",
    rfm: pd.DataFrame,
    t: int,  # prediction horizon in days
) -> pd.DataFrame:
    """
    Generate BG/NBD predictions.
    
    Returns:
        - expected_transactions: expected number of transactions in horizon
        - probability_alive: P(alive | history)
        - expected_revenue: expected_transactions * monetary_value
    """
    # Expected transactions in horizon
    expected_transactions = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, rfm["frequency"], rfm["recency"], rfm["T"]
    )
    
    # Probability alive
    prob_alive = bgf.conditional_probability_alive(
        rfm["frequency"], rfm["recency"], rfm["T"]
    )
    
    # Expected revenue
    expected_revenue = expected_transactions * rfm["monetary_value"]
    
    # Discounted expected revenue
    daily_discount = (1 + 0.10) ** (1/365) - 1  # default
    discount_factor = 1 / (1 + daily_discount) ** t
    expected_discounted_revenue = expected_revenue * discount_factor
    
    results = rfm.copy()
    results["expected_transactions"] = expected_transactions
    results["probability_alive"] = prob_alive
    results["expected_revenue"] = expected_revenue
    results["expected_discounted_revenue"] = expected_discounted_revenue
    results["bgnbd_clv"] = expected_discounted_revenue  # Revenue CLV
    
    return results


def evaluate_bgnbd(
    bgf: "BetaGeoFitter",
    rfm_test: pd.DataFrame,
    t: int,
) -> Dict:
    """
    Evaluate BG/NBD predictions against test period actuals.
    """
    # Generate predictions for test customers
    preds = predict_bgnbd(bgf, rfm_test, t)
    
    # Actual transactions in test period
    # We need to compute actual frequency in test period
    # This requires the full transaction data - simplified here
    
    # For now, return prediction statistics
    return {
        "mean_expected_transactions": float(preds["expected_transactions"].mean()),
        "mean_prob_alive": float(preds["probability_alive"].mean()),
        "mean_expected_revenue": float(preds["expected_revenue"].mean()),
        "mean_bgnbd_clv": float(preds["bgnbd_clv"].mean()),
        "median_bgnbd_clv": float(preds["bgnbd_clv"].median()),
    }


def compare_with_monte_carlo(
    bgnbd_results: pd.DataFrame,
    monte_carlo_results: pd.DataFrame,
) -> Dict:
    """
    Compare BG/NBD CLV with Monte Carlo CLV predictions.
    """
    # Merge on Customer ID
    merged = bgnbd_results[["Customer ID", "bgnbd_clv", "expected_transactions", "probability_alive"]].merge(
        monte_carlo_results[["Customer ID", "clv_mean"]],
        on="Customer ID",
        how="inner"
    )
    
    if len(merged) == 0:
        return {"error": "No overlapping customers"}
    
    from scipy.stats import spearmanr, pearsonr
    
    # Correlations
    spearman_clv, _ = spearmanr(merged["bgnbd_clv"], merged["clv_mean"])
    pearson_clv, _ = pearsonr(merged["bgnbd_clv"], merged["clv_mean"])
    
    # Summary stats
    comparison = {
        "n_customers": len(merged),
        "bgnbd_clv_mean": float(merged["bgnbd_clv"].mean()),
        "bgnbd_clv_median": float(merged["bgnbd_clv"].median()),
        "mc_clv_mean": float(merged["clv_mean"].mean()),
        "mc_clv_median": float(merged["clv_mean"].median()),
        "spearman_correlation": float(spearman_clv),
        "pearson_correlation": float(pearson_clv),
        "ratio_means": float(merged["bgnbd_clv"].mean() / merged["clv_mean"].mean()) if merged["clv_mean"].mean() > 0 else None,
    }
    
    return comparison


def main() -> None:
    args = parse_args()

    if not LIFETIMES_AVAILABLE:
        LOGGER.error("lifetimes package not available. Install with: pip install lifetimes")
        # Create empty outputs
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path("./bgnbd_benchmark_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)
        
        with open(output_dir / "model_card.json", "w") as f:
            json.dump({
                "model_name": "BG/NBD Benchmark",
                "status": "SKIPPED - lifetimes package not installed",
                "install_command": "pip install lifetimes",
            }, f, indent=2)
        return

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.clv_dir / "bgnbd_benchmark"
    product_metrics_path = config.product_analytics_dir / "product_metrics.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"

    # Load canonical transactions
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

    # Run validations
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # Prepare training data (up to cutoff)
    LOGGER.info(f"Preparing RFM data up to {args.eval_cutoff}...")
    rfm_train = prepare_rfm_data(tx, args.eval_cutoff)
    
    LOGGER.info(f"Training customers: {len(rfm_train):,}")
    LOGGER.info(f"Mean frequency: {rfm_train['frequency'].mean():.2f}")
    LOGGER.info(f"Mean recency: {rfm_train['recency'].mean():.1f}")
    LOGGER.info(f"Mean T: {rfm_train['T'].mean():.1f}")

    # Fit BG/NBD
    LOGGER.info("Fitting BG/NBD model...")
    bgf = fit_bgnbd(rfm_train)
    
    # Save model parameters
    params = bgf._unload_params()
    with open(output_dir / "bgnbd_params.json", "w") as f:
        json.dump({k: float(v) for k, v in params.items()}, f, indent=2)

    # Predict on training data (in-sample)
    t_horizon = args.horizon_days
    LOGGER.info(f"Generating predictions for {t_horizon} day horizon...")
    train_preds = predict_bgnbd(bgf, rfm_train, t_horizon)
    
    # Apply margin
    train_preds["bgnbd_clv_net"] = train_preds["bgnbd_clv"] * args.margin_rate
    
    # Save predictions
    train_preds.to_csv(output_dir / "bgnbd_predictions.csv", index=False)
    train_preds.to_parquet(output_dir / "bgnbd_predictions.parquet")

    # Evaluate against Monte Carlo CLV if available
    monte_carlo_path = config.clv_dir / "clv_customer_predictions.csv"
    comparison = {}
    if monte_carlo_path.exists():
        LOGGER.info("Comparing with Monte Carlo CLV...")
        mc_results = pd.read_csv(monte_carlo_path)
        comparison = compare_with_monte_carlo(train_preds, mc_results)
        LOGGER.info(f"Comparison: {comparison}")
    else:
        LOGGER.warning("Monte Carlo CLV results not found for comparison")

    # Model card
    model_card = {
        "model_name": "BG/NBD (Beta-Geometric/Negative Binomial Distribution)",
        "methodology": "Classical Buy 'Til You Die model for CLV",
        "eval_cutoff": args.eval_cutoff,
        "horizon_days": args.horizon_days,
        "annual_discount_rate": args.annual_discount_rate,
        "margin_rate": args.margin_rate,
        "params": {k: float(v) for k, v in params.items()},
        "training_customers": len(rfm_train),
        "training_stats": {
            "mean_frequency": float(rfm_train["frequency"].mean()),
            "mean_recency": float(rfm_train["recency"].mean()),
            "mean_T": float(rfm_train["T"].mean()),
            "mean_monetary": float(rfm_train["monetary_value"].mean()),
        },
        "prediction_stats": {
            "mean_bgnbd_clv": float(train_preds["bgnbd_clv"].mean()),
            "median_bgnbd_clv": float(train_preds["bgnbd_clv"].median()),
            "mean_prob_alive": float(train_preds["probability_alive"].mean()),
        },
        "comparison_with_monte_carlo": comparison,
    }

    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "bgnbd_predictions.csv",
            "bgnbd_predictions.parquet",
            "bgnbd_params.json",
            "model_card.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # CLV distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_preds["bgnbd_clv_net"], bins=50, alpha=0.7, edgecolor='white')
        ax.set_xlabel("BG/NBD CLV (Net)")
        ax.set_ylabel("Customers")
        ax.set_title("BG/NBD Customer CLV Distribution")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "01_bgnbd_clv_distribution.png", dpi=180)
        plt.close(fig)

        # Probability alive distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_preds["probability_alive"], bins=50, alpha=0.7, edgecolor='white')
        ax.set_xlabel("P(Alive)")
        ax.set_ylabel("Customers")
        ax.set_title("Probability Alive Distribution")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "02_prob_alive_distribution.png", dpi=180)
        plt.close(fig)

        # Expected transactions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_preds["expected_transactions"], bins=50, alpha=0.7, edgecolor='white')
        ax.set_xlabel(f"Expected Transactions in {t_horizon} days")
        ax.set_ylabel("Customers")
        ax.set_title("Expected Future Transactions")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "03_expected_transactions.png", dpi=180)
        plt.close(fig)

        # Comparison with Monte Carlo if available
        if monte_carlo_path.exists() and len(comparison) > 0 and "error" not in comparison:
            mc_results = pd.read_csv(monte_carlo_path)
            merged = train_preds[["Customer ID", "bgnbd_clv_net"]].merge(
                mc_results[["Customer ID", "clv_mean"]], on="Customer ID", how="inner"
            )
            if len(merged) > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(merged["bgnbd_clv_net"], merged["clv_mean"], alpha=0.3, s=10)
                max_val = max(merged["bgnbd_clv_net"].max(), merged["clv_mean"].max())
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label="y=x")
                ax.set_xlabel("BG/NBD CLV (Net)")
                ax.set_ylabel("Monte Carlo CLV")
                ax.set_title(f"CLV Comparison (Spearman ρ={comparison['spearman_correlation']:.3f})")
                ax.legend()
                ax.grid(alpha=0.15)
                fig.tight_layout()
                fig.savefig(output_dir / "plots" / "04_bgnbd_vs_monte_carlo.png", dpi=180)
                plt.close(fig)

    LOGGER.info("BG/NBD benchmark complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()