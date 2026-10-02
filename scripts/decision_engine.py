#!/usr/bin/env python3
"""
Decision Engine — Next-Best-Action Policy for Retail Customers

This script combines multiple predictive signals into a transparent next-best-action policy.

Inputs:
- CLV proxy (predicted future value)
- Churn probability (discrete-time hazard)
- Survival probability (3/6/12 month)
- Next purchase probability (7/30/60 day)
- Reactivation probability (for inactive customers)
- Segment
- Recommendation signals
- Confidence intervals

Outputs:
- recommended_action: protect_value, accelerate_purchase, reactivate, cross_sell, nurture, monitor
- priority_score: numerical priority for capacity allocation
- decision_confidence: confidence in the recommendation
- expected_value_proxy: expected value of taking the action
- action_reason: human-readable explanation

The policy is explicitly labeled OBSERVATIONAL until causal treatment data exists.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args


SEED = 42
np.random.seed(SEED)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("decision_engine")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision engine for next-best-action.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--clv-file", default=None, help="CLV predictions (overrides config).")
    parser.add_argument("--churn-file", default=None, help="Churn/next-purchase predictions (overrides config).")
    parser.add_argument("--reactivation-file", default=None, help="Reactivation predictions (overrides config).")
    parser.add_argument("--recommendations-file", default=None, help="Product recommendations (overrides config).")
    parser.add_argument("--segments-file", default=None, help="Customer segments (overrides config).")
    parser.add_argument("--capacity-total", type=int, default=1000, help="Total contact capacity.")
    parser.add_argument("--capacity-reactivate", type=int, default=300, help="Reactivation contact capacity.")
    parser.add_argument("--capacity-accelerate", type=int, default=400, help="Acceleration contact capacity.")
    parser.add_argument("--capacity-cross-sell", type=int, default=300, help="Cross-sell contact capacity.")
    parser.add_argument("--capacity-nurture", type=int, default=500, help="Nurture contact capacity.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence for action.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def load_predictions(file_path: Path, required_cols: List[str], rename_map: Dict[str, str] = None) -> pd.DataFrame:
    """Load and validate prediction file."""
    if not file_path.exists():
        LOGGER.warning(f"File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Rename columns if needed
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        LOGGER.warning(f"Missing columns in {file_path}: {missing}")
        return pd.DataFrame()
    
    # Ensure Customer ID is int
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Customer ID"])
    
    return df


def combine_customer_data(
    clv_df: pd.DataFrame,
    churn_df: pd.DataFrame,
    reactivation_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all customer-level predictions into a single decision-ready DataFrame."""
    
    # Start with CLV as base (has all customers)
    if clv_df.empty:
        raise ValueError("CLV predictions required as base")
    
    df = clv_df.copy()
    
    # Merge churn/next-purchase
    if not churn_df.empty:
        churn_cols = ["Customer ID", "churn_probability", "survival_3m", "survival_6m", "survival_12m",
                      "next_purchase_7d_probability", "next_purchase_30d_probability", "next_purchase_60d_probability"]
        available = [c for c in churn_cols if c in churn_df.columns]
        df = df.merge(churn_df[available], on="Customer ID", how="left", suffixes=("", "_churn"))
    
    # Merge reactivation
    if not reactivation_df.empty:
        react_cols = ["Customer ID", "reactivation_probability", "calendar_month", "lifetime_orders", 
                      "lifetime_gross_revenue", "recency_months", "consecutive_inactive_months"]
        available = [c for c in react_cols if c in reactivation_df.columns]
        df = df.merge(reactivation_df[available], on="Customer ID", how="left", suffixes=("", "_react"))
    
    # Merge recommendations (take top 3 per customer)
    if not recommendations_df.empty:
        rec_summary = recommendations_df.groupby("Customer ID").agg(
            top_recommendation=("recommended_product", "first"),
            top_score=("score", "max"),
            rec_reason=("reason", "first"),
            rec_count=("recommended_product", "count"),
        ).reset_index()
        df = df.merge(rec_summary, on="Customer ID", how="left")
    
    # Merge segments
    if not segments_df.empty:
        seg_cols = ["Customer ID", "segment", "segment_name", "segment_confidence"]
        available = [c for c in seg_cols if c in segments_df.columns]
        df = df.merge(segments_df[available], on="Customer ID", how="left")
    
    return df


def compute_decision_policy(df: pd.DataFrame, confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply the observational next-best-action policy.
    
    Policy logic:
    1. PROTECT_VALUE: High CLV + High churn risk + High confidence
    2. ACCELERATE_PURCHASE: High CLV + High next-purchase prob + Low churn
    3. REACTIVATE: Inactive + High reactivation prob + High historical value
    4. CROSS_SELL: Active + High CLV + Good recommendations + Low churn
    5. NURTURE: Medium CLV + Medium engagement
    6. MONITOR: Low priority / insufficient signals
    
    All actions are OBSERVATIONAL - not causal.
    """
    
    # Ensure required columns exist with defaults
    defaults = {
        "clv_mean": 0.0,
        "clv_p10": 0.0,
        "clv_p90": 0.0,
        "churn_probability": 0.5,
        "survival_3m": 0.5,
        "survival_6m": 0.5,
        "survival_12m": 0.5,
        "next_purchase_7d_probability": 0.0,
        "next_purchase_30d_probability": 0.0,
        "next_purchase_60d_probability": 0.0,
        "reactivation_probability": 0.0,
        "top_score": 0.0,
        "rec_reason": "none",
        "segment_name": "unknown",
        "lifetime_gross_revenue": 0.0,
        "recency_months": 999,
        "consecutive_inactive_months": 999,
    }
    
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    
    # Determine if customer is currently inactive
    df["is_inactive"] = (df["recency_months"] > 3) | (df["consecutive_inactive_months"] > 3)
    df["is_active"] = ~df["is_inactive"]
    
    # Compute action scores
    df["protect_value_score"] = (
        (df["clv_mean"] / (df["clv_mean"].max() + 1e-9)) * 0.4 +
        (df["churn_probability"]) * 0.3 +
        (1 - df["survival_3m"]) * 0.2 +
        (df.get("segment_confidence", 0.5)) * 0.1
    )
    
    df["accelerate_purchase_score"] = (
        (df["next_purchase_30d_probability"]) * 0.4 +
        (df["clv_mean"] / (df["clv_mean"].max() + 1e-9)) * 0.3 +
        (1 - df["churn_probability"]) * 0.2 +
        (df["top_score"] if "top_score" in df.columns else 0) * 0.1
    )
    
    df["reactivate_score"] = (
        (df["reactivation_probability"] if "reactivation_probability" in df.columns else 0) * 0.4 +
        (df["lifetime_gross_revenue"] / (df["lifetime_gross_revenue"].max() + 1e-9)) * 0.3 +
        (df["consecutive_inactive_months"] < 12).astype(float) * 0.2 +
        (df["is_inactive"]).astype(float) * 0.1
    )
    
    df["cross_sell_score"] = (
        (df["top_score"] if "top_score" in df.columns else 0) * 0.4 +
        (df["clv_mean"] / (df["clv_mean"].max() + 1e-9)) * 0.3 +
        (df["is_active"]).astype(float) * 0.2 +
        (1 - df["churn_probability"]) * 0.1
    )
    
    df["nurture_score"] = (
        (df["clv_mean"] / (df["clv_mean"].max() + 1e-9)) * 0.5 +
        (1 - df["churn_probability"]) * 0.3 +
        (df["is_active"]).astype(float) * 0.2
    )
    
    # Determine recommended action
    score_cols = ["protect_value_score", "accelerate_purchase_score", "reactivate_score", 
                  "cross_sell_score", "nurture_score"]
    
    def select_action(row, threshold):
        # Build action candidates with constraints
        candidates = {}
        
        if row["protect_value_score"] >= threshold and row["churn_probability"] > 0.3:
            candidates["protect_value"] = row["protect_value_score"]
        
        if row["accelerate_purchase_score"] >= threshold and row["next_purchase_30d_probability"] > 0.3:
            candidates["accelerate_purchase"] = row["accelerate_purchase_score"]
        
        if row["reactivate_score"] >= threshold and row["is_inactive"]:
            candidates["reactivate"] = row["reactivate_score"]
        
        if row["cross_sell_score"] >= threshold and row["is_active"] and row["top_score"] > 0.2:
            candidates["cross_sell"] = row["cross_sell_score"]
        
        if row["nurture_score"] >= threshold:
            candidates["nurture"] = row["nurture_score"]
        
        # Default: monitor
        if not candidates:
            return "monitor"
        
        # Return highest scoring valid action
        return max(candidates, key=candidates.get)
    
    df["recommended_action"] = df.apply(select_action, axis=1, threshold=confidence_threshold)
    
    # Priority score = score of SELECTED action only (not max of all)
    action_score_cols = {
        "protect_value": "protect_value_score",
        "accelerate_purchase": "accelerate_purchase_score",
        "reactivate": "reactivate_score",
        "cross_sell": "cross_sell_score",
        "nurture": "nurture_score",
    }
    
    conditions = [df["recommended_action"] == action for action in action_score_cols]
    choices = [df[col] for col in action_score_cols.values()]
    df["priority_score"] = np.select(conditions, choices, default=0.0)
    
    # Decision confidence = selected action score (not a probability, but a heuristic score)
    df["decision_confidence"] = df["priority_score"]
    
    # Expected value proxy
    def compute_expected_value(row):
        if row["recommended_action"] == "protect_value":
            return row["clv_mean"] * row["churn_probability"] * 0.3  # 30% value at risk
        elif row["recommended_action"] == "accelerate_purchase":
            return row["clv_mean"] * row["next_purchase_30d_probability"] * 0.2
        elif row["recommended_action"] == "reactivate":
            return row["lifetime_gross_revenue"] * row["reactivation_probability"] * 0.5
        elif row["recommended_action"] == "cross_sell":
            return row["top_score"] * row["clv_mean"] * 0.3
        elif row["recommended_action"] == "nurture":
            return row["clv_mean"] * 0.1
        else:
            return 0.0
    
    df["expected_value_proxy"] = df.apply(compute_expected_value, axis=1)
    
    # Action reason
    def action_reason(row):
        action = row["recommended_action"]
        if action == "protect_value":
            return f"High CLV (£{row['clv_mean']:.0f}) at risk (churn={row['churn_probability']:.0%}, survival_3m={row['survival_3m']:.0%})"
        elif action == "accelerate_purchase":
            return f"Ready to buy (30d prob={row['next_purchase_30d_probability']:.0%}), high value (£{row['clv_mean']:.0f})"
        elif action == "reactivate":
            return f"Inactive {row['consecutive_inactive_months']:.0f} months, reactivation prob={row.get('reactivation_probability', 0):.0%}, past value £{row['lifetime_gross_revenue']:.0f}"
        elif action == "cross_sell":
            return f"Active buyer, top recommendation score={row.get('top_score', 0):.2f}, CLV £{row['clv_mean']:.0f}"
        elif action == "nurture":
            return f"Medium CLV (£{row['clv_mean']:.0f}), low churn risk ({row['churn_probability']:.0%})"
        else:
            return "Low priority or insufficient signals"
    
    df["action_reason"] = df.apply(action_reason, axis=1)
    
    return df


def apply_capacity_constraints(
    df: pd.DataFrame,
    capacity_total: int,
    capacity_reactivate: int,
    capacity_accelerate: int,
    capacity_cross_sell: int,
    capacity_nurture: int,
) -> pd.DataFrame:
    """Apply capacity constraints with constrained optimization (greedy value-density)."""

    # Define actions we are allocating (excluding monitor for now)
    actions = ["reactivate", "accelerate_purchase", "cross_sell", "nurture", "protect_value"]
    # Map action names to capacity arguments
    action_capacities = {
        "reactivate": capacity_reactivate,
        "accelerate_purchase": capacity_accelerate,
        "cross_sell": capacity_cross_sell,
        "nurture": capacity_nurture,
        "protect_value": capacity_total,  # protect_value shares total capacity
    }

    # Helper functions to compute expected value for each action
    def expected_value_protect_value(row):
        return row["clv_mean"] * row["churn_probability"] * 0.3  # 30% value at risk

    def expected_value_accelerate_purchase(row):
        return row["clv_mean"] * row["next_purchase_30d_probability"] * 0.2

    def expected_value_reactivate(row):
        return row["lifetime_gross_revenue"] * row["reactivation_probability"] * 0.5

    def expected_value_cross_sell(row):
        return row["top_score"] * row["clv_mean"] * 0.3

    def expected_value_nurture(row):
        return row["clv_mean"] * 0.1

    ev_funcs = {
        "protect_value": expected_value_protect_value,
        "accelerate_purchase": expected_value_accelerate_purchase,
        "reactivate": expected_value_reactivate,
        "cross_sell": expected_value_cross_sell,
        "nurture": expected_value_nurture,
    }

    # Helper functions to determine eligibility for each action
    def eligible_protect_value(row):
        # High CLV + High churn risk + High confidence
        # We'll use the same conditions as in select_action for protect_value
        return (row["churn_probability"] > 0.3)  # and row["protect_value_score"] >= confidence_threshold? 
        # But we don't have confidence_threshold here. We'll use the score >= median as in the original?
        # Instead, we'll reuse the logic from select_action: we need the score and threshold.
        # Since we don't have the threshold, we'll approximate by using the score being above median.
        # However, we are going to compute the score again? Let's avoid duplication.
        # We'll compute the protect_value_score as in compute_decision_policy.
        # But note: we are not given the confidence_threshold in this function.
        # We'll assume that the score columns are still present and we can use them.
        # We'll change: we will require that the action's score is above the median (as in the original fallback).
        # This is not ideal but allows us to use the existing score columns.
        return row["protect_value_score"] >= df["protect_value_score"].median()

    def eligible_accelerate_purchase(row):
        return (row["next_purchase_30d_probability"] > 0.3) and (row["accelerate_purchase_score"] >= df["accelerate_purchase_score"].median())

    def eligible_reactivate(row):
        return row["is_inactive"] and (row["reactivate_score"] >= df["reactivate_score"].median())

    def eligible_cross_sell(row):
        return row["is_active"] and (row["top_score"] > 0.2) and (row["cross_sell_score"] >= df["cross_sell_score"].median())

    def eligible_nurture(row):
        return (row["nurture_score"] >= df["nurture_score"].median())

    eligibility_funcs = {
        "protect_value": eligible_protect_value,
        "accelerate_purchase": eligible_accelerate_purchase,
        "reactivate": eligible_reactivate,
        "cross_sell": eligible_cross_sell,
        "nurture": eligible_nurture,
    }

    # Build list of eligible customer-action pairs with expected value
    candidates = []  # list of (customer_id, action, expected_value)
    for idx, row in df.iterrows():
        customer_id = row["Customer ID"]
        for action in actions:
            if eligibility_funcs[action](row):
                ev = ev_funcs[action](row)
                candidates.append((customer_id, action, ev))

    # Sort by expected_value descending (value-density, cost=1)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Initialize allocation tracking
    allocated_per_action = {action: 0 for action in actions}
    allocated_per_customer = set()  # set of customer IDs that have been allocated an action
    total_allocated = 0
    # Map from customer_id to allocated action
    customer_allocation = {}

    # Greedy allocation
    for customer_id, action, ev in candidates:
        if customer_id in allocated_per_customer:
            continue  # customer already got an action
        if allocated_per_action[action] >= action_capacities[action]:
            continue  # action capacity exceeded
        if total_allocated >= capacity_total:
            continue  # global capacity exceeded
        # Allocate this action to this customer
        allocated_per_action[action] += 1
        allocated_per_customer.add(customer_id)
        total_allocated += 1
        customer_allocation[customer_id] = action

    # Build the output DataFrame
    df_out = df.copy()
    # Default action is monitor
    df_out["recommended_action_capped"] = "monitor"
    # Override with allocated actions
    for customer_id, action in customer_allocation.items():
        df_out.loc[df_out["Customer ID"] == customer_id, "recommended_action_capped"] = action

    # Compute diagnostics for model card (we'll add to model card later in main)
    # We'll return the diagnostics as well? But we cannot change return type.
    # Instead, we'll store them in the df_out as attributes? Not good.
    # We'll compute them in main after calling this function.
    # For now, we'll just return the df_out.
    # The caller (main) can compute the diagnostics from the returned df_out.

    return df_out


def save_plots(output_dir: Path, df: pd.DataFrame) -> None:
    """Generate decision engine visualizations."""
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Action distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    action_counts = df["recommended_action_capped"].value_counts()
    ax.barh(action_counts.index[::-1], action_counts.values[::-1])
    ax.set_xlabel("Customers")
    ax.set_title("Customer Allocation by Recommended Action (After Capacity Constraints)")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "01_action_allocation.png", dpi=180)
    plt.close(fig)
    
    # Priority score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for action in df["recommended_action_capped"].unique():
        subset = df[df["recommended_action_capped"] == action]
        ax.hist(subset["priority_score"], bins=20, alpha=0.5, label=action, density=True)
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Density")
    ax.set_title("Priority Score Distribution by Action")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "02_priority_distribution.png", dpi=180)
    plt.close(fig)
    
    # Expected value by action
    fig, ax = plt.subplots(figsize=(10, 6))
    ev_by_action = df.groupby("recommended_action_capped")["expected_value_proxy"].mean().sort_values()
    ax.barh(ev_by_action.index, ev_by_action.values)
    ax.set_xlabel("Expected Value Proxy")
    ax.set_title("Average Expected Value by Recommended Action")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "03_expected_value_by_action.png", dpi=180)
    plt.close(fig)
    
    # CLV vs Churn risk scatter colored by action
    fig, ax = plt.subplots(figsize=(10, 7))
    for action in df["recommended_action_capped"].unique():
        subset = df[df["recommended_action_capped"] == action]
        ax.scatter(subset["clv_mean"], subset["churn_probability"], 
                   s=20, alpha=0.5, label=action)
    ax.set_xlabel("CLV (Mean)")
    ax.set_ylabel("Churn Probability")
    ax.set_title("Customer Value vs Churn Risk by Recommended Action")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "04_clv_vs_churn_by_action.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.decision_engine_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    
    LOGGER.info("Loading prediction files...")
    
    clv_file = Path(args.clv_file).expanduser().resolve() if args.clv_file else config.clv_dir / "clv_customer_predictions.csv"
    churn_file = Path(args.churn_file).expanduser().resolve() if args.churn_file else config.churn_dir / "customer_churn_next_purchase.csv"
    reactivation_file = Path(args.reactivation_file).expanduser().resolve() if args.reactivation_file else config.reactivation_dir / "reactivation_predictions.csv"
    recommendations_file = Path(args.recommendations_file).expanduser().resolve() if args.recommendations_file else config.recommendations_dir / "recommendations.csv"
    segments_file = Path(args.segments_file).expanduser().resolve() if args.segments_file else config.segmentation_dir / "customer_segments.parquet"
    
    clv_df = load_predictions(
        clv_file,
        ["Customer ID", "clv_mean"],
        {"clv": "clv_mean", "clv_lower": "clv_p10", "clv_upper": "clv_p90"}
    )
    
    churn_df = load_predictions(
        churn_file,
        ["Customer ID"],
        {"churn_prob": "churn_probability", "next_purchase_30d": "next_purchase_30d_probability"}
    )
    
    reactivation_df = load_predictions(
        reactivation_file,
        ["Customer ID"],
        {"reactivation_prob": "reactivation_probability"}
    )
    
    recommendations_df = load_predictions(
        recommendations_file,
        ["Customer ID", "recommended_product", "score", "reason"],
    )
    
    segments_df = pd.DataFrame()
    if segments_file.exists():
        segments_df = pd.read_parquet(segments_file)
        segments_df["Customer ID"] = pd.to_numeric(segments_df["Customer ID"], errors="coerce").astype("Int64")
        segments_df = segments_df.dropna(subset=["Customer ID"])
    
    # -------------------------------------------------------------------------
    # Combine all customer data
    # -------------------------------------------------------------------------
    LOGGER.info("Combining customer data...")
    df = combine_customer_data(
        clv_df, churn_df, reactivation_df, recommendations_df, segments_df
    )
    
    # -------------------------------------------------------------------------
    # Apply decision policy
    # -------------------------------------------------------------------------
    LOGGER.info("Applying decision policy...")
    df = compute_decision_policy(df, confidence_threshold=args.confidence_threshold)
    
    # -------------------------------------------------------------------------
    # Apply capacity constraints
    # -------------------------------------------------------------------------
    LOGGER.info("Applying capacity constraints...")
    df = apply_capacity_constraints(
        df,
        capacity_total=args.capacity_total,
        capacity_reactivate=args.capacity_reactivate,
        capacity_accelerate=args.capacity_accelerate,
        capacity_cross_sell=args.capacity_cross_sell,
        capacity_nurture=args.capacity_nurture,
    )
    
    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    LOGGER.info("Writing outputs...")
    
    # Final decision scores
    output_cols = [
        "Customer ID", "recommended_action_capped", "priority_score", 
        "decision_confidence", "expected_value_proxy", "action_reason",
        "clv_mean", "clv_p10", "clv_p90",
        "churn_probability", "survival_3m", "survival_6m", "survival_12m",
        "next_purchase_7d_probability", "next_purchase_30d_probability", "next_purchase_60d_probability",
        "reactivation_probability", "top_score", "rec_reason", "segment_name",
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    df[available_cols].to_csv(output_dir / "customer_decision_scores.csv", index=False)
    
# Action summary
    action_summary = df.groupby("recommended_action_capped").agg(
        customers=("Customer ID", "count"),
        avg_priority=("priority_score", "mean"),
        avg_confidence=("decision_confidence", "mean"),
        avg_expected_value=("expected_value_proxy", "mean"),
        total_expected_value=("expected_value_proxy", "sum"),
    ).reset_index().rename(columns={"recommended_action_capped": "final_action"})
    action_summary.to_csv(output_dir / "action_summary.csv", index=False)
    
    # Allocation diagnostics for model card
    total_allocated = (df["recommended_action_capped"] != "monitor").sum()
    per_action_allocated = {}
    for action in ["protect_value", "accelerate_purchase", "reactivate", "cross_sell", "nurture"]:
        per_action_allocated[action] = (df["recommended_action_capped"] == action).sum()
    allocation_diagnostics = {
        "total_allocated": int(total_allocated),
        "total_capacity": args.capacity_total,
        "per_action_allocated": {k: int(v) for k, v in per_action_allocated.items()},
        "per_action_capacity": {
            "protect_value": args.capacity_total,
            "reactivate": args.capacity_reactivate,
            "accelerate_purchase": args.capacity_accelerate,
            "cross_sell": args.capacity_cross_sell,
            "nurture": args.capacity_nurture,
        },
    }

    # Model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "model_name": "Observational Next-Best-Action Decision Engine",
            "methodology": "Multi-signal scoring with capacity-aware allocation",
            "policy_type": "OBSERVATIONAL - not causal",
            "actions": ["protect_value", "accelerate_purchase", "reactivate", "cross_sell", "nurture", "monitor"],
            "capacity_constraints": {
                "total": args.capacity_total,
                "reactivate": args.capacity_reactivate,
                "accelerate_purchase": args.capacity_accelerate,
                "cross_sell": args.capacity_cross_sell,
                "nurture": args.capacity_nurture,
            },
            "confidence_threshold": args.confidence_threshold,
            "action_distribution": df["recommended_action_capped"].value_counts().to_dict(),
            "allocation_diagnostics": allocation_diagnostics,
            "known_limitations": [
                "Observational policy - does not estimate causal treatment effects",
                "No A/B test validation of action effectiveness",
                "Assumes independent additive value of actions",
                "Capacity constraints are hard caps, not optimized",
            ],
        }, f, indent=2, default=str)

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "customer_decision_scores.csv",
            "action_summary.csv",
            "model_card.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Plots
    if not args.skip_plots:
        LOGGER.info("Generating plots...")
        save_plots(output_dir, df)
    
    LOGGER.info("Decision engine complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    import argparse
    main()