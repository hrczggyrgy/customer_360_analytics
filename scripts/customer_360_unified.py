#!/usr/bin/env python3
"""
Customer 360 Unified Snapshot

This script merges ALL customer-level artifacts into a single unified parquet file
for the Streamlit dashboard. One row per customer with all available signals.

Artifacts merged:
- customer_360_current.parquet (descriptive + PIT features)
- customer_segments.parquet (HDBSCAN segments)
- clv_customer_predictions.csv (CLV proxy)
- customer_churn_next_purchase.csv (churn + survival + next-purchase)
- reactivation_predictions.csv (reactivation probability)
- recommendations.parquet (product recommendations)
- customer_decision_scores.csv (decision engine actions)
- rfm_segments.csv (RFM baseline)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args
from retail_ds.contracts import validate_customer_360


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("customer_360_unified")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create unified Customer 360 snapshot.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


# Artifact definitions (shared between functions)
ARTIFACTS = {
    "customer_360": {
        "path": None,  # Will be set in merge_artifacts
        "required": ["Customer ID"],
        "rename": {},
    },
    "segments": {
        "path": None,
        "required": ["Customer ID"],
        "rename": {"segment": "hdbscan_segment", "segment_name": "hdbscan_segment_name", "segment_confidence": "hdbscan_confidence"},
    },
    "clv": {
        "path": None,
        "required": ["Customer ID", "clv_mean"],
        "rename": {"clv": "clv_mean", "clv_lower": "clv_p10", "clv_upper": "clv_p90"},
    },
    "churn": {
        "path": None,
        "required": ["Customer ID"],
        "rename": {"churn_prob": "next_month_inactivity_risk", "next_purchase_30d": "next_purchase_30d_probability"},
    },
    "reactivation": {
        "path": None,
        "required": ["Customer ID"],
        "rename": {"reactivation_prob": "reactivation_probability"},
    },
    "recommendations": {
        "path": None,
        "required": ["Customer ID", "recommended_product", "score", "reason"],
        "rename": {},
    },
    "decisions": {
        "path": None,
        "required": ["Customer ID", "recommended_action_capped", "priority_score"],
        "rename": {},
    },
    "rfm": {
        "path": None,
        "required": ["Customer ID"],
        "rename": {},
    },
}


def load_artifact(file_path: Path, required_cols: List[str] = None, rename_map: Dict[str, str] = None) -> pd.DataFrame:
    """Load artifact with validation."""
    if not file_path.exists():
        LOGGER.warning(f"Artifact not found: {file_path}")
        return pd.DataFrame()
    
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        LOGGER.warning(f"Unsupported format: {file_path}")
        return pd.DataFrame()
    
    # Rename columns if needed
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Check required columns
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            LOGGER.warning(f"Missing columns in {file_path}: {missing}")
            return pd.DataFrame()
    
    # Ensure Customer ID is int
    if "Customer ID" in df.columns:
        df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Customer ID"])
    
    return df


def merge_artifacts(
    config: ProjectConfig,
) -> pd.DataFrame:
    """Merge all customer-level artifacts into unified snapshot."""
    
    # Set artifact paths
    artifacts = ARTIFACTS.copy()
    artifacts["customer_360"]["path"] = config.customer_360_dir / "customer_360_current.parquet"
    artifacts["segments"]["path"] = config.segmentation_dir / "customer_segments.parquet"
    artifacts["clv"]["path"] = config.clv_dir / "clv_customer_predictions.csv"
    artifacts["churn"]["path"] = config.churn_dir / "customer_churn_next_purchase.csv"
    artifacts["reactivation"]["path"] = config.reactivation_dir / "reactivation_predictions.csv"
    artifacts["recommendations"]["path"] = config.recommendations_dir / "recommendations.parquet"
    artifacts["decisions"]["path"] = config.decision_engine_dir / "customer_decision_scores.csv"
    artifacts["rfm"]["path"] = config.segmentation_dir / "rfm_segments.csv"
    
    loaded = {}
    for name, spec in artifacts.items():
        df = load_artifact(spec["path"], spec["required"], spec["rename"])
        if not df.empty:
            # Remove duplicate Customer IDs, keep first
            df = df.drop_duplicates(subset=["Customer ID"], keep="first")
            loaded[name] = df
            LOGGER.info(f"Loaded {name}: {len(df):,} customers")
        else:
            LOGGER.warning(f"Failed to load {name}")
    
    # Start with customer_360 as base (has all customers with features)
    if "customer_360" not in loaded:
        raise ValueError("Customer 360 base artifact is required")
    
    unified = loaded["customer_360"].copy()
    LOGGER.info(f"Base customer_360: {len(unified):,} customers")
    
    # Merge each artifact
    merge_order = ["segments", "clv", "churn", "reactivation", "decisions", "rfm"]
    
    for name in merge_order:
        if name in loaded:
            df = loaded[name]
            # Determine merge columns (all except Customer ID)
            merge_cols = [c for c in df.columns if c != "Customer ID"]
            LOGGER.info(f"Merging {name}: {len(merge_cols)} columns")
            unified = unified.merge(df, on="Customer ID", how="left", suffixes=("", f"_{name}"))
            LOGGER.info(f"  After merge: {len(unified):,} customers")
    
    # Handle recommendations separately (multiple rows per customer)
    if "recommendations" in loaded:
        recs = loaded["recommendations"]
        # Get top 3 recommendations per customer
        top_recs = recs.sort_values("score", ascending=False).groupby("Customer ID").head(3)
        # Pivot to wide format
        rec_pivot = top_recs.pivot_table(
            index="Customer ID",
            columns="rank" if "rank" in top_recs.columns else None,
            values=["recommended_product", "score", "reason", "association_lift"],
            aggfunc="first",
        )
        rec_pivot.columns = [f"rec_{col[0]}_{col[1]}" if col[1] else f"rec_{col[0]}" for col in rec_pivot.columns]
        rec_pivot = rec_pivot.reset_index()
        unified = unified.merge(rec_pivot, on="Customer ID", how="left")
        LOGGER.info(f"Merged recommendations: {len(unified):,} customers")
    
    return unified


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for dashboard convenience."""
    
    # Customer health score (0-100)
    if all(c in df.columns for c in ["next_month_inactivity_risk", "clv_mean", "survival_6m"]):
        clv_norm = df["clv_mean"] / (df["clv_mean"].max() + 1e-9)
        health = (
            (1 - df["next_month_inactivity_risk"]) * 0.4 +
            df["survival_6m"] * 0.3 +
            clv_norm * 0.3
        ) * 100
        df["customer_health_score"] = health.clip(0, 100).round(1)
    
    # Lifetime value tier
    if "lifetime_gross_revenue" in df.columns:
        try:
            df["value_tier"] = pd.qcut(
                df["lifetime_gross_revenue"], 
                q=4, 
                labels=["Bronze", "Silver", "Gold", "Platinum"],
                duplicates="drop"
            )
        except ValueError:
            # Fallback: use cut with equal-width bins
            df["value_tier"] = pd.cut(
                df["lifetime_gross_revenue"], 
                bins=4, 
                labels=["Bronze", "Silver", "Gold", "Platinum"]
            )
    
    # Engagement tier
    if "lifetime_orders" in df.columns:
        try:
            df["engagement_tier"] = pd.qcut(
                df["lifetime_orders"], 
                q=4, 
                labels=["Low", "Medium", "High", "Very High"],
                duplicates="drop"
            )
        except ValueError:
            df["engagement_tier"] = pd.cut(
                df["lifetime_orders"], 
                bins=4, 
                labels=["Low", "Medium", "High", "Very High"]
            )
    
    # Action priority tier
    if "priority_score" in df.columns:
        try:
            df["action_priority_tier"] = pd.qcut(
                df["priority_score"].fillna(0),
                q=4,
                labels=["Low", "Medium", "High", "Critical"],
                duplicates="drop"
            )
        except ValueError:
            df["action_priority_tier"] = pd.cut(
                df["priority_score"].fillna(0),
                bins=4,
                labels=["Low", "Medium", "High", "Critical"]
            )
    
    return df


def validate_unified_snapshot(df: pd.DataFrame) -> Dict:
    """Validate the unified snapshot."""
    results = {
        "total_customers": len(df),
        "total_columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Check for duplicates
    dup_count = df.duplicated(subset=["Customer ID"]).sum()
    results["duplicate_customers"] = int(dup_count)
    
    # Column completeness
    completeness = df.notna().mean()
    results["column_completeness"] = completeness.to_dict()
    results["avg_completeness"] = float(completeness.mean())
    
    # Key column checks
    key_cols = ["Customer ID", "clv_mean", "next_month_inactivity_risk", "hdbscan_segment"]
    for col in key_cols:
        if col in df.columns:
            results[f"{col}_completeness"] = float(df[col].notna().mean())
            if col in ["clv_mean", "next_month_inactivity_risk"]:
                results[f"{col}_mean"] = float(df[col].mean())
                results[f"{col}_median"] = float(df[col].median())
    
    return results


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.customer_360_dir / "unified"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical transactions for validation
    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"
    if canonical_path.exists():
        tx = pl.read_parquet(canonical_path)
    else:
        raw = load_raw_transactions(input_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # Run validations
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # Merge all artifacts
    LOGGER.info("Merging customer artifacts into unified snapshot...")
    unified = merge_artifacts(config)
    
    # Add derived features
    unified = add_derived_features(unified)
    
    # Validate
    validation = validate_unified_snapshot(unified)
    LOGGER.info(f"Validation: {validation['total_customers']:,} customers, {validation['total_columns']} columns, {validation['avg_completeness']:.1%} avg completeness")
    
    # Save unified snapshot
    unified.to_parquet(output_dir / "customer_360_unified.parquet", index=False)
    unified.to_csv(output_dir / "customer_360_unified.csv", index=False)
    
    # Save validation report
    with open(output_dir / "unified_validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)
    
    # Model card
    with open(output_dir / "model_card.json", "w") as f:
        json.dump({
            "artifact": "Customer 360 Unified Snapshot",
            "description": "Single-row-per-customer unified view of all predictive signals",
            "source_artifacts": list(ARTIFACTS.keys()),
            "validation": validation,
        }, f, indent=2, default=str)
    
    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "customer_360_unified.parquet",
            "customer_360_unified.csv",
            "unified_validation.json",
            "model_card.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    LOGGER.info(f"Customer 360 Unified Snapshot complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()