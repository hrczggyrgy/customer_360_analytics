#!/usr/bin/env python3
"""
Online Retail II — behavioral customer segmentation.

End-to-end pipeline:
1) Read/validate Online Retail II using shared retail_ds package.
2) Feature-engineer a multi-view customer "behavioral signature" in Polars.
3) Winsorize -> Yeo-Johnson -> RobustScaler -> balanced behavioral blocks.
4) PCA for denoising/compression.
5) HDBSCAN model selection + bootstrap stability diagnostic.
6) KMeans fallback only if HDBSCAN is unavailable.
7) Export customer scores, segment profiles, model diagnostics, loadings,
   feature dictionary, PCA coordinates, and decision-ready visuals.

This is deliberately broader than RFM. It models:
- economic intensity
- order-value concentration
- purchase cadence
- assortment breadth and concentration
- repeat-product behavior
- price behavior
- temporal behavior
- customer lifecycle
- cancellation / reversal behavior

Example:
    python customer_segmentation.py \
        --config config/project.yaml \
        --output-dir ./online_retail_segmentation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import PowerTransformer, RobustScaler

try:
    from sklearn.cluster import HDBSCAN as SKHDBSCAN
except Exception:
    SKHDBSCAN = None

try:
    import hdbscan as EXTERNAL_HDBSCAN
except Exception:
    EXTERNAL_HDBSCAN = None

from retail_ds.io import load_raw_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures
from retail_ds.customer_month import build_customer_month_panel, add_rolling_features
from retail_ds.validation import run_all_validations, assert_validations_pass
from retail_ds.config import ProjectConfig, load_config, add_config_args
from retail_ds.contracts import validate_customer_360


SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Behavioral feature architecture
# ---------------------------------------------------------------------

FEATURE_GROUPS = {
    "economic_intensity": [
        "revenue",
        "log_revenue",
        "avg_order_value",
        "median_order_value",
        "revenue_per_active_day",
        "units_per_invoice",
        "lines_per_invoice",
        "order_value_cv",
        "order_value_hhi",
        "invoice_intensity_per_month",
    ],
    "purchase_cadence": [
        "invoice_count",
        "active_purchase_days",
        "purchase_day_share",
        "median_interpurchase_days",
        "mean_interpurchase_days",
        "interpurchase_cv",
        "burstiness",
        "invoice_count_per_active_day",
    ],
    "assortment_behavior": [
        "unique_products",
        "products_per_invoice",
        "repeat_product_ratio",
        "product_hhi",
        "single_product_invoice_share",
        "multi_product_invoice_share",
    ],
    "price_behavior": [
        "weighted_avg_unit_price",
        "median_unit_price",
        "price_iqr",
        "line_price_cv",
        "premium_line_share",
        "zero_price_line_share",
    ],
    "temporal_signature": [
        "weekend_purchase_share",
        "business_hour_share",
        "hour_entropy",
        "weekday_entropy",
        "month_entropy",
    ],
    "lifecycle": [
        "tenure_days",
        "days_since_last_purchase",
        "active_span_ratio",
        "purchase_months",
        "months_since_last_purchase",
    ],
    "return_friction": [
        "return_units",
        "return_value",
        "unit_return_rate",
        "value_return_rate",
        "return_invoice_rate",
        "return_line_share",
    ],
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("customer_segmentation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral customer segmentation for Online Retail II."
    )
    add_config_args(parser)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config).",
    )

    parser.add_argument(
        "--min-customers",
        type=int,
        default=150,
        help="Minimum number of customers required to cluster.",
    )

    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.90,
        help="Target cumulative PCA variance for clustering.",
    )

    parser.add_argument(
        "--max-pca-components",
        type=int,
        default=12,
        help="Maximum number of PCA components used for clustering.",
    )

    parser.add_argument(
        "--sample-for-plots",
        type=int,
        default=10000,
        help="Maximum observations used in scatter plots.",
    )

    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=10,
        help="Bootstrap repeats used for cluster stability diagnostics.",
    )

    parser.add_argument(
        "--rfm-baseline",
        action="store_true",
        default=False,
        help="Compute and compare with RFM baseline segmentation.",
    )

    parser.add_argument(
        "--track-transitions",
        action="store_true",
        default=False,
        help="Track segment transitions across multiple time points.",
    )

    parser.add_argument(
        "--transition-dates",
        type=str,
        default=None,
        help="Comma-separated dates for segment transition tracking (ISO format).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------

def entropy_feature(
    df: pl.DataFrame,
    date_part: str,
    denominator: float,
    alias: str,
    date_column: str = "invoice_date",
) -> pl.DataFrame:

    if date_part == "hour":
        values = df.with_columns(pl.col(date_column).dt.hour().alias("bucket"))
    elif date_part == "weekday":
        values = df.with_columns(pl.col(date_column).dt.weekday().alias("bucket"))
    elif date_part == "month":
        values = df.with_columns(pl.col(date_column).dt.month().alias("bucket"))
    else:
        raise ValueError(f"Unsupported temporal component: {date_part}")

    counts = (
        values.group_by(["Customer ID", "bucket"])
        .len("n")
    )

    counts = counts.with_columns(
        (pl.col("n") / pl.col("n").sum().over("Customer ID")).alias("p")
    )

    entropy = (
        counts.group_by("Customer ID")
        .agg((-(pl.col("p") * pl.col("p").log()).sum() / math.log(denominator)).alias(alias))
    )

    return entropy


# ---------------------------------------------------------------------
# RFM Baseline Segmentation
# ---------------------------------------------------------------------

def compute_rfm_segments(
    tx: pl.DataFrame,
    reference_date: pl.Datetime,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) segments.
    
    Args:
        tx: Canonical transactions
        reference_date: Reference date for recency calculation
        n_quantiles: Number of quantile bins for each dimension (default 5 for quintiles)
    
    Returns:
        DataFrame with Customer ID and RFM scores + segment labels
    """
    sales = tx.filter(
        (pl.col("is_sale") & pl.col("is_positive_price") & (pl.col("InvoiceDate") <= reference_date))
    )
    
    if sales.height == 0:
        raise ValueError("No clean sales found for RFM computation")
    
    # Invoice-level aggregation
    invoice = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
            ]
        )
    )
    
    # Customer-level RFM
    customer_rfm = (
        invoice.group_by("Customer ID")
        .agg(
            [
                pl.col("invoice_date").max().alias("last_purchase"),
                pl.len().alias("frequency"),
                pl.col("invoice_revenue").sum().alias("monetary"),
            ]
        )
        .with_columns(
            [
                (pl.lit(reference_date) - pl.col("last_purchase")).dt.total_days().alias("recency_days"),
            ]
        )
    )
    
    # Quantile-based scoring (1 = worst, n_quantiles = best for F & M; reversed for R)
    for col in ["frequency", "monetary"]:
        customer_rfm = customer_rfm.with_columns(
            pl.col(col).qcut(n_quantiles, labels=list(range(1, n_quantiles + 1)), allow_duplicates=True)
            .cast(pl.Int64)
            .alias(f"{col}_score")
        )
    
    # Recency: lower is better (more recent), so reverse the quantiles
    customer_rfm = customer_rfm.with_columns(
        pl.col("recency_days").qcut(n_quantiles, labels=list(range(n_quantiles, 0, -1)), allow_duplicates=True)
        .cast(pl.Int64)
        .alias("recency_score")
    )
    
    # Combined RFM score
    customer_rfm = customer_rfm.with_columns(
        (pl.col("recency_score") * 100 + pl.col("frequency_score") * 10 + pl.col("monetary_score"))
        .alias("rfm_score")
    )
    
    # Segment labels based on RFM scores
    customer_rfm = customer_rfm.with_columns(
        pl.when(
            (pl.col("recency_score") >= n_quantiles - 1) & 
            (pl.col("frequency_score") >= n_quantiles - 1) & 
            (pl.col("monetary_score") >= n_quantiles - 1)
        ).then(pl.lit("Champions"))
        .when(
            (pl.col("recency_score") >= n_quantiles - 1) & 
            (pl.col("frequency_score") >= n_quantiles - 2)
        ).then(pl.lit("Loyal Customers"))
        .when(
            (pl.col("recency_score") >= n_quantiles - 2) & 
            (pl.col("frequency_score") >= n_quantiles - 2)
        ).then(pl.lit("Potential Loyalists"))
        .when(
            (pl.col("recency_score") >= n_quantiles - 1) & 
            (pl.col("frequency_score") <= 2)
        ).then(pl.lit("New Customers"))
        .when(
            (pl.col("recency_score") <= 2) & 
            (pl.col("frequency_score") >= n_quantiles - 1)
        ).then(pl.lit("At Risk"))
        .when(
            (pl.col("recency_score") <= 2) & 
            (pl.col("frequency_score") <= 2) & 
            (pl.col("monetary_score") >= n_quantiles - 2)
        ).then(pl.lit("Cannot Lose Them"))
        .when(
            (pl.col("recency_score") <= 2) & 
            (pl.col("frequency_score") <= 2)
        ).then(pl.lit("Hibernating"))
        .otherwise(pl.lit("Others"))
        .alias("rfm_segment")
    )
    
    return customer_rfm.select([
        "Customer ID", "recency_days", "frequency", "monetary",
        "recency_score", "frequency_score", "monetary_score",
        "rfm_score", "rfm_segment"
    ])


def compare_segmentations(
    hdbscan_segments: pl.DataFrame,
    rfm_segments: pl.DataFrame,
) -> Dict:
    """
    Compare HDBSCAN segments with RFM baseline.
    
    Returns:
        Dictionary with comparison metrics (ARI, NMI, confusion matrix)
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Join on Customer ID
    merged = hdbscan_segments.join(rfm_segments, on="Customer ID", how="inner")
    
    if merged.height == 0:
        return {"error": "No overlapping customers"}
    
    # Convert to numeric for ARI
    hdbscan_labels = merged["segment"].to_numpy()
    rfm_labels = merged["rfm_segment"].to_numpy()
    
    # Encode RFM segments as integers
    rfm_unique = {seg: i for i, seg in enumerate(sorted(set(rfm_labels)))}
    rfm_encoded = np.array([rfm_unique[s] for s in rfm_labels])
    
    ari = adjusted_rand_score(hdbscan_labels, rfm_encoded)
    nmi = normalized_mutual_info_score(hdbscan_labels, rfm_encoded)
    
    # Cross-tabulation
    crosstab = merged.group_by(["segment_name", "rfm_segment"]).len().sort("len", descending=True)
    
    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "overlapping_customers": merged.height,
        "hdbscan_segments": int(hdbscan_labels.max()) + 1 if hdbscan_labels.max() >= 0 else 0,
        "rfm_segments": len(rfm_unique),
        "crosstab": crosstab.to_pandas().to_dict(orient="records"),
    }


# ---------------------------------------------------------------------
# Segment Transition Tracking
# ---------------------------------------------------------------------

def compute_segment_transitions(
    tx: pl.DataFrame,
    feature_func,
    feature_groups: Dict,
    prediction_dates: List[str],
    min_cluster_size: int = 15,
    min_samples: int = 5,
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """
    Compute segment transitions across multiple prediction dates.
    
    Args:
        tx: Canonical transactions
        feature_func: Function to build customer features
        feature_groups: Feature group definitions
        prediction_dates: List of prediction dates (ISO format)
        min_cluster_size: HDBSCAN min_cluster_size
        min_samples: HDBSCAN min_samples
    
    Returns:
        Tuple of (transition_matrix, transition_summary_df)
    """
    from retail_ds.features import build_point_in_time_features
    from retail_ds.customer_month import build_customer_month_panel
    
    cm = build_customer_month_panel(tx)
    
    segment_results = []
    all_customers = set()
    
    for pred_date in prediction_dates:
        # Build features at this date
        customer_pl, _ = feature_func(tx.filter(pl.col("InvoiceDate") <= pred_date))
        
        # Robust preprocessing
        X, feature_columns, used_groups = robust_matrix(customer_pl, feature_groups)
        
        # PCA
        from sklearn.decomposition import PCA
        n_components = min(12, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        Z = pca.fit_transform(X)
        n_clustering = min(max(2, int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90)) + 1), Z.shape[1])
        Z_cluster = Z[:, :n_clustering]
        
        # Clustering
        labels, _, _, model = choose_clusters(Z_cluster, repeats=0)
        probabilities = get_probabilities(model, labels)
        
        # Store results
        df = customer_pl.select(["Customer ID"]).to_pandas()
        df["prediction_date"] = pred_date
        df["segment"] = labels
        df["segment_confidence"] = probabilities
        df["segment_name"] = [f"Segment_{int(l):02d}" if l >= 0 else "Noise" for l in labels]
        
        segment_results.append(df)
        all_customers.update(df["Customer ID"].tolist())
    
    # Combine all segments
    all_segments = pd.concat(segment_results, ignore_index=True)
    
    # Build transition matrix (between consecutive dates)
    transitions = []
    for i in range(len(prediction_dates) - 1):
        prev_date = prediction_dates[i]
        next_date = prediction_dates[i + 1]
        
        prev_seg = all_segments[all_segments["prediction_date"] == prev_date][["Customer ID", "segment", "segment_name"]].rename(
            columns={"segment": "segment_from", "segment_name": "segment_name_from"}
        )
        next_seg = all_segments[all_segments["prediction_date"] == next_date][["Customer ID", "segment", "segment_name"]].rename(
            columns={"segment": "segment_to", "segment_name": "segment_name_to"}
        )
        
        merged = prev_seg.merge(next_seg, on="Customer ID", how="outer")
        merged["transition_period"] = f"{prev_date} -> {next_date}"
        transitions.append(merged)
    
    if transitions:
        transition_df = pd.concat(transitions, ignore_index=True)
        
        # Transition matrix
        valid = transition_df[transition_df["segment_from"].notna() & transition_df["segment_to"].notna()]
        if len(valid) > 0:
            trans_matrix = pd.crosstab(
                valid["segment_name_from"], 
                valid["segment_name_to"], 
                normalize="index"
            ).fillna(0)
        else:
            trans_matrix = pd.DataFrame()
        
        # Summary stats
        summary_rows = []
        for period, group in transition_df.groupby("transition_period"):
            total = len(group)
            stable = len(group[group["segment_from"] == group["segment_to"]])
            noise_to_cluster = len(group[(group["segment_from"] == -1) & (group["segment_to"] >= 0)])
            cluster_to_noise = len(group[(group["segment_from"] >= 0) & (group["segment_to"] == -1)])
            churned = len(group[group["segment_from"].notna() & group["segment_to"].isna()])
            new = len(group[group["segment_from"].isna() & group["segment_to"].notna()])
            
            summary_rows.append({
                "period": period,
                "total_customers": total,
                "stable_pct": stable / total * 100 if total > 0 else 0,
                "noise_to_cluster_pct": noise_to_cluster / total * 100 if total > 0 else 0,
                "cluster_to_noise_pct": cluster_to_noise / total * 100 if total > 0 else 0,
                "churned_pct": churned / total * 100 if total > 0 else 0,
                "new_customers_pct": new / total * 100 if total > 0 else 0,
            })
        
        summary_df = pd.DataFrame(summary_rows)
    else:
        trans_matrix = pd.DataFrame()
        summary_df = pd.DataFrame()
    
    return trans_matrix, summary_df


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def build_customer_features(
    tx: pl.DataFrame,
) -> Tuple[pl.DataFrame, Dict[str, List[str]]]:

    LOGGER.info("Engineering customer behavioral features with Polars...")

    sales = tx.filter(
        (pl.col("is_sale") & pl.col("is_positive_price"))
    )

    all_customers = tx.select("Customer ID").unique()

    if sales.height == 0:
        raise ValueError("No positive purchase lines found.")

    # -------------------------------------------------------------
    # Invoice-level behavioral table
    # -------------------------------------------------------------

    invoices = (
        sales.group_by(["Customer ID", "Invoice"])
        .agg(
            [
                pl.col("InvoiceDate").min().alias("invoice_date"),
                pl.col("gross_merchandise_revenue").sum().alias("invoice_revenue"),
                pl.col("Quantity").sum().alias("invoice_units"),
                pl.len().alias("invoice_lines"),
                pl.col("StockCode").n_unique().alias("invoice_unique_products"),
            ]
        )
        .sort(["Customer ID", "invoice_date"])
        .with_columns(
            [
                pl.col("invoice_date").dt.weekday().alias("invoice_weekday"),
                pl.col("invoice_date").dt.hour().alias("invoice_hour"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Inter-purchase cadence
    # -------------------------------------------------------------

    invoice_dates = (
        invoices.select(["Customer ID", "invoice_date"])
        .with_columns(
            (
                pl.col("invoice_date").diff().over("Customer ID").dt.total_seconds() / 86400
            ).alias("interpurchase_days")
        )
    )

    cadence = (
        invoice_dates.group_by("Customer ID")
        .agg(
            [
                pl.col("interpurchase_days").median().alias("median_interpurchase_days"),
                pl.col("interpurchase_days").mean().alias("mean_interpurchase_days"),
                pl.col("interpurchase_days").std().alias("std_interpurchase_days"),
                pl.col("interpurchase_days").count().alias("interpurchase_count"),
                (
                    (pl.col("interpurchase_days") > 0).sum()
                    / pl.col("interpurchase_days").count()
                ).alias("positive_interpurchase_share"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("std_interpurchase_days")
                    / (pl.col("mean_interpurchase_days").abs() + 1e-9)
                )
                .fill_null(0.0)
                .alias("interpurchase_cv"),
                (
                    (pl.col("std_interpurchase_days") - pl.col("mean_interpurchase_days"))
                    / (pl.col("std_interpurchase_days") + pl.col("mean_interpurchase_days") + 1e-9)
                )
                .fill_null(0.0)
                .alias("burstiness"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Order value concentration
    # -------------------------------------------------------------

    invoice_value_shape = (
        invoices.with_columns(
            (
                pl.col("invoice_revenue")
                / pl.col("invoice_revenue").sum().over("Customer ID").clip(lower_bound=1e-9)
            ).alias("invoice_revenue_share")
        )
        .group_by("Customer ID")
        .agg((pl.col("invoice_revenue_share") ** 2).sum().alias("order_value_hhi"))
    )

    # -------------------------------------------------------------
    # Customer-level invoice features
    # -------------------------------------------------------------

    invoice_base = (
        invoices.group_by("Customer ID")
        .agg(
            [
                pl.len().alias("invoice_count"),
                pl.col("invoice_date").n_unique().alias("active_purchase_days"),
                pl.col("invoice_revenue").sum().alias("revenue"),
                pl.col("invoice_revenue").mean().alias("avg_order_value"),
                pl.col("invoice_revenue").median().alias("median_order_value"),
                pl.col("invoice_revenue").std().fill_null(0.0).alias("order_value_std"),
                (
                    pl.col("invoice_revenue").std().fill_null(0.0)
                    / (pl.col("invoice_revenue").mean().abs() + 1e-9)
                )
                .fill_null(0.0)
                .alias("order_value_cv"),
                pl.col("invoice_units").sum().alias("sales_units"),
                pl.col("invoice_lines").sum().alias("sales_lines"),
                pl.col("invoice_unique_products").sum().alias("invoice_product_count_sum"),
                pl.col("invoice_unique_products")
                .filter(pl.col("invoice_unique_products") == 1)
                .count()
                .alias("single_product_invoices"),
                pl.col("invoice_unique_products")
                .filter(pl.col("invoice_unique_products") > 1)
                .count()
                .alias("multi_product_invoices"),
                pl.col("invoice_date").min().alias("first_purchase"),
                pl.col("invoice_date").max().alias("last_purchase"),
                pl.col("invoice_date").dt.month_start().n_unique().alias("purchase_months"),
                pl.col("invoice_date").dt.date().n_unique().alias("purchase_days"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("invoice_count")
                    / (
                        (pl.col("last_purchase") - pl.col("first_purchase"))
                        .dt.total_days() / 30.4375
                    ).clip(lower_bound=1.0)
                )
                .alias("invoice_intensity_per_month"),
                (pl.col("sales_lines") / pl.col("invoice_count").clip(lower_bound=1))
                .alias("lines_per_invoice"),
                (pl.col("sales_units") / pl.col("invoice_count").clip(lower_bound=1))
                .alias("units_per_invoice"),
                (pl.col("invoice_product_count_sum") / pl.col("invoice_count").clip(lower_bound=1))
                .alias("products_per_invoice"),
                (pl.col("single_product_invoices") / pl.col("invoice_count").clip(lower_bound=1))
                .alias("single_product_invoice_share"),
                (pl.col("multi_product_invoices") / pl.col("invoice_count").clip(lower_bound=1))
                .alias("multi_product_invoice_share"),
                (pl.col("last_purchase") - pl.col("first_purchase"))
                .dt.total_days()
                .fill_null(0.0)
                .alias("tenure_days"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Product affinity and concentration
    # -------------------------------------------------------------

    customer_product = (
        sales.group_by(["Customer ID", "StockCode"])
        .agg(
            [
                pl.col("gross_merchandise_revenue").sum().alias("product_revenue"),
                pl.col("Invoice").n_unique().alias("product_invoice_count"),
            ]
        )
    )

    product_profile = (
        customer_product.with_columns(
            (
                pl.col("product_revenue")
                / pl.col("product_revenue").sum().over("Customer ID").clip(lower_bound=1e-9)
            ).alias("product_share")
        )
        .group_by("Customer ID")
        .agg(
            [
                (pl.col("product_share") ** 2).sum().alias("product_hhi"),
                pl.col("StockCode").n_unique().alias("unique_products"),
                ((pl.col("product_invoice_count") > 1).cast(pl.Float64)).mean().alias("repeat_product_ratio"),
            ]
        )
    )

    # -------------------------------------------------------------
    # Price behavior
    # -------------------------------------------------------------

    global_price_q75 = sales.select(pl.col("Price").quantile(0.75)).item()

    price_profile = (
        sales.group_by("Customer ID")
        .agg(
            [
                (
                    (pl.col("Price") * pl.col("Quantity")).sum()
                    / pl.col("Quantity").sum().clip(lower_bound=1)
                ).alias("weighted_avg_unit_price"),
                pl.col("Price").median().alias("median_unit_price"),
                (pl.col("Price").quantile(0.75) - pl.col("Price").quantile(0.25)).alias("price_iqr"),
                (pl.col("Price") > global_price_q75).mean().alias("premium_line_share"),
                (pl.col("Price") <= 0).mean().alias("zero_price_line_share"),
                pl.col("Price").mean().alias("mean_unit_price"),
                pl.col("Price").std().fill_null(0.0).alias("std_unit_price"),
            ]
        )
        .with_columns(
            (pl.col("std_unit_price") / (pl.col("mean_unit_price").abs() + 1e-9))
            .fill_null(0.0)
            .alias("line_price_cv")
        )
    )

    # -------------------------------------------------------------
    # Temporal signature
    # -------------------------------------------------------------

    temporal = (
        invoices.group_by("Customer ID")
        .agg(
            [
                (pl.col("invoice_weekday") >= 6).mean().alias("weekend_purchase_share"),
                ((pl.col("invoice_hour") >= 9) & (pl.col("invoice_hour") < 18)).mean().alias("business_hour_share"),
            ]
        )
    )

    temporal = temporal.join(
        entropy_feature(invoices, "hour", 24.0, "hour_entropy", date_column="invoice_date"),
        on="Customer ID", how="left"
    )
    temporal = temporal.join(
        entropy_feature(invoices, "weekday", 7.0, "weekday_entropy", date_column="invoice_date"),
        on="Customer ID", how="left"
    )
    temporal = temporal.join(
        entropy_feature(invoices, "month", 12.0, "month_entropy", date_column="invoice_date"),
        on="Customer ID", how="left"
    )

    # -------------------------------------------------------------
    # Returns / cancellations
    # -------------------------------------------------------------

    returns = tx.filter(pl.col("is_return") | pl.col("is_cancellation"))

    if returns.height:
        return_profile = (
            returns.group_by("Customer ID")
            .agg(
                [
                    pl.col("Quantity").filter(pl.col("Quantity") < 0).abs().sum().alias("return_units"),
                    pl.col("gross_merchandise_revenue").sum().alias("return_value"),
                    pl.col("Invoice").n_unique().alias("return_invoices"),
                    pl.len().alias("return_lines"),
                ]
            )
        )
    else:
        return_profile = pl.DataFrame(
            schema={
                "Customer ID": pl.Int64,
                "return_units": pl.Float64,
                "return_value": pl.Float64,
                "return_invoices": pl.Float64,
                "return_lines": pl.Float64,
            }
        )

    # -------------------------------------------------------------
    # Combine customer feature views
    # -------------------------------------------------------------

    features = all_customers

    for right in [
        invoice_base,
        invoice_value_shape,
        cadence,
        product_profile,
        price_profile,
        temporal,
        return_profile,
    ]:
        features = features.join(right, on="Customer ID", how="left")

    reference_date = sales.select(pl.col("InvoiceDate").max()).item()
    overall_first_date = sales.select(pl.col("InvoiceDate").min()).item()

    overall_span_days = max(
        1.0,
        float((reference_date - overall_first_date).total_seconds() / 86400.0),
    )

    features = features.with_columns(
        [
            pl.lit(reference_date).alias("reference_date"),
            ((pl.lit(reference_date) - pl.col("last_purchase")).dt.total_seconds() / 86400)
            .fill_null(0.0)
            .alias("days_since_last_purchase"),
            (
                (pl.lit(reference_date).dt.month_start() - pl.col("last_purchase").dt.month_start())
                .dt.total_days() / 30.4375
            )
            .fill_null(0.0)
            .alias("months_since_last_purchase"),
            (pl.col("active_purchase_days") / (pl.col("tenure_days") + 1.0)).alias("purchase_day_share"),
            (pl.col("tenure_days") / pl.lit(overall_span_days)).alias("active_span_ratio"),
            (pl.col("invoice_count") / pl.col("active_purchase_days").clip(lower_bound=1))
            .alias("invoice_count_per_active_day"),
            (pl.col("revenue") / pl.col("active_purchase_days").clip(lower_bound=1))
            .alias("revenue_per_active_day"),
            pl.col("revenue").log1p().alias("log_revenue"),
        ]
    )

    features = features.with_columns(
        [
            pl.col("return_units").fill_null(0.0),
            pl.col("return_value").fill_null(0.0),
            pl.col("return_invoices").fill_null(0.0),
            pl.col("return_lines").fill_null(0.0),
        ]
    )

    features = features.with_columns(
        [
            (pl.col("return_units") / (pl.col("sales_units") + pl.col("return_units")).clip(lower_bound=1.0))
            .alias("unit_return_rate"),
            (pl.col("return_value") / (pl.col("revenue") + pl.col("return_value")).clip(lower_bound=1.0))
            .alias("value_return_rate"),
            (pl.col("return_invoices") / (pl.col("invoice_count") + pl.col("return_invoices")).clip(lower_bound=1.0))
            .alias("return_invoice_rate"),
            (pl.col("return_lines") / (pl.col("sales_lines") + pl.col("return_lines")).clip(lower_bound=1.0))
            .alias("return_line_share"),
        ]
    )

    numeric_features = sorted(
        {feature for group in FEATURE_GROUPS.values() for feature in group}
    )

    features = features.with_columns(
        [
            pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0)
            for column in numeric_features
            if column in features.columns
        ]
    )

    features = features.with_columns(pl.col("Customer ID").cast(pl.Int64))

    feature_groups = {
        group: [feature for feature in features_list if feature in features.columns]
        for group, features_list in FEATURE_GROUPS.items()
    }

    LOGGER.info(
        "Customers: %s | engineered features: %s",
        f"{features.height:,}",
        len(numeric_features),
    )

    return features, feature_groups


# ---------------------------------------------------------------------
# Robust preprocessing + block balancing
# ---------------------------------------------------------------------

def robust_matrix(
    features: pl.DataFrame,
    feature_groups: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:

    columns = [
        feature
        for group in feature_groups.values()
        for feature in group
        if feature in features.columns
    ]

    X = features.select(columns).to_pandas().astype(float).to_numpy()
    feature_columns = columns

    # Winsorization.
    lower = np.nanpercentile(X, 1.0, axis=0)
    upper = np.nanpercentile(X, 99.0, axis=0)
    X = np.clip(X, lower, upper)

    # Remove zero-variance columns.
    stds = np.nanstd(X, axis=0)
    keep = stds > 1e-12
    X = X[:, keep]
    feature_columns = [column for column, keep_flag in zip(feature_columns, keep) if keep_flag]

    # Yeo-Johnson handles non-negative, zero-heavy, skewed, and negative return features.
    power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    X = power_transformer.fit_transform(X)

    robust_scaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=True)
    X = robust_scaler.fit_transform(X)

    # Equalize the behavioral views before PCA.
    used_groups = {}
    for group_name, group_columns in feature_groups.items():
        active = [column for column in group_columns if column in feature_columns]
        if not active:
            continue
        indices = [feature_columns.index(column) for column in active]
        X[:, indices] /= math.sqrt(len(indices))
        used_groups[group_name] = active

    return X, feature_columns, used_groups


# ---------------------------------------------------------------------
# HDBSCAN helpers
# ---------------------------------------------------------------------

def build_hdbscan(min_cluster_size: int, min_samples: int):
    if SKHDBSCAN is not None:
        return SKHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            n_jobs=-1,
        )
    if EXTERNAL_HDBSCAN is not None:
        return EXTERNAL_HDBSCAN.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
    return None


def get_probabilities(model, labels: np.ndarray) -> np.ndarray:
    probabilities = getattr(model, "probabilities_", None)
    if probabilities is None:
        output = np.ones(labels.shape[0], dtype=float)
        output[labels < 0] = 0.0
        return output
    return np.asarray(probabilities, dtype=float)


def make_hdbscan_candidates(n: int) -> List[Tuple[int, int]]:
    if n < 30:
        return [(max(5, n // 5), max(3, n // 10))]

    raw_sizes = [
        max(15, int(round(n * 0.005))),
        max(20, int(round(n * 0.01))),
        max(30, int(round(n * 0.02))),
        max(50, int(round(n * 0.03))),
        max(75, int(round(n * 0.05))),
    ]

    max_size = max(10, min(250, n // 3))
    cluster_sizes = sorted(
        set(min(max(10, size), max_size) for size in raw_sizes)
    )

    candidates = []
    for cluster_size in cluster_sizes:
        for fraction in (0.5, 1.0):
            min_samples = max(5, min(cluster_size, int(round(cluster_size * fraction))))
            candidates.append((cluster_size, min_samples))
    return candidates


def evaluate_hdbscan(X: np.ndarray, model) -> Dict[str, float]:
    labels = np.asarray(model.labels_)
    clustered = labels >= 0

    number_clusters = len(set(labels[clustered])) if clustered.any() else 0
    coverage = float(clustered.mean())
    probabilities = get_probabilities(model, labels)
    mean_probability = float(probabilities[clustered].mean()) if clustered.any() else 0.0

    if number_clusters >= 2 and clustered.sum() >= 20:
        sample_size = min(3000, int(clustered.sum()))
        candidate_indices = np.flatnonzero(clustered)
        if candidate_indices.size > sample_size:
            rng = np.random.default_rng(SEED)
            indices = rng.choice(candidate_indices, size=sample_size, replace=False)
        else:
            indices = candidate_indices
        y = labels[indices]
        try:
            silhouette = float(silhouette_score(X[indices], y, metric="euclidean"))
        except Exception:
            silhouette = -1.0
        try:
            davies_bouldin = float(davies_bouldin_score(X[indices], y))
        except Exception:
            davies_bouldin = float("inf")
    else:
        silhouette = -1.0
        davies_bouldin = float("inf")

    normalized_silhouette = (silhouette + 1.0) / 2.0
    normalized_db = (
        1.0 / (1.0 + max(davies_bouldin, 0.0)) if np.isfinite(davies_bouldin) else 0.0
    )

    quality = (
        0.45 * normalized_silhouette
        + 0.20 * normalized_db
        + 0.20 * coverage
        + 0.15 * mean_probability
    )

    if number_clusters < 2:
        quality *= 0.20
    if number_clusters > 20:
        quality *= 0.85

    return {
        "n_clusters": int(number_clusters),
        "coverage": coverage,
        "noise_share": 1.0 - coverage,
        "mean_probability": mean_probability,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin if np.isfinite(davies_bouldin) else None,
        "quality": float(quality),
    }


# ---------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------

def bootstrap_stability(
    X: np.ndarray,
    labels_full: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    repeats: int,
) -> float:
    if build_hdbscan(min_cluster_size, min_samples) is None or repeats <= 0:
        return float("nan")

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    sample_size = max(min(n - 1, int(0.8 * n)), min_cluster_size * 2)
    sample_size = min(sample_size, n - 1)

    scores = []
    for repeat in range(repeats):
        indices = rng.choice(n, size=sample_size, replace=False)
        model = build_hdbscan(min_cluster_size, min_samples)
        try:
            model.fit(X[indices])
            bootstrap_labels = np.asarray(model.labels_)
            full_labels_subset = labels_full[indices]
            ari = float(adjusted_rand_score(full_labels_subset, bootstrap_labels))
            scores.append(ari)
        except Exception as exc:
            LOGGER.warning("Bootstrap stability repeat %s failed: %s", repeat + 1, exc)

    if not scores:
        return float("nan")
    return float(np.mean(scores))


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def choose_clusters(
    X: np.ndarray,
    repeats: int = 10,
) -> Tuple[np.ndarray, Dict, pd.DataFrame, object]:

    n = X.shape[0]
    candidates = make_hdbscan_candidates(n)
    hdbscan_available = build_hdbscan(10, 5) is not None

    if not hdbscan_available:
        LOGGER.warning("HDBSCAN unavailable; using KMeans fallback")
        # KMeans fallback
        fallback_rows = []
        best_solution = None
        max_k = min(10, max(2, n // 25 + 1))
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, n_init=25, random_state=SEED, max_iter=500)
            labels = model.fit_predict(X)
            sample_size = min(3000, n)
            if n > sample_size:
                indices = np.random.default_rng(SEED).choice(n, size=sample_size, replace=False)
            else:
                indices = np.arange(n)
            sil = float(silhouette_score(X[indices], labels[indices]))
            quality = (sil + 1) / 2
            fallback_rows.append({"algorithm": "KMeans_fallback", "k": k, "silhouette": sil, "quality": quality})
            if best_solution is None or sil > best_solution[0]:
                best_solution = (sil, model, labels)
        candidate_df = pd.DataFrame(fallback_rows).sort_values("quality", ascending=False).reset_index(drop=True)
        best_silhouette, best_model, labels = best_solution
        metadata = {
            "algorithm": "KMeans_fallback",
            "k": int(best_model.n_clusters),
            "n_clusters": int(best_model.n_clusters),
            "coverage": 1.0,
            "noise_share": 0.0,
            "mean_probability": 1.0,
            "silhouette": float(best_silhouette),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "bootstrap_ari": None,
            "selection_score": float((best_silhouette + 1) / 2),
        }
        return labels, metadata, candidate_df, best_model

    LOGGER.info("HDBSCAN available; evaluating %s candidate parameterizations.", len(candidates))

    candidate_rows = []
    fitted_models = []

    for min_cluster_size, min_samples in candidates:
        model = build_hdbscan(min_cluster_size, min_samples)
        try:
            model.fit(X)
            metrics = evaluate_hdbscan(X, model)
            metrics.update(
                {
                    "algorithm": "HDBSCAN",
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                }
            )
            candidate_rows.append(metrics)
            fitted_models.append((metrics, model))
        except Exception as exc:
            LOGGER.warning("HDBSCAN candidate failed mcs=%s ms=%s: %s", min_cluster_size, min_samples, exc)

    if not candidate_rows:
        raise RuntimeError("No HDBSCAN candidates succeeded.")

    candidate_df = (
        pd.DataFrame(candidate_rows)
        .sort_values("quality", ascending=False)
        .reset_index(drop=True)
    )

    top_candidates = candidate_df.head(min(3, len(candidate_df)))
    stability_results = []

    for _, row in top_candidates.iterrows():
        min_cluster_size = int(row["min_cluster_size"])
        min_samples = int(row["min_samples"])
        matching_model = next(
            model
            for metrics, model in fitted_models
            if int(metrics["min_cluster_size"]) == min_cluster_size
            and int(metrics["min_samples"]) == min_samples
        )
        stability = bootstrap_stability(X, np.asarray(matching_model.labels_), min_cluster_size, min_samples, repeats)
        stability_results.append((min_cluster_size, min_samples, stability))

    candidate_df["bootstrap_ari"] = np.nan
    for min_cluster_size, min_samples, stability in stability_results:
        mask = (candidate_df.min_cluster_size == min_cluster_size) & (candidate_df.min_samples == min_samples)
        candidate_df.loc[mask, "bootstrap_ari"] = stability

    candidate_df["final_selection_score"] = candidate_df["quality"]
    valid_stability = candidate_df["bootstrap_ari"].notna()
    candidate_df.loc[valid_stability, "final_selection_score"] = (
        0.70 * candidate_df.loc[valid_stability, "quality"]
        + 0.30 * candidate_df.loc[valid_stability, "bootstrap_ari"].clip(lower=0.0, upper=1.0)
    )

    candidate_df = (
        candidate_df.sort_values("final_selection_score", ascending=False).reset_index(drop=True)
    )

    best = candidate_df.iloc[0]
    best_min_cluster_size = int(best["min_cluster_size"])
    best_min_samples = int(best["min_samples"])

    best_model = build_hdbscan(best_min_cluster_size, best_min_samples)
    best_model.fit(X)

    best_labels = np.asarray(best_model.labels_)
    bootstrap_ari_value = (
        None if pd.isna(best.get("bootstrap_ari")) else float(best["bootstrap_ari"])
    )

    metadata = {
        "algorithm": "HDBSCAN",
        "min_cluster_size": best_min_cluster_size,
        "min_samples": best_min_samples,
        "n_clusters": int(best["n_clusters"]),
        "coverage": float(best["coverage"]),
        "noise_share": float(best["noise_share"]),
        "mean_probability": float(best["mean_probability"]),
        "silhouette": float(best["silhouette"]),
        "davies_bouldin": best["davies_bouldin"],
        "bootstrap_ari": bootstrap_ari_value,
        "selection_score": float(best["final_selection_score"]),
    }

    return best_labels, metadata, candidate_df, best_model


# ---------------------------------------------------------------------
# Profile scoring
# ---------------------------------------------------------------------

def robust_profile_scores(features: pl.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X = (
        features.select(feature_columns)
        .to_pandas()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    scaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=True)
    Z = scaler.fit_transform(X)
    return pd.DataFrame(Z, columns=feature_columns)


def make_profile_tables(
    feature_df: pd.DataFrame,
    profile_z: pd.DataFrame,
    labels: np.ndarray,
    feature_groups: Dict[str, List[str]],
    probabilities: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    group_names = list(feature_groups.keys())
    group_scores = pd.DataFrame(index=np.arange(len(feature_df)))

    for group, columns in feature_groups.items():
        active_columns = [column for column in columns if column in profile_z.columns]
        if active_columns:
            group_scores[group] = profile_z[active_columns].median(axis=1)
        else:
            group_scores[group] = 0.0

    group_scores["cluster"] = labels
    group_scores["confidence"] = probabilities

    non_noise = group_scores[group_scores["cluster"] >= 0]

    if non_noise.empty:
        group_profile = pd.DataFrame()
    else:
        aggregation = {
            "customers": ("cluster", "size"),
            "confidence_mean": ("confidence", "mean"),
        }
        for group in group_names:
            aggregation[group] = (group, "mean")

        group_profile = (
            non_noise.groupby("cluster").agg(**aggregation).reset_index()
        )
        group_profile["share"] = group_profile["customers"] / len(feature_df)
        group_profile = group_profile.sort_values("customers", ascending=False).reset_index(drop=True)

    feature_cluster = profile_z.copy()
    feature_cluster["cluster"] = labels
    feature_cluster = feature_cluster[feature_cluster["cluster"] >= 0]

    if feature_cluster.empty:
        feature_medians = pd.DataFrame()
    else:
        feature_medians = (
            feature_cluster.groupby("cluster")[profile_z.columns].median().reset_index()
        )

    return group_profile, feature_medians


# ---------------------------------------------------------------------
# Visual diagnostics
# ---------------------------------------------------------------------

def save_visuals(
    out_dir: Path,
    features_pd: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    pca_xy: np.ndarray,
    pca_variance: np.ndarray,
    group_profile: pd.DataFrame,
    sample_for_plots: int,
) -> None:

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    n = len(features_pd)

    if n > sample_for_plots:
        indices = rng.choice(n, size=sample_for_plots, replace=False)
    else:
        indices = np.arange(n)

    # 01. PCA segment map
    fig, ax = plt.subplots(figsize=(11, 8))
    unique_labels = sorted(np.unique(labels))
    for cluster in unique_labels:
        cluster_indices = indices[labels[indices] == cluster]
        if cluster == -1:
            ax.scatter(
                pca_xy[cluster_indices, 0],
                pca_xy[cluster_indices, 1],
                s=9, alpha=0.20, label="Noise",
            )
        else:
            ax.scatter(
                pca_xy[cluster_indices, 0],
                pca_xy[cluster_indices, 1],
                s=12, alpha=0.55, label=f"Cluster {cluster}",
            )
    ax.set_title(
        f"Customer Behavioral Segments — PCA Space "
        f"({pca_variance[0] * 100:.1f}% / {pca_variance[1] * 100:.1f}% variance)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "01_pca_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 02. Segment size
    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sizes))
    ax.bar(x, sizes.values)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Noise" if label == -1 else f"Cluster {label}" for label in sizes.index],
        rotation=45, ha="right",
    )
    ax.set_ylabel("Customers")
    ax.set_title("Segment Size and Density-Noise Population")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_segment_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 03. Behavioral view heatmap
    if group_profile is not None and not group_profile.empty:
        heatmap_columns = [c for c in FEATURE_GROUPS.keys() if c in group_profile.columns]
        heatmap = group_profile.set_index("cluster")[heatmap_columns]
        array = heatmap.to_numpy()
        fig, ax = plt.subplots(
            figsize=(12, max(4.5, 0.55 * len(heatmap) + 2))
        )
        image = ax.imshow(array, aspect="auto")
        ax.set_yticks(np.arange(len(heatmap)))
        ax.set_yticklabels([f"Cluster {cluster}" for cluster in heatmap.index])
        ax.set_xticks(np.arange(len(heatmap.columns)))
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in heatmap.columns],
            rotation=30, ha="right",
        )
        ax.set_title("Behavioral View Profile — Cluster Median Robust Scores")
        for row_index in range(array.shape[0]):
            for column_index in range(array.shape[1]):
                ax.text(column_index, row_index, f"{array[row_index, column_index]:.2f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8, label="Cluster median robust score")
        fig.tight_layout()
        fig.savefig(plots_dir / "03_behavior_view_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 04. Economic intensity vs cadence
    revenue = features_pd["revenue"].to_numpy()
    invoice_count = features_pd["invoice_count"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        np.log1p(revenue[indices]),
        np.log1p(invoice_count[indices]),
        c=labels[indices], s=12, alpha=0.35,
    )
    ax.set_xlabel("log(1 + customer revenue)")
    ax.set_ylabel("log(1 + invoice count)")
    ax.set_title("Economic Intensity vs Purchase Cadence")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "04_value_vs_cadence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 05. Return/reversal behavior
    return_rate = features_pd["value_return_rate"].to_numpy()
    marker_size = 15 + 80 * probabilities[indices]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        np.log1p(revenue[indices]),
        return_rate[indices],
        c=labels[indices],
        s=marker_size,
        alpha=0.35,
    )
    ax.set_xlabel("log(1 + customer revenue)")
    ax.set_ylabel("Observed value return / reversal rate")
    ax.set_title("Customer Value vs Reversal Behavior (marker size = clustering confidence)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "05_value_vs_returns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    # Resolve output directory
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.segmentation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Load canonical transactions using shared package
    canonical_path = config.data_quality_dir / "canonical_transactions.parquet"
    if canonical_path.exists():
        LOGGER.info("Loading canonical transactions from %s", canonical_path)
        tx = pl.read_parquet(canonical_path)
    else:
        LOGGER.info("Building canonical transactions from raw data...")
        raw = load_raw_transactions(config.raw_data_path, args.sheet)
        tx = clean_transactions(raw)
        tx = add_calendar_fields(tx)
        tx = classify_transactions(tx)
        tx = compute_financial_measures(tx)

    # Run validations
    validation_results = run_all_validations(tx=tx)
    assert_validations_pass(validation_results)

    # Feature engineering
    customer_pl, feature_groups = build_customer_features(tx)

    customer_pd = customer_pl.to_pandas()

    if len(customer_pd) < args.min_customers:
        raise ValueError(
            f"Only {len(customer_pd):,} customers are available; need at least {args.min_customers:,}."
        )

    # Robust feature matrix
    X, feature_columns, used_groups = robust_matrix(customer_pl, feature_groups)
    LOGGER.info("Matrix shape after robust preprocessing: %s", X.shape)

    # PCA
    full_pca = PCA(
        n_components=min(args.max_pca_components, X.shape[1]),
        svd_solver="full",
        random_state=SEED,
    )
    Z = full_pca.fit_transform(X)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    requested_components = int(np.searchsorted(cumulative_variance, args.pca_variance)) + 1

    if Z.shape[1] >= 2:
        n_components_for_clustering = min(max(2, requested_components), Z.shape[1])
    else:
        n_components_for_clustering = 1

    Z_cluster = Z[:, :n_components_for_clustering]
    LOGGER.info(
        "PCA: %s features -> %s PCs for clustering; cumulative variance %.2f%%",
        X.shape[1], n_components_for_clustering,
        cumulative_variance[n_components_for_clustering - 1] * 100,
    )

    # Clustering
    labels, cluster_metadata, candidate_models, cluster_model = choose_clusters(
        Z_cluster, repeats=args.stability_repeats
    )
    probabilities = get_probabilities(cluster_model, labels)

    LOGGER.info(
        "Selected %s | clusters=%s | coverage=%.1f%% | noise=%.1f%% | silhouette=%.3f",
        cluster_metadata.get("algorithm"),
        cluster_metadata.get("n_clusters"),
        cluster_metadata.get("coverage", 0) * 100,
        cluster_metadata.get("noise_share", 0) * 100,
        cluster_metadata.get("silhouette", float("nan")),
    )

    # Customer segment output
    customer_pd["segment"] = labels
    customer_pd["segment_confidence"] = probabilities
    segment_names = []
    for label in labels:
        if label == -1:
            segment_names.append("Noise / low-density")
        else:
            segment_names.append(f"Segment_{int(label):02d}")
    customer_pd["segment_name"] = segment_names

    customer_pd.to_csv(output_dir / "customer_segments.csv", index=False)

    try:
        segmented_polars = customer_pl.with_columns(
            [pl.Series("segment", labels), pl.Series("segment_confidence", probabilities)]
        )
        segmented_polars.write_parquet(output_dir / "customer_segments.parquet")
    except Exception as exc:
        LOGGER.warning("Parquet export failed: %s", exc)

    # PCA coordinates
    pca_data = {f"PC{i + 1}": Z[:, i] for i in range(Z.shape[1])}
    pca_pd = pd.DataFrame(pca_data)
    pca_pd.insert(0, "Customer ID", customer_pd["Customer ID"].to_numpy())
    pca_pd["segment"] = labels
    pca_pd["segment_confidence"] = probabilities
    pca_pd.to_csv(output_dir / "pca_coordinates.csv", index=False)

    # PCA loadings
    loadings = pd.DataFrame(
        full_pca.components_.T,
        index=feature_columns,
        columns=[f"PC{i + 1}" for i in range(full_pca.components_.shape[0])],
    )
    loadings.to_csv(output_dir / "pca_feature_loadings.csv")

    # Cluster profiles
    profile_z = robust_profile_scores(customer_pl, feature_columns)
    group_profile, feature_profile = make_profile_tables(
        customer_pd, profile_z, labels, used_groups, probabilities
    )
    group_profile.to_csv(output_dir / "segment_profiles.csv", index=False)
    feature_profile.to_csv(output_dir / "segment_feature_medians_robust.csv", index=False)

    raw_medians = customer_pd.assign(segment=labels)
    raw_medians = (
        raw_medians[raw_medians["segment"] >= 0]
        .groupby("segment")[feature_columns]
        .median()
        .reset_index()
    )
    raw_medians.to_csv(output_dir / "segment_feature_medians_raw.csv", index=False)

    # Model candidate diagnostics
    candidate_models.to_csv(output_dir / "cluster_model_candidates.csv", index=False)

    # Visuals
    if Z.shape[1] >= 2:
        pca_visual = Z[:, :2]
        pca_visual_variance = np.array([full_pca.explained_variance_ratio_[0], full_pca.explained_variance_ratio_[1]])
    else:
        pca_visual = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])
        pca_visual_variance = np.array([full_pca.explained_variance_ratio_[0], 0.0])

    # RFM Baseline comparison
    rfm_comparison = None
    if args.rfm_baseline:
        LOGGER.info("Computing RFM baseline segmentation...")
        reference_date = customer_pd["reference_date"].iloc[0]
        rfm_segments = compute_rfm_segments(tx, reference_date)
        rfm_segments.write_parquet(output_dir / "rfm_segments.parquet")
        rfm_segments.to_pandas().to_csv(output_dir / "rfm_segments.csv", index=False)
        
        hdbscan_segments = pl.DataFrame({
            "Customer ID": customer_pd["Customer ID"],
            "segment": labels,
            "segment_name": customer_pd["segment_name"]
        })
        
        rfm_comparison = compare_segmentations(hdbscan_segments, rfm_segments)
        with open(output_dir / "rfm_comparison.json", "w") as f:
            json.dump(rfm_comparison, f, indent=2, default=str)
        
        LOGGER.info(
            "RFM Comparison: ARI=%.3f, NMI=%.3f, Overlapping=%d",
            rfm_comparison.get("adjusted_rand_index", 0),
            rfm_comparison.get("normalized_mutual_info", 0),
            rfm_comparison.get("overlapping_customers", 0),
        )

    # Segment Transition Tracking
    transition_summary = None
    if args.track_transitions:
        LOGGER.info("Computing segment transitions...")
        if args.transition_dates:
            pred_dates = [d.strip() for d in args.transition_dates.split(",")]
        else:
            pred_dates = ["2010-09-30", "2010-12-31", "2011-03-31", "2011-06-30", "2011-09-30"]
        
        trans_matrix, transition_summary = compute_segment_transitions(
            tx, build_customer_features, FEATURE_GROUPS, pred_dates
        )
        
        if not trans_matrix.empty:
            trans_matrix.to_csv(output_dir / "segment_transition_matrix.csv")
        if not transition_summary.empty:
            transition_summary.to_csv(output_dir / "segment_transition_summary.csv", index=False)
        
        LOGGER.info("Segment transition tracking completed")

    save_visuals(
        out_dir=output_dir,
        features_pd=customer_pd,
        labels=labels,
        probabilities=probabilities,
        pca_xy=pca_visual,
        pca_variance=pca_visual_variance,
        group_profile=group_profile,
        sample_for_plots=args.sample_for_plots,
    )

    # Feature dictionary
    feature_dictionary = []
    for group, columns in used_groups.items():
        for column in columns:
            feature_dictionary.append({"feature": column, "behavioral_view": group})
    pd.DataFrame(feature_dictionary).to_csv(output_dir / "feature_dictionary.csv", index=False)

    # Model metadata
    metadata = {
        "input": str(config.raw_data_path),
        "rows_after_cleaning": int(tx.height),
        "customers": int(customer_pd.shape[0]),
        "reference_date_for_recency": str(customer_pd["reference_date"].iloc[0]),
        "feature_count_before_variance_filter": int(
            sum(len(f) for f in feature_groups.values())
        ),
        "feature_count_after_variance_filter": int(len(feature_columns)),
        "feature_groups": used_groups,
        "pca_components_full": int(Z.shape[1]),
        "pca_components_for_clustering": int(n_components_for_clustering),
        "pca_cumulative_variance_for_clustering": float(
            cumulative_variance[n_components_for_clustering - 1]
        ),
        "pca_explained_variance_ratio": [float(v) for v in full_pca.explained_variance_ratio_],
        "cluster_model": cluster_metadata,
        "seed": SEED,
        "config_hash": config.config_hash,
        "run_id": getattr(args, "run_id", None),
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Run manifest
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,  # Simplified
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "files": [
            "customer_segments.csv",
            "customer_segments.parquet",
            "pca_coordinates.csv",
            "pca_feature_loadings.csv",
            "segment_profiles.csv",
            "segment_feature_medians_robust.csv",
            "segment_feature_medians_raw.csv",
            "cluster_model_candidates.csv",
            "feature_dictionary.csv",
            "model_metadata.json",
            "plots/01_pca_clusters.png",
            "plots/02_segment_sizes.png",
            "plots/03_behavior_view_heatmap.png",
            "plots/04_value_vs_cadence.png",
            "plots/05_value_vs_returns.png",
        ],
    }
    
    if args.rfm_baseline:
        manifest["files"].extend(["rfm_segments.csv", "rfm_segments.parquet", "rfm_comparison.json"])
    
    if args.track_transitions:
        manifest["files"].extend(["segment_transition_matrix.csv", "segment_transition_summary.csv"])
    
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("Finished. Outputs written to: %s", output_dir)


if __name__ == "__main__":
    main()