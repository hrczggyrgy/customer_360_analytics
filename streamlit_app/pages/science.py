"""
Science & Governance workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_row,
    render_missing,
)
from ..ui.scope import apply_scope_to_dataframe, ScopeApplicability
from ..app_data import get_registry
from ..app_formatting import format_count


def render() -> None:
    """Render the Science & Governance workspace."""
    registry = get_registry()
    
    # =============================================================================
    # SECTION 1: DATA QUALITY
    # =============================================================================
    render_section_label("Data Quality")
    
    quality = registry.load_dataframe("data_quality", "validation_results.json")
    schema = registry.load_dataframe("data_quality", "schema_report.csv")
    missingness = registry.load_dataframe("data_quality", "missingness_report.csv")
    duplicate = registry.load_dataframe("data_quality", "duplicate_report.csv")
    txn_type = registry.load_dataframe("data_quality", "transaction_type_report.csv")
    recon = registry.load_dataframe("data_quality", "reconciliation_report.json")
    
    if quality is not None:
        if isinstance(quality, dict):
            # JSON loaded as dict
            for key, val in quality.items():
                st.metric(key.replace("_", " ").title(), str(val))
        else:
            st.dataframe(quality, use_container_width=True, hide_index=True)
    
    if schema is not None:
        with st.expander("Schema report"):
            st.dataframe(schema, use_container_width=True, hide_index=True)
    
    if missingness is not None:
        with st.expander("Missingness report"):
            st.dataframe(missingness, use_container_width=True, hide_index=True)
    
    if duplicate is not None:
        with st.expander("Duplicate report"):
            st.dataframe(duplicate, use_container_width=True, hide_index=True)
    
    if txn_type is not None:
        with st.expander("Transaction type classification"):
            st.dataframe(txn_type, use_container_width=True, hide_index=True)
    
    if recon is not None:
        with st.expander("Financial reconciliation"):
            if isinstance(recon, dict):
                st.json(recon)
            else:
                st.dataframe(recon, use_container_width=True, hide_index=True)
    
    if all(x is None for x in [quality, schema, missingness, duplicate, txn_type, recon]):
        render_missing("Data quality outputs not available.")
    
    # =============================================================================
    # SECTION 2: FEATURE GOVERNANCE
    # =============================================================================
    render_section_label("Feature Governance")
    
    features = registry.load_dataframe("customer_360", "feature_dictionary.json")
    feature_groups = registry.load_dataframe("customer_360", "feature_groups.json")
    validation = registry.load_dataframe("customer_360", "validation_results.json")
    unified_validation = registry.load_dataframe("customer_360", "unified_validation.json")
    pit_metadata = registry.load_dataframe("customer_360", "customer_360_metadata.json")
    
    if features is not None:
        if isinstance(features, dict):
            st.metric("Total features", len(features))
            with st.expander("Feature dictionary"):
                st.json(features)
        else:
            st.dataframe(features, use_container_width=True, hide_index=True)
    
    if feature_groups is not None:
        with st.expander("Feature groups (blocks)"):
            if isinstance(feature_groups, dict):
                st.json(feature_groups)
            else:
                st.dataframe(feature_groups, use_container_width=True, hide_index=True)
    
    if validation is not None:
        with st.expander("Feature validation"):
            if isinstance(validation, dict):
                st.json(validation)
            else:
                st.dataframe(validation, use_container_width=True, hide_index=True)
    
    if unified_validation is not None:
        with st.expander("Unified snapshot validation"):
            if isinstance(unified_validation, dict):
                st.json(unified_validation)
            else:
                st.dataframe(unified_validation, use_container_width=True, hide_index=True)
    
    if pit_metadata is not None:
        with st.expander("Point-in-time metadata"):
            if isinstance(pit_metadata, dict):
                st.json(pit_metadata)
            else:
                st.dataframe(pit_metadata, use_container_width=True, hide_index=True)
    
    render_science_card(
        "Point-in-time feature engine",
        "All predictive features are constructed using only information available "
        "at the prediction date. The feature engine enforces temporal ordering "
        "through explicit `point_in_time_safe` flags and rolling-window "
        "computations that never peek at future data."
    )
    
    # =============================================================================
    # SECTION 3: MODEL PERFORMANCE
    # =============================================================================
    render_section_label("Model Performance")
    
    models = [
        ("Segmentation", "segmentation"),
        ("CLV Proxy", "clv"),
        ("Churn / Survival", "churn"),
        ("Next Purchase", "churn"),
        ("Reactivation", "reactivation"),
        ("Recommendations", "recommendations"),
        ("Decision Engine", "decision_engine"),
        ("Market Basket", "market_basket"),
    ]
    
    model_diagnostics = []
    
    for label, module in models:
        card = registry.load_model_card(module)
        
        if card and "metrics" in card:
            metrics = card.get("metrics", {})
            model_diagnostics.append({
                "Model": label,
                "Status": "Available",
                "ROC-AUC": f"{metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') else "—",
                "PR-AUC": f"{metrics.get('pr_auc', 0):.3f}" if metrics.get('pr_auc') else "—",
                "Brier": f"{metrics.get('brier', 0):.4f}" if metrics.get('brier') else "—",
                "Log Loss": f"{metrics.get('log_loss', 0):.4f}" if metrics.get('log_loss') else "—",
                "Calibration": "Available" if "calibration" in card else "—",
            })
        elif card:
            model_diagnostics.append({
                "Model": label,
                "Status": "Available (no metrics)",
            })
        else:
            model_diagnostics.append({
                "Model": label,
                "Status": "Not available",
            })
    
    if model_diagnostics:
        st.dataframe(pd.DataFrame(model_diagnostics), use_container_width=True, hide_index=True)
    
    # =============================================================================
    # SECTION 4: CALIBRATION
    # =============================================================================
    render_section_label("Calibration")
    
    for label, module in [("Churn / Inactivity Risk", "churn"), ("Next Purchase", "churn")]:
        card = registry.load_model_card(module)
        
        if card and "calibration" in card:
            st.markdown(f"#### {label}")
            cal = card["calibration"]
            
            if isinstance(cal, dict) and "bins" in cal:
                cal_df = pd.DataFrame(cal["bins"])
                st.dataframe(cal_df, use_container_width=True, hide_index=True)
            elif isinstance(cal, list):
                cal_df = pd.DataFrame(cal)
                st.dataframe(cal_df, use_container_width=True, hide_index=True)
        else:
            st.caption(f"{label}: Calibration data not available.")
    
    render_science_card(
        "Calibration importance",
        "A well-calibrated model produces probabilities that match observed frequencies. "
        "If the model predicts 30% churn risk, ~30% of those customers should actually "
        "become inactive. Isotonic and Platt calibration are applied to ensure "
        "probabilities are decision-ready."
    )
    
    # =============================================================================
    # SECTION 5: CLV VALIDATION
    # =============================================================================
    render_section_label("CLV Validation")
    
    clv_card = registry.load_model_card("clv")
    bgnbd_card = registry.load_model_card("clv")  # bgnbd benchmark in same dir
    
    if clv_card:
        if "validation" in clv_card:
            val = clv_card["validation"]
            if isinstance(val, dict):
                for k, v in val.items():
                    st.metric(k.replace("_", " ").title(), str(v))
        
        if "temporal_split" in clv_card:
            with st.expander("Temporal split details"):
                st.json(clv_card["temporal_split"])
    
    # BG/NBD benchmark
    bgnbd_path = registry.load_dataframe("clv", "bgnbd_benchmark/bgnbd_predictions.parquet")
    if bgnbd_path is not None:
        with st.expander("BG/NBD Benchmark"):
            st.metric("Customers", len(bgnbd_path))
            if "clv_mean" in bgnbd_path.columns:
                st.metric("Mean BG/NBD CLV", format_currency(bgnbd_path["clv_mean"].mean()))
    
    render_science_card(
        "CLV validation approach",
        "CLV Proxy is validated using temporal holdouts: train on early periods, "
        "predict on later periods. Uncertainty intervals (p10-p90) are generated "
        "via Monte Carlo simulation. BG/NBD provides a probabilistic benchmark "
        "for comparison. Note: this is a proxy — no margin data, no true lifetime model."
    )
    
    # =============================================================================
    # SECTION 6: RECOMMENDATION EVALUATION
    # =============================================================================
    render_section_label("Recommendation Evaluation")
    
    rec_card = registry.load_model_card("recommendations")
    
    if rec_card:
        if "temporal_evaluation" in rec_card:
            with st.expander("Temporal holdout evaluation"):
                st.json(rec_card["temporal_evaluation"])
        
        if "baseline_comparison" in rec_card:
            with st.expander("Baseline comparison"):
                st.json(rec_card["baseline_comparison"])
    
    challenger_results = registry.load_dataframe("recommendations", "recommendation_challenger/challenger_results.parquet")
    if challenger_results is not None:
        with st.expander("Challenger model results"):
            st.dataframe(challenger_results, use_container_width=True, hide_index=True)
    
    render_science_card(
        "Recommendation evaluation",
        "Recommendations are evaluated on temporal holdout: models trained on data "
        "before cutoff date, evaluated on purchases after cutoff. "
        "Hit rate, recall, MRR, and NDCG measure ranking quality. "
        "Popularity baseline provides a lower bound for model utility."
    )
    
    # =============================================================================
    # SECTION 7: SEGMENTATION STABILITY
    # =============================================================================
    render_section_label("Segmentation Stability")
    
    seg_meta = registry.load_model_card("segmentation")
    
    if seg_meta:
        if "stability" in seg_meta:
            st.json(seg_meta["stability"])
        
        if "ari_scores" in seg_meta:
            with st.expander("Bootstrap ARI scores"):
                st.json(seg_meta["ari_scores"])
    
    candidates = registry.load_dataframe("segmentation", "cluster_model_candidates.csv")
    if candidates is not None:
        with st.expander("Cluster model candidates"):
            st.dataframe(candidates, use_container_width=True, hide_index=True)
    
    render_science_card(
        "Segmentation stability",
        "Behavioral segmentation uses HDBSCAN with bootstrap stability assessment. "
        "ARI (Adjusted Rand Index) > 0.95 across 10 repeats indicates stable clusters. "
        "Feature blocks are balanced before PCA to prevent any single domain from dominating."
    )
    
    # =============================================================================
    # SECTION 8: RUN LINEAGE
    # =============================================================================
    render_section_label("Run Lineage")
    
    statuses = registry.get_all_module_statuses()
    
    lineage_data = []
    for module, status in statuses.items():
        if status.primary_artifact:
            art = status.primary_artifact
            lineage_data.append({
                "Module": module,
                "Status": status.overall_status.title(),
                "Run ID": art.run_id[:8] + "..." if art.run_id else "—",
                "Data Version": art.data_version or "—",
                "Code Version": art.code_version or "—",
                "Model Version": art.model_version or "—",
                "Generated": art.generated_at.strftime("%Y-%m-%d %H:%M") if art.generated_at else "—",
                "Freshness": status.freshness_note,
            })
    
    if lineage_data:
        st.dataframe(pd.DataFrame(lineage_data), use_container_width=True, hide_index=True)
        
        # Cross-run consistency check
        run_ids = set()
        data_versions = set()
        code_versions = set()
        
        for item in lineage_data:
            if item["Run ID"] != "—":
                run_ids.add(item["Run ID"])
            if item["Data Version"] != "—":
                data_versions.add(item["Data Version"])
            if item["Code Version"] != "—":
                code_versions.add(item["Code Version"])
        
        if len(run_ids) > 1:
            st.warning(f"⚠️ Multiple run IDs detected: {', '.join(run_ids)}")
        if len(data_versions) > 1:
            st.warning(f"⚠️ Multiple data versions detected: {', '.join(data_versions)}")
        if len(code_versions) > 1:
            st.warning(f"⚠️ Multiple code versions detected: {', '.join(code_versions)}")
        
        if len(run_ids) == 1 and len(data_versions) == 1 and len(code_versions) == 1:
            st.success("✅ All artifacts from a single consistent run")
    
    # =============================================================================
    # SECTION 9: METHODOLOGY
    # =============================================================================
    render_section_label("Methodology")
    
    steps = [
        (
            "01",
            "Data Quality",
            "Standardize transactions, identify reversals/cancellations, validate identifiers, and preserve a clear observation window.",
        ),
        (
            "02",
            "Customer 360",
            "Create one reusable analytical customer mart containing economics, cadence, assortment, price behavior, temporal behavior, lifecycle, and returns.",
        ),
        (
            "03",
            "Behavioral Segmentation",
            "Robustly transform heterogeneous customer features, balance behavioral blocks, denoise with PCA, and cluster in latent behavioral space.",
        ),
        (
            "04",
            "Cohort Analysis",
            "Measure customer retention and economic decay by acquisition cohort age, keeping return-only months visible to avoid understating commercial friction.",
        ),
        (
            "05",
            "Predicted Future Net Revenue (CLV Proxy)",
            "Project customer economics into the future and retain an uncertainty representation rather than reducing value to one deterministic number.",
        ),
        (
            "06",
            "Retention and Purchase Propensity",
            "Model forward-looking customer state using temporal splits so future purchase behavior does not leak into model development.",
        ),
        (
            "07",
            "Product Recommendations",
            "Generate co-purchase based product recommendations with transparent scoring and popularity fallback.",
        ),
        (
            "08",
            "Market Basket Analysis",
            "Discover transaction-level association rules (support, confidence, lift, conviction, leverage) distinct from customer-level affinity.",
        ),
        (
            "09",
            "Commercial Decision Engine",
            "Combine value, risk, propensity, behavioral context, confidence, and capacity into a transparent next-best-action policy.",
        ),
        (
            "10",
            "Experimentation",
            "The causal layer comes last: actual interventions need randomized or otherwise defensible treatment/control data before incremental impact is claimed.",
        ),
    ]
    
    for number_, title, body in steps:
        st.markdown(
            f"""
            <div style='border-left: 3px solid var(--primary); padding: 9px 0 9px 14px; margin: 8px 0;'>
                <strong style='color: var(--text-primary);'>{number_} · {title}</strong>
                <div style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.45; margin-top: 4px;'>{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    render_science_card(
        "Important scientific boundary",
        "Transaction data is excellent for understanding observed customer "
        "behavior and forecasting future behavior. It is not, by itself, enough "
        "to establish that a campaign, discount, reminder, or recommendation "
        "caused an incremental outcome. The portfolio should explicitly preserve "
        "this distinction and introduce experimentation for the final causal layer."
    )
    
    # =============================================================================
    # SECTION 10: LIMITATIONS
    # =============================================================================
    render_section_label("Known Limitations")
    
    limitations = [
        "NRR > 100%: Retail revenue index semantics need clearer documentation",
        "CLV uncertainty: Empirical coverage of p10/p90 intervals not verified",
        "Margin scenarios: Assume constant rate; no COGS data in Online Retail II",
        "Recommendation recall: Low recall expected for sparse retail data (baseline popularity outperforms on hit rate)",
        "Reactivation model: Framework exists (is_reactivation flag), dedicated model not implemented",
        "Causal claims: All model outputs are observational; no experimental validation layer",
        "Single-country focus: Model trained on UK-dominant data; generalization untested",
        "Temporal scope: Observation window limited to dataset time range",
    ]
    
    for lim in limitations:
        st.markdown(f"- {lim}")