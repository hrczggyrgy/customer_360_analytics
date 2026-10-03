"""
Segments workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_row,
    render_insight,
    render_missing,
)
from ..ui.scope import apply_scope_to_dataframe
from ..app_data import get_registry
from ..app_formatting import format_currency, format_probability, format_percent, format_count, format_ratio
from ..ui.charts import (
    plot_horizontal_bar,
    plot_segment_heatmap,
    plot_transition_matrix,
    plot_scatter,
    PLOTLY_CONFIG,
    apply_plotly_theme,
)


def render() -> None:
    """Render the Segments workspace."""
    registry = get_registry()
    
    # Load data
    segments = registry.load_dataframe("segmentation")
    profiles = registry.load_dataframe("segmentation", "segment_profiles.csv")
    pca = registry.load_dataframe("segmentation", "pca_coordinates.csv")
    transition = registry.load_dataframe("segmentation", "segment_transition_matrix.csv")
    transition_summary = registry.load_dataframe("segmentation", "segment_transition_summary.csv")
    rfm = registry.load_dataframe("segmentation", "rfm_segments.csv")
    combined = registry.load_dataframe("customer_360")
    
    if segments is None and profiles is None and pca is None:
        render_missing("Run customer_segmentation.py to populate this workspace.")
        st.stop()
    
    # Apply global scope
    if combined is not None:
        combined = apply_scope_to_dataframe(combined)
    
    # =============================================================================
    # SECTION 1: SEGMENT PORTFOLIO MAP
    # =============================================================================
    render_section_label("Segment portfolio map")
    
    # Use hdbscan_segment from unified customer_360 (matches segmentation artifact)
    seg_col = "hdbscan_segment" if combined is not None and "hdbscan_segment" in combined.columns else None
    
    if combined is not None and seg_col and seg_col in combined.columns:
        # Find value and activity columns
        clv_col = None
        for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
            if c in combined.columns:
                clv_col = c
                break
        
        churn_col = None
        for c in ["churn_probability", "churn_prob", "prob_churn"]:
            if c in combined.columns:
                churn_col = c
                break
        
        np_col = None
        for c in ["next_purchase_30d_probability", "next_purchase_probability", "purchase_probability_30d"]:
            if c in combined.columns:
                np_col = c
                break
        
        if clv_col and churn_col and np_col:
            # Aggregate by segment
            seg_col = "hdbscan_segment"
            seg_data = combined.dropna(subset=[seg_col]).copy()
            seg_data[clv_col] = pd.to_numeric(seg_data[clv_col], errors="coerce")
            seg_data[churn_col] = pd.to_numeric(seg_data[churn_col], errors="coerce")
            seg_data[np_col] = pd.to_numeric(seg_data[np_col], errors="coerce")
            
            profile = (
                seg_data.dropna(subset=[clv_col, churn_col, np_col])
                .groupby(seg_col)
                .agg(
                    customers=(clv_col, "size"),
                    revenue=(clv_col, "sum"),
                    value_per_customer=(clv_col, "mean"),
                    median_clv=(clv_col, "median"),
                    retention=(np_col, "mean"),
                    risk=(churn_col, "mean"),
                    avg_aov=("aov", "mean") if "aov" in seg_data.columns else (clv_col, "mean"),
                    product_breadth=("unique_products", "mean") if "unique_products" in seg_data.columns else (clv_col, "mean"),
                )
                .reset_index()
                .sort_values("revenue", ascending=False)
            )
            
            # Portfolio map: X = retention/activity, Y = value per customer
            fig = plot_scatter(
                profile,
                x_col="retention",
                y_col="value_per_customer",
                size_col="customers",
                color_col=seg_col,
                title="Segment Portfolio: Retention vs Value per Customer",
                labels={"retention": "Purchase Propensity (activity)", "value_per_customer": "Value per Customer"},
                max_points=50,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.caption("Bubble size = customer count. X-axis = purchase propensity (retention proxy). Y-axis = mean predicted value per customer.")
        else:
            render_missing("CLV, churn, and purchase propensity columns needed for portfolio map.")
    else:
        render_missing("Combined customer data with hdbscan_segment required.")
    
    # =============================================================================
    # SECTION 2: SEGMENT SCORECARD
    # =============================================================================
    render_section_label("Segment scorecard")
    
    if profiles is not None:
        p = profiles.copy()
        seg = "cluster" if "cluster" in p.columns else ("segment" if "segment" in p.columns else "segment_name")
        exclude = {seg, "customers", "confidence_mean", "share"}
        numeric = p.select_dtypes(include=["number"]).columns.tolist()
        metric_cols = [c for c in numeric if c not in exclude]
        
        # Default display columns
        default_display = [seg, "customers"]
        for c in ["total_revenue", "revenue", "avg_revenue", "median_clv", "retention", "risk", "avg_products", "aov"]:
            if c in p.columns:
                default_display.append(c)
        
        # Sort selector
        sort_options = [c for c in default_display if c != seg and c in p.columns]
        sort_by = st.selectbox("Sort by", sort_options, index=0, key="segment_sort")
        ascending = st.checkbox("Ascending", value=False, key="segment_sort_asc")
        
        display_df = p[default_display]
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(sort_by, ascending=ascending)
        
        # Format for display
        for col in display_df.columns:
            if col in p.columns:
                sem_type = col.lower()
                if any(kw in sem_type for kw in ["revenue", "clv", "value", "aov", "price"]):
                    display_df[col] = display_df[col].apply(format_currency)
                elif any(kw in sem_type for kw in ["retention", "risk", "rate", "share"]):
                    display_df[col] = display_df[col].apply(format_probability)
                elif "count" in sem_type or "customers" in sem_type or "products" in sem_type:
                    display_df[col] = display_df[col].apply(format_count)
                elif "ratio" in sem_type or "lift" in sem_type:
                    display_df[col] = display_df[col].apply(format_ratio)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        render_missing("Segment profiles not available.")
    
    # =============================================================================
    # SECTION 3: SEGMENT COMPOSITION
    # =============================================================================
    render_section_label("Segment composition — Behavioral profile heatmap")
    
    if profiles is not None:
        p = profiles.copy()
        seg = "cluster" if "cluster" in p.columns else ("segment" if "segment" in p.columns else "segment_name")
        exclude = {seg, "customers", "confidence_mean", "share"}
        numeric = p.select_dtypes(include=["number"]).columns.tolist()
        metric_cols = [c for c in numeric if c not in exclude]
        
        if seg and metric_cols:
            # Group metrics by feature block
            block_prefixes = {
                "Economic": ["revenue", "value", "clv", "aov", "order", "spend"],
                "Cadence": ["interpurchase", "frequency", "recency", "cadence", "tenure"],
                "Assortment": ["product", "breadth", "unique", "hhi", "repeat"],
                "Pricing": ["price", "premium", "unit_price"],
                "Temporal": ["month", "weekday", "hour", "entropy", "seasonal"],
                "Lifecycle": ["lifecycle", "active", "inactive", "reactivation", "churn"],
                "Returns": ["return", "cancellation", "refund"],
            }
            
            selected_blocks = st.multiselect(
                "Feature blocks",
                options=list(block_prefixes.keys()),
                default=list(block_prefixes.keys()),
                key="segment_blocks_select",
            )
            
            selected_metrics = []
            for block in selected_blocks:
                for prefix in block_prefixes[block]:
                    for col in metric_cols:
                        if prefix.lower() in col.lower():
                            selected_metrics.append(col)
            
            selected_metrics = list(dict.fromkeys(selected_metrics))  # deduplicate preserving order
            
            if selected_metrics:
                fig = plot_segment_heatmap(p, segment_col=seg, metric_cols=selected_metrics[:20], title="Segment behavioral profiles (normalized)")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                
                st.caption(
                    "Features normalized relative to portfolio median. "
                    "Red = above median, Blue = below median. "
                    "Feature blocks balanced before PCA to prevent scale dominance."
                )
        else:
            render_missing("No behavioral metrics available for heatmap.")
    else:
        render_missing("Segment profiles not available.")
    
    # =============================================================================
    # SECTION 4: SEGMENT MOVEMENT
    # =============================================================================
    render_section_label("Segment movement — Transition matrix")
    
    if transition is not None:
        # Determine columns
        from_col = None
        to_col = None
        count_col = None
        
        for c in ["from_segment", "prior_segment", "segment_from", "from"]:
            if c in transition.columns:
                from_col = c
                break
        
        for c in ["to_segment", "current_segment", "segment_to", "to"]:
            if c in transition.columns:
                to_col = c
                break
        
        for c in ["customers", "count", "n", "customer_count"]:
            if c in transition.columns:
                count_col = c
                break
        
        if from_col and to_col and count_col:
            view = st.radio("View", ["Count", "Row % (outflow share)", "Col % (inflow share)"], horizontal=True, key="transition_view")
            
            normalize = "count"
            if view == "Row % (outflow share)":
                normalize = "row_pct"
            elif view == "Col % (inflow share)":
                normalize = "col_pct"
            
            fig = plot_transition_matrix(
                transition,
                from_col=from_col,
                to_col=to_col,
                count_col=count_col,
                title=f"Segment Transitions ({view})",
                normalize=normalize,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Summary insights
            if transition_summary is not None:
                st.markdown("#### Transition summary")
                st.dataframe(transition_summary, use_container_width=True, hide_index=True)
            
            # Largest flows
            st.markdown("#### Largest segment flows")
            top_flows = transition.nlargest(10, count_col)
            st.dataframe(top_flows[[from_col, to_col, count_col]], use_container_width=True, hide_index=True)
        else:
            render_missing("Transition matrix columns not recognized.")
    else:
        render_missing("Segment transition matrix not available.")
    
    # =============================================================================
    # SECTION 5: RFM BENCHMARK
    # =============================================================================
    render_section_label("RFM benchmark — Interpretable reference")
    
    if rfm is not None:
        # Determine columns
        rfm_col = None
        for c in ["rfm_segment", "segment", "rfm_label"]:
            if c in rfm.columns:
                rfm_col = c
                break
        
        if rfm_col:
            rfm_summary = (
                rfm.groupby(rfm_col, dropna=False)
                .size()
                .reset_index()
            )
            rfm_summary.columns = [rfm_col, "customers"]
            
            # Add revenue/value if available in combined
            if combined is not None and rfm_col in combined.columns:
                clv_col = None
                for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
                    if c in combined.columns:
                        clv_col = c
                        break
                
                if clv_col:
                    combined_vals = combined.dropna(subset=[rfm_col, clv_col]).copy()
                    combined_vals[clv_col] = pd.to_numeric(combined_vals[clv_col], errors="coerce")
                    
                    rfm_value = (
                        combined_vals.groupby(rfm_col)
                        .agg(
                            total_value=(clv_col, "sum"),
                            median_value=(clv_col, "median"),
                        )
                        .reset_index()
                    )
                    rfm_summary = rfm_summary.merge(rfm_value, on=rfm_col, how="left")
            
            rfm_summary = rfm_summary.sort_values("customers", ascending=True)
            
            fig = plot_horizontal_bar(
                rfm_summary,
                x_col="customers",
                y_col=rfm_col,
                title="Customer distribution across RFM segments",
                labels={"customers": "Customers", rfm_col: "RFM Segment"},
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.dataframe(rfm_summary, use_container_width=True, hide_index=True)
            
            render_science_card(
                "RFM vs Behavioral Segmentation",
                "RFM provides an **interpretable benchmark** based on three explicit rules: "
                "Recency, Frequency, Monetary. Behavioral segmentation (HDBSCAN) discovers "
                "multidimensional structure across economic intensity, cadence, assortment, "
                "pricing, temporal behavior, lifecycle, and returns. "
                "They serve different purposes: RFM for operational clarity, "
                "behavioral clusters for discovering hidden structure."
            )
        else:
            render_missing("RFM segment column not found.")
    else:
        render_missing("RFM segments not available (run customer_segmentation.py with --rfm-baseline).")
    
    # =============================================================================
    # PCA (moved to supporting evidence)
    # =============================================================================
    with st.expander("Statistical model diagnostics — PCA"):
        if pca is not None:
            pc1 = "PC1" if "PC1" in pca.columns else "pc1"
            pc2 = "PC2" if "PC2" in pca.columns else "pc2"
            ps = "segment" if "segment" in pca.columns else ("cluster" if "cluster" in pca.columns else "segment_name")
            
            if pc1 in pca.columns and pc2 in pca.columns:
                from ..ui.charts import plot_pca_scatter
                fig = plot_pca_scatter(pca, pc1_col=pc1, pc2_col=pc2, color_col=ps)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.caption("PCA coordinates colored by segment. This is supporting evidence for the behavioral segmentation model, not the primary retail story.")
        else:
            render_missing("PCA coordinates not available.")