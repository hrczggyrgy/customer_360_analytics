"""
Segmentation page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..app_components import (
    render_section_label,
    render_science_card,
    render_kpi_card,
)
from ..app_data import get_registry
from ..app_formatting import format_percent, format_count, auto_format
from ..app_charts import (
    base_layout,
    plot_missing,
    plot_pca_scatter,
    plot_segment_heatmap,
    plot_horizontal_bar,
    PLOTLY_CONFIG,
)


def render():
    """Render the Segmentation page."""
    registry = get_registry()
    
    segments = registry.load_dataframe("segmentation")
    profiles = registry.load_dataframe("segmentation", "segment_profiles.csv")
    pca = registry.load_dataframe("segmentation", "pca_coordinates.csv")
    
    if segments is None and profiles is None and pca is None:
        st.warning("No segmentation outputs found. Run customer_segmentation.py first.")
        st.stop()
    
    if segments is not None:
        seg_col = "segment" if "segment" in segments.columns else "segment_name"
        
        seg_summary = (
            segments.groupby(seg_col, dropna=False)
            .size()
            .reset_index(name="customers")
        )
        
        seg_cols = st.columns(4)
        with seg_cols[0]:
            render_kpi_card("Customers", len(segments), formatter="count")
        with seg_cols[1]:
            render_kpi_card("Segments", seg_summary[seg_col].nunique(), formatter="count")
        with seg_cols[2]:
            render_kpi_card("Largest segment", seg_summary["customers"].max(), formatter="count")
        with seg_cols[3]:
            # Noise detection - use numeric segment == -1
            if "segment" in segments.columns:
                noise_mask = segments["segment"] == -1
            else:
                noise_mask = segments[seg_col].astype(str).str.contains("noise", case=False, na=False)
            render_kpi_card("Noise / low-density", noise_mask.mean(), formatter="percent")
        
        st.markdown("#### Segment size")
        seg_summary = seg_summary.sort_values("customers", ascending=True)
        fig = plot_horizontal_bar(
            seg_summary,
            x_col="customers",
            y_col=seg_col,
            title="Customer distribution across behavioral segments",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    left, right = st.columns([1.1, 0.9])
    
    with left:
        st.markdown("#### Latent behavioral map")
        
        if pca is not None:
            pc1 = "PC1" if "PC1" in pca.columns else "pc1"
            pc2 = "PC2" if "PC2" in pca.columns else "pc2"
            ps = "segment_name" if "segment_name" in pca.columns else ("segment" if "segment" in pca.columns else "cluster")
            
            if pc1 in pca.columns and pc2 in pca.columns:
                fig = plot_pca_scatter(pca, pc1_col=pc1, pc2_col=pc2, color_col=ps)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                plot_missing("PCA output was found but PC1 / PC2 were not detected.")
        else:
            plot_missing("PCA coordinates are not available.")
    
    with right:
        st.markdown("#### Behavioral views")
        
        if profiles is not None:
            p = profiles.copy()
            seg = "cluster" if "cluster" in p.columns else ("segment" if "segment" in p.columns else "segment_name")
            exclude = {seg, "customers", "confidence_mean", "share"}
            numeric = p.select_dtypes(include=["number"]).columns.tolist()
            metric_cols = [c for c in numeric if c not in exclude]
            
            if seg and metric_cols:
                choices = metric_cols
                selected = st.multiselect(
                    "Behavioral dimensions",
                    choices,
                    default=choices[:min(8, len(choices))],
                    key="segment_metrics_select",
                )
                
                if selected:
                    fig = plot_segment_heatmap(p, segment_col=seg, metric_cols=selected)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            st.caption(
                "The feature blocks are balanced before PCA so that economic intensity, "
                "cadence, assortment, price, temporal behavior, lifecycle, and returns "
                "each contribute to the latent space rather than letting one block dominate."
            )
        else:
            plot_missing("segment_profiles.csv is not available.")
    
    if profiles is not None:
        st.markdown("#### Segment profile table")
        st.dataframe(profiles, use_container_width=True, hide_index=True)
    
    render_science_card(
        "Feature blocks balanced before PCA",
        "The behavioral features are grouped into seven blocks (economic, cadence, "
        "assortment, pricing, temporal, lifecycle, returns) and each block is normalized "
        "independently before PCA. This prevents any single block from dominating the "
        "latent space simply due to scale."
    )