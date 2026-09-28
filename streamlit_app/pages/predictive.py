"""
Predictive Value (CLV Proxy) page for Retail Customer Intelligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..app_components import render_section_label, render_science_card, render_kpi_card, render_missing
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count
from ..app_charts import (
    base_layout,
    plot_missing,
    plot_histogram_with_marginal,
    plot_uncertainty_band,
    plot_clv_by_segment,
    PLOTLY_CONFIG,
)


def render():
    """Render the Predictive Value (CLV) page."""
    registry = get_registry()
    
    clv = registry.load_dataframe("clv")
    
    # Fallback to combined customer_360
    if clv is None:
        combined = registry.load_dataframe("customer_360")
        if combined is not None:
            clv_col = None
            for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
                if c in combined.columns:
                    clv_col = c
                    break
            if clv_col:
                clv = combined.copy()
    
    if clv is None or clv.empty:
        st.warning("No CLV output was detected. Run clv_analysis.py first.")
        st.stop()
    
    # Find CLV column
    ccol = None
    for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
        if c in clv.columns:
            ccol = c
            break
    
    if ccol is None:
        st.error("CLV file found, but no CLV field was recognized.")
        st.stop()
    
    clv_values = pd.to_numeric(clv[ccol], errors="coerce")
    
    # Uncertainty columns
    lower_col = None
    for c in ["clv_lower", "clv_p10", "clv_lower_bound"]:
        if c in clv.columns:
            lower_col = c
            break
    
    upper_col = None
    for c in ["clv_upper", "clv_p90", "clv_upper_bound"]:
        if c in clv.columns:
            upper_col = c
            break
    
    clean = clv_values.dropna()
    
    # KPIs
    kpis = st.columns(5)
    with kpis[0]:
        render_kpi_card("Customers", clean.size, formatter="count")
    with kpis[1]:
        render_kpi_card("Median CLV Proxy", clean.median(), formatter="currency")
    with kpis[2]:
        render_kpi_card("Mean CLV Proxy", clean.mean(), formatter="currency")
    with kpis[3]:
        render_kpi_card("Total CLV Proxy", clean.sum(), formatter="currency")
    with kpis[4]:
        render_kpi_card("90th percentile", clean.quantile(0.90), formatter="currency")
    
    left, right = st.columns([1.15, 0.85])
    
    with left:
        st.markdown("#### CLV Proxy distribution")
        plot_df = pd.DataFrame({"CLV Proxy": clean})
        
        fig = plot_histogram_with_marginal(clean.values, "CLV Proxy", "Customer value distribution")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with right:
        st.markdown("#### Value uncertainty")
        
        if lower_col and upper_col:
            lower = pd.to_numeric(clv[lower_col], errors="coerce")
            upper = pd.to_numeric(clv[upper_col], errors="coerce")
            
            valid = pd.DataFrame({
                "clv": clv_values,
                "lower": lower,
                "upper": upper,
            }).dropna()
            
            valid = valid.sort_values("clv").reset_index(drop=True)
            
            if len(valid) > 1000:
                valid = valid.iloc[::max(1, len(valid) // 1000)]
            
            fig = plot_uncertainty_band(
                x_vals=np.arange(len(valid)),
                expected=valid["clv"].values,
                lower=valid["lower"].values,
                upper=valid["upper"].values,
                title="Expected value with uncertainty",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing("No lower/upper CLV uncertainty fields were detected in the output.")
    
    # CLV by segment
    seg_col = "segment_name" if "segment_name" in clv.columns else ("segment" if "segment" in clv.columns else None)
    if seg_col:
        st.markdown("#### CLV by behavioral segment")
        
        tmp = clv.copy()
        tmp[ccol] = pd.to_numeric(tmp[ccol], errors="coerce")
        profile = (
            tmp.dropna(subset=[ccol])
            .groupby(seg_col)
            .agg(
                customers=(ccol, "size"),
                total_clv=(ccol, "sum"),
                median_clv=(ccol, "median"),
            )
            .reset_index()
            .sort_values("total_clv", ascending=False)
        )
        
        from ..app_charts import plot_clv_by_segment
        fig = plot_clv_by_segment(
            profile,
            segment_col=seg_col,
            clv_col="total_clv",
            title="Where is customer value concentrated?",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    render_science_card(
        "How to interpret CLV Proxy scientifically",
        "CLV Proxy is a forward economic estimate (Predicted Future Net Revenue), not a historical revenue total. "
        "It combines expected future activity, expected spend, persistence, returns, and model uncertainty. "
        "It is not a true economic CLV (no margin data, no probabilistic lifetime model). "
        "For portfolio decisions, the uncertainty band should be treated as part "
        "of the signal rather than ignored. The term 'CLV' is used as a proxy label only."
    )