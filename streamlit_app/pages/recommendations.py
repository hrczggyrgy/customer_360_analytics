"""
Recommendations page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..app_components import render_section_label, render_science_card, render_kpi_card, render_missing
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count, format_probability
from ..app_charts import (
    base_layout,
    plot_missing,
    plot_histogram,
    plot_horizontal_bar,
    PLOTLY_CONFIG,
)


def render():
    """Render the Recommendations page."""
    registry = get_registry()
    
    recs = registry.load_dataframe("recommendations")
    product_metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
    
    if recs is None or recs.empty:
        st.warning("No recommendation outputs found. Run recommendation_engine.py first.")
        st.stop()
    
    # KPIs
    n_customers = recs["Customer ID"].nunique()
    n_recs = len(recs)
    avg_recs = n_recs / n_customers if n_customers > 0 else 0
    
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Customers with recommendations", n_customers, formatter="count")
    with k2:
        render_kpi_card("Total recommendations", n_recs, formatter="count")
    with k3:
        render_kpi_card("Avg per customer", avg_recs, formatter="count")
    
    # Reason distribution
    if "reason" in recs.columns:
        st.markdown("#### Recommendation strategy distribution")
        reason_counts = recs["reason"].value_counts().reset_index()
        reason_counts.columns = ["Reason", "Count"]
        
        fig = plot_horizontal_bar(
            reason_counts,
            x_col="Count",
            y_col="Reason",
            title="How recommendations were generated",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Score distribution
    if "score" in recs.columns:
        st.markdown("#### Recommendation score distribution")
        fig = plot_histogram(
            pd.to_numeric(recs["score"], errors="coerce").dropna(),
            "Score",
            "Distribution of recommendation scores",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Top recommended products
    st.markdown("#### Top recommended products")
    top_products = recs["recommended_product"].value_counts().head(20).reset_index()
    top_products.columns = ["Product", "Count"]
    
    # Enrich with product info if available
    if product_metrics is not None:
        prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price"]].copy()
        prod_info.columns = ["recommended_product", "Description", "Product Role", "Avg Price"]
        top_products = top_products.merge(prod_info, on="recommended_product", how="left")
    
    st.dataframe(top_products, use_container_width=True, hide_index=True)
    
    # Customer-level view
    st.markdown("#### Customer recommendations")
    
    # Customer selector
    customer_ids = sorted(recs["Customer ID"].dropna().astype(int).unique().tolist())
    
    selected_customer = st.selectbox(
        "Select customer",
        options=customer_ids,
        format_func=lambda x: f"Customer {x}",
        key="rec_customer_select",
    )
    
    if selected_customer:
        customer_recs = recs[recs["Customer ID"] == selected_customer].copy()
        customer_recs = customer_recs.sort_values("score", ascending=False)
        
        # Enrich with product info
        if product_metrics is not None:
            prod_cols = ["StockCode", "Description", "product_role", "avg_price", "total_revenue"]
            prod_info = product_metrics[prod_cols].copy()
            prod_info.columns = ["recommended_product", "Description", "Product Role", "Avg Price", "Total Revenue"]
            customer_recs = customer_recs.merge(prod_info, on="recommended_product", how="left")
        
        # Format for display
        display_cols = ["recommended_product", "Description", "Product Role", "score", "reason", "support", "association_lift"]
        if "Avg Price" in customer_recs.columns:
            display_cols.extend(["Avg Price", "Total Revenue"])
        
        display = customer_recs[display_cols].copy()
        display["score"] = display["score"].apply(lambda x: f"{x:.4f}")
        if "association_lift" in display.columns:
            display["association_lift"] = display["association_lift"].apply(lambda x: f"{x:.2f}")
        if "Avg Price" in display.columns:
            display["Avg Price"] = display["Avg Price"].apply(lambda x: f"£{x:.2f}" if pd.notna(x) else "—")
        if "Total Revenue" in display.columns:
            display["Total Revenue"] = display["Total Revenue"].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "—")
        
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(display, use_container_width=True, hide_index=True)
    
    render_science_card(
        "Recommendation methodology",
        "Recommendations are generated using product co-purchase affinity. "
        "For each customer, we identify products frequently bought together with "
        "their past purchases. These are scored by co-occurrence strength and "
        "blended with product popularity as a fallback. The 'reason' field "
        "indicates whether the recommendation came from co-purchase affinity "
        "or popularity fallback."
    )