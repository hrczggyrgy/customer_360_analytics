"""
Recommendations page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_card,
    render_kpi_row,
    render_missing,
    render_formatted_dataframe,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count, format_probability
from ..ui.charts import (
    plot_missing,
    plot_histogram,
    plot_horizontal_bar,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Recommendations page."""
    registry = get_registry()
    
    recs = registry.load_dataframe("recommendations")
    product_metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
    
    if recs is None or recs.empty:
        from ..ui import render_empty_state
        render_empty_state(
            "No recommendations data",
            "No recommendation outputs found.",
            "Run the pipeline",
            "python scripts/recommendation_engine.py",
        )
        st.stop()
    
    # KPIs
    n_customers = recs["Customer ID"].nunique()
    n_recs = len(recs)
    avg_recs = n_recs / n_customers if n_customers > 0 else 0
    
    render_kpi_row([
        {"label": "Customers with recommendations", "value": n_customers, "formatter": "count"},
        {"label": "Total recommendations", "value": n_recs, "formatter": "count"},
        {"label": "Avg per customer", "value": avg_recs, "formatter": "count"},
    ])
    
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
    top_products.columns = ["recommended_product", "Count"]
    
    # Enrich with product info if available
    if product_metrics is not None:
        prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price"]].copy()
        prod_info.columns = ["recommended_product", "Description", "Product Role", "Avg Price"]
        top_products = top_products.merge(prod_info, on="recommended_product", how="left")
    
    # Use formatted dataframe
    render_formatted_dataframe(top_products)
    
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
        
        # Enrich with product info if not already present
        if product_metrics is not None:
            # Check if product columns already exist in recs
            existing_product_cols = {"Description", "product_role", "avg_price", "total_revenue"}
            if not existing_product_cols.issubset(customer_recs.columns):
                prod_cols = ["StockCode", "Description", "product_role", "avg_price", "total_revenue"]
                prod_info = product_metrics[prod_cols].copy()
                prod_info.columns = ["recommended_product", "Description", "Product Role", "Avg Price", "Total Revenue"]
                customer_recs = customer_recs.merge(prod_info, on="recommended_product", how="left")
            else:
                # Rename existing columns for display
                rename_map = {
                    "Description": "Description",
                    "product_role": "Product Role",
                    "avg_price": "Avg Price",
                    "total_revenue": "Total Revenue",
                }
                customer_recs = customer_recs.rename(columns=rename_map)
        
        # Format for display
        display_cols = ["recommended_product", "Description", "Product Role", "score", "reason", "support", "association_lift"]
        if "Avg Price" in customer_recs.columns:
            display_cols.extend(["Avg Price", "Total Revenue"])
        
        display = customer_recs[display_cols].copy()
        
        # Use formatted dataframe
        render_formatted_dataframe(display)
    
    render_science_card(
        "Recommendation methodology",
        "Recommendations are generated using product co-purchase affinity. "
        "For each customer, we identify products frequently bought together with "
        "their past purchases. These are scored by co-occurrence strength and "
        "blended with product popularity as a fallback. The 'reason' field "
        "indicates whether the recommendation came from co-purchase affinity "
        "or popularity fallback."
    )