"""
Recommendations page — Product co-purchase based recommendations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_section_label,
    render_kpi_row,
    render_science_card,
    render_missing_data,
    render_customer_selector,
    render_download_button,
)
from app_charts import (
    plot_distribution_histogram,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import format_count, format_score, format_currency


def render_recommendations_page() -> None:
    """Render the Product Recommendations page."""
    registry = get_registry()

    recs = registry.load_dataframe("recommendations", "recommendations")
    model_card = registry.load_model_card("recommendations")
    product_metrics = registry.load_dataframe("product_analytics", "product_metrics")

    if recs is None or recs.empty:
        st.warning("No recommendation outputs found. Run recommendation_engine.py first.")
        st.stop()

    # Page hero
    render_page_hero(
        title="Product Recommendations",
        description="Customer-specific product recommendations based on co-purchase affinity and behavioral signals.",
        kicker="PRODUCT INTELLIGENCE",
    )

    # ---- KPIs ----
    n_customers = recs["Customer ID"].nunique() if "Customer ID" in recs.columns else 0
    n_products = recs["recommended_product"].nunique() if "recommended_product" in recs.columns else 0
    avg_recs_per_cust = len(recs) / n_customers if n_customers > 0 else 0

    kpis = [
        {"label": "Customers with Recommendations", "value": format_count(n_customers)},
        {"label": "Unique Products Recommended", "value": format_count(n_products)},
        {"label": "Avg Recommendations/Customer", "value": f"{avg_recs_per_cust:.1f}"},
        {"label": "Total Recommendations", "value": format_count(len(recs))},
    ]
    render_kpi_row(kpis, columns=4)

    # ---- REASON DISTRIBUTION ----
    if "reason" in recs.columns:
        st.markdown("#### Recommendation Strategy Distribution")
        reason_counts = recs["reason"].value_counts().reset_index()
        reason_counts.columns = ["Reason", "Count"]

        import plotly.express as px
        fig = px.pie(
            reason_counts,
            values="Count",
            names="Reason",
            title="Recommendation Basis",
            color_discrete_sequence=["#315efb", "#218739", "#a56600", "#c53d32", "#7b1fa2"],
        )
        render_chart_responsive(fig, "rec_reasons")

    # ---- SCORE DISTRIBUTION ----
    if "score" in recs.columns:
        st.markdown("#### Recommendation Score Distribution")
        scores = pd.to_numeric(recs["score"], errors="coerce").dropna()

        fig = plot_distribution_histogram(
            scores.tolist(),
            title="Recommendation Score Distribution",
            height=400,
            x_title="Score",
            nbins=30,
        )
        render_chart_responsive(fig, "rec_scores")

    # ---- TOP RECOMMENDED PRODUCTS ----
    if "recommended_product" in recs.columns:
        st.markdown("#### Top Recommended Products")
        top_products = recs["recommended_product"].value_counts().head(20).reset_index()
        top_products.columns = ["Product", "Recommendation Count"]

        # Enrich with product details
        if product_metrics is not None:
            prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price", "total_revenue"]].copy()
            prod_info = prod_info.rename(columns={"StockCode": "Product"})
            top_products = top_products.merge(prod_info, on="Product", how="left")

        st.dataframe(top_products, use_container_width=True, hide_index=True)

    # ---- CUSTOMER-LEVEL FILTERING ----
    st.markdown("---")
    st.markdown("#### Customer-Level Recommendations")

    customer_df = registry.load_dataframe("customer_360", "customer_360_current")
    if customer_df is None:
        customer_df = registry.load_dataframe("customer_360", "customer_360")

    selected_id = render_customer_selector(customer_df, key_prefix="rec")

    if selected_id is not None:
        cust_recs = recs[recs["Customer ID"] == selected_id]
        if not cust_recs.empty:
            st.markdown(f"**Recommendations for Customer {selected_id}**")

            display_cols = ["recommended_product", "score", "reason", "support", "lift"]
            display_cols = [c for c in display_cols if c in cust_recs.columns]

            if product_metrics is not None:
                prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price"]].copy()
                prod_info = prod_info.rename(columns={"StockCode": "recommended_product"})
                cust_recs = cust_recs.merge(prod_info, on="recommended_product", how="left")
                display_cols = ["recommended_product", "Description", "product_role", "avg_price"] + display_cols

            st.dataframe(cust_recs[display_cols].sort_values("score", ascending=False), use_container_width=True, hide_index=True)

            render_download_button(
                cust_recs[display_cols],
                f"customer_{selected_id}_recommendations.csv",
                "Download Recommendations (CSV)",
            )
        else:
            st.info("No recommendations generated for this customer.")

    # ---- DOWNLOAD ALL ----
    st.markdown("---")
    render_download_button(
        recs,
        "all_recommendations.csv",
        "Download All Recommendations (CSV)",
    )

    # ---- SCIENTIFIC CONTEXT ----
    st.markdown("---")
    render_science_card(
        "Recommendation Basis",
        "Recommendations are based on **observed co-purchase affinity** from historical transactions. "
        "A high score indicates products frequently purchased together with this customer's historical basket. "
        "This is not a guarantee of future purchase — it reflects historical association patterns. "
        "The engine combines co-purchase signals with product popularity as a fallback for customers with sparse history.",
    )

    render_science_card(
        "Limitations",
        "• Observational co-purchase only — no causal recommendation effect\n"
        "• No explicit personalization beyond historical basket overlap\n"
        "• No real-time session/context awareness\n"
        "• Popularity fallback may recommend globally popular but irrelevant products\n"
        "• No A/B test validation of recommendation effectiveness",
    )


if __name__ == "__main__":
    render_recommendations_page()