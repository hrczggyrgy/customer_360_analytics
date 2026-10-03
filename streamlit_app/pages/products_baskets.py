"""
Products & Baskets workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_row,
    render_missing,
    render_audience_table,
)
from ..ui.scope import apply_scope_to_dataframe
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count, format_probability, format_ratio
from ..ui.charts import (
    plot_horizontal_bar,
    plot_scatter,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Products & Baskets workspace."""
    registry = get_registry()
    
    # Load data
    product_metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
    co_purchase = registry.load_dataframe("product_analytics", "co_purchase_matrix.parquet")
    association_rules = registry.load_dataframe("market_basket", "association_rules.parquet")
    frequent_itemsets = registry.load_dataframe("market_basket", "frequent_itemsets.parquet")
    recommendations = registry.load_dataframe("recommendations")
    combined = registry.load_dataframe("customer_360")
    
    if product_metrics is None and association_rules is None:
        render_missing("Run product_analytics.py and market_basket_analysis.py to populate this workspace.")
        st.stop()
    
    # Apply global scope
    if combined is not None:
        combined = apply_scope_to_dataframe(combined)
    
    # =============================================================================
    # SECTION 1: PRODUCT PORTFOLIO
    # =============================================================================
    render_section_label("Product portfolio")
    
    if product_metrics is not None:
        n_products = len(product_metrics)
        total_revenue = product_metrics["total_revenue"].sum() if "total_revenue" in product_metrics.columns else 0
        n_customers = product_metrics["unique_customers"].sum() if "unique_customers" in product_metrics.columns else 0
        repeat_rate = product_metrics["repeat_customer_rate"].mean() if "repeat_customer_rate" in product_metrics.columns else 0
        
        render_kpi_row([
            {"label": "Products", "value": n_products, "formatter": "count"},
            {"label": "Total Revenue", "value": total_revenue, "formatter": "currency"},
            {"label": "Unique Customers", "value": n_customers, "formatter": "count"},
            {"label": "Avg Repeat Rate", "value": repeat_rate, "formatter": "probability"},
        ])
        
        # Product roles
        if "product_role" in product_metrics.columns:
            role_counts = product_metrics["product_role"].value_counts().reset_index()
            role_counts.columns = ["Product Role", "Count"]
            
            fig = plot_horizontal_bar(
                role_counts,
                x_col="Count",
                y_col="Product Role",
                title="Product role distribution",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        render_missing("Product metrics not available.")
    
    # =============================================================================
    # SECTION 2: BASKET OPPORTUNITY MAP
    # =============================================================================
    render_section_label("Basket opportunity map — Support vs Lift")
    
    if association_rules is not None:
        rules = association_rules.copy()
        
        # Ensure required columns
        req_cols = ["antecedent", "consequent", "support", "confidence", "lift"]
        if all(c in rules.columns for c in req_cols):
            # Sample for performance
            if len(rules) > 5000:
                rules = rules.sample(5000, random_state=42)
            
            fig = plot_scatter(
                rules,
                x_col="support",
                y_col="lift",
                size_col="confidence",
                color_col="product_role" if "product_role" in rules.columns else None,
                title="Basket Association Rules: Support vs Lift (bubble size = Confidence)",
                labels={
                    "support": "Support (transaction frequency)",
                    "lift": "Lift (association strength)",
                    "confidence": "Confidence",
                },
                max_points=5000,
            )
            
            # Add quadrant lines
            fig.add_vline(x=rules["support"].median(), line_dash="dot", line_color="rgba(100,100,100,0.5)")
            fig.add_hline(y=rules["lift"].median(), line_dash="dot", line_color="rgba(100,100,100,0.5)")
            
            # Quadrant annotations
            annotations = [
                dict(x=rules["support"].max()*0.8, y=rules["lift"].max()*0.8, text="High support + High lift<br>Broad commercial association", showarrow=False, font=dict(size=10, color="rgba(100,100,100,0.7)")),
                dict(x=rules["support"].min()*1.1, y=rules["lift"].max()*0.8, text="Low support + High lift<br>Niche affinity", showarrow=False, font=dict(size=10, color="rgba(100,100,100,0.7)")),
                dict(x=rules["support"].max()*0.8, y=rules["lift"].min()*1.1, text="High support + Low lift<br>Common but weak", showarrow=False, font=dict(size=10, color="rgba(100,100,100,0.7)")),
                dict(x=rules["support"].min()*1.1, y=rules["lift"].min()*1.1, text="Low support + Low lift<br>Low priority", showarrow=False, font=dict(size=10, color="rgba(100,100,100,0.7)")),
            ]
            fig.update_layout(annotations=annotations)
            
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Interpretation
            st.caption("""
            **Interpretation:**
            - **High support + High lift**: Products frequently bought together with strong association — broad commercial opportunity
            - **Low support + High lift**: Niche but strong affinity — targeted cross-sell opportunity
            - **High support + Low lift**: Common co-occurrence but weak distinctive association
            - **Low support + Low lift**: Minimal commercial signal
            """)
        else:
            render_missing("Association rules missing required columns.")
    else:
        render_missing("Market basket analysis not available (run market_basket_analysis.py).")
    
    # =============================================================================
    # SECTION 3: ASSOCIATION RULES TABLE
    # =============================================================================
    render_section_label("Association rules — Detailed view")
    
    if association_rules is not None:
        rules = association_rules.copy()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Min support", 0.0, float(rules["support"].max()), 0.0, key="min_support")
        with col2:
            min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.0, key="min_confidence")
        with col3:
            min_lift = st.slider("Min lift", 0.0, float(rules["lift"].max()), 1.0, key="min_lift")
        
        filtered = rules[
            (rules["support"] >= min_support) &
            (rules["confidence"] >= min_confidence) &
            (rules["lift"] >= min_lift)
        ].sort_values("lift", ascending=False)
        
        # Display columns
        display_cols = ["antecedent", "consequent", "support", "confidence", "lift"]
        for c in ["conviction", "leverage", "consequent_description", "product_role"]:
            if c in filtered.columns:
                display_cols.append(c)
        
        st.caption(f"Showing {len(filtered):,} of {len(rules):,} rules")
        
        # Format for display
        display_df = filtered[display_cols].head(200).copy()
        for col in ["support", "confidence"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        for col in ["lift", "conviction", "leverage"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        render_audience_table(
            filtered[display_cols],
            download_filename="association_rules.csv",
            download_label="Export filtered rules",
        )
    else:
        render_missing("Association rules not available.")
    
    # =============================================================================
    # SECTION 4: CUSTOMER AFFINITY VS BASKET ASSOCIATION
    # =============================================================================
    render_section_label("Customer affinity vs Basket association")
    
    st.markdown("""
    **Scientific distinction:**
    
    | Aspect | Basket Association | Customer Affinity |
    |--------|-------------------|-------------------|
    | **Definition** | Items bought together in the **same transaction** | Products purchased by the **same customer across time** |
    | **Grain** | Transaction-level | Customer-level |
    | **Use case** | Bundle offers, shelf placement, POS recommendations | Cross-sell campaigns, lifecycle marketing, personalized recommendations |
    | **Method** | Apriori / FP-Growth on transactions | Co-purchase matrix on customer histories |
    | **This project** | `market_basket_analysis.py` → `association_rules.parquet` | `product_analytics.py` → `co_purchase_matrix.parquet` |
    """)
    
    # Show both if available
    if co_purchase is not None and association_rules is not None:
        st.markdown("#### Top customer affinity pairs (co-purchase matrix)")
        
        # Get top affinity pairs
        cp = co_purchase.copy()
        if "lift" in cp.columns and "product_a" in cp.columns and "product_b" in cp.columns:
            top_affinity = cp.nlargest(20, "lift")[["product_a", "product_b", "support", "confidence", "lift", "cooccurrence"]]
            
            # Enrich with descriptions
            if product_metrics is not None:
                prod_info = product_metrics[["StockCode", "Description"]].copy()
                prod_info.columns = ["product_a", "Description A"]
                top_affinity = top_affinity.merge(prod_info, on="product_a", how="left")
                prod_info.columns = ["product_b", "Description B"]
                top_affinity = top_affinity.merge(prod_info, on="product_b", how="left")
            
            st.dataframe(top_affinity, use_container_width=True, hide_index=True)
        
        st.markdown("#### Top basket association rules (market basket)")
        if "lift" in association_rules.columns:
            top_basket = association_rules.nlargest(20, "lift")[["antecedent", "consequent", "support", "confidence", "lift"]]
            st.dataframe(top_basket, use_container_width=True, hide_index=True)
    elif co_purchase is not None:
        st.caption("Customer affinity data available. Market basket analysis not run.")
    elif association_rules is not None:
        st.caption("Market basket data available. Customer affinity (co-purchase matrix) not available.")
    else:
        render_missing("Neither customer affinity nor market basket data available.")
    
    # =============================================================================
    # SECTION 5: PRODUCT OPPORTUNITY TABLE
    # =============================================================================
    render_section_label("Product opportunity")
    
    if product_metrics is not None:
        pm = product_metrics.copy()
        
        # Select and format columns
        opp_cols = []
        for c in ["StockCode", "Description", "product_role", "total_revenue", "unique_customers", "repeat_customer_rate", "avg_price"]:
            if c in pm.columns:
                opp_cols.append(c)
        
        # Add basket association strength if available
        if association_rules is not None:
            # Convert list/array columns to strings for grouping
            ar = association_rules.copy()
            ar["antecedent_str"] = ar["antecedent"].apply(lambda x: str(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else str(x))
            ar["consequent_str"] = ar["consequent"].apply(lambda x: str(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else str(x))
            
            # Compute max lift per product as antecedent
            antecedent_lift = ar.groupby("antecedent_str")["lift"].max().reset_index()
            antecedent_lift.columns = ["StockCode", "max_basket_lift"]
            
            # Compute max lift per product as consequent
            consequent_lift = ar.groupby("consequent_str")["lift"].max().reset_index()
            consequent_lift.columns = ["StockCode", "max_affinity_lift"]
            
            pm = pm.merge(antecedent_lift, on="StockCode", how="left")
            pm = pm.merge(consequent_lift, on="StockCode", how="left")
            
            if "max_basket_lift" in pm.columns:
                opp_cols.append("max_basket_lift")
            if "max_affinity_lift" in pm.columns:
                opp_cols.append("max_affinity_lift")
        
        display_df = pm[opp_cols].sort_values("total_revenue", ascending=False) if "total_revenue" in pm.columns else pm[opp_cols]
        
        st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
        
        render_audience_table(
            display_df,
            download_filename="product_opportunities.csv",
            download_label="Export product opportunities",
        )
    else:
        render_missing("Product metrics not available.")