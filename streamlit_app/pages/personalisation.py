"""
Personalisation workspace for Retail Customer Intelligence.
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
    render_formatted_dataframe,
    apply_global_scope,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count, format_probability
from ..ui.charts import (
    plot_horizontal_bar,
    plot_histogram,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Personalisation workspace."""
    registry = get_registry()
    
    # Load data
    recs = registry.load_dataframe("recommendations")
    challenger = registry.load_dataframe("recommendations", "recommendation_challenger/challenger_results.parquet")
    challenger_card = registry.load_model_card("recommendations")
    product_metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
    combined = registry.load_dataframe("customer_360")
    
    if recs is None or recs.empty:
        render_missing("Run recommendation_engine.py to populate this workspace.")
        st.stop()
    
    # Apply global scope
    if combined is not None:
        combined = apply_global_scope(combined)
    
    # =============================================================================
    # SECTION 1: RECOMMENDATION COVERAGE
    # =============================================================================
    render_section_label("Recommendation coverage")
    
    n_customers = recs["Customer ID"].nunique()
    n_recs = len(recs)
    avg_recs = n_recs / n_customers if n_customers > 0 else 0
    
    # Catalog coverage
    catalog_products = recs["recommended_product"].nunique()
    total_products = product_metrics["StockCode"].nunique() if product_metrics is not None else 0
    catalog_coverage = catalog_products / total_products if total_products > 0 else 0
    
    render_kpi_row([
        {"label": "Customers with recommendations", "value": n_customers, "formatter": "count"},
        {"label": "Total recommendations", "value": n_recs, "formatter": "count"},
        {"label": "Avg per customer", "value": avg_recs, "formatter": "count", "column_name": "avg_recs"},
        {"label": "Catalog coverage", "value": catalog_coverage, "formatter": "percent"},
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
    
    # =============================================================================
    # SECTION 2: RECOMMENDATION CHALLENGER BENCHMARK
    # =============================================================================
    render_section_label("Recommendation benchmark — Challenger evaluation")
    
    if challenger is not None:
        # Expected columns: model, hit_rate_5, hit_rate_10, recall_10, mrr_10, ndcg_10, coverage
        metric_cols = [c for c in challenger.columns if c not in ["model", "run_id"]]
        
        if metric_cols:
            display_df = challenger[["model"] + metric_cols].copy()
            
            # Format metrics
            for col in metric_cols:
                if "rate" in col.lower() or "recall" in col.lower() or "coverage" in col.lower():
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                elif "mrr" in col.lower() or "ndcg" in col.lower():
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.caption("""
            **Metrics explained:**
            - **Hit Rate @K**: Share of customers with at least 1 relevant recommendation in top-K
            - **Recall @K**: Share of relevant items recovered in top-K
            - **MRR @K**: Mean Reciprocal Rank — rewards higher-ranked relevant items
            - **NDCG @K**: Normalized Discounted Cumulative Gain — rewards ranking quality
            - **Coverage**: Share of catalog appearing in recommendations
            
            Higher is better for all metrics. No automatic 'winner' — choose based on business objective.
            """)
            
            # Model card details
            if challenger_card and "challenger" in str(challenger_card).lower():
                with st.expander("Challenger model card"):
                    st.json(challenger_card)
        else:
            render_missing("Challenger results missing metric columns.")
    else:
        render_missing("Challenger evaluation not available (run recommendation_challenger.py).")
    
    # =============================================================================
    # SECTION 3: RECOMMENDATION QUALITY
    # =============================================================================
    render_section_label("Recommendation quality")
    
    if recs is not None:
        # Score distribution by reason
        if "reason" in recs.columns and "score" in recs.columns:
            st.markdown("**Score distribution by strategy**")
            
            for reason in recs["reason"].unique():
                reason_recs = recs[recs["reason"] == reason]["score"]
                reason_recs = pd.to_numeric(reason_recs, errors="coerce").dropna()
                if len(reason_recs) > 0:
                    st.write(f"**{reason}** ({len(reason_recs):,} recs): "
                           f"median={reason_recs.median():.3f}, "
                           f"mean={reason_recs.mean():.3f}, "
                           f"p90={reason_recs.quantile(0.9):.3f}")
        
        # Catalog coverage detail
        if product_metrics is not None:
            recommended_products = recs["recommended_product"].unique()
            catalog_recs = product_metrics[product_metrics["StockCode"].isin(recommended_products)]
            
            st.markdown("**Recommended product profile**")
            if "product_role" in catalog_recs.columns:
                role_dist = catalog_recs["product_role"].value_counts().reset_index()
                role_dist.columns = ["Product Role", "Count"]
                
                fig = plot_horizontal_bar(
                    role_dist,
                    x_col="Count",
                    y_col="Product Role",
                    title="Product roles in recommendations",
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # =============================================================================
    # SECTION 4: CUSTOMER RECOMMENDATIONS
    # =============================================================================
    render_section_label("Customer recommendations")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Customer selector
        customer_ids = sorted(recs["Customer ID"].dropna().astype(int).unique().tolist())
        selected_customer = st.selectbox(
            "Select customer",
            options=["All"] + customer_ids,
            format_func=lambda x: f"Customer {x}" if x != "All" else "All customers",
            key="rec_customer_select",
        )
    
    with col2:
        # Segment filter
        segment_filter = "All"
        if combined is not None:
            seg_col = "hdbscan_segment" if "hdbscan_segment" in combined.columns else "segment_name"
            if seg_col in combined.columns:
                segments = ["All"] + sorted(combined[seg_col].dropna().unique().tolist())
                segment_filter = st.selectbox("Segment", segments, key="rec_segment_filter")
    
    with col3:
        # Reason filter
        reasons = ["All"] + sorted(recs["reason"].dropna().unique().tolist()) if "reason" in recs.columns else ["All"]
        reason_filter = st.selectbox("Reason", reasons, key="rec_reason_filter")
    
    with col4:
        # Score threshold
        score_threshold = st.number_input("Min score", 0.0, 1.0, 0.0, 0.01, key="rec_score_threshold")
    
    # Apply filters
    filtered_recs = recs.copy()
    
    if selected_customer != "All":
        filtered_recs = filtered_recs[filtered_recs["Customer ID"] == selected_customer]
    
    if segment_filter != "All" and combined is not None:
        seg_col = "hdbscan_segment" if "hdbscan_segment" in combined.columns else "segment_name"
        if seg_col in combined.columns:
            seg_customers = combined[combined[seg_col] == segment_filter]["Customer ID"].unique()
            filtered_recs = filtered_recs[filtered_recs["Customer ID"].isin(seg_customers)]
    
    if reason_filter != "All":
        filtered_recs = filtered_recs[filtered_recs["reason"] == reason_filter]
    
    if score_threshold > 0:
        filtered_recs = filtered_recs[pd.to_numeric(filtered_recs["score"], errors="coerce") >= score_threshold]
    
    # Display
    if not filtered_recs.empty:
        # Enrich with product info
        if product_metrics is not None:
            prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price", "total_revenue"]].copy()
            prod_info.columns = ["recommended_product", "Description", "Product Role", "Avg Price", "Total Revenue"]
            filtered_recs = filtered_recs.merge(prod_info, on="recommended_product", how="left")
        
        # Sort
        if "score" in filtered_recs.columns:
            filtered_recs = filtered_recs.sort_values("score", ascending=False)
        
        # Display columns
        display_cols = ["Customer ID", "recommended_product", "Description", "Product Role", "rank", "score", "reason"]
        for c in ["support", "association_lift", "Avg Price", "Total Revenue"]:
            if c in filtered_recs.columns:
                display_cols.append(c)
        
        available_cols = [c for c in display_cols if c in filtered_recs.columns]
        display = filtered_recs[available_cols].copy()
        
        # Format
        if "score" in display.columns:
            display["score"] = display["score"].apply(lambda x: f"{x:.3f}")
        if "association_lift" in display.columns:
            display["association_lift"] = display["association_lift"].apply(lambda x: f"{x:.2f}")
        if "Avg Price" in display.columns:
            display["Avg Price"] = display["Avg Price"].apply(format_currency)
        if "Total Revenue" in display.columns:
            display["Total Revenue"] = display["Total Revenue"].apply(format_currency)
        
        st.dataframe(display, use_container_width=True, hide_index=True)
        
        # Export
        render_audience_table(
            filtered_recs[available_cols],
            download_filename="customer_recommendations.csv",
            download_label="Export filtered recommendations",
        )
    else:
        render_missing("No recommendations match the current filters.")
    
    render_science_card(
        "Recommendation methodology",
        "Recommendations are generated using product co-purchase affinity. "
        "For each customer, we identify products frequently bought together with "
        "their past purchases. These are scored by co-occurrence strength and "
        "blended with product popularity as a fallback. The 'reason' field "
        "indicates whether the recommendation came from co-purchase affinity "
        "or popularity fallback. The challenger framework evaluates multiple "
        "candidate models (co-purchase+popularity, item-based CF, content-based, "
        "popularity-only) on temporal holdout to ensure out-of-sample validity."
    )