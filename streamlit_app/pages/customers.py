"""
Customers workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_customer_header,
    render_customer_metric_row,
    render_customer_selector,
    render_evidence_table,
    render_peer_benchmark,
    render_audience_table,
    render_missing,
    apply_global_scope,
)
from ..app_data import get_registry
from ..app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_count,
    format_days,
    auto_format,
)
from ..ui.charts import (
    base_layout,
    plot_missing,
    plot_dual_axis_line_bar,
    plot_peer_benchmark,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Customers workspace."""
    registry = get_registry()
    
    # Load data
    customer_df = registry.load_dataframe("customer_360")
    monthly = registry.load_dataframe("customer_360", "customer_month.parquet")
    recommendations = registry.load_dataframe("recommendations")
    decision = registry.load_dataframe("decision_engine")
    segments = registry.load_dataframe("segmentation")
    
    if customer_df is None or customer_df.empty:
        render_missing("Run the pipeline to populate customer data.")
        st.stop()
    
    # Apply global scope
    customer_df = apply_global_scope(customer_df)
    
    id_col = "Customer ID"
    
    # Check for deep linking via query params
    query_params = st.query_params
    default_customer = None
    if "customer_id" in query_params:
        try:
            default_customer = int(query_params["customer_id"])
        except (ValueError, TypeError):
            pass
    
    st.caption(f"Data source: **Customer 360** — `{len(customer_df):,}` customers (after scope filters)")
    
    # Customer selector
    selected_id = render_customer_selector(customer_df, id_col, default_customer=default_customer)
    
    if selected_id is None:
        st.stop()
    
    row = customer_df[customer_df[id_col].astype(int) == int(selected_id)]
    
    if row.empty:
        render_missing("Customer not found.")
        st.stop()
    
    row = row.iloc[0]
    
    # Safe extraction helper
    def safe_get(row, *keys):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None
    
    # Extract key values
    segment_value = safe_get(row, "segment_name", "segment")
    clv_value = safe_get(row, "clv_mean", "clv", "predicted_clv", "customer_clv")
    churn_value = safe_get(row, "churn_probability", "churn_prob", "prob_churn")
    purchase_value = safe_get(row, "next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d")
    action_value = safe_get(row, "recommended_action_capped", "final_action", "recommended_action")
    lifecycle_value = safe_get(row, "lifecycle_state", "customer_state")
    value_tier = safe_get(row, "value_tier")
    engagement_tier = safe_get(row, "engagement_tier")
    priority_tier = safe_get(row, "action_priority_tier")
    rfm_segment = safe_get(row, "rfm_segment")
    
    # =============================================================================
    # CUSTOMER HEADER
    # =============================================================================
    render_customer_header(
        customer_id=int(selected_id),
        segment=segment_value,
        action=action_value,
        clv=pd.to_numeric(clv_value, errors="coerce") if clv_value is not None else None,
        churn_risk=pd.to_numeric(churn_value, errors="coerce") if churn_value is not None else None,
    )
    
    # Additional context badges
    badges = []
    if value_tier:
        badges.append(f"Value: {value_tier}")
    if engagement_tier:
        badges.append(f"Engagement: {engagement_tier}")
    if priority_tier:
        badges.append(f"Priority: {priority_tier}")
    if lifecycle_value:
        badges.append(f"Lifecycle: {lifecycle_value}")
    
    if badges:
        st.markdown(" · ".join(badges))
    
    # =============================================================================
    # KEY METRICS
    # =============================================================================
    revenue_value = safe_get(row, "net_revenue", "revenue", "gross_revenue")
    orders_value = safe_get(row, "orders", "invoice_count")
    recency_value = safe_get(row, "recency_days", "days_since_last_purchase")
    products_value = safe_get(row, "unique_products", "product_breadth")
    
    render_customer_metric_row([
        ("Predicted Future Value", clv_value, "currency"),
        ("Inactivity Risk", churn_value, "probability"),
        ("30d Purchase Propensity", purchase_value, "probability"),
        ("Net Revenue", revenue_value, "currency"),
        ("Orders", orders_value, "count"),
        ("Recency (days)", recency_value, "days"),
        ("Product Breadth", products_value, "count"),
    ])
    
    # =============================================================================
    # CUSTOMER TRAJECTORY + MODEL EVIDENCE
    # =============================================================================
    left, right = st.columns([1.2, 0.8])
    
    with left:
        st.markdown("#### Customer purchase trajectory")
        
        if monthly is not None:
            m_id_col = "Customer ID"
            month_col = "calendar_month"
            revenue_col = "net_revenue"
            order_col = "orders"
            
            if all(c in monthly.columns for c in [m_id_col, month_col, revenue_col]):
                hist = monthly[
                    pd.to_numeric(monthly[m_id_col], errors="coerce").round() == int(selected_id)
                ].copy()
                
                if not hist.empty:
                    hist[month_col] = pd.to_datetime(hist[month_col], errors="coerce")
                    hist[revenue_col] = pd.to_numeric(hist[revenue_col], errors="coerce")
                    hist = hist.dropna(subset=[month_col]).sort_values(month_col)
                    
                    fig = plot_dual_axis_line_bar(
                        hist,
                        x_col=month_col,
                        bar_col=revenue_col,
                        line_col=order_col,
                        title="Observed monthly customer economics",
                        bar_name="Net revenue",
                        line_name="Orders",
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    plot_missing("No monthly event history was found for this customer.")
            else:
                plot_missing("Customer-month columns were not recognized.")
        else:
            plot_missing("Run customer_360.py to populate monthly customer history.")
    
    with right:
        st.markdown("#### Model evidence")
        
        evidence_rows = []
        
        field_map = [
            ("Behavioral segment", ["segment_name", "segment"]),
            ("Lifecycle state", ["lifecycle_state", "customer_state"]),
            ("Value tier", ["value_tier"]),
            ("Engagement tier", ["engagement_tier"]),
            ("Action priority", ["action_priority_tier"]),
            ("Predicted Future Value", ["clv_mean", "clv", "predicted_clv", "customer_clv"]),
            ("CLV lower (p10)", ["clv_lower", "clv_p10", "clv_lower_bound"]),
            ("CLV upper (p90)", ["clv_upper", "clv_p90", "clv_upper_bound"]),
            ("Inactivity risk", ["churn_probability", "churn_prob", "prob_churn"]),
            ("Survival (3m)", ["survival_3m"]),
            ("Survival (6m)", ["survival_6m"]),
            ("Survival (12m)", ["survival_12m"]),
            ("30d Purchase propensity", ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]),
            ("7d Purchase propensity", ["next_purchase_7d_probability"]),
            ("60d Purchase propensity", ["next_purchase_60d_probability"]),
            ("Reactivation probability", ["reactivation_probability"]),
            ("Expected days to next purchase", ["expected_days_to_next_purchase"]),
            ("Recency pressure", ["recency_pressure"]),
            ("Decision confidence", ["decision_confidence"]),
            ("Recommended action", ["recommended_action_capped", "final_action", "recommended_action"]),
            ("Action reason", ["action_reason"]),
        ]
        
        for label, candidates in field_map:
            value = None
            for c in candidates:
                if c in row and pd.notna(row[c]):
                    value = row[c]
                    break
            if value is None:
                continue
            
            if any(kw in label.lower() for kw in ["probability", "risk", "propensity", "survival", "confidence"]):
                display = format_probability(value)
            elif "value" in label.lower() or "clv" in label.lower() or "revenue" in label.lower():
                display = format_currency(value)
            elif "days" in label.lower():
                display = format_days(value)
            else:
                display = auto_format(value)
            
            evidence_rows.append({"Metric": label, "Value": display})
        
        if evidence_rows:
            render_evidence_table(evidence_rows)
        
        render_science_card(
            "How to read this customer",
            "The dashboard intentionally separates **value**, **risk**, and **propensity**. "
            "A high-value customer is not automatically a good intervention target; "
            "the action layer looks for a commercially meaningful signal and a "
            "sufficiently strong model-confidence context."
        )
    
    # =============================================================================
    # CUSTOMER VS PEERS
    # =============================================================================
    st.markdown("---")
    render_section_label("Customer vs peers")
    
    # Build peer group (same segment) or full population
    peer_group = customer_df.copy()
    if segment_value and "segment_name" in peer_group.columns:
        peer_group = peer_group[peer_group["segment_name"] == segment_value]
    
    # Define metrics for comparison
    peer_metrics = []
    
    metric_defs = [
        ("Frequency (Orders)", "orders", True),
        ("Value (Net Revenue)", "net_revenue", True),
        ("Recency (days)", "recency_days", False),
        ("Product Breadth", "unique_products", True),
        ("Repeat Rate", "repeat_product_ratio", True),
        ("AOV", "aov", True),
        ("Return Rate", "unit_return_rate", False),
        ("Purchase Cadence (days)", "median_interpurchase_days", False),
    ]
    
    for label, col, higher_better in metric_defs:
        if col in row and col in peer_group.columns:
            cust_val = pd.to_numeric(row[col], errors="coerce")
            peer_vals = pd.to_numeric(peer_group[col], errors="coerce").dropna()
            
            if pd.notna(cust_val) and len(peer_vals) > 0:
                peer_metrics.append({
                    "label": label,
                    "value": cust_val,
                    "peer_median": peer_vals.median(),
                    "higher_is_better": higher_better,
                    "formatter": "currency" if "revenue" in col.lower() or "aov" in col.lower() else 
                                   "count" if col in ["orders", "unique_products"] else
                                   "days" if "days" in col.lower() else
                                   "ratio" if "ratio" in col.lower() or "rate" in col.lower() else "auto",
                    "column_name": col,
                })
    
    if peer_metrics:
        render_peer_benchmark(peer_metrics, title=f"vs {'Segment peers' if segment_value else 'All customers'}")
    else:
        render_missing("Insufficient data for peer comparison.")
    
    # =============================================================================
    # VALUE/RISK PROFILE
    # =============================================================================
    st.markdown("---")
    render_section_label("Value / Risk / Propensity profile")
    
    profile_cols = st.columns(3)
    
    with profile_cols[0]:
        # Value percentile
        if "clv_mean" in customer_df.columns:
            clv_vals = pd.to_numeric(customer_df["clv_mean"], errors="coerce").dropna()
            cust_clv = pd.to_numeric(clv_value, errors="coerce")
            if pd.notna(cust_clv):
                pct = (clv_vals <= cust_clv).mean() * 100
                st.metric("Predicted Future Value", f"{pct:.0f}th percentile", format_currency(cust_clv))
    
    with profile_cols[1]:
        # Risk percentile
        if "churn_probability" in customer_df.columns:
            churn_vals = pd.to_numeric(customer_df["churn_probability"], errors="coerce").dropna()
            cust_churn = pd.to_numeric(churn_value, errors="coerce")
            if pd.notna(cust_churn):
                pct = (churn_vals <= cust_churn).mean() * 100
                st.metric("Inactivity Risk", f"{pct:.0f}th percentile", format_probability(cust_churn))
    
    with profile_cols[2]:
        # Propensity percentile
        if "next_purchase_30d_probability" in customer_df.columns:
            prop_vals = pd.to_numeric(customer_df["next_purchase_30d_probability"], errors="coerce").dropna()
            cust_prop = pd.to_numeric(purchase_value, errors="coerce")
            if pd.notna(cust_prop):
                pct = (prop_vals <= cust_prop).mean() * 100
                st.metric("30d Purchase Propensity", f"{pct:.0f}th percentile", format_probability(cust_prop))
    
    # Visual bars
    st.markdown("**Relative position**")
    
    bars = []
    if "clv_mean" in customer_df.columns:
        clv_vals = pd.to_numeric(customer_df["clv_mean"], errors="coerce").dropna()
        cust_clv = pd.to_numeric(clv_value, errors="coerce")
        if pd.notna(cust_clv):
            pct = (clv_vals <= cust_clv).mean() * 100
            bars.append(("VALUE", pct))
    
    if "churn_probability" in customer_df.columns:
        churn_vals = pd.to_numeric(customer_df["churn_probability"], errors="coerce").dropna()
        cust_churn = pd.to_numeric(churn_value, errors="coerce")
        if pd.notna(cust_churn):
            pct = (churn_vals <= cust_churn).mean() * 100
            bars.append(("RISK", pct))
    
    if "next_purchase_30d_probability" in customer_df.columns:
        prop_vals = pd.to_numeric(customer_df["next_purchase_30d_probability"], errors="coerce").dropna()
        cust_prop = pd.to_numeric(purchase_value, errors="coerce")
        if pd.notna(cust_prop):
            pct = (prop_vals <= cust_prop).mean() * 100
            bars.append(("PROPENSITY", pct))
    
    for label, pct in bars:
        filled = int(pct / 5)
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        st.markdown(f"`{label:12s} {bar} {pct:.0f}th`")
    
    # =============================================================================
    # CUSTOMER PRODUCT INTELLIGENCE
    # =============================================================================
    st.markdown("---")
    render_section_label("Product intelligence")
    
    prod_left, prod_right = st.columns([1, 1])
    
    with prod_left:
        st.markdown("**Top purchased products**")
        # Get customer's purchase history from monthly
        if monthly is not None:
            m_id_col = "Customer ID"
            if m_id_col in monthly.columns:
                cust_monthly = monthly[
                    pd.to_numeric(monthly[m_id_col], errors="coerce").round() == int(selected_id)
                ]
                if "product_breadth" in cust_monthly.columns or "unique_products" in cust_monthly.columns:
                    # Show from customer_360 current
                    prod_cols = [c for c in row.index if "top_product" in c.lower() or "share_top" in c.lower()]
                    if prod_cols:
                        for col in sorted(prod_cols)[:5]:
                            val = row[col]
                            if pd.notna(val) and val > 0:
                                st.write(f"• {col.replace('share_top_product_pool_', '')}: {val:.1%}")
                    else:
                        st.caption("Product share data not available in current snapshot.")
                else:
                    st.caption("Monthly product data not available.")
            else:
                st.caption("No purchase history found.")
        else:
            st.caption("Monthly data not loaded.")
    
    with prod_right:
        st.markdown("**Recommendations**")
        if recommendations is not None:
            cust_recs = recommendations[recommendations["Customer ID"] == selected_id]
            if not cust_recs.empty:
                display_cols = ["recommended_product", "score", "reason", "supporting_products", "association_lift"]
                available_cols = [c for c in display_cols if c in cust_recs.columns]
                if available_cols:
                    disp = cust_recs[available_cols].head(5).copy()
                    # Enrich with product info
                    product_metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
                    if product_metrics is not None:
                        prod_info = product_metrics[["StockCode", "Description", "product_role"]].copy()
                        prod_info.columns = ["recommended_product", "Description", "Product Role"]
                        disp = disp.merge(prod_info, on="recommended_product", how="left")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.caption("Recommendation columns not recognized.")
            else:
                st.caption("No recommendations for this customer.")
        else:
            st.caption("Recommendations not available.")
    
    # =============================================================================
    # CUSTOMER ACTION CARD
    # =============================================================================
    st.markdown("---")
    render_section_label("Customer opportunity")
    
    if decision is not None:
        cust_decision = decision[decision["Customer ID"] == selected_id]
        if not cust_decision.empty:
            drow = cust_decision.iloc[0]
            
            action = safe_get(drow, "recommended_action_capped", "final_action", "recommended_action")
            priority = safe_get(drow, "priority_score")
            confidence = safe_get(drow, "decision_confidence")
            reason = safe_get(drow, "action_reason")
            expected_value = safe_get(drow, "expected_value_proxy")
            eligible = safe_get(drow, "eligible")
            suppressed = safe_get(drow, "suppressed")
            suppression_reason = safe_get(drow, "suppression_reason")
            
            st.markdown(f"""
            <div style='
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius-lg);
                padding: 20px;
            '>
                <div style='display:flex;gap:16px;flex-wrap:wrap;'>
                    <div style='flex:1;min-width:200px;'>
                        <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;'>Recommended Action</div>
                        <div style='color:var(--text-primary);font-size:var(--text-xl);font-weight:700;margin-top:4px;'>{action or '—'}</div>
                    </div>
                    <div style='flex:1;min-width:200px;'>
                        <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;'>Priority Score</div>
                        <div style='color:var(--text-primary);font-size:var(--text-xl);font-weight:700;margin-top:4px;'>{format_score(priority) if priority is not None else '—'}</div>
                    </div>
                    <div style='flex:1;min-width:200px;'>
                        <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;'>Decision Confidence</div>
                        <div style='color:var(--text-primary);font-size:var(--text-xl);font-weight:700;margin-top:4px;'>{format_probability(confidence) if confidence is not None else '—'}</div>
                    </div>
                    <div style='flex:1;min-width:200px;'>
                        <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;'>Expected Value Proxy</div>
                        <div style='color:var(--text-primary);font-size:var(--text-xl);font-weight:700;margin-top:4px;'>{format_currency(expected_value) if expected_value is not None else '—'}</div>
                    </div>
                </div>
                <div style='margin-top:16px;padding-top:16px;border-top:1px solid var(--border);'>
                    <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;'>Reason</div>
                    <div style='color:var(--text-primary);margin-top:4px;'>{reason or 'No reason recorded'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if suppressed:
                st.warning(f"⚠️ Suppressed: {suppression_reason or 'Capacity/eligibility constraints'}")
            elif eligible is False:
                st.info("ℹ️ Not eligible for current action allocation")
        else:
            st.caption("No decision engine record for this customer.")
    else:
        render_missing("Decision engine data not available.")