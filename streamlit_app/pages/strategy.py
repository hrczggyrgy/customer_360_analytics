"""
Strategy workspace for Retail Customer Intelligence.
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
    render_insight_row,
    render_missing,
    render_audience_table,
    apply_global_scope,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_probability, format_percent, format_count, auto_format
from ..ui.charts import (
    plot_concentration_curve,
    plot_action_allocation,
    plot_opportunity_matrix,
    plot_value_concentration_curve,
    PLOTLY_CONFIG,
    apply_plotly_theme,
)


def render() -> None:
    """Render the Strategy workspace."""
    registry = get_registry()
    
    # Load data
    combined = registry.load_dataframe("customer_360")
    segments = registry.load_dataframe("segmentation")
    action_summary = registry.load_dataframe("decision_engine", "action_summary.csv")
    decision = registry.load_dataframe("decision_engine")
    
    if combined is None:
        render_missing("Run customer_360.py to populate the strategy view.")
        st.stop()
    
    # Apply global scope
    combined = apply_global_scope(combined)
    
    # =============================================================================
    # SECTION 1: PORTFOLIO HEADLINE KPIs
    # =============================================================================
    n_customers = len(combined)
    
    # CLV total
    clv_total = None
    clv_col = None
    for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
        if c in combined.columns:
            clv_col = c
            break
    if clv_col:
        clv_total = pd.to_numeric(combined[clv_col], errors="coerce").sum()
    
    # Mean churn risk
    churn_mean = None
    churn_col = None
    for c in ["churn_probability", "churn_prob", "prob_churn"]:
        if c in combined.columns:
            churn_col = c
            break
    if churn_col:
        churn_mean = pd.to_numeric(combined[churn_col], errors="coerce").mean()
    
    # Mean next-purchase probability
    np_mean = None
    np_col = None
    for c in ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]:
        if c in combined.columns:
            np_col = c
            break
    if np_col:
        np_mean = pd.to_numeric(combined[np_col], errors="coerce").mean()
    
    # Value at risk (high churn risk * high CLV)
    value_at_risk = None
    if clv_col and churn_col:
        high_risk = pd.to_numeric(combined[churn_col], errors="coerce") >= 0.7
        high_clv = pd.to_numeric(combined[clv_col], errors="coerce")
        value_at_risk = high_clv[high_risk].sum()
    
    # 30-day purchase opportunity (high propensity * CLV)
    purchase_opportunity = None
    if clv_col and np_col:
        high_prop = pd.to_numeric(combined[np_col], errors="coerce") >= 0.5
        high_clv = pd.to_numeric(combined[clv_col], errors="coerce")
        purchase_opportunity = high_clv[high_prop].sum()
    
    render_kpi_row([
        {"label": "Customers", "value": n_customers, "formatter": "count"},
        {"label": "Predicted Future Value", "value": clv_total, "formatter": "currency"},
        {"label": "Mean Inactivity Risk", "value": churn_mean, "formatter": "probability"},
        {"label": "Mean Purchase Propensity", "value": np_mean, "formatter": "probability"},
        {"label": "Value at Risk", "value": value_at_risk, "formatter": "currency"},
        {"label": "Purchase Opportunity", "value": purchase_opportunity, "formatter": "currency"},
    ])
    
    # =============================================================================
    # SECTION 2: WHAT CHANGED? - Customer movement
    # =============================================================================
    render_section_label("What changed? — Customer movement")
    
    if "lifecycle_state" in combined.columns and "customer_state" in combined.columns:
        # Current vs prior lifecycle comparison
        current_lifecycle = combined["lifecycle_state"].value_counts().reset_index()
        current_lifecycle.columns = ["Lifecycle", "Customers"]
        
        prior_lifecycle = combined["customer_state"].value_counts().reset_index()
        prior_lifecycle.columns = ["Lifecycle", "Customers"]
        
        # Merge for comparison
        movement = current_lifecycle.merge(prior_lifecycle, on="Lifecycle", how="outer", suffixes=("_current", "_prior")).fillna(0)
        movement["Net change"] = movement["Customers_current"] - movement["Customers_prior"]
        
        left, right = st.columns(2)
        
        with left:
            fig = plot_action_allocation(
                current_lifecycle,
                action_col="Lifecycle",
                count_col="Customers",
                title="Current lifecycle distribution",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        with right:
            fig = plot_action_allocation(
                prior_lifecycle,
                action_col="Lifecycle",
                count_col="Customers",
                title="Prior lifecycle distribution",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Net movement table
        st.markdown("#### Net customer movement")
        display_movement = movement[["Lifecycle", "Customers_current", "Customers_prior", "Net change"]].copy()
        display_movement.columns = ["Lifecycle", "Current", "Prior", "Net change"]
        display_movement = display_movement.sort_values("Net change", ascending=False)
        st.dataframe(display_movement, use_container_width=True, hide_index=True)
    else:
        render_missing("Lifecycle state data not available for movement analysis.")
    
    # =============================================================================
    # SECTION 3: WHERE IS THE VALUE? - Concentration curve
    # =============================================================================
    render_section_label("Where is the value? — Concentration")
    
    if clv_col:
        clv_values = pd.to_numeric(combined[clv_col], errors="coerce").dropna()
        
        left, right = st.columns([1.2, 0.8])
        
        with left:
            fig = plot_value_concentration_curve(
                clv_values.values,
                title="Where is customer value concentrated?",
                annotate_thresholds=[0.1, 0.2, 0.4],
                label_prefix="Top",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        with right:
            # Top decile/quintile callouts
            total = clv_values.sum()
            sorted_vals = clv_values.sort_values(ascending=False)
            
            top_10 = sorted_vals.head(int(len(sorted_vals) * 0.1)).sum() / total * 100
            top_20 = sorted_vals.head(int(len(sorted_vals) * 0.2)).sum() / total * 100
            top_40 = sorted_vals.head(int(len(sorted_vals) * 0.4)).sum() / total * 100
            
            render_insight(
                label="VALUE CONCENTRATION",
                headline=f"Top 10% of customers hold {top_10:.0f}% of predicted future value",
                detail=f"Top 20% → {top_20:.0f}%  |  Top 40% → {top_40:.0f}%",
                evidence=f"Based on {len(clv_values):,} customers with CLV proxy values",
                severity="high",
            )
            
            render_science_card(
                "How to read this",
                "The concentration curve shows the cumulative share of predicted future value "
                "accounted for by the top X% of customers. A steep curve means value is highly "
                "concentrated in a small customer population.",
            )
    else:
        render_missing("CLV data not available for concentration analysis.")
    
    # =============================================================================
    # SECTION 4: HERO VISUAL - OPPORTUNITY MATRIX
    # =============================================================================
    render_section_label("Where is the opportunity? — Propensity vs Risk matrix")
    
    # Prepare data for opportunity matrix
    if decision is not None:
        plot_df = apply_global_scope(decision)
        
        # Ensure required columns
        req_cols = ["clv_mean", "churn_probability", "next_purchase_30d_probability", "recommended_action_capped"]
        available = all(c in plot_df.columns for c in req_cols)
        
        if available:
            # Sample for performance
            if len(plot_df) > 5000:
                plot_df = plot_df.sample(5000, random_state=42)
            
            fig = plot_opportunity_matrix(
                plot_df,
                x_col="next_purchase_30d_probability",
                y_col="churn_probability",
                size_col="clv_mean",
                color_col="recommended_action_capped",
                x_label="30-day purchase propensity",
                y_label="Inactivity risk",
                title="Opportunity Matrix: Propensity vs Risk (bubble size = Predicted Future Value)",
                quadrant_labels={
                    "top_right": "PROTECT\nHigh value, high risk",
                    "top_left": "NURTURE\nLower value, high risk",
                    "bottom_right": "GROW\nHigh value, low risk",
                    "bottom_left": "ACCELERATE\nLower value, low risk",
                },
                hover_cols=["Customer ID", "segment_name", "priority_score"],
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Quadrant summary
            st.markdown("#### Quadrant summary")
            x_med = plot_df["next_purchase_30d_probability"].median()
            y_med = plot_df["churn_probability"].median()
            
            quadrants = {
                "PROTECT (High value, high risk)": plot_df[
                    (plot_df["next_purchase_30d_probability"] >= x_med) & 
                    (plot_df["churn_probability"] >= y_med)
                ],
                "GROW (High value, low risk)": plot_df[
                    (plot_df["next_purchase_30d_probability"] >= x_med) & 
                    (plot_df["churn_probability"] < y_med)
                ],
                "NURTURE (Lower value, high risk)": plot_df[
                    (plot_df["next_purchase_30d_probability"] < x_med) & 
                    (plot_df["churn_probability"] >= y_med)
                ],
                "ACCELERATE (Lower value, low risk)": plot_df[
                    (plot_df["next_purchase_30d_probability"] < x_med) & 
                    (plot_df["churn_probability"] < y_med)
                ],
            }
            
            quad_cols = st.columns(4)
            for (label, qdf), col in zip(quadrants.items(), quad_cols):
                with col:
                    n = len(qdf)
                    val = qdf["clv_mean"].sum() if "clv_mean" in qdf.columns else 0
                    st.metric(label, f"{n:,} customers", f"£{val:,.0f}")
        else:
            render_missing("Decision engine data required for opportunity matrix.")
    else:
        render_missing("Run decision_engine.py to populate the opportunity matrix.")
    
    # =============================================================================
    # SECTION 5: PRIORITY OPPORTUNITY TABLE
    # =============================================================================
    render_section_label("Priority opportunities")
    
    if decision is not None:
        plot_df = apply_global_scope(decision)
        
        # Build opportunity cohorts
        opportunities = []
        
        # High-value at-risk
        if all(c in plot_df.columns for c in ["clv_mean", "churn_probability", "recommended_action_capped", "segment_name"]):
            hv_at_risk = plot_df[
                (pd.to_numeric(plot_df["clv_mean"], errors="coerce") >= plot_df["clv_mean"].quantile(0.75)) &
                (pd.to_numeric(plot_df["churn_probability"], errors="coerce") >= 0.7)
            ]
            if len(hv_at_risk) > 0:
                opportunities.append({
                    "Opportunity": "High-value at-risk",
                    "Customers": len(hv_at_risk),
                    "Value": hv_at_risk["clv_mean"].sum(),
                    "Value Share": hv_at_risk["clv_mean"].sum() / plot_df["clv_mean"].sum() * 100,
                    "Avg Risk": hv_at_risk["churn_probability"].mean(),
                    "Avg Propensity": hv_at_risk.get("next_purchase_30d_probability", pd.Series([0])).mean(),
                    "Primary Segment": hv_at_risk["segment_name"].mode().iloc[0] if not hv_at_risk["segment_name"].mode().empty else "—",
                    "Recommended Attention": "Protect / Retain",
                })
            
            # High-value ready-to-buy
            hv_ready = plot_df[
                (pd.to_numeric(plot_df["clv_mean"], errors="coerce") >= plot_df["clv_mean"].quantile(0.75)) &
                (pd.to_numeric(plot_df.get("next_purchase_30d_probability", 0), errors="coerce") >= 0.5)
            ]
            if len(hv_ready) > 0:
                opportunities.append({
                    "Opportunity": "High-value ready-to-buy",
                    "Customers": len(hv_ready),
                    "Value": hv_ready["clv_mean"].sum(),
                    "Value Share": hv_ready["clv_mean"].sum() / plot_df["clv_mean"].sum() * 100,
                    "Avg Risk": hv_ready["churn_probability"].mean(),
                    "Avg Propensity": hv_ready.get("next_purchase_30d_probability", pd.Series([0])).mean(),
                    "Primary Segment": hv_ready["segment_name"].mode().iloc[0] if not hv_ready["segment_name"].mode().empty else "—",
                    "Recommended Attention": "Accelerate / Cross-sell",
                })
            
            # Valuable dormant
            dormant = plot_df[
                (pd.to_numeric(plot_df["clv_mean"], errors="coerce") >= plot_df["clv_mean"].quantile(0.5)) &
                (plot_df.get("lifecycle_state", "").isin(["dormant", "stale"]))
            ]
            if len(dormant) > 0:
                opportunities.append({
                    "Opportunity": "Valuable dormant",
                    "Customers": len(dormant),
                    "Value": dormant["clv_mean"].sum(),
                    "Value Share": dormant["clv_mean"].sum() / plot_df["clv_mean"].sum() * 100,
                    "Avg Risk": dormant["churn_probability"].mean(),
                    "Avg Propensity": dormant.get("next_purchase_30d_probability", pd.Series([0])).mean(),
                    "Primary Segment": dormant["segment_name"].mode().iloc[0] if not dormant["segment_name"].mode().empty else "—",
                    "Recommended Attention": "Reactivate",
                })
            
            # Active cross-sell
            active = plot_df[
                (plot_df.get("lifecycle_state", "").isin(["active_repeat"])) &
                (pd.to_numeric(plot_df.get("next_purchase_30d_probability", 0), errors="coerce") >= 0.3)
            ]
            if len(active) > 0:
                opportunities.append({
                    "Opportunity": "Active cross-sell",
                    "Customers": len(active),
                    "Value": active["clv_mean"].sum(),
                    "Value Share": active["clv_mean"].sum() / plot_df["clv_mean"].sum() * 100,
                    "Avg Risk": active["churn_probability"].mean(),
                    "Avg Propensity": active.get("next_purchase_30d_probability", pd.Series([0])).mean(),
                    "Primary Segment": active["segment_name"].mode().iloc[0] if not active["segment_name"].mode().empty else "—",
                    "Recommended Attention": "Cross-sell / Nurture",
                })
        
        if opportunities:
            opp_df = pd.DataFrame(opportunities)
            render_audience_table(
                opp_df,
                columns=["Opportunity", "Customers", "Value", "Value Share", "Avg Risk", "Avg Propensity", "Primary Segment", "Recommended Attention"],
                download_filename="strategy_opportunities.csv",
                download_label="Export opportunities",
            )
        else:
            render_missing("Insufficient data to build opportunity cohorts.")
    else:
        render_missing("Decision engine data required for priority opportunities.")
    
    # =============================================================================
    # SECTION 6: STRATEGIC INSIGHT PANEL
    # =============================================================================
    render_section_label("Strategic insights")
    
    insights = []
    
    if clv_col:
        clv_values = pd.to_numeric(combined[clv_col], errors="coerce").dropna()
        total = clv_values.sum()
        sorted_vals = clv_values.sort_values(ascending=False)
        top_20_share = sorted_vals.head(int(len(sorted_vals) * 0.2)).sum() / total * 100
        
        insights.append({
            "label": "VALUE CONCENTRATION",
            "headline": f"Value is highly concentrated: top 20% hold {top_20_share:.0f}% of predicted future value",
            "detail": f"Median CLV: {format_currency(clv_values.median())}  |  Mean: {format_currency(clv_values.mean())}",
            "evidence": f"CLV proxy distribution across {len(clv_values):,} customers",
            "severity": "high",
        })
    
    if churn_col and "segment_name" in combined.columns:
        # Risk concentration by segment
        churn_vals = pd.to_numeric(combined[churn_col], errors="coerce")
        seg_risk = combined.groupby("segment_name")[churn_col].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()).sort_values(ascending=False)
        if len(seg_risk) > 0:
            top_seg = seg_risk.index[0]
            top_risk = seg_risk.iloc[0]
            insights.append({
                "label": "RISK CONCENTRATION",
                "headline": f"Inactivity risk concentrated in '{top_seg}' segment ({format_probability(top_risk)})",
                "detail": f"Segment average risk ranges from {format_probability(seg_risk.min())} to {format_probability(seg_risk.max())}",
                "evidence": f"Churn probability by behavioral segment (n={len(seg_risk)} segments)",
                "severity": "high",
            })
    
    if "lifecycle_state" in combined.columns:
        lifecycle_dist = combined["lifecycle_state"].value_counts(normalize=True)
        if "at_risk" in lifecycle_dist.index:
            at_risk_pct = lifecycle_dist["at_risk"] * 100
            insights.append({
                "label": "LIFECYCLE SHIFT",
                "headline": f"{at_risk_pct:.0f}% of customers are in 'at risk' lifecycle state",
                "detail": f"Active repeat: {lifecycle_dist.get('active_repeat', 0)*100:.0f}%  |  Dormant: {lifecycle_dist.get('dormant', 0)*100:.0f}%  |  New: {lifecycle_dist.get('new_single_order', 0)*100:.0f}%",
                "evidence": f"Customer lifecycle state distribution (n={len(combined):,})",
                "severity": "medium",
            })
    
    if action_summary is not None:
        action_col = None
        for c in ["final_action", "recommended_action", "action"]:
            if c in action_summary.columns:
                action_col = c
                break
        customer_col = None
        for c in ["customers", "customer_count"]:
            if c in action_summary.columns:
                customer_col = c
                break
        if action_col and customer_col:
            protect_pct = action_summary[action_summary[action_col].str.contains("protect", case=False, na=False)][customer_col].sum() / action_summary[customer_col].sum() * 100
            monitor_pct = action_summary[action_summary[action_col].str.contains("monitor", case=False, na=False)][customer_col].sum() / action_summary[customer_col].sum() * 100
            insights.append({
                "label": "ACTION BALANCE",
                "headline": f"Protection actions cover {protect_pct:.0f}% of customers; {monitor_pct:.0f}% in monitor",
                "detail": "Decision engine allocates capacity across protect, reactivate, accelerate, cross-sell, nurture, monitor",
                "evidence": "Decision engine action allocation summary",
                "severity": "info",
            })
    
    if insights:
        render_insight_row(insights, max_cols=2)
    else:
        render_missing("Insufficient data to generate strategic insights.")